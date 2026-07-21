#!/usr/bin/env python3
"""
Collide-2V adaptation of Maciej's `eval_quality.py` composite latent-quality score.

`eval_quality.py` was written for Maciej's own HH_4b anomaly challenge (a single
`test.parquet` + `labels.csv`, `SM_BKGS = {Wjet, DY, QCD, ttbar}`, anomaly = HH_4b).
This project's Collide-2V dataset has a different schema (per-class parquet trees,
classes MinBias/Top/QCD/DY, `L1T_PUPPIPart_*` branches) and no HH_4b analogue, so
this module reuses `eval_quality.py`'s scoring core (alignment loss, participation
ratio, GMM anomaly gap, Mahalanobis distance, geometric-mean composite) unchanged
via import, and supplies Collide-2V-specific data loading and background/anomaly
framing on top. `eval_quality.py` itself is not modified.

Background/anomaly convention (confirmed with the project owner): MinBias is the
background/reference class (GMM + Mahalanobis reference fit on it, PCA whitening
fit on it), DY is the anomaly/signal-like class scored against that reference.
Top and QCD are still streamed (so alignment loss and participation ratio, which
run over the whole diverse eval set, benefit from sample diversity) but are not
part of the GMM-gap/Mahalanobis terms -- this mirrors the project's actual
DY-vs-MinBias research question rather than Maciej's HH_4b/SM-background framing.

Usage (CLI):
    python eval_quality_collide2v.py /path/to/mha_nodes_aligned_best_for_eval.pt \\
        --n-events-per-class 2000 --output scores.json

Usage (module):
    from ngt_model.evaluation.eval_quality_collide2v import EvalCollide2V
    ev = EvalCollide2V(model, n_events_per_class=2000)
    print(ev.score, ev.scores)
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch
from sklearn.covariance import LedoitWolf
from sklearn.mixture import GaussianMixture

from ngt_model.evaluation.eval_quality import (
    SEED,
    SEQ_LEN,
    EPS,
    _build_ref_ranges,
    _infer_latent_dim,
    _load_model,
    alignment_loss,
    composite_geometric_mean,
    create_augmented_pairs,
    extract_latents,
    fit_pca_whiten,
    normalize_metrics,
    participation_ratio,
)
from ngt_model.datasets.collide2v_data import (
    CLASS_NAMES,
    SUPPORTED_CLASS_NAMES,
    Collide2VConfig,
    Collide2VLoader,
    build_pid_mapping,
    build_split_indices,
    compute_normalization_constants,
    preprocess_dataset,
)

BACKGROUND_CLASSES = ["minbias"]
ANOMALY_CLASS = "dy"

GMM_KWARGS = dict(
    n_components=2,
    covariance_type="full",
    random_state=SEED,
    max_iter=500,
    reg_covar=1e-5,
    init_params="kmeans",
)


# ═══════════════════════════════════════════════════════════════════════════════
#  metrics specific to the MinBias(background)-vs-DY(anomaly) convention
# ═══════════════════════════════════════════════════════════════════════════════

def _expected_input_features(model: torch.nn.Module) -> int | None:
    """Return the feature dimension expected by the saved encoder."""
    if hasattr(model, "input_proj") and hasattr(model.input_proj, "in_features"):
        return int(model.input_proj.in_features)
    if hasattr(model, "encoder"):
        return _expected_input_features(model.encoder)
    return None


def _align_feature_dim(X: np.ndarray, expected_features: int | None, name: str) -> np.ndarray:
    """Pad or truncate the last feature axis to match the saved model input layer."""
    if expected_features is None:
        return X

    actual_features = X.shape[-1]
    if actual_features == expected_features:
        return X

    if actual_features < expected_features:
        pad = expected_features - actual_features
        print(
            f"  WARNING: {name} has {actual_features} features but model expects "
            f"{expected_features}; padding {pad} zero feature(s). This usually means "
            "the small eval sample missed one or more PID categories."
        )
        return np.pad(X, ((0, 0), (0, 0), (0, pad)), mode="constant")

    print(
        f"  WARNING: {name} has {actual_features} features but model expects "
        f"{expected_features}; truncating extra feature(s)."
    )
    return X[:, :, :expected_features]


def gmm_anomaly_gap_collide2v(Z_whitened, anomaly_mask, bg_mask, background_label: str, anomaly_label: str):
    """Fits one GMM on the selected background latents and scores the anomaly class."""
    Z_anom = Z_whitened[anomaly_mask]
    Z_bg = Z_whitened[bg_mask]

    if len(Z_bg) < 20:
        print(f"  WARNING: too few {background_label} events for GMM, skipping")
        return 0.0, {}

    gmm = GaussianMixture(**GMM_KWARGS)
    gmm.fit(Z_bg)

    nll_anomaly = -gmm.score_samples(Z_anom)
    avg_nll = float(np.mean(nll_anomaly))
    nll_bg = -gmm.score_samples(Z_bg)
    avg_nll_bg = float(np.mean(nll_bg))

    print(f"  GMM {background_label}->{anomaly_label:40s}  "
          f"NLL(anom)={avg_nll:.2f}  NLL(bg)={avg_nll_bg:.2f}")
    return avg_nll, {anomaly_label: avg_nll}


def mahalanobis_anomaly_collide2v(Z_whitened, anomaly_mask, bg_mask):
    """Mahalanobis distance between the anomaly and background centroids."""
    Z_anom = Z_whitened[anomaly_mask]
    Z_bg = Z_whitened[bg_mask]
    if len(Z_anom) < 2 or len(Z_bg) < 2:
        return 0.0

    mu_a = np.mean(Z_anom, axis=0)
    mu_b = np.mean(Z_bg, axis=0)
    cov = LedoitWolf().fit(Z_bg).covariance_
    cov_inv = np.linalg.pinv(cov)
    delta = mu_a - mu_b
    return float(np.sqrt(delta @ cov_inv @ delta))


# ═══════════════════════════════════════════════════════════════════════════════
#  Eval class
# ═══════════════════════════════════════════════════════════════════════════════

class EvalCollide2V:
    def __init__(
        self,
        model,
        n_events_per_class: int = 2000,
        data_source: str = "collide2v_updated",
        feature_mode: str = "l1_only",
        candidate_selection: str = "top_pt",
        n_constituents: int = SEQ_LEN,
        device=None,
        output_path=None,
        collide2v_cfg: Collide2VConfig | None = None,
        background_classes: list[str] | None = None,
        anomaly_class: str = ANOMALY_CLASS,
    ):
        if isinstance(model, str):
            self.model_ = _load_model(model, device)
        elif isinstance(model, torch.nn.Module):
            self.model_ = model
        else:
            raise TypeError(f"model must be str (path) or nn.Module, got {type(model)}")

        self.device = next(self.model_.parameters()).device
        self.n_events_per_class = n_events_per_class
        self.background_classes = list(background_classes or BACKGROUND_CLASSES)
        self.anomaly_class = anomaly_class
        if self.anomaly_class in self.background_classes:
            raise ValueError(
                f"anomaly_class={self.anomaly_class} must not also be in background_classes={self.background_classes}"
            )
        unknown = [
            cls for cls in [*self.background_classes, self.anomaly_class]
            if cls not in SUPPORTED_CLASS_NAMES
        ]
        if unknown:
            raise ValueError(f"Unsupported eval classes {unknown}; supported classes are {SUPPORTED_CLASS_NAMES}")
        class_names = []
        for cls in [*self.background_classes, self.anomaly_class]:
            if cls not in class_names:
                class_names.append(cls)
        self.cfg = collide2v_cfg or Collide2VConfig(
            data_source=data_source,
            n_constituents=n_constituents,
            feature_mode=feature_mode,
            candidate_selection=candidate_selection,
            class_names=class_names,
        )
        if self.cfg.class_names != class_names:
            missing = [cls for cls in class_names if cls not in self.cfg.class_names]
            if missing:
                raise ValueError(
                    f"collide2v_cfg.class_names={self.cfg.class_names} is missing required eval classes {missing}"
                )

        n_params = sum(p.numel() for p in self.model_.parameters())
        print(f"EvalCollide2V: model loaded  params={n_params:,}  "
              f"n_events_per_class={n_events_per_class}  device={self.device}")

        self.scores = self._run(output_path)
        self.score = self.scores["score"]

    def _run(self, output_path=None):
        print("=" * 60)
        print("Collide-2V Representation-Quality Evaluation")
        print("=" * 60)

        loader = Collide2VLoader(self.cfg)
        label_map = self.cfg.label_map
        events_per_class = {cls: self.n_events_per_class for cls in self.cfg.class_names}
        raw_tensor, y, sample_name, _ = loader.load_classes(events_per_class, cache_prefix="eval")
        bg_labels = [label_map[cls] for cls in self.background_classes]
        bg_mask_all = np.isin(y, bg_labels)
        anomaly_label = label_map[self.anomaly_class]
        print(f"\n  total={len(raw_tensor)}  background({'+'.join(self.background_classes)})={int(np.sum(bg_mask_all))}  "
              f"anomaly({self.anomaly_class})={int(np.sum(y == anomaly_label))}")

        train_idx, val_idx, eval_idx = build_split_indices(
            y,
            class_names=self.cfg.class_names,
            label_map=label_map,
            seed=SEED,
        )

        norm = compute_normalization_constants(raw_tensor[train_idx])
        # Build the PID vocabulary from all loaded eval events. With very small
        # samples, the train split alone can miss a rare PID and produce a
        # narrower one-hot tensor than the saved checkpoint expects.
        pid_mapping = build_pid_mapping(raw_tensor)
        X_all, pid_mapping = preprocess_dataset(raw_tensor, norm=norm, pid_mapping=pid_mapping, add_pid_ohe=True)
        expected_features = _expected_input_features(self.model_)
        X_all = _align_feature_dim(X_all, expected_features, "preprocessed eval tensor")
        n_features = X_all.shape[-1]

        X_eval = X_all[eval_idx]
        y_eval = y[eval_idx]
        X_eval_raw = raw_tensor[eval_idx]

        self.latent_dim = _infer_latent_dim(self.model_, self.device, n_features)
        print(f"  latent dim = {self.latent_dim}")

        print("\nExtracting original latents ...")
        Z_all = extract_latents(self.model_, X_eval, self.device)

        print("Augmenting eval set ...")
        X_aug_raw = create_augmented_pairs(X_eval_raw)
        Z_aug_list = []
        for a in range(X_aug_raw.shape[0]):
            X_aug_pp, _ = preprocess_dataset(X_aug_raw[a], norm=norm, pid_mapping=pid_mapping, add_pid_ohe=True)
            X_aug_pp = _align_feature_dim(X_aug_pp, expected_features, f"augmented eval tensor {a}")
            Z_aug_list.append(extract_latents(self.model_, X_aug_pp, self.device))

        anomaly_mask = y_eval == anomaly_label
        bg_mask = np.isin(y_eval, bg_labels)

        background_label = "+".join(self.background_classes)
        print(f"PCA whitening (fit on {background_label}) ...")
        pca = fit_pca_whiten(Z_all[bg_mask])
        Z_whitened = pca.transform(Z_all)
        Z_aug_whitened = [pca.transform(Z_aug) for Z_aug in Z_aug_list]

        print("\nComputing metrics ...")
        print("-" * 50)

        align = alignment_loss(Z_whitened, Z_aug_whitened)
        print(f"  alignment loss       : {align:.6f}")

        pr = participation_ratio(Z_whitened)
        pr_norm = pr / self.latent_dim
        print(f"  participation ratio  : {pr:.2f}  (norm = {pr_norm:.4f})")

        gmm, per_class = gmm_anomaly_gap_collide2v(
            Z_whitened,
            anomaly_mask,
            bg_mask,
            background_label,
            self.anomaly_class,
        )
        print(f"  GMM anomaly gap      : {gmm:.2f}  ({background_label} vs {self.anomaly_class})")

        md = mahalanobis_anomaly_collide2v(Z_whitened, anomaly_mask, bg_mask)
        print(f"  Mahalanobis distance : {md:.4f}  ({self.anomaly_class} vs {background_label})")

        ref_ranges = _build_ref_ranges(self.latent_dim)
        raw = {"align_loss": align, "pr": pr, "gmm_gap": gmm, "md": md}
        norm_metrics = normalize_metrics(raw, ref_ranges)

        Q_geo = composite_geometric_mean(norm_metrics)

        bar = "-" * 50
        print(f"\n{bar}")
        print("  normalised sub-scores:")
        print(f"    align_loss  = {norm_metrics['align_loss_norm']:.4f}")
        print(f"    pr          = {norm_metrics['pr_norm']:.4f}")
        print(f"    gmm_gap     = {norm_metrics['gmm_gap_norm']:.4f}")
        print(f"    md          = {norm_metrics['md_norm']:.4f}")
        print(f"  COMPOSITE SCORE (geo-mean) = {Q_geo:.6f}")
        print(f"{bar}")

        scores = {
            "score": Q_geo,
            "align_loss": round(float(align), 6),
            "pr": round(float(pr), 4),
            "pr_norm": round(float(pr_norm), 4),
            "gmm_gap": round(float(gmm), 4),
            "mahalanobis": round(float(md), 4),
            "background_classes": self.background_classes,
            "anomaly_class": self.anomaly_class,
            "class_names": self.cfg.class_names,
            "n_events_per_class": self.n_events_per_class,
            "n_eval_events": int(len(y_eval)),
            "align_loss_norm": round(float(norm_metrics["align_loss_norm"]), 4),
            "pr_norm_norm": round(float(norm_metrics["pr_norm"]), 4),
            "gmm_gap_norm": round(float(norm_metrics["gmm_gap_norm"]), 4),
            "md_norm": round(float(norm_metrics["md_norm"]), 4),
            "n_params": sum(p.numel() for p in self.model_.parameters()),
            "latent_dim": self.latent_dim,
            "n_pid_classes": len(pid_mapping),
        }

        if output_path:
            with open(output_path, "w") as f:
                json.dump(scores, f, indent=2, default=str)
            print(f"\nScores written -> {output_path}")

        return scores


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate representation quality of a Collide-2V PyTorch model.")
    parser.add_argument("model", help="Path to a full nn.Module .pt file (torch.save(model, path))")
    parser.add_argument("--n-events-per-class", type=int, default=2000)
    parser.add_argument("--background-classes", default=",".join(BACKGROUND_CLASSES),
                        help="Comma-separated background classes. For HH4B use minbias,top,qcd,dy.")
    parser.add_argument("--anomaly-class", default=ANOMALY_CLASS,
                        help="Anomaly/signal class. For the first dedicated test use hh4b.")
    parser.add_argument("--data-source", default="collide2v_updated",
                        choices=["collide2v_updated", "samples_json", "hf"])
    parser.add_argument("--feature-mode", default="l1_only", choices=["l1_only"],
                        help="Only L1 candidate columns are allowed for NGT runs.")
    parser.add_argument("--candidate-selection", default="top_pt")
    parser.add_argument("--collide2v-path", default=None, help="Override COLLIDE2V_PATH")
    parser.add_argument("--output", default=None, help="Path to save scores.json")
    parser.add_argument("--device", default=None, help="Device (cuda/cpu), auto-detected if unset")
    args = parser.parse_args()
    background_classes = [part.strip().lower() for part in args.background_classes.split(",") if part.strip()]
    anomaly_class = args.anomaly_class.strip().lower()
    class_names = []
    for cls in [*background_classes, anomaly_class]:
        if cls not in class_names:
            class_names.append(cls)

    cfg_kwargs = dict(
        data_source=args.data_source,
        feature_mode=args.feature_mode,
        candidate_selection=args.candidate_selection,
        class_names=class_names,
    )
    if args.collide2v_path:
        cfg_kwargs["collide2v_path"] = args.collide2v_path
    cfg = Collide2VConfig(**cfg_kwargs)

    EvalCollide2V(
        args.model,
        n_events_per_class=args.n_events_per_class,
        device=args.device,
        output_path=args.output,
        collide2v_cfg=cfg,
        background_classes=background_classes,
        anomaly_class=anomaly_class,
    )


if __name__ == "__main__":
    main()
