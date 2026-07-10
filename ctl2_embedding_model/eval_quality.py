#!/usr/bin/env python3
"""
Representation-quality evaluation for PyTorch models.

Usage (CLI):
    python eval.py /path/to/model.pt [--data TEST.parquet] [--labels LABELS.csv]

Usage (module):
    from eval import Eval
    evaluator = Eval("/path/to/model.pt")
    print(evaluator.score)   # single float
    print(evaluator.scores)  # full dict
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.covariance import LedoitWolf

SEED = 42
SEQ_LEN = 100
N_AUG = 4
BATCH_SIZE = 256

ANOMALY_KEY = "HH_4b"

DEFAULT_DATA = ("/eos/user/m/maglowac/maciej_mlchallange/test_data/"
                "test/test.parquet")
DEFAULT_LABELS = ("/eos/user/m/maglowac/maciej_mlchallange/test_data/"
                  "labels/test_labels.csv")

GMM_KWARGS = dict(
    n_components=2,
    covariance_type="full",
    random_state=SEED,
    max_iter=500,
    reg_covar=1e-5,
    init_params="kmeans",
)

EPS = np.finfo(np.float64).eps

SM_BKGS = {"Wjet", "DY", "QCD", "ttbar"}

NORM_CONSTANTS_DEFAULT = {
    "pt_min": 0.0,
    "pt_max": 8.3179,
    "eta_min": -5.0309,
    "eta_max": 5.0353,
    "phi_min": -np.pi,
    "phi_max": np.pi,
    "dxy_min": -3.1416,
    "dxy_max": 3.1406,
}

RAW_COLS = [
    "L1T_PFPart_PT",    # 0 : pT
    "L1T_PFPart_Eta",   # 1 : eta
    "L1T_PFPart_Phi",   # 2 : phi
    "L1T_PFPart_D0",    # 3 : d0 / dxy
    "L1T_PFPart_PID",   # 4 : particle ID
]
N_RAW_FEATURES = len(RAW_COLS)


# ═══════════════════════════════════════════════════════════════════════════════
#  data loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_test_data(test_parquet, labels_csv, seq_len, raw_cols):
    df = pd.read_parquet(test_parquet)
    labels_df = pd.read_csv(labels_csv)

    y = labels_df["label"].values.astype(np.int64)
    src = labels_df["source_file"].values

    missing = [c for c in raw_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Columns not found in parquet: {missing}. "
            f"Available: {list(df.columns)}")

    n_raw = len(raw_cols)
    n_events = len(df)

    raw_arrays = []
    raw_arrays_pt = []
    for ci, col in enumerate(raw_cols):
        col_data = df[col].values
        raw_arrays.append(col_data)
        if ci == 0:
            raw_arrays_pt = col_data

    X = np.zeros((n_events, seq_len, n_raw), dtype=np.float32)
    for i in range(n_events):
        n_part = len(raw_arrays_pt[i])
        if n_part == 0:
            continue
        idx = np.argsort(raw_arrays_pt[i])[::-1][:seq_len]
        for ci in range(n_raw):
            part_data = raw_arrays[ci][i]
            X[i, :len(idx), ci] = np.asarray(part_data, dtype=np.float32)[idx]
    return X, y, src


def split_by_source(y, src):
    anomaly_mask = np.array([ANOMALY_KEY in s for s in src])
    bg_mask = ~anomaly_mask

    bg_sources = {}
    all_unique = np.unique(src[bg_mask])
    for s in all_unique:
        base_name = os.path.splitext(os.path.basename(s))[0]
        if not any(bkg.lower() in base_name.lower() for bkg in SM_BKGS):
            print(f"  skipping non-SM background: {s}")
            continue
        bg_sources[s] = bg_mask & (src == s)

    if not bg_sources:
        raise ValueError(
            f"No SM background classes ({SM_BKGS}) found in data. "
            f"Available sources: {list(all_unique)}")

    return anomaly_mask, bg_mask, bg_sources


# ═══════════════════════════════════════════════════════════════════════════════
#  preprocessing  (raw 5-ch → 4 kine + PID-OHE = 12-ch)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_normalization_constants(data: np.ndarray) -> dict:
    pt = data[:, :, 0]
    eta = data[:, :, 1]
    dxy = data[:, :, 3]

    pt_log = np.log1p(pt)
    return {
        "pt_min": pt_log.min(),
        "pt_max": pt_log.max(),
        "eta_min": eta.min(),
        "eta_max": eta.max(),
        "phi_min": -np.pi,
        "phi_max": np.pi,
        "dxy_min": dxy.min(),
        "dxy_max": dxy.max(),
    }


def normalize_particles(x: np.ndarray, norm: dict) -> np.ndarray:
    pt = np.log1p(x[..., 0])
    pt = (pt - norm["pt_min"]) / (norm["pt_max"] - norm["pt_min"] + 1e-8)
    eta = (x[..., 1] - norm["eta_min"]) / (norm["eta_max"] - norm["eta_min"] + 1e-8)
    phi = (x[..., 2] + np.pi) / (2 * np.pi)

    mask_pad = x[..., 0] == 0
    pt[mask_pad] = 0.0
    eta[mask_pad] = 0.0
    phi[mask_pad] = 0.0

    out = np.stack([
        pt, eta, phi,
        x[..., 3].copy(),
        x[..., 4].copy(),
    ], axis=-1)
    out[mask_pad, :4] = 0.0
    return out


def build_pid_mapping(data: np.ndarray) -> dict:
    pid_tensor = data[:, :, 4]
    mask_valid = data[:, :, 0] > 0
    valid_pids = pid_tensor[mask_valid].astype(np.int32)
    unique_pids = np.unique(valid_pids)
    return {int(pid): idx for idx, pid in enumerate(unique_pids)}


def map_pid(pid: np.ndarray, pid_mapping: dict) -> np.ndarray:
    mapped = np.full_like(pid, -1, dtype=np.int32)
    for k, v in pid_mapping.items():
        mapped[pid == k] = v
    return mapped


def preprocess_dataset(
    data: np.ndarray,
    norm: dict | None = None,
    pid_mapping: dict | None = None,
) -> np.ndarray:
    if norm is None:
        norm = NORM_CONSTANTS_DEFAULT
    normed = normalize_particles(data, norm)
    kine = normed[:, :, :4]  # pt, eta, phi, d0

    if pid_mapping is None:
        pid_mapping = build_pid_mapping(data)
    pid = normed[:, :, 4].astype(np.int32)
    pid_idx = map_pid(pid, pid_mapping)
    n_classes = len(pid_mapping)
    pid_ohe = np.zeros((*pid_idx.shape, n_classes), dtype=np.float32)
    valid = pid_idx >= 0
    pid_ohe[valid] = np.eye(n_classes, dtype=np.float32)[pid_idx[valid]]
    return np.concatenate([kine, pid_ohe], axis=-1).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
#  model loading
# ═══════════════════════════════════════════════════════════════════════════════

def _load_model(model_path, device):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    obj = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(obj, dict) and "state_dict" in obj:
        raise ValueError(
            "Detected a state_dict. Please save the full model with "
            "torch.save(model, path), not model.state_dict()."
        )

    if callable(obj) or isinstance(obj, torch.nn.Module):
        model = obj
    elif isinstance(obj, dict):
        raise ValueError(
            "Detected a plain dict. Please save the full model object."
        )
    else:
        raise ValueError(
            f"Unexpected type from torch.load: {type(obj)}. "
            "Expected a torch.nn.Module."
        )

    model.to(device)
    model.eval()
    return model


def _infer_latent_dim(model, device, n_features):
    dummy = torch.zeros(1, SEQ_LEN, n_features, device=device)
    with torch.no_grad():
        z = model(dummy)
    return z.shape[-1]


# ═══════════════════════════════════════════════════════════════════════════════
#  augmentations  (applied on **raw** 5-channel data)
# ═══════════════════════════════════════════════════════════════════════════════

def augment_event(x, rng):
    x_aug = x.copy()
    dphi = rng.uniform(-0.8, 0.8)
    x_aug[:, 2] = (x_aug[:, 2] + dphi + np.pi) % (2.0 * np.pi) - np.pi
    noise = rng.normal(0.0, 0.015, size=x_aug.shape).astype(np.float32)
    x_aug += noise
    return x_aug


def create_augmented_pairs(X):
    X_aug = np.zeros((N_AUG, *X.shape), dtype=np.float32)
    for a in range(N_AUG):
        rng_a = np.random.default_rng(SEED + 100 * (a + 1))
        for i in range(len(X)):
            X_aug[a, i] = augment_event(X[i], rng_a)
    return X_aug


# ═══════════════════════════════════════════════════════════════════════════════
#  inference
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_latents(model, X, device):
    latents = []
    for i in range(0, len(X), BATCH_SIZE):
        batch = torch.as_tensor(X[i: i + BATCH_SIZE], device=device)
        z = model(batch)
        latents.append(z.cpu().numpy())
    return np.concatenate(latents, axis=0).astype(np.float64)


# ═══════════════════════════════════════════════════════════════════════════════
#  PCA whitening
# ═══════════════════════════════════════════════════════════════════════════════

def fit_pca_whiten(Z_bg):
    pca = PCA(whiten=True, random_state=SEED)
    pca.fit(Z_bg)
    return pca


# ═══════════════════════════════════════════════════════════════════════════════
#  metrics
# ═══════════════════════════════════════════════════════════════════════════════

def alignment_loss(Z_orig, Z_aug_list):
    n_orig = Z_orig / (np.linalg.norm(Z_orig, axis=1, keepdims=True) + EPS)
    losses = []
    for Z_aug in Z_aug_list:
        n_aug = Z_aug / (np.linalg.norm(Z_aug, axis=1, keepdims=True) + EPS)
        losses.append(float(np.mean(1.0 - np.sum(n_orig * n_aug, axis=1))))
    return float(np.mean(losses))


def participation_ratio(Z):
    cov = np.cov(Z.T)
    ev = np.maximum(np.linalg.eigvalsh(cov), 0.0)
    s = float(np.sum(ev))
    if s < EPS:
        return 0.0
    return float(s ** 2 / np.sum(ev ** 2))


def gmm_anomaly_gap(Z_whitened, anomaly_mask, bg_sources):
    Z_anom = Z_whitened[anomaly_mask]

    per_class_nlls = {}
    for bg_name, bg_idx in bg_sources.items():
        Z_bg_class = Z_whitened[bg_idx]
        if len(Z_bg_class) < 20:
            print(f"  WARNING: too few events for {bg_name}, skipping")
            continue

        gmm = GaussianMixture(**GMM_KWARGS)
        gmm.fit(Z_bg_class)

        nll_anomaly = -gmm.score_samples(Z_anom)
        avg_nll = float(np.mean(nll_anomaly))
        per_class_nlls[bg_name] = avg_nll

        nll_bg = -gmm.score_samples(Z_bg_class)
        avg_nll_bg = float(np.mean(nll_bg))

        print(f"  GMM {bg_name:40s}  "
              f"NLL(anom)={avg_nll:.2f}  NLL(bg)={avg_nll_bg:.2f}")

    if not per_class_nlls:
        return 0.0, {}
    return float(np.mean(list(per_class_nlls.values()))), per_class_nlls


def mahalanobis_anomaly(Z_whitened, anomaly_mask, bg_sources):
    Z_anom = Z_whitened[anomaly_mask]

    minbias_key = None
    for key in bg_sources:
        if "qcd" in key.lower():
            minbias_key = key
            break
    if minbias_key is None:
        print("  WARNING: no minbias/QCD class found, skipping Mahalanobis")
        return 0.0

    Z_mb = Z_whitened[bg_sources[minbias_key]]
    if len(Z_anom) < 2 or len(Z_mb) < 2:
        return 0.0

    mu_a = np.mean(Z_anom, axis=0)
    mu_b = np.mean(Z_mb, axis=0)

    cov = LedoitWolf().fit(Z_mb).covariance_
    cov_inv = np.linalg.pinv(cov)
    delta = mu_a - mu_b
    return float(np.sqrt(delta @ cov_inv @ delta))


# ═══════════════════════════════════════════════════════════════════════════════
#  composite score
# ═══════════════════════════════════════════════════════════════════════════════

def _build_ref_ranges(latent_dim):
    return {
        "align_loss": (0.0, 1.0),
        "pr": (1.0, latent_dim),
        "gmm_gap": (0.0, 500.0),
        "md": (0.0, 50.0),
    }


def normalize_metrics(raw, ref_ranges):
    out = {}
    for key, (lo, hi) in ref_ranges.items():
        if key == "align_loss":
            out["align_loss_norm"] = float(np.clip(
                1.0 - (raw[key] - lo) / (hi - lo), 0.0, 1.0))
        else:
            out[f"{key}_norm"] = float(np.clip(
                (raw[key] - lo) / (hi - lo), 0.0, 1.0))
    return out


def composite_geometric_mean(norm):
    cols = ["align_loss_norm", "pr_norm", "gmm_gap_norm", "md_norm"]
    vals = np.array([norm[c] + EPS for c in cols])
    return float(np.exp(np.mean(np.log(vals))))


# ═══════════════════════════════════════════════════════════════════════════════
#  Eval class
# ═══════════════════════════════════════════════════════════════════════════════

class Eval:
    def __init__(self, model, test_parquet=None, labels_csv=None,
                 output_path=None, device=None, seq_len=SEQ_LEN,
                 raw_cols=RAW_COLS, norm_constants=None):
        if isinstance(model, str):
            self.model_ = _load_model(model, device)
        elif isinstance(model, torch.nn.Module):
            self.model_ = model
        else:
            raise TypeError(
                f"model must be str (path) or nn.Module, got {type(model)}")

        self.device = next(self.model_.parameters()).device
        self.seq_len = seq_len
        self.raw_cols = raw_cols
        self.norm_constants = norm_constants or NORM_CONSTANTS_DEFAULT

        n_params = sum(p.numel() for p in self.model_.parameters())
        print(f"Eval: model loaded  params={n_params:,}  "
              f"seq_len={self.seq_len}  device={self.device}")

        self.scores = self._run(test_parquet, labels_csv, output_path)
        self.score = self.scores["score"]

    # ------------------------------------------------------------------
    #  internal: full evaluation pipeline
    # ------------------------------------------------------------------

    def _run(self, test_parquet=None, labels_csv=None, output_path=None):
        test_parquet = test_parquet or DEFAULT_DATA
        labels_csv = labels_csv or DEFAULT_LABELS

        print("=" * 60)
        print("Representation-Quality Evaluation")
        print("=" * 60)

        # -- load raw data -----------------------------------------------
        print(f"\nLoading test data: {test_parquet}")
        X_raw, y_true, src = load_test_data(
            test_parquet, labels_csv, self.seq_len, self.raw_cols)
        anomaly_mask, bg_mask, bg_sources = split_by_source(y_true, src)

        n_anom = int(np.sum(anomaly_mask))
        n_bg = int(np.sum(bg_mask))
        print(f"  total={len(X_raw)}  anomaly({ANOMALY_KEY})={n_anom}  "
              f"bg={n_bg}")
        print(f"  background classes: {list(bg_sources.keys())}")

        # -- build PID mapping & normalisation constants -----------------
        print("\nBuilding PID mapping ...")
        pid_mapping = build_pid_mapping(X_raw)
        n_pid_classes = len(pid_mapping)
        n_features = 4 + n_pid_classes
        print(f"  PID classes = {n_pid_classes} "
              f"→ preprocessed dims = (N, {self.seq_len}, {n_features})")
        print(f"  PID mapping = {pid_mapping}")

        # infer latent dim from the model after we know n_features
        self.latent_dim = _infer_latent_dim(
            self.model_, self.device, n_features)
        print(f"  latent dim = {self.latent_dim}")

        # -- preprocess ---------------------------------------------------
        print("\nPreprocessing (kine norm + PID OHE) ...")
        norm = self.norm_constants
        X_pp = preprocess_dataset(X_raw, norm=norm, pid_mapping=pid_mapping)

        # -- extract latents (original) ----------------------------------
        print("Extracting original latents ...")
        Z_all = extract_latents(self.model_, X_pp, self.device)

        # -- augmentations ------------------------------------------------
        print(f"Augmenting x{N_AUG} ...")
        X_aug_raw = create_augmented_pairs(X_raw)
        Z_aug_list = []
        for a in range(N_AUG):
            X_aug_pp = preprocess_dataset(
                X_aug_raw[a], norm=norm, pid_mapping=pid_mapping)
            Z_aug_list.append(
                extract_latents(self.model_, X_aug_pp, self.device))

        # -- PCA whitening ------------------------------------------------
        print("PCA whitening (fit on backgrounds only) ...")
        pca = fit_pca_whiten(Z_all[bg_mask])
        Z_whitened = pca.transform(Z_all)
        Z_aug_whitened = [pca.transform(Z_aug) for Z_aug in Z_aug_list]

        # -- metrics ------------------------------------------------------
        print("\nComputing metrics ...")
        print("-" * 50)

        align = alignment_loss(Z_whitened, Z_aug_whitened)
        print(f"  alignment loss       : {align:.6f}")

        pr = participation_ratio(Z_whitened)
        pr_norm = pr / self.latent_dim
        print(f"  participation ratio  : {pr:.2f}  "
              f"(norm = {pr_norm:.4f})")

        gmm, per_class = gmm_anomaly_gap(Z_whitened, anomaly_mask, bg_sources)
        print(f"  GMM anomaly gap      : {gmm:.2f}  "
              f"(mean NLL across {len(per_class)} bkg classes)")

        md = mahalanobis_anomaly(Z_whitened, anomaly_mask, bg_sources)
        print(f"  Mahalanobis distance : {md:.4f}  "
              f"(anomaly vs minbias/QCD)")

        ref_ranges = _build_ref_ranges(self.latent_dim)
        raw = {"align_loss": align, "pr": pr, "gmm_gap": gmm, "md": md}
        norm_metrics = normalize_metrics(raw, ref_ranges)

        Q_geo = composite_geometric_mean(norm_metrics)

        bar = "-" * 50
        print(f"\n{bar}")
        print(f"  normalised sub-scores:")
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
            "n_bkg_classes": len(bg_sources),
            "bkg_classes": list(bg_sources.keys()),
            "gmm_per_class": {k: round(float(v), 2)
                              for k, v in per_class.items()},
            "align_loss_norm": round(float(norm_metrics["align_loss_norm"]), 4),
            "pr_norm_norm": round(float(norm_metrics["pr_norm"]), 4),
            "gmm_gap_norm": round(float(norm_metrics["gmm_gap_norm"]), 4),
            "md_norm": round(float(norm_metrics["md_norm"]), 4),
            "n_params": sum(p.numel() for p in self.model_.parameters()),
            "n_pid_classes": n_pid_classes,
            "pid_mapping": {int(k): int(v)
                            for k, v in pid_mapping.items()},
        }

        if output_path:
            with open(output_path, "w") as f:
                json.dump(scores, f, indent=2, default=str)
            print(f"\nScores written -> {output_path}")

        return scores


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def _validate_paths(model_path, data_path, labels_path):
    for path, name in [(model_path, "model"), (data_path, "data"),
                        (labels_path, "labels")]:
        if not os.path.exists(path):
            print(f"ERROR: {name} not found: {path}", file=sys.stderr)
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate representation quality of a PyTorch model.")
    parser.add_argument("model", help="Path to .pt model file")
    parser.add_argument("--data", default=DEFAULT_DATA,
                        help="Path to test.parquet")
    parser.add_argument("--labels", default=DEFAULT_LABELS,
                        help="Path to test_labels.csv")
    parser.add_argument("--output", default=None,
                        help="Path to save scores.json")
    parser.add_argument("--device", default=None,
                        help="Device (cuda/cpu), auto-detected if unset")
    args = parser.parse_args()

    _validate_paths(args.model, args.data, args.labels)

    Eval(args.model, test_parquet=args.data, labels_csv=args.labels,
         output_path=args.output, device=args.device)


if __name__ == "__main__":
    main()
