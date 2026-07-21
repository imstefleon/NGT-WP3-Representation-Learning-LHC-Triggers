#!/usr/bin/env python3
"""
Scaling sweep: train small MHAEmbeddingFullPrecision models at increasing
(model size, samples-per-class) and watch eval_quality_collide2v.py's composite
latent-quality score. Intended as a quick sanity check that the composite score
behaves sensibly (increases with capacity/data) before committing to larger,
longer runs on CERN infra.

Usage:
    PYTHONPATH=ngt_model/src python -m ngt_model.evaluation.scale_test \
        --grid scaling --epochs 5 --output ngt_mha_nodes_aligned_outputs/scale_tests/scaling_eval_quality.csv

    PYTHONPATH=ngt_model/src python -m ngt_model.evaluation.scale_test \
        --synthetic --output sweep_results_synthetic.csv   # local smoke test, no /eos needed

Each row of the output CSV is one (tier, n_events_per_class) run:
    tier, latent_dim, ff_dim, num_heads, n_events_per_class, n_params,
    score, align_loss, pr, gmm_gap, mahalanobis
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ngt_model.models.mha_model import MHAEmbeddingFullPrecision, LatentOnlyWrapper
from ngt_model.datasets.collide2v_data import (
    CLASS_NAMES,
    Collide2VConfig,
    Collide2VLoader,
    build_pid_mapping,
    build_split_indices,
    compute_normalization_constants,
    preprocess_dataset,
)
from ngt_model.evaluation.eval_quality_collide2v import EvalCollide2V

SEED = 42
N_CONSTITUENTS = 100
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 256


@dataclass
class SizeTier:
    name: str
    latent_dim: int
    ff_dim: int
    num_heads: int
    events_per_class: int


QUICK_GRID: List[SizeTier] = [
    SizeTier("tiny", latent_dim=8, ff_dim=16, num_heads=2, events_per_class=500),
    SizeTier("small", latent_dim=12, ff_dim=32, num_heads=2, events_per_class=2_000),
    SizeTier("medium", latent_dim=24, ff_dim=64, num_heads=4, events_per_class=10_000),
    SizeTier("large", latent_dim=32, ff_dim=128, num_heads=8, events_per_class=50_000),
]

SCALING_GRID: List[SizeTier] = [
    SizeTier("tiny_1k", latent_dim=24, ff_dim=64, num_heads=4, events_per_class=1_000),
    SizeTier("small_10k", latent_dim=64, ff_dim=128, num_heads=4, events_per_class=10_000),
    SizeTier("medium_50k", latent_dim=128, ff_dim=128, num_heads=8, events_per_class=50_000),
    SizeTier("reasonable_100k", latent_dim=128, ff_dim=128, num_heads=8, events_per_class=100_000),
]

SYNTHETIC_GRID: List[SizeTier] = [
    SizeTier("tiny", latent_dim=8, ff_dim=16, num_heads=2, events_per_class=64),
    SizeTier("small", latent_dim=12, ff_dim=32, num_heads=2, events_per_class=128),
]

GRIDS = {
    "quick": QUICK_GRID,
    "scaling": SCALING_GRID,
}


class TensorDatasetTorch(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32, copy=False))
        self.y = torch.from_numpy(y.astype(np.int64, copy=False))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def make_synthetic_raw(events_per_class: int, n_constituents: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Random data shaped like the real Collide-2V raw tensor, for wiring smoke tests only."""
    rng = np.random.default_rng(seed)
    n_classes = len(CLASS_NAMES)
    raw = np.zeros((n_classes * events_per_class, n_constituents, 5), dtype=np.float32)
    y = np.zeros(n_classes * events_per_class, dtype=np.int64)
    pid_choices = np.array([11, -11, 13, -13, 22, 130, 211, -211], dtype=np.float32)
    for cls_idx in range(n_classes):
        sl = slice(cls_idx * events_per_class, (cls_idx + 1) * events_per_class)
        n_valid = rng.integers(20, n_constituents, size=events_per_class)
        for i, nv in enumerate(n_valid):
            row = sl.start + i
            raw[row, :nv, 0] = rng.exponential(5.0, size=nv).astype(np.float32) + 0.1  # pt > 0
            raw[row, :nv, 1] = rng.uniform(-5, 5, size=nv).astype(np.float32)  # eta
            raw[row, :nv, 2] = rng.uniform(-np.pi, np.pi, size=nv).astype(np.float32)  # phi
            raw[row, :nv, 3] = rng.uniform(-1, 1, size=nv).astype(np.float32)  # dxy
            raw[row, :nv, 4] = rng.choice(pid_choices, size=nv)  # pid
        y[sl] = cls_idx
    return raw, y


def run_epoch(model, loader, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, total_correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        with torch.set_grad_enabled(is_train):
            out = model(xb)
            loss = F.cross_entropy(out["logits"], yb)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        total_loss += float(loss.item()) * len(xb)
        total_correct += int((out["logits"].argmax(dim=1) == yb).sum().item())
        total += len(xb)
    return total_loss / max(total, 1), total_correct / max(total, 1)


def train_one(tier: SizeTier, cfg: Collide2VConfig, epochs: int, device: torch.device,
              synthetic: bool) -> tuple[MHAEmbeddingFullPrecision, dict]:
    events_per_class = {cls: tier.events_per_class for cls in CLASS_NAMES}

    if synthetic:
        raw_tensor, y = make_synthetic_raw(tier.events_per_class, cfg.n_constituents, seed=SEED)
        sample_name = None
    else:
        loader = Collide2VLoader(cfg)
        raw_tensor, y, sample_name, _ = loader.load_classes(events_per_class, cache_prefix=f"sweep_{tier.name}")

    train_idx, val_idx, eval_idx = build_split_indices(y, seed=SEED)
    norm = compute_normalization_constants(raw_tensor[train_idx])
    pid_mapping = build_pid_mapping(raw_tensor[train_idx])
    X_all, pid_mapping = preprocess_dataset(raw_tensor, norm=norm, pid_mapping=pid_mapping, add_pid_ohe=True)

    X_train, y_train = X_all[train_idx], y[train_idx]
    X_val, y_val = X_all[val_idx], y[val_idx]

    model = MHAEmbeddingFullPrecision(
        n_constituents=cfg.n_constituents,
        n_features=X_all.shape[-1],
        latent_dim=tier.latent_dim,
        n_classes=len(CLASS_NAMES),
        ff_dim=tier.ff_dim,
        num_heads=tier.num_heads,
        pooling_mode="attention",
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    train_loader = DataLoader(TensorDatasetTorch(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDatasetTorch(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    best_val_loss = float("inf")
    best_state = None
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, device, optimizer=None)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"    [{tier.name}] epoch {epoch}/{epochs} "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
              f"({time.time()-t0:.1f}s)")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    return model, {"n_params": n_params, "best_val_loss": best_val_loss}


def _load_mlflow_credentials_from_local_conf():
    credentials_path = Path(__file__).resolve().parents[3] / "conf" / "local" / "credentials.yml"
    if not credentials_path.exists():
        return

    section = None
    for raw_line in credentials_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if not raw_line.startswith((" ", "\t")) and line.endswith(":"):
            section = line[:-1]
            continue
        if section == "mlflow_auth" and ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip().strip("'\"")
            if key in {"MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD"} and value:
                os.environ.setdefault(key, value)


def setup_mlflow(args, grid: List[SizeTier], eval_events: int, device: torch.device):
    _load_mlflow_credentials_from_local_conf()
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError("MLflow logging requested, but the 'mlflow' package is not installed.") from exc

    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment)
    mlflow.start_run(run_name=args.mlflow_run_name or f"scale_test_{args.grid}")
    mlflow.log_params({
        "sweep_grid": args.grid,
        "epochs_per_tier": args.epochs,
        "eval_events_per_class": eval_events,
        "data_source": args.data_source,
        "synthetic": args.synthetic,
        "device": str(device),
        "n_tiers": len(grid),
    })
    for idx, tier in enumerate(grid):
        mlflow.log_params({
            f"tier_{idx}_name": tier.name,
            f"tier_{idx}_events_per_class": tier.events_per_class,
            f"tier_{idx}_latent_dim": tier.latent_dim,
            f"tier_{idx}_ff_dim": tier.ff_dim,
            f"tier_{idx}_num_heads": tier.num_heads,
        })
    return mlflow


def run_sweep(grid: List[SizeTier], cfg: Collide2VConfig, epochs: int, eval_events_per_class: int,
              synthetic: bool, device: torch.device, mlflow_client=None) -> pd.DataFrame:
    rows = []
    for step, tier in enumerate(grid):
        print(f"\n=== tier={tier.name}  latent_dim={tier.latent_dim} ff_dim={tier.ff_dim} "
              f"num_heads={tier.num_heads} events_per_class={tier.events_per_class} ===")
        model, train_info = train_one(tier, cfg, epochs=epochs, device=device, synthetic=synthetic)
        wrapped = LatentOnlyWrapper(model).to(device)
        wrapped.eval()

        if synthetic:
            # EvalCollide2V streams real data via Collide2VLoader; for the synthetic
            # smoke test we exercise the same metric pipeline directly instead.
            from ngt_model.evaluation.eval_quality_collide2v import (
                gmm_anomaly_gap_collide2v, mahalanobis_anomaly_collide2v, BACKGROUND_CLASSES, ANOMALY_CLASS,
            )
            from ngt_model.evaluation.eval_quality import (
                alignment_loss, participation_ratio, fit_pca_whiten, normalize_metrics,
                composite_geometric_mean, create_augmented_pairs, extract_latents,
                _build_ref_ranges, _infer_latent_dim,
            )
            from ngt_model.datasets.collide2v_data import LABEL_MAP

            raw_eval, y_eval = make_synthetic_raw(eval_events_per_class, cfg.n_constituents, seed=SEED + 999)
            norm = compute_normalization_constants(raw_eval)
            pid_mapping = build_pid_mapping(raw_eval)
            X_eval, pid_mapping = preprocess_dataset(raw_eval, norm=norm, pid_mapping=pid_mapping, add_pid_ohe=True)

            latent_dim = _infer_latent_dim(wrapped, device, X_eval.shape[-1])
            Z_all = extract_latents(wrapped, X_eval, device)
            X_aug_raw = create_augmented_pairs(raw_eval)
            Z_aug_list = []
            for a in range(X_aug_raw.shape[0]):
                X_aug_pp, _ = preprocess_dataset(X_aug_raw[a], norm=norm, pid_mapping=pid_mapping, add_pid_ohe=True)
                Z_aug_list.append(extract_latents(wrapped, X_aug_pp, device))

            background_label = "+".join(BACKGROUND_CLASSES)
            anomaly_mask = y_eval == LABEL_MAP[ANOMALY_CLASS]
            bg_mask = np.isin(y_eval, [LABEL_MAP[cls] for cls in BACKGROUND_CLASSES])
            pca = fit_pca_whiten(Z_all[bg_mask])
            Z_whitened = pca.transform(Z_all)
            Z_aug_whitened = [pca.transform(z) for z in Z_aug_list]

            align = alignment_loss(Z_whitened, Z_aug_whitened)
            pr = participation_ratio(Z_whitened)
            gmm, _ = gmm_anomaly_gap_collide2v(
                Z_whitened,
                anomaly_mask,
                bg_mask,
                background_label,
                ANOMALY_CLASS,
            )
            md = mahalanobis_anomaly_collide2v(Z_whitened, anomaly_mask, bg_mask)
            ref_ranges = _build_ref_ranges(latent_dim)
            norm_metrics = normalize_metrics({"align_loss": align, "pr": pr, "gmm_gap": gmm, "md": md}, ref_ranges)
            score = composite_geometric_mean(norm_metrics)
            eval_scores = {"score": score, "align_loss": align, "pr": pr, "gmm_gap": gmm, "mahalanobis": md}
        else:
            ev = EvalCollide2V(wrapped, n_events_per_class=eval_events_per_class, collide2v_cfg=cfg)
            eval_scores = ev.scores

        row = {
            "tier": tier.name,
            "latent_dim": tier.latent_dim,
            "ff_dim": tier.ff_dim,
            "num_heads": tier.num_heads,
            "n_events_per_class": tier.events_per_class,
            "n_params": train_info["n_params"],
            "best_val_loss": train_info["best_val_loss"],
            "score": eval_scores["score"],
            "align_loss": eval_scores.get("align_loss"),
            "pr": eval_scores.get("pr"),
            "gmm_gap": eval_scores.get("gmm_gap"),
            "mahalanobis": eval_scores.get("mahalanobis"),
        }
        rows.append(row)
        if mlflow_client is not None:
            safe_tier = tier.name.replace("-", "_")
            mlflow_client.log_metrics({
                f"{safe_tier}_score": float(row["score"]),
                f"{safe_tier}_best_val_loss": float(row["best_val_loss"]),
                f"{safe_tier}_n_params": float(row["n_params"]),
                f"{safe_tier}_n_events_per_class": float(row["n_events_per_class"]),
            }, step=step)
            for metric_name in ["align_loss", "pr", "gmm_gap", "mahalanobis"]:
                if row.get(metric_name) is not None:
                    mlflow_client.log_metric(f"{safe_tier}_{metric_name}", float(row[metric_name]), step=step)
            mlflow_client.log_metric("scale_score_by_tier", float(row["score"]), step=step)
            mlflow_client.log_metric("scale_events_per_class", float(row["n_events_per_class"]), step=step)
        print(f"    -> score={eval_scores['score']:.4f}")

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Model-size x sample-count scaling sweep for the MHA encoder.")
    parser.add_argument("--output", default="scale_test_results.csv")
    parser.add_argument("--grid", default="quick", choices=sorted(GRIDS),
                        help="quick is shorter; scaling is the small-to-reasonable sweep.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--eval-events-per-class", type=int, default=2000)
    parser.add_argument("--data-source", default="collide2v_updated",
                        choices=["collide2v_updated", "samples_json", "hf"])
    parser.add_argument("--collide2v-path", default=None)
    parser.add_argument("--synthetic", action="store_true",
                        help="Use random synthetic data instead of real Collide-2V data "
                             "(wiring smoke test only, no /eos access needed).")
    parser.add_argument("--device", default=None)
    parser.add_argument("--mlflow-enable", action="store_true",
                        help="Log the scale-test table and tier metrics to MLflow.")
    parser.add_argument("--mlflow-tracking-uri", default="https://ml.cern.ch/models")
    parser.add_argument("--mlflow-experiment", default="ngt_model")
    parser.add_argument("--mlflow-run-name", default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    cfg_kwargs = dict(data_source=args.data_source, n_constituents=N_CONSTITUENTS)
    if args.collide2v_path:
        cfg_kwargs["collide2v_path"] = args.collide2v_path
    cfg = Collide2VConfig(**cfg_kwargs)

    grid = SYNTHETIC_GRID if args.synthetic else GRIDS[args.grid]
    eval_events = min(args.eval_events_per_class, 32) if args.synthetic else args.eval_events_per_class
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    mlflow_client = setup_mlflow(args, grid, eval_events, device) if args.mlflow_enable else None
    if mlflow_client is not None:
        print(f"mlflow=enabled uri={args.mlflow_tracking_uri} experiment={args.mlflow_experiment}")
    else:
        print("mlflow=disabled (add --mlflow-enable to log this sweep)")

    try:
        df = run_sweep(grid, cfg, epochs=args.epochs, eval_events_per_class=eval_events,
                       synthetic=args.synthetic, device=device, mlflow_client=mlflow_client)
        df.to_csv(args.output, index=False)
        if mlflow_client is not None:
            mlflow_client.log_artifact(args.output)
        print(f"\nSaved sweep results -> {args.output}")
        print(df.to_string(index=False))
    finally:
        if mlflow_client is not None:
            mlflow_client.end_run()


if __name__ == "__main__":
    main()
