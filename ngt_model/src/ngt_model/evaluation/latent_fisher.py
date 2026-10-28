#!/usr/bin/env python3
"""
Empirical Fisher diagnostic in latent space.

This module estimates which latent dimensions are task-relevant by computing
the mean squared gradient of the classification loss with respect to each
latent coordinate:

    F_j = mean_events((d loss / d z_j) ** 2)

It is a lightweight first approximation before attempting a full Hessian. The
diagnostic is intended to complement PCA/covariance latent diagnostics: PCA
shows where the representation has variance, while this Fisher diagonal shows
which latent directions affect the supervised classification objective.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

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


class TensorDatasetTorch(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32, copy=False))
        self.y = torch.from_numpy(y.astype(np.int64, copy=False))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def parse_csv_list(raw: str) -> list[str]:
    values = [part.strip().lower() for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("CSV list must contain at least one value")
    return values


def _safe_entropy_effective_rank(weights: np.ndarray) -> float:
    positive = weights[weights > 0]
    if len(positive) == 0:
        return 0.0
    entropy = -np.sum(positive * np.log(positive))
    return float(np.exp(entropy))


def _n_components_for(cumulative: np.ndarray, threshold: float) -> int:
    idx = np.searchsorted(cumulative, threshold, side="left")
    return int(min(idx + 1, len(cumulative)))


def _expected_input_features(model: torch.nn.Module) -> int | None:
    if hasattr(model, "input_proj") and hasattr(model.input_proj, "in_features"):
        return int(model.input_proj.in_features)
    if hasattr(model, "encoder"):
        return _expected_input_features(model.encoder)
    return None


def _align_feature_dim(X: np.ndarray, expected_features: int | None, name: str) -> np.ndarray:
    if expected_features is None:
        return X

    actual_features = X.shape[-1]
    if actual_features == expected_features:
        return X
    if actual_features < expected_features:
        pad = expected_features - actual_features
        print(
            f"WARNING: {name} has {actual_features} features but model expects "
            f"{expected_features}; padding {pad} zero feature(s)."
        )
        return np.pad(X, ((0, 0), (0, 0), (0, pad)), mode="constant")

    print(
        f"WARNING: {name} has {actual_features} features but model expects "
        f"{expected_features}; truncating {actual_features - expected_features} feature(s)."
    )
    return X[:, :, :expected_features]


def _load_model(model_path: Path, device: torch.device) -> torch.nn.Module:
    try:
        model = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        model = torch.load(model_path, map_location=device)
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"Expected a saved torch.nn.Module, got {type(model)} from {model_path}")
    model.to(device)
    model.eval()
    return model


def infer_run_context(target: Path, model_arg: str | None) -> tuple[Path | None, Path]:
    if target.is_dir():
        run_dir = target
        model_path = Path(model_arg) if model_arg else run_dir / "models" / "mha_nodes_aligned_best.pt"
        return run_dir, model_path
    return None, target


def load_run_config(run_dir: Path | None) -> dict:
    if run_dir is None:
        return {}
    config_path = run_dir / "run_config.json"
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def prepare_eval_data(args, run_config: dict) -> tuple[np.ndarray, np.ndarray, list[str]]:
    class_names = parse_csv_list(args.class_names) if args.class_names else run_config.get("class_names", CLASS_NAMES)
    unknown = [cls for cls in class_names if cls not in SUPPORTED_CLASS_NAMES]
    if unknown:
        raise ValueError(f"Unsupported classes {unknown}; supported classes are {SUPPORTED_CLASS_NAMES}")

    cfg_kwargs = dict(
        data_source=args.data_source or run_config.get("data_source", "collide2v_updated"),
        n_constituents=int(args.n_constituents or run_config.get("n_constituents", 100)),
        feature_mode=args.feature_mode or run_config.get("feature_mode", "l1_only"),
        candidate_selection=args.candidate_selection or run_config.get("candidate_selection", "top_pt"),
        class_names=class_names,
    )
    if args.collide2v_path:
        cfg_kwargs["collide2v_path"] = Path(args.collide2v_path)

    cfg = Collide2VConfig(**cfg_kwargs)
    loader = Collide2VLoader(cfg)
    events_per_class = {cls: args.n_events_per_class for cls in class_names}
    raw_tensor, y, _, _ = loader.load_classes(events_per_class, cache_prefix="latent_fisher")

    train_idx, _, eval_idx = build_split_indices(
        y,
        class_names=class_names,
        label_map=cfg.label_map,
        seed=args.seed,
    )
    norm = compute_normalization_constants(raw_tensor[train_idx])
    pid_mapping = build_pid_mapping(raw_tensor)
    X_all, _ = preprocess_dataset(raw_tensor, norm=norm, pid_mapping=pid_mapping, add_pid_ohe=True)
    return X_all[eval_idx], y[eval_idx], class_names


def compute_latent_fisher(
    model: torch.nn.Module,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[dict, pd.DataFrame]:
    expected_features = _expected_input_features(model)
    X_eval = _align_feature_dim(X_eval, expected_features, "preprocessed eval tensor")

    loader = DataLoader(TensorDatasetTorch(X_eval, y_eval), batch_size=batch_size, shuffle=False, num_workers=0)
    n_events = 0
    loss_sum = 0.0
    correct = 0
    fisher_sum = None
    abs_grad_sum = None
    abs_grad_max = None
    z_sum = None
    z2_sum = None

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        model.zero_grad(set_to_none=True)
        out = model(xb)
        if not isinstance(out, dict) or "latent" not in out or "logits" not in out:
            raise TypeError("Model forward must return a dict with 'latent' and 'logits'. Use the full model checkpoint.")

        z = out["latent"]
        z.retain_grad()
        logits = out["logits"]
        loss = F.cross_entropy(logits, yb)
        loss.backward()

        grad = z.grad.detach()
        z_detached = z.detach()
        bs = len(xb)

        grad_sq = grad.pow(2).sum(dim=0).cpu().double().numpy()
        grad_abs = grad.abs()
        z_batch = z_detached.cpu().double().numpy()

        if fisher_sum is None:
            latent_dim = grad.shape[1]
            fisher_sum = np.zeros(latent_dim, dtype=np.float64)
            abs_grad_sum = np.zeros(latent_dim, dtype=np.float64)
            abs_grad_max = np.zeros(latent_dim, dtype=np.float64)
            z_sum = np.zeros(latent_dim, dtype=np.float64)
            z2_sum = np.zeros(latent_dim, dtype=np.float64)

        fisher_sum += grad_sq
        abs_grad_sum += grad_abs.sum(dim=0).cpu().double().numpy()
        abs_grad_max = np.maximum(abs_grad_max, grad_abs.max(dim=0).values.cpu().double().numpy())
        z_sum += z_batch.sum(axis=0)
        z2_sum += (z_batch ** 2).sum(axis=0)
        n_events += bs
        loss_sum += float(loss.item()) * bs
        correct += int((logits.argmax(dim=1) == yb).sum().item())

    if fisher_sum is None or n_events == 0:
        raise RuntimeError("No events were processed for Fisher computation")

    fisher_diag = fisher_sum / n_events
    grad_abs_mean = abs_grad_sum / n_events
    latent_mean = z_sum / n_events
    latent_variance = np.maximum((z2_sum / n_events) - latent_mean ** 2, 0.0)

    fisher_total = float(fisher_diag.sum())
    fisher_mass = fisher_diag / fisher_total if fisher_total > 0 else np.zeros_like(fisher_diag)
    fisher_cumulative = np.cumsum(np.sort(fisher_mass)[::-1])
    variance_total = float(latent_variance.sum())
    variance_mass = latent_variance / variance_total if variance_total > 0 else np.zeros_like(latent_variance)

    fisher_pr = float((fisher_total ** 2) / max(float(np.sum(fisher_diag ** 2)), 1e-30))
    fisher_erank = _safe_entropy_effective_rank(fisher_mass)
    latent_dim = int(len(fisher_diag))

    order = np.argsort(fisher_diag)[::-1]
    rank = np.empty_like(order)
    rank[order] = np.arange(1, latent_dim + 1)

    by_dim = pd.DataFrame({
        "latent_dim_index": np.arange(latent_dim, dtype=int),
        "fisher_diag": fisher_diag,
        "fisher_mass": fisher_mass,
        "fisher_rank": rank,
        "latent_variance": latent_variance,
        "latent_variance_mass": variance_mass,
        "grad_abs_mean": grad_abs_mean,
        "grad_abs_max": abs_grad_max,
        "latent_mean": latent_mean,
    }).sort_values("fisher_rank").reset_index(drop=True)
    by_dim["cumulative_fisher_mass"] = by_dim["fisher_mass"].cumsum()

    summary = {
        "n_events": int(n_events),
        "latent_dim": latent_dim,
        "mean_loss": float(loss_sum / n_events),
        "accuracy": float(correct / n_events),
        "fisher_total": fisher_total,
        "fisher_effective_rank_entropy": fisher_erank,
        "fisher_participation_ratio": fisher_pr,
        "fisher_participation_ratio_norm": fisher_pr / max(latent_dim, 1),
        "fisher_top1_mass": float(fisher_cumulative[0]) if len(fisher_cumulative) else 0.0,
        "fisher_top5_mass": float(fisher_cumulative[min(4, len(fisher_cumulative) - 1)]),
        "fisher_top10_mass": float(fisher_cumulative[min(9, len(fisher_cumulative) - 1)]),
        "fisher_dims_50pct": _n_components_for(fisher_cumulative, 0.50),
        "fisher_dims_80pct": _n_components_for(fisher_cumulative, 0.80),
        "fisher_dims_90pct": _n_components_for(fisher_cumulative, 0.90),
        "fisher_dims_95pct": _n_components_for(fisher_cumulative, 0.95),
        "latent_variance_total": variance_total,
        "active_dims_fisher_gt_1pct": int(np.sum(fisher_mass > 0.01)),
        "active_dims_variance_gt_1pct": int(np.sum(variance_mass > 0.01)),
    }
    return summary, by_dim


def save_plot(by_dim: pd.DataFrame, output_path: Path):
    import matplotlib.pyplot as plt

    ordered = by_dim.sort_values("fisher_rank")
    by_index = by_dim.sort_values("latent_dim_index")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(np.arange(1, len(ordered) + 1), ordered["fisher_mass"], marker="o", markersize=2)
    axes[0].set_title("Fisher mass by rank")
    axes[0].set_xlabel("ranked latent dimension")
    axes[0].set_ylabel("Fisher mass")
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(np.arange(1, len(ordered) + 1), ordered["cumulative_fisher_mass"], marker="o", markersize=2)
    axes[1].axhline(0.90, color="tab:orange", linestyle="--", linewidth=1)
    axes[1].axhline(0.95, color="tab:red", linestyle="--", linewidth=1)
    axes[1].set_title("Cumulative Fisher mass")
    axes[1].set_xlabel("ranked latent dimension")
    axes[1].set_ylabel("cumulative mass")
    axes[1].set_ylim(0, 1.02)
    axes[1].grid(True, alpha=0.3)

    axes[2].scatter(by_index["latent_variance"], by_index["fisher_diag"], s=16, alpha=0.8)
    axes[2].set_title("Variance vs Fisher")
    axes[2].set_xlabel("latent variance")
    axes[2].set_ylabel("Fisher diagonal")
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Compute empirical Fisher diagonal in latent space.")
    parser.add_argument("target", help="Run directory or full-model checkpoint path")
    parser.add_argument("--model", default=None, help="Checkpoint path when target is a run directory")
    parser.add_argument("--n-events-per-class", type=int, default=2000)
    parser.add_argument("--class-names", default=None,
                        help="Comma-separated class names. Defaults to run_config.json when available.")
    parser.add_argument("--data-source", default=None, choices=["collide2v_updated", "samples_json", "hf"])
    parser.add_argument("--feature-mode", default=None, choices=["l1_only"])
    parser.add_argument("--candidate-selection", default=None)
    parser.add_argument("--collide2v-path", default=None)
    parser.add_argument("--n-constituents", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-prefix", default="latent_fisher")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    target = Path(args.target)
    run_dir, model_path = infer_run_context(target, args.model)
    run_config = load_run_config(run_dir)
    output_dir = run_dir if run_dir is not None else model_path.parent

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print("Latent empirical-Fisher diagnostics")
    print("=" * 50)
    print(f"target: {target}")
    print(f"model: {model_path}")
    print(f"device: {device}")

    model = _load_model(model_path, device)
    X_eval, y_eval, class_names = prepare_eval_data(args, run_config)
    summary, by_dim = compute_latent_fisher(model, X_eval, y_eval, device, args.batch_size)
    summary["model_path"] = str(model_path)
    summary["class_names"] = class_names
    summary["n_events_per_class"] = args.n_events_per_class

    summary_path = output_dir / f"{args.output_prefix}_summary.json"
    by_dim_path = output_dir / f"{args.output_prefix}_by_dim.csv"
    plot_path = output_dir / "figures" / f"{args.output_prefix}_diag.png"

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    by_dim.to_csv(by_dim_path, index=False)
    if not args.no_plot:
        save_plot(by_dim, plot_path)

    print(f"n_events: {summary['n_events']:,}")
    print(f"latent_dim: {summary['latent_dim']}")
    print(f"mean_loss: {summary['mean_loss']:.6f}")
    print(f"accuracy: {summary['accuracy']:.4f}")
    print(f"fisher_effective_rank_entropy: {summary['fisher_effective_rank_entropy']:.4f}")
    print(f"fisher_participation_ratio: {summary['fisher_participation_ratio']:.4f}")
    print(f"fisher_top5_mass: {summary['fisher_top5_mass']:.4f}")
    print(f"fisher_dims_90pct: {summary['fisher_dims_90pct']}")
    print(f"active_dims_fisher_gt_1pct: {summary['active_dims_fisher_gt_1pct']}")
    print(f"summary -> {summary_path}")
    print(f"by_dim -> {by_dim_path}")
    if not args.no_plot:
        print(f"plot -> {plot_path}")


if __name__ == "__main__":
    main()
