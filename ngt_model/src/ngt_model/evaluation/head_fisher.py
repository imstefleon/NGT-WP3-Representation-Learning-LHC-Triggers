#!/usr/bin/env python3
"""
Empirical Fisher diagnostic for the classifier head.

This is a lightweight Hessian/Fisher approximation on model parameters. It
complements `latent_fisher.py`: latent Fisher asks which latent coordinates
matter for the loss, while this module asks how concentrated the classifier
head curvature is across its own trainable parameters.

The estimate uses squared gradients of the batch-mean cross-entropy loss:

    F_theta ~= mean_batches((d loss_batch / d theta) ** 2)

It is not a full Hessian eigensystem, but it is cheap enough to run repeatedly
and useful as a first check before heavier Hessian/Fisher approximations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ngt_model.evaluation.latent_fisher import (
    TensorDatasetTorch,
    _align_feature_dim,
    _expected_input_features,
    _load_model,
    _n_components_for,
    _safe_entropy_effective_rank,
    infer_run_context,
    load_run_config,
    prepare_eval_data,
)


SCOPE_PREFIXES = {
    "classifier_head": ("cls_hidden.", "class_logits."),
    "latent_and_classifier": ("latent.", "cls_hidden.", "class_logits."),
    "all_trainable": ("",),
}


def select_parameters(model: torch.nn.Module, scope: str) -> list[tuple[str, torch.nn.Parameter]]:
    prefixes = SCOPE_PREFIXES[scope]
    selected = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(name.startswith(prefix) for prefix in prefixes):
            selected.append((name, param))
    if not selected:
        raise RuntimeError(f"No trainable parameters matched scope={scope!r}")
    return selected


def _metric_summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    total = float(values.sum())
    mass = values / total if total > 0 else np.zeros_like(values)
    ordered = np.sort(mass)[::-1]
    cumulative = np.cumsum(ordered)
    n = len(values)
    return {
        "fisher_total": total,
        "fisher_effective_rank_entropy": _safe_entropy_effective_rank(mass),
        "fisher_participation_ratio": float((total ** 2) / max(float(np.sum(values ** 2)), 1e-30)),
        "fisher_participation_ratio_norm": float((total ** 2) / max(float(np.sum(values ** 2)), 1e-30)) / max(n, 1),
        "fisher_top1_mass": float(cumulative[0]) if len(cumulative) else 0.0,
        "fisher_top10_mass": float(cumulative[min(9, len(cumulative) - 1)]) if len(cumulative) else 0.0,
        "fisher_top100_mass": float(cumulative[min(99, len(cumulative) - 1)]) if len(cumulative) else 0.0,
        "fisher_dims_50pct": _n_components_for(cumulative, 0.50) if len(cumulative) else 0,
        "fisher_dims_80pct": _n_components_for(cumulative, 0.80) if len(cumulative) else 0,
        "fisher_dims_90pct": _n_components_for(cumulative, 0.90) if len(cumulative) else 0,
        "fisher_dims_95pct": _n_components_for(cumulative, 0.95) if len(cumulative) else 0,
        "active_params_fisher_gt_1pct": int(np.sum(mass > 0.01)),
    }


def compute_head_fisher(
    model: torch.nn.Module,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    device: torch.device,
    batch_size: int,
    scope: str,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    expected_features = _expected_input_features(model)
    X_eval = _align_feature_dim(X_eval, expected_features, "preprocessed eval tensor")

    params = select_parameters(model, scope)
    param_names = [name for name, _ in params]
    param_sizes = np.array([int(param.numel()) for _, param in params], dtype=np.int64)
    param_offsets = np.cumsum(np.concatenate([[0], param_sizes[:-1]]))
    n_params = int(param_sizes.sum())

    fisher_diag = np.zeros(n_params, dtype=np.float64)
    abs_grad_sum = np.zeros(n_params, dtype=np.float64)
    max_grad = np.zeros(n_params, dtype=np.float64)
    group_fisher = np.zeros(len(params), dtype=np.float64)

    loader = DataLoader(TensorDatasetTorch(X_eval, y_eval), batch_size=batch_size, shuffle=False, num_workers=0)
    n_events = 0
    n_batches = 0
    loss_sum = 0.0
    correct = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        model.zero_grad(set_to_none=True)
        out = model(xb)
        if not isinstance(out, dict) or "logits" not in out:
            raise TypeError("Model forward must return a dict with 'logits'. Use the full model checkpoint.")
        logits = out["logits"]
        loss = F.cross_entropy(logits, yb)
        loss.backward()

        cursor = 0
        for group_idx, (_, param) in enumerate(params):
            size = int(param.numel())
            grad = param.grad
            if grad is None:
                values = np.zeros(size, dtype=np.float64)
            else:
                values = grad.detach().reshape(-1).cpu().double().numpy()
            sq = values ** 2
            fisher_diag[cursor:cursor + size] += sq
            abs_grad_sum[cursor:cursor + size] += np.abs(values)
            max_grad[cursor:cursor + size] = np.maximum(max_grad[cursor:cursor + size], np.abs(values))
            group_fisher[group_idx] += float(sq.sum())
            cursor += size

        bs = len(xb)
        n_events += bs
        n_batches += 1
        loss_sum += float(loss.item()) * bs
        correct += int((logits.argmax(dim=1) == yb).sum().item())

    if n_batches == 0:
        raise RuntimeError("No batches were processed for classifier-head Fisher computation")

    fisher_diag /= n_batches
    abs_grad_sum /= n_batches
    max_grad = max_grad
    group_fisher /= n_batches

    total = float(fisher_diag.sum())
    group_mass = group_fisher / total if total > 0 else np.zeros_like(group_fisher)
    summary = {
        "scope": scope,
        "n_events": int(n_events),
        "n_batches": int(n_batches),
        "n_params_scope": int(n_params),
        "mean_loss": float(loss_sum / n_events),
        "accuracy": float(correct / n_events),
        **_metric_summary(fisher_diag),
    }

    group_rows = []
    for name, offset, size, value, mass in zip(param_names, param_offsets, param_sizes, group_fisher, group_mass):
        group_rows.append({
            "parameter_name": name,
            "offset": int(offset),
            "n_params": int(size),
            "fisher_sum": float(value),
            "fisher_mass": float(mass),
            "fisher_mean": float(value / max(int(size), 1)),
        })
    by_parameter = pd.DataFrame(group_rows).sort_values("fisher_mass", ascending=False).reset_index(drop=True)
    by_parameter["cumulative_fisher_mass"] = by_parameter["fisher_mass"].cumsum()

    order = np.argsort(fisher_diag)[::-1]
    spectrum = pd.DataFrame({
        "rank": np.arange(1, n_params + 1, dtype=int),
        "flat_parameter_index": order.astype(int),
        "fisher_diag": fisher_diag[order],
        "fisher_mass": fisher_diag[order] / total if total > 0 else np.zeros(n_params),
        "grad_abs_mean": abs_grad_sum[order],
        "grad_abs_max": max_grad[order],
    })
    spectrum["cumulative_fisher_mass"] = spectrum["fisher_mass"].cumsum()
    return summary, by_parameter, spectrum


def save_plot(by_parameter: pd.DataFrame, spectrum: pd.DataFrame, output_path: Path):
    import matplotlib.pyplot as plt

    top_groups = by_parameter.head(12).sort_values("fisher_mass", ascending=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].barh(top_groups["parameter_name"], top_groups["fisher_mass"])
    axes[0].set_title("Classifier-head Fisher mass")
    axes[0].set_xlabel("mass by parameter tensor")
    axes[0].grid(True, axis="x", alpha=0.3)

    axes[1].plot(spectrum["rank"], spectrum["fisher_diag"])
    axes[1].set_title("Fisher diagonal spectrum")
    axes[1].set_xlabel("ranked parameter")
    axes[1].set_ylabel("Fisher diagonal")
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(spectrum["rank"], spectrum["cumulative_fisher_mass"])
    axes[2].axhline(0.90, color="tab:orange", linestyle="--", linewidth=1)
    axes[2].axhline(0.95, color="tab:red", linestyle="--", linewidth=1)
    axes[2].set_title("Cumulative Fisher mass")
    axes[2].set_xlabel("ranked parameter")
    axes[2].set_ylabel("cumulative mass")
    axes[2].set_ylim(0, 1.02)
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Compute empirical Fisher diagonal for classifier-head parameters.")
    parser.add_argument("target", help="Run directory or full-model checkpoint path")
    parser.add_argument("--model", default=None, help="Checkpoint path when target is a run directory")
    parser.add_argument("--scope", default="classifier_head", choices=sorted(SCOPE_PREFIXES))
    parser.add_argument("--n-events-per-class", type=int, default=2000)
    parser.add_argument("--class-names", default=None)
    parser.add_argument("--data-source", default=None, choices=["collide2v_updated", "samples_json", "hf"])
    parser.add_argument("--feature-mode", default=None, choices=["l1_only"])
    parser.add_argument("--candidate-selection", default=None)
    parser.add_argument("--collide2v-path", default=None)
    parser.add_argument("--n-constituents", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-prefix", default="head_fisher")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    target = Path(args.target)
    run_dir, model_path = infer_run_context(target, args.model)
    run_config = load_run_config(run_dir)
    output_dir = run_dir if run_dir is not None else model_path.parent

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print("Classifier-head empirical-Fisher diagnostics")
    print("=" * 56)
    print(f"target: {target}")
    print(f"model: {model_path}")
    print(f"scope: {args.scope}")
    print(f"device: {device}")

    model = _load_model(model_path, device)
    X_eval, y_eval, class_names = prepare_eval_data(args, run_config)
    summary, by_parameter, spectrum = compute_head_fisher(
        model=model,
        X_eval=X_eval,
        y_eval=y_eval,
        device=device,
        batch_size=args.batch_size,
        scope=args.scope,
    )
    summary["model_path"] = str(model_path)
    summary["class_names"] = class_names
    summary["n_events_per_class"] = args.n_events_per_class

    summary_path = output_dir / f"{args.output_prefix}_summary.json"
    by_parameter_path = output_dir / f"{args.output_prefix}_by_parameter.csv"
    spectrum_path = output_dir / f"{args.output_prefix}_spectrum.csv"
    plot_path = output_dir / "figures" / f"{args.output_prefix}_diag.png"

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    by_parameter.to_csv(by_parameter_path, index=False)
    spectrum.to_csv(spectrum_path, index=False)
    if not args.no_plot:
        save_plot(by_parameter, spectrum, plot_path)

    print(f"n_events: {summary['n_events']:,}")
    print(f"n_params_scope: {summary['n_params_scope']:,}")
    print(f"mean_loss: {summary['mean_loss']:.6f}")
    print(f"accuracy: {summary['accuracy']:.4f}")
    print(f"fisher_effective_rank_entropy: {summary['fisher_effective_rank_entropy']:.4f}")
    print(f"fisher_participation_ratio: {summary['fisher_participation_ratio']:.4f}")
    print(f"fisher_top100_mass: {summary['fisher_top100_mass']:.4f}")
    print(f"fisher_dims_90pct: {summary['fisher_dims_90pct']}")
    print(f"summary -> {summary_path}")
    print(f"by_parameter -> {by_parameter_path}")
    print(f"spectrum -> {spectrum_path}")
    if not args.no_plot:
        print(f"plot -> {plot_path}")


if __name__ == "__main__":
    main()
