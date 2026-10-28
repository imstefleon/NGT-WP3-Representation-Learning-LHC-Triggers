#!/usr/bin/env python3
"""
Latent-space effective-dimensionality diagnostics.

This module reads a saved `eval_latents.npy` file from a run directory and
computes the covariance eigenvalue spectrum, PCA explained variance,
effective rank, and participation ratio. It is intended to diagnose whether a
nominally high-dimensional latent space is using many independent directions or
collapsing onto a small number of axes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _safe_entropy_effective_rank(explained_variance_ratio: np.ndarray) -> float:
    positive = explained_variance_ratio[explained_variance_ratio > 0]
    if len(positive) == 0:
        return 0.0
    entropy = -np.sum(positive * np.log(positive))
    return float(np.exp(entropy))


def _n_components_for(cumulative: np.ndarray, threshold: float) -> int:
    idx = np.searchsorted(cumulative, threshold, side="left")
    return int(min(idx + 1, len(cumulative)))


def compute_latent_diagnostics(latents: np.ndarray) -> tuple[dict, pd.DataFrame]:
    if latents.ndim != 2:
        raise ValueError(f"Expected a 2D latent array, got shape={latents.shape}")
    if len(latents) < 2:
        raise ValueError("At least two latent vectors are required")

    z = np.asarray(latents, dtype=np.float64)
    z_centered = z - z.mean(axis=0, keepdims=True)
    cov = np.cov(z_centered, rowvar=False)

    eigenvalues = np.linalg.eigvalsh(cov)[::-1]
    eigenvalues = np.maximum(eigenvalues, 0.0)
    total_variance = float(eigenvalues.sum())
    if total_variance <= 0:
        explained = np.zeros_like(eigenvalues)
    else:
        explained = eigenvalues / total_variance
    cumulative = np.cumsum(explained)

    participation_ratio = float((total_variance ** 2) / max(float(np.sum(eigenvalues ** 2)), 1e-12))
    effective_rank = _safe_entropy_effective_rank(explained)
    latent_dim = int(z.shape[1])

    summary = {
        "n_events": int(z.shape[0]),
        "latent_dim": latent_dim,
        "total_variance": total_variance,
        "effective_rank_entropy": effective_rank,
        "participation_ratio": participation_ratio,
        "participation_ratio_norm": participation_ratio / max(latent_dim, 1),
        "n_components_50pct": _n_components_for(cumulative, 0.50),
        "n_components_80pct": _n_components_for(cumulative, 0.80),
        "n_components_90pct": _n_components_for(cumulative, 0.90),
        "n_components_95pct": _n_components_for(cumulative, 0.95),
        "n_components_99pct": _n_components_for(cumulative, 0.99),
        "top1_explained_variance": float(explained[0]) if len(explained) else 0.0,
        "top2_cumulative_explained_variance": float(cumulative[1]) if len(cumulative) > 1 else float(cumulative[0]),
        "top5_cumulative_explained_variance": float(cumulative[min(4, len(cumulative) - 1)]),
        "top10_cumulative_explained_variance": float(cumulative[min(9, len(cumulative) - 1)]),
        "top20_cumulative_explained_variance": float(cumulative[min(19, len(cumulative) - 1)]),
    }

    spectrum = pd.DataFrame({
        "component": np.arange(1, len(eigenvalues) + 1, dtype=int),
        "eigenvalue": eigenvalues,
        "explained_variance_ratio": explained,
        "cumulative_explained_variance": cumulative,
    })
    return summary, spectrum


def save_plot(spectrum: pd.DataFrame, output_path: Path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(spectrum["component"], spectrum["explained_variance_ratio"], marker="o", markersize=2)
    axes[0].set_title("Latent covariance spectrum")
    axes[0].set_xlabel("component")
    axes[0].set_ylabel("explained variance ratio")
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(spectrum["component"], spectrum["cumulative_explained_variance"], marker="o", markersize=2)
    axes[1].axhline(0.90, color="tab:orange", linestyle="--", linewidth=1)
    axes[1].axhline(0.95, color="tab:red", linestyle="--", linewidth=1)
    axes[1].set_title("Cumulative explained variance")
    axes[1].set_xlabel("component")
    axes[1].set_ylabel("cumulative explained variance")
    axes[1].set_ylim(0, 1.02)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Compute latent effective-dimensionality diagnostics.")
    parser.add_argument("run_dir", help="Run directory containing eval_latents.npy")
    parser.add_argument("--latents-file", default="eval_latents.npy")
    parser.add_argument("--output-prefix", default="latent_effective_dim")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    latents_path = run_dir / args.latents_file
    if not latents_path.exists():
        raise FileNotFoundError(f"Missing latent file: {latents_path}")

    latents = np.load(latents_path)
    summary, spectrum = compute_latent_diagnostics(latents)

    summary_path = run_dir / f"{args.output_prefix}_summary.json"
    spectrum_path = run_dir / f"{args.output_prefix}_spectrum.csv"
    plot_path = run_dir / "figures" / f"{args.output_prefix}_spectrum.png"

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    spectrum.to_csv(spectrum_path, index=False)
    if not args.no_plot:
        save_plot(spectrum, plot_path)

    print("Latent effective-dimensionality diagnostics")
    print("=" * 50)
    print(f"run_dir: {run_dir}")
    print(f"latents: {latents_path}")
    print(f"n_events: {summary['n_events']:,}")
    print(f"latent_dim: {summary['latent_dim']}")
    print(f"effective_rank_entropy: {summary['effective_rank_entropy']:.4f}")
    print(f"participation_ratio: {summary['participation_ratio']:.4f}")
    print(f"top1 explained variance: {summary['top1_explained_variance']:.4f}")
    print(f"top5 cumulative explained variance: {summary['top5_cumulative_explained_variance']:.4f}")
    print(f"top10 cumulative explained variance: {summary['top10_cumulative_explained_variance']:.4f}")
    print(f"n_components_90pct: {summary['n_components_90pct']}")
    print(f"n_components_95pct: {summary['n_components_95pct']}")
    print(f"summary -> {summary_path}")
    print(f"spectrum -> {spectrum_path}")
    if not args.no_plot:
        print(f"plot -> {plot_path}")


if __name__ == "__main__":
    main()
