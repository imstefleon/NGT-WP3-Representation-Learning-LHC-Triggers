#!/usr/bin/env python3
"""Aggregate epoch-wise external anomaly injection metrics."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_RUN_DIR = (
    "outputs/scaling_study/large_100m_80m/one_24h_100m/runs/2026-08-24/"
    "scaling_4class_80m_dmodel2016_ff8064_100m_400ep_noaug_ce_infonce005_lr1e5_seed42"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    return parser.parse_args()


def infer_epoch(path: Path) -> int:
    match = re.search(r"epoch_(\d+)", str(path))
    if not match:
        raise ValueError(f"Cannot infer epoch from {path}")
    return int(match.group(1))


def load_metrics(run_dir: Path) -> pd.DataFrame:
    rows = []
    for metrics_path in sorted(run_dir.glob("figures/anomaly_injection_epoch_*/external_higgs_anomaly_metrics.csv")):
        df = pd.read_csv(metrics_path)
        df["epoch"] = infer_epoch(metrics_path)
        df["metrics_path"] = str(metrics_path)
        rows.append(df)
    if not rows:
        raise FileNotFoundError(f"No anomaly injection metrics found under {run_dir / 'figures'}")
    return pd.concat(rows, ignore_index=True).sort_values(["signal", "epoch"])


def plot_metric_grid(df: pd.DataFrame, out_path: Path) -> None:
    metrics = [
        ("roc_auc_vs_sm_background", "AUC vs SM background"),
        ("eff_at_95pct_bg", "eff @ 95% bkg rejection"),
        ("eff_at_99pct_bg", "eff @ 99% bkg rejection"),
        ("eff_at_99_5pct_bg", "eff @ 99.5% bkg rejection"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for ax, (column, title) in zip(axes.ravel(), metrics):
        for signal, sdf in df.groupby("signal"):
            ax.plot(sdf["epoch"], sdf[column], marker="o", label=signal)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.set_ylabel(column)
        ax.grid(True, alpha=0.25)
        ax.set_ylim(bottom=0.0)
    axes[0, 0].legend()
    fig.suptitle("External anomaly injection over training epochs", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    run_dir = Path(parse_args().run_dir)
    out_dir = run_dir / "figures" / "anomaly_injection_evolution"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_metrics(run_dir)
    df.to_csv(out_dir / "anomaly_injection_evolution_summary.csv", index=False)
    plot_metric_grid(df, out_dir / "anomaly_injection_evolution_metrics.png")
    print(f"saved: {out_dir / 'anomaly_injection_evolution_summary.csv'}")
    print(f"saved: {out_dir / 'anomaly_injection_evolution_metrics.png'}")


if __name__ == "__main__":
    main()
