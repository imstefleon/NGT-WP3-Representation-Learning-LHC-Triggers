#!/usr/bin/env python3
"""Collect power-law sweep results and fit log-log trends.

Run from the repository root after the four runs and external Higgs evaluations
are complete:

    python scaling_study/power_law_param_sample/commands/collect_and_fit_power_law.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "scaling_study" / "power_law_param_sample"
MANIFEST = STUDY / "power_law_manifest.csv"
OUTDIR = STUDY / "summaries"


def latest_run_dir(output_root: str, run_name: str) -> Path:
    base = ROOT / output_root / "runs"
    candidates = sorted(base.glob(f"*/{run_name}"))
    if not candidates:
        raise FileNotFoundError(f"missing run directory for {run_name} under {base}")
    return candidates[-1]


def read_metric(run_dir: Path, signal: str) -> dict:
    metrics = pd.read_csv(run_dir / "figures" / "external_higgs_anomaly" / "external_higgs_anomaly_metrics.csv")
    row = metrics.loc[metrics["signal"] == signal].iloc[0].to_dict()
    if "eff_at_99_5pct_bg" not in row:
        scores = pd.read_csv(run_dir / "figures" / "external_higgs_anomaly" / "external_higgs_event_scores.csv")
        bg = scores.loc[scores["score_split"] == "background_eval", "minbias_distance_score"].to_numpy(float)
        sig = scores.loc[scores["sample_class"] == signal, "minbias_distance_score"].to_numpy(float)
        row["eff_at_99_5pct_bg"] = float(np.mean(sig > np.percentile(bg, 99.5)))
    return row


def fit_loglog(x: np.ndarray, y: np.ndarray) -> dict:
    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return {"slope": np.nan, "intercept": np.nan, "n_points": int(mask.sum())}
    slope, intercept = np.polyfit(np.log10(x[mask]), np.log10(y[mask]), deg=1)
    return {"slope": float(slope), "intercept": float(intercept), "n_points": int(mask.sum())}


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST)
    rows = []
    for _, point in manifest.iterrows():
        run_dir = latest_run_dir(point["output_root"], point["run_name"])
        for signal in ["hh4b", "vbfh"]:
            metric = read_metric(run_dir, signal)
            rows.append(
                {
                    "point": point["point"],
                    "signal": signal,
                    "actual_params": int(point["actual_params"]),
                    "total_samples": int(point["total_samples"]),
                    "ff_dim": int(point["ff_dim"]),
                    "run_dir": str(run_dir),
                    "auc": float(metric["roc_auc_vs_sm_background"]),
                    "eff_at_95pct_bg": float(metric["eff_at_95pct_bg"]),
                    "eff_at_99pct_bg": float(metric["eff_at_99pct_bg"]),
                    "eff_at_99_5pct_bg": float(metric["eff_at_99_5pct_bg"]),
                }
            )
    results = pd.DataFrame(rows)
    results.to_csv(OUTDIR / "power_law_results.csv", index=False)

    fits = []
    for signal, part in results.groupby("signal"):
        for metric in ["eff_at_95pct_bg", "eff_at_99pct_bg", "eff_at_99_5pct_bg"]:
            fits.append({"signal": signal, "metric": metric, "x": "params", **fit_loglog(part["actual_params"].to_numpy(float), part[metric].to_numpy(float))})
            fits.append({"signal": signal, "metric": metric, "x": "samples", **fit_loglog(part["total_samples"].to_numpy(float), part[metric].to_numpy(float))})
    (OUTDIR / "power_law_fits.json").write_text(json.dumps(fits, indent=2), encoding="utf-8")

    for metric in ["eff_at_99pct_bg", "eff_at_99_5pct_bg"]:
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        for signal, part in results.groupby("signal"):
            ax.plot(part["actual_params"], part[metric], marker="o", linewidth=2.0, label=signal.upper())
            for _, row in part.iterrows():
                ax.text(row["actual_params"], row[metric] + 0.004, f"ff{int(row['ff_dim'])}", ha="center", fontsize=8)
        ax.set_xscale("log")
        ax.set_xlabel("trainable parameters")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} vs parameters, samples scaled in tandem")
        ax.grid(alpha=0.25, which="both")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(OUTDIR / f"{metric}_vs_params.png", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    main()
