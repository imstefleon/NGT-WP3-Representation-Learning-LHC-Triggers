#!/usr/bin/env python3
"""Collect completed scaling-study runs into one analysis table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


ANOMALY_OUTPUT_SUBDIR = Path("figures") / "external_higgs_anomaly"
MASTER_COLUMNS = [
    "run_status",
    "run_dir",
    "run_name",
    "n_classes",
    "class_configuration",
    "class_names",
    "samples_per_class",
    "model_size",
    "n_parameters",
    "n_encoder_params",
    "n_classifier_params",
    "latent_dim",
    "ff_dim",
    "num_heads",
    "num_attention_blocks",
    "pooling_mode",
    "classifier_head",
    "probe_architecture",
    "seed",
    "signal",
    "anomaly_method",
    "median_anomaly_score",
    "p90_anomaly_score",
    "p95_anomaly_score",
    "auc_vs_background",
    "eff_at_95_bg_rejection",
    "eff_at_99_bg_rejection",
    "pca_dims_90",
    "pca_dims_95",
    "pca_dims_99",
    "effective_rank",
    "participation_ratio",
]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"WARNING: failed to read {path}: {exc}")
        return {}


def find_run_dir(output_root: Path, run_name: str) -> Path | None:
    candidates = sorted((output_root / "runs").glob(f"*/{run_name}"))
    if not candidates:
        return None
    return candidates[-1]


def collect_one(manifest_row: pd.Series) -> tuple[list[dict[str, Any]], str]:
    output_root = Path(str(manifest_row["output_root"]))
    run_name = str(manifest_row["run_name"])
    run_dir = find_run_dir(output_root, run_name)
    if run_dir is None:
        return [], f"missing run directory for {run_name}"

    config = read_json(run_dir / "run_config.json")
    latent_diag = read_json(run_dir / "latent_diagnostics_summary.json")
    metrics_path = run_dir / ANOMALY_OUTPUT_SUBDIR / "external_higgs_anomaly_metrics_by_method.csv"
    if not metrics_path.exists():
        return [], f"missing anomaly metrics for {run_name}: {metrics_path}"

    metrics = pd.read_csv(metrics_path)
    rows = []
    class_names = config.get("class_names") or str(manifest_row.get("class_names", "")).split(",")
    for _, metric in metrics.iterrows():
        rows.append(
            {
                "run_status": "completed",
                "run_dir": str(run_dir),
                "run_name": run_name,
                "n_classes": int(config.get("n_classes", len(class_names))),
                "class_configuration": manifest_row["class_configuration"],
                "class_names": ",".join(class_names) if isinstance(class_names, list) else str(class_names),
                "samples_per_class": int(config.get("n_events_per_class", manifest_row["samples_per_class"])),
                "model_size": manifest_row["model_size"],
                "n_parameters": config.get("n_params", ""),
                "n_encoder_params": config.get("n_encoder_params", ""),
                "n_classifier_params": config.get("n_classifier_params", ""),
                "latent_dim": config.get("latent_dim", manifest_row["latent_dim"]),
                "ff_dim": config.get("ff_dim", manifest_row["ff_dim"]),
                "num_heads": config.get("num_heads", manifest_row["num_heads"]),
                "num_attention_blocks": config.get(
                    "num_attention_blocks",
                    manifest_row["num_attention_blocks"],
                ),
                "pooling_mode": config.get("pooling_mode", ""),
                "classifier_head": config.get("classifier_head", ""),
                "probe_architecture": config.get("classifier_probe_architecture", ""),
                "seed": config.get("seed", ""),
                "signal": metric["signal"],
                "anomaly_method": metric["anomaly_method"],
                "median_anomaly_score": metric["median_anomaly_score"],
                "p90_anomaly_score": metric["p90_anomaly_score"],
                "p95_anomaly_score": metric["p95_anomaly_score"],
                "auc_vs_background": metric["auc_vs_background"],
                "eff_at_95_bg_rejection": metric["eff_at_95_bg_rejection"],
                "eff_at_99_bg_rejection": metric["eff_at_99_bg_rejection"],
                "pca_dims_90": latent_diag.get("n_components_90pct", ""),
                "pca_dims_95": latent_diag.get("n_components_95pct", ""),
                "pca_dims_99": latent_diag.get("n_components_99pct", ""),
                "effective_rank": latent_diag.get("effective_rank_entropy", ""),
                "participation_ratio": latent_diag.get("participation_ratio", ""),
            }
        )
    return rows, ""


def parse_args():
    parser = argparse.ArgumentParser(description="Build master CSV from completed scaling-study runs.")
    parser.add_argument("--manifest", default="outputs/scaling_study/scaling_study_manifest.csv")
    parser.add_argument("--output", default="outputs/scaling_study/master_results_table.csv")
    parser.add_argument("--missing-output", default="outputs/scaling_study/master_results_missing.txt")
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    rows = []
    missing = []
    for _, manifest_row in manifest.iterrows():
        new_rows, reason = collect_one(manifest_row)
        rows.extend(new_rows)
        if reason:
            missing.append(reason)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=MASTER_COLUMNS).to_csv(output, index=False)
    Path(args.missing_output).write_text("\n".join(missing), encoding="utf-8")

    print(f"collected rows: {len(rows)}")
    print(f"missing runs/artifacts: {len(missing)}")
    print(f"master table: {output}")
    if missing:
        print(f"missing details: {args.missing_output}")


if __name__ == "__main__":
    main()
