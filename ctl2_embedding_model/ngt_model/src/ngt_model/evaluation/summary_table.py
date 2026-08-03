#!/usr/bin/env python3
"""
Build a unified experiment-summary table from run directories and sweep CSVs.

The goal is to keep one comparison table with training metrics, class-level
metrics, latent diagnostics, Fisher diagnostics, eval_quality scores, and
samples/parameter ratios whenever those artifacts exist.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ORDERED_COLUMNS = [
    "source_type",
    "source_path",
    "run_status",
    "missing_artifacts",
    "run_name",
    "run_date",
    "seed",
    "class_names",
    "background_classes",
    "anomaly_class",
    "data_source",
    "n_events_per_class",
    "n_classes",
    "total_events",
    "n_train",
    "n_train_aug",
    "n_val",
    "n_eval",
    "n_params",
    "samples_per_param_total",
    "samples_per_param_train",
    "samples_per_param_train_aug",
    "latent_dim",
    "ff_dim",
    "num_heads",
    "num_attention_blocks",
    "pooling_mode",
    "classifier_head",
    "loss_mode",
    "label_smoothing",
    "focal_gamma",
    "supcon_weight",
    "supcon_temperature",
    "latent_var_reg_weight",
    "latent_var_target",
    "latent_cov_reg_weight",
    "epochs",
    "best_epoch",
    "best_val_loss",
    "best_val_acc",
    "best_val_ce_loss",
    "best_val_latent_var_reg",
    "best_val_latent_cov_reg",
    "best_val_supcon_loss",
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "minbias_precision",
    "minbias_recall",
    "minbias_f1",
    "top_precision",
    "top_recall",
    "top_f1",
    "qcd_precision",
    "qcd_recall",
    "qcd_f1",
    "dy_precision",
    "dy_recall",
    "dy_f1",
    "hh4b_precision",
    "hh4b_recall",
    "hh4b_f1",
    "effective_rank_entropy",
    "participation_ratio",
    "n_components_90pct",
    "n_components_95pct",
    "top1_explained_variance",
    "top5_cumulative_explained_variance",
    "top10_cumulative_explained_variance",
    "latent_fisher_effective_rank_entropy",
    "latent_fisher_participation_ratio",
    "latent_fisher_dims_90pct",
    "active_dims_fisher_gt_1pct",
    "head_fisher_effective_rank_entropy",
    "head_fisher_participation_ratio",
    "head_fisher_dims_90pct",
    "active_params_fisher_gt_1pct",
    "eval_quality_score",
    "eval_quality_align_loss",
    "eval_quality_pr",
    "eval_quality_gmm_gap",
    "eval_quality_mahalanobis",
]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"WARNING: could not read JSON {path}: {exc}")
        return {}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"WARNING: could not read CSV {path}: {exc}")
        return pd.DataFrame()


def _as_csv(values: Any) -> str:
    if values is None:
        return ""
    if isinstance(values, str):
        return values
    if isinstance(values, (list, tuple)):
        return ",".join(str(v) for v in values)
    return str(values)


def _float_or_blank(value: Any) -> Any:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _best_history(run_dir: Path) -> dict[str, Any]:
    history = _read_csv(run_dir / "training_history.csv")
    if history.empty:
        return {}
    row = history.loc[history["val_loss"].idxmin()].to_dict() if "val_loss" in history else history.iloc[-1].to_dict()
    return {
        "best_epoch": int(row.get("epoch", len(history))),
        "best_val_loss": _float_or_blank(row.get("val_loss")),
        "best_val_acc": _float_or_blank(row.get("val_acc")),
        "best_val_ce_loss": _float_or_blank(row.get("val_ce_loss")),
        "best_val_latent_var_reg": _float_or_blank(row.get("val_latent_var_reg")),
        "best_val_latent_cov_reg": _float_or_blank(row.get("val_latent_cov_reg")),
        "best_val_supcon_loss": _float_or_blank(row.get("val_supcon_loss")),
        "epochs": int(history["epoch"].max()) if "epoch" in history else len(history),
    }


def _report_metrics(run_dir: Path) -> dict[str, Any]:
    report = _read_csv(run_dir / "classification_report_eval.csv")
    if report.empty:
        return {}
    label_col = report.columns[0]
    report = report.rename(columns={label_col: "label"}).set_index("label")
    row: dict[str, Any] = {}
    if "accuracy" in report.index:
        row["accuracy"] = _float_or_blank(report.loc["accuracy"].dropna().iloc[0])
    if "macro avg" in report.index:
        row["macro_f1"] = _float_or_blank(report.loc["macro avg"].get("f1-score"))
    if "weighted avg" in report.index:
        row["weighted_f1"] = _float_or_blank(report.loc["weighted avg"].get("f1-score"))
    for cls in ["minbias", "top", "qcd", "dy", "hh4b"]:
        if cls in report.index:
            row[f"{cls}_precision"] = _float_or_blank(report.loc[cls].get("precision"))
            row[f"{cls}_recall"] = _float_or_blank(report.loc[cls].get("recall"))
            row[f"{cls}_f1"] = _float_or_blank(report.loc[cls].get("f1-score"))
    return row


def _first_json_matching(run_dir: Path, pattern: str) -> dict[str, Any]:
    matches = sorted(run_dir.glob(pattern))
    if not matches:
        return {}
    return _read_json(matches[-1])


def _diagnostic_metrics(run_dir: Path) -> dict[str, Any]:
    row: dict[str, Any] = {}
    latent = _read_json(run_dir / "latent_effective_dim_summary.json")
    for key in [
        "effective_rank_entropy",
        "participation_ratio",
        "n_components_90pct",
        "n_components_95pct",
        "top1_explained_variance",
        "top5_cumulative_explained_variance",
        "top10_cumulative_explained_variance",
    ]:
        if key in latent:
            row[key] = _float_or_blank(latent[key])

    latent_fisher = _read_json(run_dir / "latent_fisher_summary.json")
    fisher_map = {
        "fisher_effective_rank_entropy": "latent_fisher_effective_rank_entropy",
        "fisher_participation_ratio": "latent_fisher_participation_ratio",
        "fisher_dims_90pct": "latent_fisher_dims_90pct",
        "active_dims_fisher_gt_1pct": "active_dims_fisher_gt_1pct",
    }
    for src, dst in fisher_map.items():
        if src in latent_fisher:
            row[dst] = _float_or_blank(latent_fisher[src])

    head_fisher = _read_json(run_dir / "head_fisher_summary.json")
    head_map = {
        "fisher_effective_rank_entropy": "head_fisher_effective_rank_entropy",
        "fisher_participation_ratio": "head_fisher_participation_ratio",
        "fisher_dims_90pct": "head_fisher_dims_90pct",
        "active_params_fisher_gt_1pct": "active_params_fisher_gt_1pct",
    }
    for src, dst in head_map.items():
        if src in head_fisher:
            row[dst] = _float_or_blank(head_fisher[src])

    quality = _first_json_matching(run_dir, "eval_quality*.json")
    quality_map = {
        "score": "eval_quality_score",
        "align_loss": "eval_quality_align_loss",
        "pr": "eval_quality_pr",
        "gmm_gap": "eval_quality_gmm_gap",
        "mahalanobis": "eval_quality_mahalanobis",
    }
    for src, dst in quality_map.items():
        if src in quality:
            row[dst] = _float_or_blank(quality[src])
    if "n_params" in quality:
        row["n_params"] = _float_or_blank(quality["n_params"])
    return row


def _add_ratio_columns(row: dict[str, Any]) -> dict[str, Any]:
    n_params = row.get("n_params", "")
    if n_params in ("", None):
        return row
    try:
        n_params_f = float(n_params)
    except (TypeError, ValueError):
        return row
    for src, dst in [
        ("total_events", "samples_per_param_total"),
        ("n_train", "samples_per_param_train"),
        ("n_train_aug", "samples_per_param_train_aug"),
    ]:
        value = row.get(src)
        if value in ("", None):
            continue
        try:
            row[dst] = float(value) / n_params_f
        except (TypeError, ValueError, ZeroDivisionError):
            pass
    return row


def summarize_run(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    config = _read_json(run_dir / "run_config.json")
    class_names = config.get("class_names", [])
    n_classes = len(class_names) if isinstance(class_names, list) else None
    n_events_per_class = config.get("n_events_per_class")
    n_params = config.get("n_params", "")
    total_events = n_events_per_class * n_classes if n_events_per_class and n_classes else ""
    required_artifacts = [
        "run_config.json",
        "training_history.csv",
        "classification_report_eval.csv",
        "confusion_matrix_eval.csv",
        "eval_latents.npy",
    ]
    missing_artifacts = [name for name in required_artifacts if not (run_dir / name).exists()]
    run_status = "complete" if not missing_artifacts else "incomplete"

    anomaly_class = "hh4b" if "hh4b" in class_names else ("dy" if "dy" in class_names else "")
    background_classes = [cls for cls in class_names if cls != anomaly_class] if anomaly_class and isinstance(class_names, list) else []

    row: dict[str, Any] = {
        "source_type": "run",
        "source_path": str(run_dir),
        "run_status": run_status,
        "missing_artifacts": _as_csv(missing_artifacts),
        "run_name": run_dir.name,
        "run_date": run_dir.parent.name if run_dir.parent else "",
        "seed": config.get("seed", ""),
        "class_names": _as_csv(class_names),
        "background_classes": _as_csv(background_classes),
        "anomaly_class": anomaly_class,
        "data_source": config.get("data_source", ""),
        "n_events_per_class": n_events_per_class or "",
        "n_classes": n_classes or "",
        "total_events": total_events,
        "n_train": config.get("n_train", ""),
        "n_train_aug": config.get("n_train_aug", ""),
        "n_val": config.get("n_val", ""),
        "n_eval": config.get("n_eval", ""),
        "n_params": n_params,
        "latent_dim": config.get("latent_dim", ""),
        "ff_dim": config.get("ff_dim", ""),
        "num_heads": config.get("num_heads", ""),
        "num_attention_blocks": config.get("num_attention_blocks", 2),
        "pooling_mode": config.get("pooling_mode", ""),
        "classifier_head": config.get("classifier_head", "mlp"),
        "loss_mode": config.get("loss_mode", "ce"),
        "label_smoothing": config.get("label_smoothing", 0.0),
        "focal_gamma": config.get("focal_gamma", ""),
        "supcon_weight": config.get("supcon_weight", 0.0),
        "supcon_temperature": config.get("supcon_temperature", ""),
        "latent_var_reg_weight": config.get("latent_var_reg_weight", 0.0),
        "latent_var_target": config.get("latent_var_target", ""),
        "latent_cov_reg_weight": config.get("latent_cov_reg_weight", 0.0),
    }

    row.update(_best_history(run_dir))
    row.update(_report_metrics(run_dir))
    row.update(_diagnostic_metrics(run_dir))
    return _add_ratio_columns(row)


def discover_runs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    if (root / "run_config.json").exists():
        return [root]
    return sorted(path.parent for path in root.rglob("run_config.json"))


def summarize_sweep_csv(path: Path) -> list[dict[str, Any]]:
    df = _read_csv(path)
    if df.empty:
        return []
    file_date = pd.to_datetime(path.stat().st_mtime, unit="s").strftime("%Y-%m-%d") if path.exists() else ""
    rows = []
    for idx, original in df.iterrows():
        row = original.to_dict()
        n_params = row.get("n_params", "")
        n_events = row.get("n_events_per_class", "")
        class_names = row.get("class_names", "")
        if not class_names:
            if "hh4b" in path.stem or any(str(col).startswith("hh4b_") for col in df.columns):
                class_names = "minbias,top,qcd,dy,hh4b"
            elif "scaling_eval_quality" in path.stem:
                class_names = "minbias,top,qcd,dy"
        n_classes = len(str(class_names).split(",")) if class_names else row.get("n_classes", "")
        anomaly_class = "hh4b" if "hh4b" in str(class_names).split(",") else ("dy" if "dy" in str(class_names).split(",") else "")
        background_classes = ",".join([cls for cls in str(class_names).split(",") if cls and cls != anomaly_class])
        total_events = row.get("total_events", "")
        if total_events in ("", None) and n_events not in ("", None, "") and n_classes not in ("", None, ""):
            try:
                total_events = float(n_events) * float(n_classes)
            except (TypeError, ValueError):
                total_events = ""
        out = {
            "source_type": "sweep",
            "source_path": str(path),
            "run_status": "table_row",
            "missing_artifacts": "",
            "run_name": f"{path.stem}:row{idx}",
            "run_date": row.get("run_date", file_date),
            "seed": row.get("seed", 42),
            "class_names": class_names,
            "background_classes": background_classes,
            "anomaly_class": anomaly_class,
            "data_source": row.get("data_source", "collide2v_updated"),
            "n_events_per_class": n_events,
            "n_classes": n_classes,
            "total_events": total_events,
            "num_attention_blocks": row.get("num_attention_blocks", ""),
            "pooling_mode": row.get("pooling_mode", ""),
            "classifier_head": row.get("classifier_head", ""),
            "loss_mode": row.get("loss_mode", ""),
            "label_smoothing": row.get("label_smoothing", ""),
            "focal_gamma": row.get("focal_gamma", ""),
            "supcon_weight": row.get("supcon_weight", ""),
            "supcon_temperature": row.get("supcon_temperature", ""),
            "latent_var_reg_weight": row.get("latent_var_reg_weight", ""),
            "latent_var_target": row.get("latent_var_target", ""),
            "latent_cov_reg_weight": row.get("latent_cov_reg_weight", ""),
            **row,
        }
        sweep_aliases = {
            "score": "eval_quality_score",
            "align_loss": "eval_quality_align_loss",
            "pr": "eval_quality_pr",
            "gmm_gap": "eval_quality_gmm_gap",
            "mahalanobis": "eval_quality_mahalanobis",
        }
        for src, dst in sweep_aliases.items():
            if src in row and dst not in out:
                out[dst] = row[src]
        if n_params not in ("", None) and total_events not in ("", None, ""):
            try:
                out["samples_per_param_total"] = float(total_events) / float(n_params)
            except (TypeError, ValueError, ZeroDivisionError):
                pass
        out = _add_ratio_columns(out)
        rows.append(out)
    return rows


def normalize_columns(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for col in ORDERED_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    remaining = [col for col in df.columns if col not in ORDERED_COLUMNS]
    return df[ORDERED_COLUMNS + remaining]


def parse_args():
    parser = argparse.ArgumentParser(description="Build a unified run/sweep summary table.")
    parser.add_argument("--runs", nargs="*", default=[], help="Run directories to summarize.")
    parser.add_argument("--run-root", action="append", default=[], help="Root directory to scan for run_config.json files.")
    parser.add_argument("--sweep-csvs", nargs="*", default=[], help="Existing sweep CSV files to append.")
    parser.add_argument("--output", default="ngt_mha_nodes_aligned_outputs/scale_tests/unified_experiment_summary.csv")
    parser.add_argument("--complete-only", action="store_true", help="Exclude incomplete run directories.")
    return parser.parse_args()


def main():
    args = parse_args()
    rows: list[dict[str, Any]] = []

    run_dirs = [Path(path) for path in args.runs]
    for root in args.run_root:
        run_dirs.extend(discover_runs(Path(root)))
    if not run_dirs and not args.sweep_csvs:
        run_dirs = discover_runs(Path("ngt_mha_nodes_aligned_outputs") / "runs")

    seen = set()
    for run_dir in run_dirs:
        key = str(run_dir.resolve())
        if key in seen:
            continue
        seen.add(key)
        rows.append(summarize_run(run_dir))

    for csv_path in args.sweep_csvs:
        rows.extend(summarize_sweep_csv(Path(csv_path)))

    table = normalize_columns(rows)
    if args.complete_only and not table.empty and "run_status" in table.columns:
        table = table[table["run_status"].isin(["complete", "table_row"])].copy()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    print(f"rows: {len(table)}")
    print(f"summary -> {output}")
    if not table.empty:
        preview_cols = [col for col in ["source_type", "run_name", "latent_dim", "n_events_per_class", "accuracy", "macro_f1", "hh4b_f1", "eval_quality_score"] if col in table.columns]
        print(table[preview_cols].tail(12).to_string(index=False))


if __name__ == "__main__":
    main()
