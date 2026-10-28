#!/usr/bin/env python3
"""Log an already completed NGT run directory to MLflow."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd


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


def _log_params_from_config(mlflow, run_dir: Path):
    config_path = run_dir / "run_config.json"
    if not config_path.exists():
        return
    config = json.loads(config_path.read_text(encoding="utf-8"))
    keys = [
        "data_source",
        "feature_mode",
        "candidate_selection",
        "n_constituents",
        "n_events_per_class",
        "epochs",
        "early_stopping_patience",
        "batch_size",
        "lr",
        "weight_decay",
        "latent_dim",
        "ff_dim",
        "num_heads",
        "pooling_mode",
        "augmentation_copies",
        "class_weight_mode",
        "seed",
        "n_features",
        "n_train",
        "n_train_aug",
        "n_val",
        "n_eval",
        "distributed",
        "world_size",
    ]
    params = {key: config[key] for key in keys if key in config}
    if params:
        mlflow.log_params(params)


def _log_training_history(mlflow, run_dir: Path):
    history_path = run_dir / "training_history.csv"
    if not history_path.exists():
        return
    history = pd.read_csv(history_path)
    for _, row in history.iterrows():
        step = int(row["epoch"]) if "epoch" in row else None
        metrics = {}
        for key in ["train_loss", "train_acc", "val_loss", "val_acc", "elapsed_s"]:
            if key in row and pd.notna(row[key]):
                metric_name = "epoch_elapsed_s" if key == "elapsed_s" else key
                metrics[metric_name] = float(row[key])
        if metrics:
            mlflow.log_metrics(metrics, step=step)
    if "val_loss" in history:
        mlflow.log_metric("best_val_loss", float(history["val_loss"].min()))


def _log_eval_report(mlflow, run_dir: Path):
    report_path = run_dir / "classification_report_eval.csv"
    if not report_path.exists():
        return
    report = pd.read_csv(report_path, index_col=0)
    for cls_name, row in report.iterrows():
        for col, suffix in [("precision", "precision"), ("recall", "recall"), ("f1-score", "f1")]:
            if col in row and pd.notna(row[col]):
                safe_cls = str(cls_name).replace(" ", "_")
                mlflow.log_metric(f"eval_{safe_cls}_{suffix}", float(row[col]))


def _log_eval_quality(mlflow, run_dir: Path):
    for path in run_dir.glob("eval_quality_scores_n*.json"):
        scores = json.loads(path.read_text(encoding="utf-8"))
        if "score" in scores:
            mlflow.log_metric("eval_quality_score", float(scores["score"]))
        mlflow.log_artifact(str(path))


def main():
    parser = argparse.ArgumentParser(description="Log an existing NGT run directory to MLflow.")
    parser.add_argument("run_dir", help="Run directory containing run_config.json/training_history.csv")
    parser.add_argument("--mlflow-tracking-uri", default="https://ml.cern.ch/models")
    parser.add_argument("--mlflow-experiment", default="ngt_model")
    parser.add_argument("--mlflow-run-name", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    _load_mlflow_credentials_from_local_conf()
    import mlflow

    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment)
    run_name = args.mlflow_run_name or run_dir.name

    with mlflow.start_run(run_name=run_name):
        _log_params_from_config(mlflow, run_dir)
        _log_training_history(mlflow, run_dir)
        _log_eval_report(mlflow, run_dir)
        _log_eval_quality(mlflow, run_dir)
        for artifact in [
            "run_config.json",
            "training_history.csv",
            "classification_report_eval.csv",
            "confusion_matrix_eval.csv",
        ]:
            path = run_dir / artifact
            if path.exists():
                mlflow.log_artifact(str(path))
    print(f"Logged existing run to MLflow: {run_name}")


if __name__ == "__main__":
    main()
