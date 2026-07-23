#!/usr/bin/env python3
"""
Dataset-size sweep for the MHA workflow.

This keeps model architecture and label granularity fixed while varying the
number of events per class. It is the clean experiment for separating data
scaling from latent/model-capacity scaling.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from ngt_model.evaluation.summary_table import normalize_columns, summarize_run


def parse_int_list(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one integer")
    return values


def module_command(module: str, *args: str) -> list[str]:
    return [sys.executable, "-m", module, *args]


def run_command(cmd: list[str], dry_run: bool):
    print(" ".join(cmd), flush=True)
    if dry_run:
        return
    env = os.environ.copy()
    src = str(Path("ngt_model") / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(cmd, check=True, env=env)


def build_train_command(args, event_size: int, run_name: str) -> list[str]:
    cmd = module_command(
        "ngt_model.training.train_mha",
        "--data-source", args.data_source,
        "--class-names", args.class_names,
        "--n-events-per-class", str(event_size),
        "--epochs", str(args.epochs),
        "--early-stopping-patience", str(args.early_stopping_patience),
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--latent-dim", str(args.latent_dim),
        "--ff-dim", str(args.ff_dim),
        "--num-heads", str(args.num_heads),
        "--num-attention-blocks", str(args.num_attention_blocks),
        "--pooling-mode", args.pooling_mode,
        "--classifier-head", args.classifier_head,
        "--augmentation-copies", str(args.augmentation_copies),
        "--class-weight-mode", args.class_weight_mode,
        "--seed", str(args.seed),
        "--output-root", args.output_root,
        "--run-name", run_name,
        "--eval-quality-events", str(args.eval_quality_events),
    )
    if args.device:
        cmd.extend(["--device", args.device])
    if args.mlflow_enable:
        cmd.append("--mlflow-enable")
        cmd.extend(["--mlflow-experiment", args.mlflow_experiment])
        cmd.extend(["--mlflow-run-name", run_name])
        if args.mlflow_tracking_uri:
            cmd.extend(["--mlflow-tracking-uri", args.mlflow_tracking_uri])
    return cmd


def run_optional_diagnostics(args, run_dir: Path, dry_run: bool):
    if args.run_latent_diagnostics:
        run_command(module_command("ngt_model.evaluation.latent_diagnostics", str(run_dir)), dry_run)
    if args.latent_fisher_events_per_class > 0:
        run_command(
            module_command(
                "ngt_model.evaluation.latent_fisher",
                str(run_dir),
                "--n-events-per-class", str(args.latent_fisher_events_per_class),
                "--batch-size", str(args.batch_size),
            ),
            dry_run,
        )
    if args.head_fisher_events_per_class > 0:
        run_command(
            module_command(
                "ngt_model.evaluation.head_fisher",
                str(run_dir),
                "--n-events-per-class", str(args.head_fisher_events_per_class),
                "--batch-size", str(args.batch_size),
            ),
            dry_run,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Run fixed-architecture dataset-size MHA experiments.")
    parser.add_argument("--event-sizes", default="1000,5000,25000,100000,400000")
    parser.add_argument("--class-names", default="minbias,top,qcd,dy,hh4b")
    parser.add_argument("--data-source", default="collide2v_updated", choices=["collide2v_updated", "samples_json", "hf"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--ff-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-attention-blocks", type=int, default=2)
    parser.add_argument("--pooling-mode", default="attention", choices=["mean", "attention"])
    parser.add_argument("--classifier-head", default="mlp", choices=["mlp", "linear"])
    parser.add_argument("--augmentation-copies", type=int, default=1)
    parser.add_argument("--class-weight-mode", default="none", choices=["none", "balanced"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-root", default="ngt_mha_nodes_aligned_outputs")
    parser.add_argument("--run-prefix", default="dataset_size_sweep_hh4b_latent32")
    parser.add_argument("--output", default="ngt_mha_nodes_aligned_outputs/scale_tests/dataset_size_sweep_hh4b_latent32.csv")
    parser.add_argument("--eval-quality-events", type=int, default=2000)
    parser.add_argument("--run-latent-diagnostics", action="store_true", default=True)
    parser.add_argument("--latent-fisher-events-per-class", type=int, default=0)
    parser.add_argument("--head-fisher-events-per-class", type=int, default=0)
    parser.add_argument("--mlflow-enable", action="store_true")
    parser.add_argument("--mlflow-tracking-uri", default="https://ml.cern.ch/models")
    parser.add_argument("--mlflow-experiment", default="ngt_model")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    event_sizes = parse_int_list(args.event_sizes)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    run_date = datetime.now().strftime("%Y-%m-%d")
    for event_size in event_sizes:
        run_name = f"{args.run_prefix}_n{event_size}"
        run_dir = Path(args.output_root) / "runs" / run_date / run_name
        if args.skip_existing and (run_dir / "run_config.json").exists():
            print(f"Skipping existing run: {run_dir}")
        else:
            run_command(build_train_command(args, event_size, run_name), args.dry_run)
        if not args.dry_run:
            run_optional_diagnostics(args, run_dir, args.dry_run)
            row = summarize_run(run_dir)
            row["source_type"] = "dataset_size_sweep"
            row["n_events_per_class"] = event_size
            rows.append(row)

    if args.dry_run:
        print("Dry run complete; no CSV written.")
        return

    table = normalize_columns(rows)
    table.to_csv(output, index=False)
    print(f"dataset-size sweep -> {output}")
    preview_cols = [
        col for col in [
            "run_name", "n_events_per_class", "n_params", "samples_per_param_total",
            "accuracy", "macro_f1", "hh4b_f1", "dy_f1", "eval_quality_score",
            "effective_rank_entropy", "participation_ratio",
        ] if col in table.columns
    ]
    print(table[preview_cols].to_string(index=False))


if __name__ == "__main__":
    main()
