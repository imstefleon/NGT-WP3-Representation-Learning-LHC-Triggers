#!/usr/bin/env python3
"""
Generate a reproducible command manifest for the MHA latent scaling study.

The script intentionally does not launch training jobs. It creates one row per
planned experiment plus shell commands that can be run manually or submitted to
the usual batch/MLOps workflow.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


CLASS_CONFIGS = {
    "4class": ["minbias", "qcd", "dy", "top"],
    # Background-only granular configuration. This reaches >10 classes without
    # training on the Higgs-like signals used later for anomaly injection.
    "10plus_sm": [
        "minbias",
        "qcd",
        "dy",
        "tt_hadronic",
        "tt_leptonic",
        "tt_semileptonic",
        "wjets",
        "ww",
        "wz",
        "ttw",
        "ttz",
    ],
    "granular_sm15": [
        "minbias",
        "qcd",
        "dy_ee",
        "dy_mumu",
        "dy_tautau",
        "tt_hadronic",
        "tt_leptonic",
        "tt_semileptonic",
        "wjets",
        "ww_hadronic",
        "ww_leptonic",
        "ww_semileptonic",
        "wz",
        "ttw",
        "ttz",
    ],
    # Kept only for comparison with older project results where HH4B was a
    # supervised reference class. Do not use this as the primary AD baseline.
    "legacy_10class_hh4b_ref": [
        "minbias",
        "qcd",
        "dy",
        "top",
        "wjets",
        "ww",
        "wz",
        "ttw",
        "ttz",
        "hh4b",
    ],
}


MODEL_SCHEDULE = {
    "small": {"latent_dim": 64, "ff_dim": 128, "num_heads": 8, "num_attention_blocks": 2},
    "medium": {"latent_dim": 128, "ff_dim": 256, "num_heads": 8, "num_attention_blocks": 2},
    "large": {"latent_dim": 128, "ff_dim": 256, "num_heads": 8, "num_attention_blocks": 4},
    "xl": {"latent_dim": 256, "ff_dim": 512, "num_heads": 8, "num_attention_blocks": 4},
}


SIZE_TO_MODEL = {
    50_000: "small",
    100_000: "medium",
    150_000: "large",
    200_000: "xl",
}


@dataclass(frozen=True)
class Experiment:
    class_configuration: str
    class_names: list[str]
    samples_per_class: int
    model_size: str
    classifier_head: str
    latent_dim: int
    ff_dim: int
    num_heads: int
    num_attention_blocks: int
    output_root: str
    run_name: str
    train_command: str
    run_dir_bash: str
    diagnostics_commands: list[str]


def parse_int_list(raw: str) -> list[int]:
    values = [int(part.strip().replace("_", "")) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one integer value")
    return values


def shell_join(parts: Iterable[str]) -> str:
    quoted = []
    for part in parts:
        if not part:
            continue
        if any(ch.isspace() for ch in part) or any(ch in part for ch in ['"', "'", "$", "(", ")"]):
            escaped = part.replace('"', '\\"')
            quoted.append(f'"{escaped}"')
        else:
            quoted.append(part)
    return " ".join(quoted)


def parse_str_list(raw: str) -> list[str]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one value")
    return values


def build_train_command(
    args,
    class_names: list[str],
    n_events: int,
    model_size: str,
    classifier_head: str,
    run_name: str,
    output_root: str,
) -> str:
    dims = MODEL_SCHEDULE[model_size]
    parts = [
        "${PY:-python}",
        "-m",
        "ngt_model.training.train_mha",
        "--data-source",
        args.data_source,
        "--class-names",
        ",".join(class_names),
        "--n-events-per-class",
        str(n_events),
        "--epochs",
        str(args.epochs),
        "--early-stopping-patience",
        str(args.early_stopping_patience),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--latent-dim",
        str(dims["latent_dim"]),
        "--ff-dim",
        str(dims["ff_dim"]),
        "--num-heads",
        str(dims["num_heads"]),
        "--num-attention-blocks",
        str(dims["num_attention_blocks"]),
        "--pooling-mode",
        args.pooling_mode,
        "--classifier-head",
        classifier_head,
        "--loss-mode",
        args.loss_mode,
        "--latent-var-reg-weight",
        str(args.latent_var_reg_weight),
        "--latent-cov-reg-weight",
        str(args.latent_cov_reg_weight),
        "--augmentation-copies",
        str(args.augmentation_copies),
        "--class-weight-mode",
        args.class_weight_mode,
        "--seed",
        str(args.seed),
        "--output-root",
        output_root,
        "--run-name",
        run_name,
        "--eval-quality-events",
        str(args.eval_quality_events),
    ]
    if args.device:
        parts.extend(["--device", args.device])
    if args.mlflow:
        parts.extend(
            [
                "--mlflow-enable",
                "--mlflow-tracking-uri",
                args.mlflow_tracking_uri,
                "--mlflow-experiment",
                args.mlflow_experiment,
                "--mlflow-run-name",
                run_name,
            ]
        )
    return shell_join(parts)


def build_diagnostics_commands(args, run_dir_var: str, class_names: list[str]) -> list[str]:
    signals = [sig for sig in args.signals.split(",") if sig]
    excluded_background_classes = [cls for cls in class_names if cls in signals]
    anomaly_parts = [
        "${PY:-python}",
        "-m",
        "ngt_model.evaluation.external_higgs_anomaly",
        "--run-dir",
        run_dir_var,
        "--signals",
        ",".join(signals),
        "--n-background-per-class",
        str(args.diagnostic_events_per_class),
        "--n-signal-per-class",
        str(args.diagnostic_events_per_class),
        "--device",
        args.diagnostic_device,
        "--batch-size",
        str(args.batch_size),
        "--tsne-max-events-per-class",
        str(args.tsne_max_events_per_class),
        "--tsne-perplexity",
        str(args.tsne_perplexity),
        "--tsne-n-iter",
        str(args.tsne_n_iter),
        "--raise-errors",
    ]
    if excluded_background_classes:
        anomaly_parts.extend(["--exclude-background-classes", ",".join(excluded_background_classes)])
    commands = [
        shell_join(["${PY:-python}", "-m", "ngt_model.evaluation.latent_diagnostics", run_dir_var]),
    ]
    if args.run_fisher:
        commands.extend(
            [
                shell_join(
                    [
                        "${PY:-python}",
                        "-m",
                        "ngt_model.evaluation.latent_fisher",
                        run_dir_var,
                        "--n-events-per-class",
                        str(args.fisher_events_per_class),
                        "--device",
                        args.diagnostic_device,
                        "--batch-size",
                        str(args.batch_size),
                    ]
                ),
                shell_join(
                    [
                        "${PY:-python}",
                        "-m",
                        "ngt_model.evaluation.head_fisher",
                        run_dir_var,
                        "--scope",
                        "classifier_head",
                        "--n-events-per-class",
                        str(args.fisher_events_per_class),
                        "--device",
                        args.diagnostic_device,
                        "--batch-size",
                        str(args.batch_size),
                    ]
                ),
            ]
        )
    commands.extend(
        [
        shell_join(
            [
                "${PY:-python}",
                "-m",
                "ngt_model.evaluation.external_classifier_response",
                run_dir_var,
                "--external-classes",
                ",".join(signals),
                "--n-events-per-class",
                str(args.diagnostic_events_per_class),
                "--device",
                args.diagnostic_device,
                "--batch-size",
                str(args.batch_size),
            ]
        ),
        shell_join(anomaly_parts),
        ]
    )
    return commands


def selected_class_configs(args) -> dict[str, list[str]]:
    names = [part.strip() for part in args.class_configurations.split(",") if part.strip()]
    unknown = [name for name in names if name not in CLASS_CONFIGS]
    if unknown:
        raise ValueError(f"Unknown class configurations: {unknown}. Available: {sorted(CLASS_CONFIGS)}")
    return {name: CLASS_CONFIGS[name] for name in names}


def planned_model_sizes(args, n_events: int) -> list[str]:
    if args.full_capacity_grid:
        return list(MODEL_SCHEDULE)
    return [SIZE_TO_MODEL[n_events]]


def build_experiments(args) -> list[Experiment]:
    experiments: list[Experiment] = []
    for config_name, class_names in selected_class_configs(args).items():
        for n_events in parse_int_list(args.samples_per_class):
            if n_events not in SIZE_TO_MODEL:
                raise ValueError(f"No model-size schedule is defined for {n_events} samples/class")
            sample_label = f"{n_events // 1000}k"
            for model_size in planned_model_sizes(args, n_events):
                for classifier_head in parse_str_list(args.classifier_heads):
                    dims = MODEL_SCHEDULE[model_size]
                    output_root = (Path(args.output_root) / config_name / sample_label / model_size).as_posix()
                    run_name = (
                        f"scaling_{config_name}_{sample_label}_{model_size}_"
                        f"{classifier_head}_seed{args.seed}"
                    )
                    run_dir_bash = f"{output_root}/runs/${{RUN_DATE:-$(date +%F)}}/{run_name}"
                    train_command = build_train_command(
                        args,
                        class_names,
                        n_events,
                        model_size,
                        classifier_head,
                        run_name,
                        output_root,
                    )
                    experiments.append(
                        Experiment(
                            class_configuration=config_name,
                            class_names=class_names,
                            samples_per_class=n_events,
                            model_size=model_size,
                            classifier_head=classifier_head,
                            latent_dim=dims["latent_dim"],
                            ff_dim=dims["ff_dim"],
                            num_heads=dims["num_heads"],
                            num_attention_blocks=dims["num_attention_blocks"],
                            output_root=output_root,
                            run_name=run_name,
                            train_command=train_command,
                            run_dir_bash=run_dir_bash,
                            diagnostics_commands=build_diagnostics_commands(args, run_dir_bash, class_names),
                        )
                    )
    return experiments


def write_outputs(args, experiments: list[Experiment]) -> None:
    out_dir = Path(args.manifest_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "scaling_study_manifest.csv"
    json_path = out_dir / "scaling_study_manifest.json"
    bash_path = out_dir / "scaling_study_commands.sh"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "class_configuration",
            "class_names",
            "n_classes",
            "samples_per_class",
            "model_size",
            "classifier_head",
            "latent_dim",
            "ff_dim",
            "num_heads",
            "num_attention_blocks",
            "output_root",
            "run_name",
            "run_dir_bash",
            "train_command",
            "diagnostics_commands",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for exp in experiments:
            writer.writerow(
                {
                    **asdict(exp),
                    "class_names": ",".join(exp.class_names),
                    "n_classes": len(exp.class_names),
                    "diagnostics_commands": " && ".join(exp.diagnostics_commands),
                }
            )

    json_path.write_text(json.dumps([asdict(exp) for exp in experiments], indent=2), encoding="utf-8")

    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'export PYTHONPATH="${PWD}/ngt_model/src:${PYTHONPATH:-}"',
        'export MPLBACKEND="${MPLBACKEND:-Agg}"',
        "",
    ]
    for exp in experiments:
        lines.extend(
            [
                f"# {exp.class_configuration} | {exp.samples_per_class} per class | {exp.model_size}",
                f'RUN_NAME="{exp.run_name}"',
                f'RUN_DIR="{exp.run_dir_bash}"',
                exp.train_command,
                *exp.diagnostics_commands,
                "",
            ]
        )
    bash_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {len(experiments)} planned experiments")
    print(f"CSV:  {csv_path}")
    print(f"JSON: {json_path}")
    print(f"Bash: {bash_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Create scaling-study manifests and run commands.")
    parser.add_argument("--class-configurations", default="4class,10plus_sm")
    parser.add_argument("--samples-per-class", default="50000,100000,150000,200000")
    parser.add_argument("--full-capacity-grid", action="store_true")
    parser.add_argument("--data-source", default="collide2v_updated")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pooling-mode", default="mean", choices=["mean", "attention"])
    parser.add_argument("--classifier-heads", default="mlp2")
    parser.add_argument("--loss-mode", default="ce", choices=["ce", "label_smoothing", "focal"])
    parser.add_argument("--latent-var-reg-weight", type=float, default=0.01)
    parser.add_argument("--latent-cov-reg-weight", type=float, default=0.001)
    parser.add_argument("--augmentation-copies", type=int, default=1)
    parser.add_argument("--class-weight-mode", default="none", choices=["none", "balanced"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-root", default="outputs/scaling_study")
    parser.add_argument("--manifest-dir", default="outputs/scaling_study")
    parser.add_argument("--eval-quality-events", type=int, default=2000)
    parser.add_argument("--diagnostic-events-per-class", type=int, default=2000)
    parser.add_argument("--run-fisher", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fisher-events-per-class", type=int, default=2000)
    parser.add_argument("--tsne-max-events-per-class", type=int, default=600)
    parser.add_argument("--tsne-perplexity", type=float, default=40.0)
    parser.add_argument("--tsne-n-iter", type=int, default=1000)
    parser.add_argument("--diagnostic-device", default="cpu")
    parser.add_argument("--signals", default="hh4b,hh,vbfh,ggh")
    parser.add_argument("--mlflow", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mlflow-tracking-uri", default="https://ml.cern.ch/models")
    parser.add_argument("--mlflow-experiment", default="ngt_model_scaling_study")
    return parser.parse_args()


def main():
    args = parse_args()
    experiments = build_experiments(args)
    write_outputs(args, experiments)


if __name__ == "__main__":
    main()
