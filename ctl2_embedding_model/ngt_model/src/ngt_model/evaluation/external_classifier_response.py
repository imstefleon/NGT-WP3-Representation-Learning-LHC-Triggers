"""Evaluate how trained classifiers map external Higgs-like signals.

This script is for signals that were not necessarily part of the trained label
space. For unseen classes, the output is a classifier-response matrix, not a
confusion matrix: each row shows where events from that external sample are
mapped among the trained classes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from ngt_model.datasets.collide2v_data import (
    Collide2VConfig,
    Collide2VLoader,
    build_pid_mapping,
    compute_normalization_constants,
    preprocess_dataset,
)
from ngt_model.evaluation.eval_quality import _load_model
from ngt_model.evaluation.eval_quality_collide2v import (
    _align_feature_dim,
    _expected_input_features,
)


DEFAULT_TRAINED_CLASSES = [
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
]

DEFAULT_EXTERNAL_CLASSES = ["hh4b", "hh", "vbfh", "ggh"]


def _parse_classes(raw: str | Iterable[str]) -> list[str]:
    if isinstance(raw, str):
        return [part.strip().lower() for part in raw.split(",") if part.strip()]
    return [str(part).strip().lower() for part in raw if str(part).strip()]


def _unique_ordered(values: Iterable[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _read_run_config(run_dir: Path) -> dict:
    config_path = run_dir / "run_config.json"
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _trained_classes_from_config(run_dir: Path, fallback: list[str]) -> list[str]:
    config = _read_run_config(run_dir)
    for key in ("class_names", "classes"):
        value = config.get(key)
        if isinstance(value, list):
            return _parse_classes(value)
        if isinstance(value, str):
            return _parse_classes(value)
    return fallback


def _default_model_path(run_dir: Path) -> Path:
    candidates = [
        run_dir / "models" / "mha_nodes_aligned_best.pt",
        run_dir / "models" / "mha_nodes_aligned_best_for_eval.pt",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "No model checkpoint found. Expected one of: "
        + ", ".join(str(path) for path in candidates)
    )


def _predict_probabilities(
    model: torch.nn.Module,
    X: np.ndarray,
    batch_size: int,
    device: str,
) -> np.ndarray:
    probs = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[start : start + batch_size].astype(np.float32)).to(device)
            out = model(batch)
            if isinstance(out, dict):
                logits = out.get("logits")
            else:
                logits = getattr(out, "logits", None)
            if logits is None:
                raise RuntimeError(
                    "The loaded checkpoint did not return logits. Use the full model "
                    "checkpoint, usually models/mha_nodes_aligned_best.pt."
                )
            probs.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
    return np.concatenate(probs, axis=0)


def _matrix_from_predictions(
    true_names: np.ndarray,
    pred_names: np.ndarray,
    rows: list[str],
    cols: list[str],
) -> pd.DataFrame:
    matrix = pd.DataFrame(0, index=rows, columns=cols, dtype=int)
    for true_name, pred_name in zip(true_names, pred_names):
        if true_name in matrix.index and pred_name in matrix.columns:
            matrix.loc[true_name, pred_name] += 1
    return matrix


def _plot_matrix(
    matrix: pd.DataFrame,
    out_path: Path,
    title: str,
    colorbar_label: str,
    fmt: str = "percent",
) -> None:
    fig_w = max(8, 0.72 * len(matrix.columns))
    fig_h = max(4, 0.65 * len(matrix.index))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(matrix.values, cmap="Blues", vmin=0, vmax=max(float(matrix.values.max()), 1e-12))
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label)
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticklabels(matrix.index)
    ax.set_xlabel("Predicted trained class")
    ax.set_ylabel("Evaluated sample")
    ax.set_title(title)

    threshold = float(matrix.values.max()) * 0.55 if matrix.size else 0.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = float(matrix.iloc[i, j])
            if fmt == "percent":
                label = f"{100 * value:.1f}%"
            else:
                label = f"{value:.3f}"
            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=8,
            )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_hh4b_tag_rate(summary: pd.DataFrame, out_path: Path, model_label: str) -> None:
    if "hh4b_tag_rate" not in summary.columns:
        return
    if summary["hh4b_tag_rate"].isna().all():
        print("Skipping HH4B tag-rate plot because hh4b is not a trained classifier output.")
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = summary["true_external_class"].tolist()
    values = summary["hh4b_tag_rate"].to_numpy(dtype=float)
    bars = ax.bar(labels, values, color=["#2f7ed8", "#f27e4b", "#2ca02c", "#9467bd"][: len(labels)])
    ax.set_ylim(0, max(1.0, float(np.nanmax(values)) * 1.2 if len(values) else 1.0))
    ax.set_ylabel("fraction predicted as HH4B")
    ax.set_xlabel("evaluated sample")
    ax.set_title(f"HH4B classifier tag rate: {model_label}")
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create classifier-response plots for external Higgs-like signals."
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--model-label", default=None)
    parser.add_argument("--trained-classes", default=",".join(DEFAULT_TRAINED_CLASSES))
    parser.add_argument("--external-classes", default=",".join(DEFAULT_EXTERNAL_CLASSES))
    parser.add_argument("--n-events-per-class", type=int, default=2000)
    parser.add_argument("--data-source", default="collide2v_updated")
    parser.add_argument("--feature-mode", default="l1_only")
    parser.add_argument("--candidate-selection", default="top_pt")
    parser.add_argument("--n-constituents", type=int, default=100)
    parser.add_argument("--cache-prefix", default="eval")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir
    model_path = args.model if args.model is not None else _default_model_path(run_dir)
    model_label = args.model_label or run_dir.name
    output_dir = args.output_dir or run_dir / "figures" / "external_classifier_response"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    trained_classes = _trained_classes_from_config(run_dir, _parse_classes(args.trained_classes))
    external_classes = _parse_classes(args.external_classes)
    context_classes = _unique_ordered([*trained_classes, *external_classes])

    print("External classifier-response evaluation")
    print("=" * 58)
    print(f"run_dir={run_dir}")
    print(f"model={model_path}")
    print(f"trained_classes={trained_classes}")
    print(f"external_classes={external_classes}")
    print(f"device={device}")

    model = _load_model(model_path, device)
    expected_features = _expected_input_features(model)

    cfg = Collide2VConfig(
        data_source=args.data_source,
        class_names=context_classes,
        feature_mode=args.feature_mode,
        candidate_selection=args.candidate_selection,
        n_constituents=args.n_constituents,
    )
    loader = Collide2VLoader(cfg)
    events_per_class = {cls: args.n_events_per_class for cls in context_classes}
    raw_tensor, _y, sample_names, _raw_by_class = loader.load_classes(
        events_per_class,
        cache_prefix=args.cache_prefix,
    )
    norm = compute_normalization_constants(raw_tensor)
    pid_mapping = build_pid_mapping(raw_tensor)
    X, _ = preprocess_dataset(raw_tensor, norm=norm, pid_mapping=pid_mapping)
    X = _align_feature_dim(X.astype(np.float32), expected_features, "external classifier response")

    probs = _predict_probabilities(model, X, args.batch_size, device)
    if probs.shape[1] != len(trained_classes):
        raise RuntimeError(
            f"Logit dimension {probs.shape[1]} does not match trained class count "
            f"{len(trained_classes)}: {trained_classes}"
        )

    pred_idx = np.argmax(probs, axis=1)
    pred_names = np.array([trained_classes[i] for i in pred_idx])
    sample_names = np.asarray(sample_names)
    keep = np.isin(sample_names, external_classes)
    sample_names = sample_names[keep]
    pred_names = pred_names[keep]
    probs = probs[keep]

    counts = _matrix_from_predictions(sample_names, pred_names, external_classes, trained_classes)
    fractions = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    mean_probs = pd.DataFrame(0.0, index=external_classes, columns=trained_classes)
    for cls in external_classes:
        mask = sample_names == cls
        if np.any(mask):
            mean_probs.loc[cls] = probs[mask].mean(axis=0)

    summary_rows = []
    hh4b_idx = trained_classes.index("hh4b") if "hh4b" in trained_classes else None
    for cls in external_classes:
        row = fractions.loc[cls]
        n_events = int(counts.loc[cls].sum())
        top_pred = str(row.idxmax()) if n_events else ""
        top_frac = float(row.max()) if n_events else float("nan")
        hh4b_tag_rate = float(row["hh4b"]) if "hh4b" in row.index else float("nan")
        mean_hh4b_prob = (
            float(mean_probs.loc[cls, "hh4b"]) if "hh4b" in mean_probs.columns else float("nan")
        )
        true_recall = float(row[cls]) if cls in row.index else float("nan")
        summary_rows.append(
            {
                "true_external_class": cls,
                "n_events": n_events,
                "top_predicted_class": top_pred,
                "top_fraction": top_frac,
                "hh4b_tag_rate": hh4b_tag_rate,
                "mean_hh4b_probability": mean_hh4b_prob,
                "true_class_recall_if_trained": true_recall,
            }
        )
    summary = pd.DataFrame(summary_rows)

    counts.to_csv(output_dir / "classifier_response_counts.csv")
    fractions.to_csv(output_dir / "classifier_response_fractions.csv")
    mean_probs.to_csv(output_dir / "classifier_response_mean_probabilities.csv")
    summary.to_csv(output_dir / "classifier_response_summary.csv", index=False)
    with (output_dir / "classifier_response_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model_label": model_label,
                "run_dir": str(run_dir),
                "model": str(model_path),
                "trained_classes": trained_classes,
                "external_classes": external_classes,
                "n_events_per_class": args.n_events_per_class,
                "summary": summary.to_dict(orient="records"),
            },
            f,
            indent=2,
        )

    _plot_matrix(
        fractions,
        output_dir / "classifier_response_matrix.png",
        f"Classifier response for external signals: {model_label}",
        "row-normalized fraction",
        fmt="percent",
    )
    _plot_matrix(
        mean_probs,
        output_dir / "classifier_mean_probability_matrix.png",
        f"Mean classifier probabilities: {model_label}",
        "mean probability",
        fmt="prob",
    )
    _plot_hh4b_tag_rate(summary, output_dir / "hh4b_tag_rate_by_signal.png", model_label)

    print("\nSummary")
    print(summary.to_string(index=False))
    print(f"\nOutputs -> {output_dir}")


if __name__ == "__main__":
    main()
