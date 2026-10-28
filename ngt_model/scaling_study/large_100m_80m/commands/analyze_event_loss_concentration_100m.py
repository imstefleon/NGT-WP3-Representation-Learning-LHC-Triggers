#!/usr/bin/env python3
"""Diagnose whether early training loss is concentrated in a few events.

For each saved 100M checkpoint, this script evaluates the same fixed cached
event subset and saves per-event CE, SupCon/InfoNCE-style loss contributions,
confidence, entropy, latent norm, and positive/negative latent similarity
statistics. It also summarizes how much of the total loss is carried by the
hardest 1%, 5%, and 10% of events.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ngt_model.datasets.collide2v_data import preprocess_dataset
from ngt_model.models.mha_model import MHAEmbeddingFullPrecision


DEFAULT_RUN_DIR = (
    "outputs/scaling_study/large_100m_80m/one_24h_100m/runs/2026-08-24/"
    "scaling_4class_80m_dmodel2016_ff8064_100m_400ep_noaug_ce_infonce005_lr1e5_seed42"
)
DEFAULT_CACHE_DIR = "outputs/scaling_study/large_100m_80m/one_24h_100m/cache"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--events-per-class", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--supcon-temperature", type=float, default=None)
    parser.add_argument("--supcon-weight", type=float, default=None)
    parser.add_argument("--similarity-chunk-size", type=int, default=512)
    parser.add_argument("--checkpoint-glob", default="checkpoint_epoch_*.pt")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def checkpoint_epoch(path: Path) -> int:
    match = re.search(r"epoch_(\d+)", path.name)
    if not match:
        raise ValueError(f"Cannot infer epoch from checkpoint name: {path}")
    return int(match.group(1))


def discover_checkpoints(run_dir: Path, checkpoint_glob: str) -> list[tuple[int, Path]]:
    checkpoints = sorted(
        ((checkpoint_epoch(path), path) for path in (run_dir / "models").glob(checkpoint_glob)),
        key=lambda item: item[0],
    )
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints matching {checkpoint_glob} under {run_dir / 'models'}")
    return checkpoints


def cache_path_for_class(cache_dir: Path, config: dict, class_name: str, shard: int) -> Path:
    loaded = int(config["loaded_events_per_class"][class_name])
    feature_mode = config.get("feature_mode", "l1_only")
    selection = config.get("candidate_selection", "top_pt")
    source = config.get("data_source", "collide2v_updated")
    stride = int(config.get("data_event_stride", config.get("world_size", 1)))
    shard_suffix = f"_shard{shard:04d}of{stride:04d}" if stride > 1 else ""
    name = f"train_mha_{class_name}_{loaded}_{feature_mode}_{selection}{shard_suffix}_{source}_rawtensor.npy"
    return cache_dir / name


def load_fixed_sample(run_dir: Path, cache_dir: Path, events_per_class: int, shard: int, seed: int):
    config = read_json(run_dir / "run_config.json")
    class_names = list(config["class_names"])
    label_map = {name: int(idx) for name, idx in config["label_map"].items()}
    rng = np.random.default_rng(seed)

    raw_parts = []
    label_parts = []
    class_parts = []
    source_index_parts = []
    sample_meta = []
    for class_name in class_names:
        path = cache_path_for_class(cache_dir, config, class_name, shard)
        if not path.exists():
            raise FileNotFoundError(f"Missing cache for {class_name}: {path}")
        raw_mmap = np.load(path, mmap_mode="r")
        n_take = min(events_per_class, len(raw_mmap))
        indices = np.sort(rng.choice(len(raw_mmap), size=n_take, replace=False))
        raw = np.asarray(raw_mmap[indices], dtype=np.float32)
        raw_parts.append(raw)
        label_parts.append(np.full(n_take, label_map[class_name], dtype=np.int64))
        class_parts.extend([class_name] * n_take)
        source_index_parts.extend(indices.astype(int).tolist())
        sample_meta.append(
            {
                "class_name": class_name,
                "cache_path": str(path),
                "n_available": int(len(raw_mmap)),
                "n_selected": int(n_take),
                "first_selected_index": int(indices[0]),
                "last_selected_index": int(indices[-1]),
            }
        )

    raw_all = np.concatenate(raw_parts, axis=0)
    y = np.concatenate(label_parts, axis=0)
    class_names_per_event = np.asarray(class_parts, dtype=object)
    source_indices = np.asarray(source_index_parts, dtype=np.int64)
    event_ids = np.asarray(
        [f"{cls}:shard{shard:04d}:{idx}" for cls, idx in zip(class_names_per_event, source_indices)],
        dtype=object,
    )

    order = rng.permutation(len(y))
    raw_all = raw_all[order]
    y = y[order]
    class_names_per_event = class_names_per_event[order]
    source_indices = source_indices[order]
    event_ids = event_ids[order]

    add_pid_ohe = config.get("pid_encoding", "onehot") == "onehot"
    norm = {key: float(value) for key, value in config["norm"].items()}
    pid_mapping = {int(key): int(value) for key, value in config.get("pid_mapping", {}).items()}
    X, _ = preprocess_dataset(raw_all, norm=norm, pid_mapping=pid_mapping, add_pid_ohe=add_pid_ohe)
    return (
        X.astype(np.float32, copy=False),
        y,
        class_names_per_event,
        source_indices,
        event_ids,
        class_names,
        config,
        sample_meta,
    )


def build_model_from_config(config: dict, device: torch.device) -> MHAEmbeddingFullPrecision:
    model = MHAEmbeddingFullPrecision(
        n_constituents=int(config.get("n_constituents", 100)),
        n_features=int(config.get("n_features", 5)),
        latent_dim=int(config.get("latent_dim", 128)),
        n_classes=len(config["class_names"]),
        ff_dim=int(config.get("ff_dim", 32)),
        d_model=config.get("d_model"),
        num_heads=int(config.get("num_heads", 2)),
        pooling_mode=config.get("pooling_mode", "mean"),
        classifier_head=config.get("classifier_head", "mlp"),
        num_attention_blocks=int(config.get("num_attention_blocks", 2)),
    )
    model.to(device)
    return model


def strip_module_prefix(state_dict: dict) -> dict:
    if not any(str(key).startswith("module.") for key in state_dict):
        return state_dict
    return {str(key).removeprefix("module."): value for key, value in state_dict.items()}


def load_checkpoint_model(model_path: Path, config: dict, device: torch.device) -> torch.nn.Module:
    obj = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(obj, dict) and "model_state_dict" in obj:
        model = build_model_from_config(config, device)
        model.load_state_dict(strip_module_prefix(obj["model_state_dict"]))
    elif isinstance(obj, torch.nn.Module):
        model = obj.to(device)
    else:
        raise TypeError(f"Unsupported checkpoint object in {model_path}: {type(obj)}")
    model.eval()
    return model


@torch.no_grad()
def extract_outputs(model: torch.nn.Module, X: np.ndarray, device: torch.device, batch_size: int):
    latents = []
    logits = []
    for start in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[start : start + batch_size]).to(device)
        out = model(xb)
        if not isinstance(out, dict) or "latent" not in out or "logits" not in out:
            raise TypeError("Expected model forward to return {'latent', 'logits'}")
        latents.append(out["latent"].detach().cpu().numpy())
        logits.append(out["logits"].detach().cpu().numpy())
    return np.concatenate(latents, axis=0), np.concatenate(logits, axis=0)


def softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def ce_per_event(logits: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    probs = softmax_np(logits.astype(np.float64))
    true_prob = np.clip(probs[np.arange(len(y)), y.astype(int)], 1e-300, 1.0)
    ce = -np.log(true_prob)
    confidence = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    entropy = -np.sum(np.clip(probs, 1e-300, 1.0) * np.log(np.clip(probs, 1e-300, 1.0)), axis=1)
    return ce.astype(np.float32), confidence.astype(np.float32), entropy.astype(np.float32), pred.astype(np.int64)


def supcon_diagnostics(
    Z: np.ndarray,
    y: np.ndarray,
    temperature: float,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    z = Z.astype(np.float32, copy=False)
    z_norm = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
    labels = y.astype(np.int64)
    n = len(labels)
    losses = np.full(n, np.nan, dtype=np.float32)
    max_pos = np.full(n, np.nan, dtype=np.float32)
    max_neg = np.full(n, np.nan, dtype=np.float32)
    margin = np.full(n, np.nan, dtype=np.float32)

    all_indices = np.arange(n)
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        anchors = z_norm[start:stop]
        sim = anchors @ z_norm.T
        logits = sim / temperature

        row_indices = all_indices[start:stop]
        self_mask = row_indices[:, None] == all_indices[None, :]
        pos_mask = (labels[start:stop, None] == labels[None, :]) & ~self_mask
        neg_mask = labels[start:stop, None] != labels[None, :]
        denom_mask = ~self_mask

        max_logits = np.where(denom_mask, logits, -np.inf).max(axis=1, keepdims=True)
        stable_logits = logits - max_logits
        exp_logits = np.exp(stable_logits) * denom_mask
        log_prob = stable_logits - np.log(np.maximum(exp_logits.sum(axis=1, keepdims=True), 1e-30))

        pos_count = pos_mask.sum(axis=1)
        valid = pos_count > 0
        if np.any(valid):
            chunk_loss = -(log_prob * pos_mask).sum(axis=1) / np.maximum(pos_count, 1)
            losses[start:stop] = chunk_loss.astype(np.float32)

        max_pos[start:stop] = np.where(pos_mask, sim, -np.inf).max(axis=1).astype(np.float32)
        max_neg[start:stop] = np.where(neg_mask, sim, -np.inf).max(axis=1).astype(np.float32)
        margin[start:stop] = max_pos[start:stop] - max_neg[start:stop]

    return losses, max_pos, max_neg, margin


def top_share(values: np.ndarray, fraction: float) -> float:
    clean = np.asarray(values, dtype=np.float64)
    clean = clean[np.isfinite(clean)]
    clean = np.maximum(clean, 0.0)
    if len(clean) == 0 or clean.sum() <= 0:
        return 0.0
    k = max(1, int(math.ceil(len(clean) * fraction)))
    return float(np.sort(clean)[-k:].sum() / clean.sum())


def effective_event_fraction(values: np.ndarray) -> float:
    clean = np.asarray(values, dtype=np.float64)
    clean = clean[np.isfinite(clean)]
    clean = np.maximum(clean, 0.0)
    if len(clean) == 0 or np.sum(clean**2) <= 0:
        return 0.0
    eff_n = float((clean.sum() ** 2) / np.sum(clean**2))
    return eff_n / len(clean)


def summarize_epoch(df: pd.DataFrame) -> dict:
    total_loss = df["total_loss"].to_numpy(dtype=float)
    ce = df["ce_loss"].to_numpy(dtype=float)
    supcon = df["supcon_loss"].to_numpy(dtype=float)
    return {
        "epoch": int(df["epoch"].iloc[0]),
        "n_events": int(len(df)),
        "accuracy": float((df["pred_label"] == df["true_label"]).mean()),
        "mean_ce_loss": float(np.mean(ce)),
        "median_ce_loss": float(np.median(ce)),
        "p90_ce_loss": float(np.percentile(ce, 90)),
        "p99_ce_loss": float(np.percentile(ce, 99)),
        "mean_supcon_loss": float(np.nanmean(supcon)),
        "median_supcon_loss": float(np.nanmedian(supcon)),
        "mean_total_loss": float(np.nanmean(total_loss)),
        "median_total_loss": float(np.nanmedian(total_loss)),
        "p90_total_loss": float(np.nanpercentile(total_loss, 90)),
        "p99_total_loss": float(np.nanpercentile(total_loss, 99)),
        "top_1pct_total_loss_share": top_share(total_loss, 0.01),
        "top_5pct_total_loss_share": top_share(total_loss, 0.05),
        "top_10pct_total_loss_share": top_share(total_loss, 0.10),
        "effective_event_fraction_total_loss": effective_event_fraction(total_loss),
        "mean_confidence": float(df["confidence"].mean()),
        "mean_entropy": float(df["entropy"].mean()),
        "mean_latent_norm": float(df["latent_norm"].mean()),
        "mean_pos_neg_margin": float(df["pos_neg_margin"].mean()),
    }


def hard_overlap(rows: pd.DataFrame, fractions: list[float]) -> pd.DataFrame:
    output = []
    epochs = sorted(rows["epoch"].unique())
    event_ids_by_epoch = {epoch: rows[rows["epoch"] == epoch].set_index("event_id") for epoch in epochs}
    for fraction in fractions:
        sets = {}
        for epoch, df_epoch in event_ids_by_epoch.items():
            k = max(1, int(math.ceil(len(df_epoch) * fraction)))
            sets[epoch] = set(df_epoch.nlargest(k, "total_loss").index)
        epoch1 = epochs[0]
        for epoch in epochs:
            prev = epochs[max(0, epochs.index(epoch) - 1)]
            current = sets[epoch]
            with_epoch1 = len(current & sets[epoch1]) / max(len(current | sets[epoch1]), 1)
            with_prev = len(current & sets[prev]) / max(len(current | sets[prev]), 1)
            output.append(
                {
                    "fraction": fraction,
                    "epoch": int(epoch),
                    "jaccard_vs_epoch1": float(with_epoch1),
                    "jaccard_vs_previous": float(with_prev),
                    "n_hard_events": int(len(current)),
                }
            )
    return pd.DataFrame(output)


def plot_loss_distributions(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for epoch, epoch_df in df.groupby("epoch"):
        axes[0].hist(epoch_df["ce_loss"], bins=60, histtype="step", linewidth=1.5, label=f"epoch {epoch}")
        axes[1].hist(epoch_df["total_loss"], bins=60, histtype="step", linewidth=1.5, label=f"epoch {epoch}")
    axes[0].set_title("Per-event CE loss")
    axes[1].set_title("Per-event total loss = CE + w * InfoNCE")
    for ax in axes:
        ax.set_xlabel("loss")
        ax.set_ylabel("events")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_loss_concentration(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    fractions = np.linspace(0.001, 0.25, 120)
    for epoch, epoch_df in df.groupby("epoch"):
        values = np.sort(epoch_df["total_loss"].to_numpy(dtype=float))[::-1]
        values = np.maximum(values[np.isfinite(values)], 0.0)
        cumsum = np.cumsum(values)
        total = cumsum[-1] if len(cumsum) else 0.0
        shares = []
        for fraction in fractions:
            k = max(1, int(math.ceil(len(values) * fraction)))
            shares.append(float(cumsum[k - 1] / total) if total > 0 else 0.0)
        ax.plot(fractions * 100.0, shares, label=f"epoch {epoch}")
    ax.plot(fractions * 100.0, fractions, color="black", linestyle="--", linewidth=1, label="uniform")
    ax.set_title("How much loss is carried by the hardest events?")
    ax.set_xlabel("hardest events kept (%)")
    ax.set_ylabel("fraction of total loss")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_summary(summary: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes = axes.ravel()
    axes[0].plot(summary["epoch"], summary["mean_ce_loss"], marker="o", label="mean CE")
    axes[0].plot(summary["epoch"], summary["p90_ce_loss"], marker="o", label="p90 CE")
    axes[0].plot(summary["epoch"], summary["p99_ce_loss"], marker="o", label="p99 CE")
    axes[0].set_title("CE loss distribution")
    axes[0].legend()

    axes[1].plot(summary["epoch"], summary["top_1pct_total_loss_share"], marker="o", label="top 1%")
    axes[1].plot(summary["epoch"], summary["top_5pct_total_loss_share"], marker="o", label="top 5%")
    axes[1].plot(summary["epoch"], summary["top_10pct_total_loss_share"], marker="o", label="top 10%")
    axes[1].set_title("Total loss concentration")
    axes[1].legend()

    axes[2].plot(summary["epoch"], summary["effective_event_fraction_total_loss"], marker="o")
    axes[2].set_title("Effective event fraction")
    axes[2].set_ylabel("effective N / N")

    axes[3].plot(summary["epoch"], summary["accuracy"], marker="o", label="subset accuracy")
    axes[3].plot(summary["epoch"], summary["mean_confidence"], marker="o", label="mean confidence")
    axes[3].set_title("Prediction behavior")
    axes[3].legend()

    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_hard_overlap(overlap: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for fraction, df_fraction in overlap.groupby("fraction"):
        label = f"top {100 * fraction:g}%"
        axes[0].plot(df_fraction["epoch"], df_fraction["jaccard_vs_epoch1"], marker="o", label=label)
        axes[1].plot(df_fraction["epoch"], df_fraction["jaccard_vs_previous"], marker="o", label=label)
    axes[0].set_title("Hard-event overlap vs epoch 1")
    axes[1].set_title("Hard-event overlap vs previous epoch")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.set_ylabel("Jaccard overlap")
        ax.grid(True, alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_class_composition(df: pd.DataFrame, out: Path) -> None:
    rows = []
    for epoch, epoch_df in df.groupby("epoch"):
        k = max(1, int(math.ceil(len(epoch_df) * 0.05)))
        hard = epoch_df.nlargest(k, "total_loss")
        for class_name, frac in hard["class_name"].value_counts(normalize=True).sort_index().items():
            rows.append({"epoch": epoch, "class_name": class_name, "fraction": frac})
    comp = pd.DataFrame(rows)
    pivot = comp.pivot_table(index="epoch", columns="class_name", values="fraction", fill_value=0.0)
    ax = pivot.plot(kind="bar", stacked=True, figsize=(8, 5))
    ax.set_title("Class composition of top 5% loss events")
    ax.set_xlabel("epoch")
    ax.set_ylabel("fraction")
    ax.legend(title="class", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(True, axis="y", alpha=0.25)
    ax.figure.tight_layout()
    ax.figure.savefig(out, dpi=180)
    plt.close(ax.figure)


def plot_ce_vs_supcon(df: pd.DataFrame, out: Path) -> None:
    epochs = [int(df["epoch"].min()), int(df["epoch"].max())]
    fig, axes = plt.subplots(1, len(epochs), figsize=(6 * len(epochs), 5), squeeze=False)
    for ax, epoch in zip(axes.ravel(), epochs):
        epoch_df = df[df["epoch"] == epoch]
        ax.scatter(epoch_df["ce_loss"], epoch_df["supcon_loss"], s=5, alpha=0.25)
        ax.set_title(f"epoch {epoch}")
        ax.set_xlabel("CE loss")
        ax.set_ylabel("InfoNCE/SupCon anchor loss")
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    cache_dir = Path(args.cache_dir)
    out_dir = run_dir / "figures" / "event_loss_concentration"
    out_dir.mkdir(parents=True, exist_ok=True)

    (
        X,
        y,
        class_names_per_event,
        source_indices,
        event_ids,
        class_names,
        config,
        sample_meta,
    ) = load_fixed_sample(run_dir, cache_dir, args.events_per_class, args.shard, args.seed)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    supcon_temperature = float(
        args.supcon_temperature if args.supcon_temperature is not None else config.get("supcon_temperature", 0.1)
    )
    supcon_weight = float(args.supcon_weight if args.supcon_weight is not None else config.get("supcon_weight", 0.0))
    checkpoints = discover_checkpoints(run_dir, args.checkpoint_glob)

    all_rows = []
    summary_rows = []
    for epoch, model_path in checkpoints:
        print(f"analyzing epoch {epoch}: {model_path}", flush=True)
        model = load_checkpoint_model(model_path, config, device)
        Z, logits = extract_outputs(model, X, device=device, batch_size=args.batch_size)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        ce, confidence, entropy, pred = ce_per_event(logits, y)
        supcon, max_pos, max_neg, margin = supcon_diagnostics(
            Z,
            y,
            temperature=supcon_temperature,
            chunk_size=args.similarity_chunk_size,
        )
        total = ce + supcon_weight * supcon
        epoch_df = pd.DataFrame(
            {
                "epoch": int(epoch),
                "event_id": event_ids,
                "class_name": class_names_per_event,
                "source_index": source_indices,
                "true_label": y.astype(int),
                "pred_label": pred.astype(int),
                "pred_class": [class_names[int(idx)] for idx in pred],
                "ce_loss": ce,
                "supcon_loss": supcon,
                "weighted_supcon_loss": supcon_weight * supcon,
                "total_loss": total,
                "confidence": confidence,
                "entropy": entropy,
                "latent_norm": np.linalg.norm(Z, axis=1),
                "max_positive_similarity": max_pos,
                "max_negative_similarity": max_neg,
                "pos_neg_margin": margin,
            }
        )
        all_rows.append(epoch_df)
        summary_rows.append(summarize_epoch(epoch_df))

    per_event = pd.concat(all_rows, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    overlap = hard_overlap(per_event, fractions=[0.01, 0.05, 0.10])

    per_event.to_csv(out_dir / "event_loss_by_epoch.csv", index=False)
    summary.to_csv(out_dir / "event_loss_summary_by_epoch.csv", index=False)
    overlap.to_csv(out_dir / "hard_event_overlap.csv", index=False)
    (out_dir / "event_loss_sample_metadata.json").write_text(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "cache_dir": str(cache_dir),
                "events_per_class": int(args.events_per_class),
                "seed": int(args.seed),
                "shard": int(args.shard),
                "class_names": class_names,
                "supcon_temperature": supcon_temperature,
                "supcon_weight": supcon_weight,
                "sample_meta": sample_meta,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    plot_loss_distributions(per_event, out_dir / "loss_distribution_by_epoch.png")
    plot_loss_concentration(per_event, out_dir / "loss_concentration_by_epoch.png")
    plot_summary(summary, out_dir / "event_loss_summary.png")
    plot_hard_overlap(overlap, out_dir / "hard_event_overlap.png")
    plot_class_composition(per_event, out_dir / "top5pct_loss_class_composition.png")
    plot_ce_vs_supcon(per_event, out_dir / "ce_vs_supcon_epoch_first_last.png")

    print(f"saved: {out_dir / 'event_loss_by_epoch.csv'}")
    print(f"saved: {out_dir / 'event_loss_summary_by_epoch.csv'}")
    print(f"saved: {out_dir / 'hard_event_overlap.csv'}")
    print(f"saved: {out_dir / 'loss_concentration_by_epoch.png'}")
    print(f"saved: {out_dir / 'event_loss_summary.png'}")


if __name__ == "__main__":
    main()
