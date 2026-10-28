#!/usr/bin/env python3
"""Plot latent-space evolution for the 100M / 80M checkpoint sequence.

The script uses a fixed, small sample from the materialized raw-tensor cache and
extracts latents from each epoch checkpoint. This keeps epoch-to-epoch plots
comparable without reloading the full 80M-event training set.
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
from sklearn.decomposition import PCA

from ngt_model.datasets.collide2v_data import preprocess_dataset


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
    parser.add_argument("--save-latents", action="store_true")
    parser.add_argument(
        "--checkpoint-glob",
        default="mha_nodes_aligned_epoch_*_for_eval.pt",
        help="Glob under RUN_DIR/models. Use latent-only eval checkpoints by default.",
    )
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
    if stride > 1:
        shard_suffix = f"_shard{shard:04d}of{stride:04d}"
    else:
        shard_suffix = ""
    name = f"train_mha_{class_name}_{loaded}_{feature_mode}_{selection}{shard_suffix}_{source}_rawtensor.npy"
    return cache_dir / name


def load_fixed_sample(run_dir: Path, cache_dir: Path, events_per_class: int, shard: int, seed: int):
    config = read_json(run_dir / "run_config.json")
    class_names = list(config["class_names"])
    label_map = {name: int(idx) for name, idx in config["label_map"].items()}
    rng = np.random.default_rng(seed)

    raw_parts = []
    label_parts = []
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
    order = rng.permutation(len(y))
    raw_all = raw_all[order]
    y = y[order]

    add_pid_ohe = config.get("pid_encoding", "onehot") == "onehot"
    norm = {key: float(value) for key, value in config["norm"].items()}
    pid_mapping = {int(key): int(value) for key, value in config.get("pid_mapping", {}).items()}
    X, _ = preprocess_dataset(raw_all, norm=norm, pid_mapping=pid_mapping, add_pid_ohe=add_pid_ohe)
    return X.astype(np.float32, copy=False), y, class_names, config, sample_meta


@torch.no_grad()
def extract_latents(model_path: Path, X: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.to(device)
    model.eval()

    latents = []
    for start in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[start : start + batch_size]).to(device)
        out = model(xb)
        if isinstance(out, dict):
            z = out["latent"]
        else:
            z = out
        latents.append(z.detach().cpu().numpy())

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return np.concatenate(latents, axis=0)


def effective_rank(explained: np.ndarray) -> float:
    positive = explained[explained > 0]
    if len(positive) == 0:
        return 0.0
    return float(np.exp(-np.sum(positive * np.log(positive))))


def latent_metrics(epoch: int, Z: np.ndarray, y: np.ndarray, class_names: list[str]) -> dict:
    centered = Z - Z.mean(axis=0, keepdims=True)
    cov = np.cov(centered, rowvar=False)
    eig = np.maximum(np.linalg.eigvalsh(cov)[::-1], 0.0)
    total_var = float(eig.sum())
    explained = eig / total_var if total_var > 0 else np.zeros_like(eig)

    centroids = []
    within_sum = 0.0
    within_n = 0
    for label, _class_name in enumerate(class_names):
        zc = Z[y == label]
        centroid = zc.mean(axis=0)
        centroids.append(centroid)
        within_sum += float(((zc - centroid) ** 2).sum())
        within_n += int(zc.shape[0])
    centroids = np.stack(centroids)
    global_centroid = Z.mean(axis=0)
    between = float(((centroids - global_centroid) ** 2).sum(axis=1).mean())
    within = within_sum / max(within_n, 1)

    norms = np.linalg.norm(Z, axis=1)
    return {
        "epoch": int(epoch),
        "n_events": int(len(Z)),
        "latent_dim": int(Z.shape[1]),
        "mean_norm": float(norms.mean()),
        "std_norm": float(norms.std()),
        "total_variance": total_var,
        "effective_rank_entropy": effective_rank(explained),
        "participation_ratio": float((total_var**2) / max(float(np.sum(eig**2)), 1e-12)),
        "top1_explained_variance": float(explained[0]) if len(explained) else 0.0,
        "top5_cumulative_explained_variance": float(explained[:5].sum()),
        "top10_cumulative_explained_variance": float(explained[:10].sum()),
        "within_class_mse": within,
        "between_class_mse": between,
        "separation_ratio": between / max(within, 1e-12),
    }


def plot_pca_epochs(coords_by_epoch: dict[int, np.ndarray], y: np.ndarray, class_names: list[str], out: Path):
    epochs = list(coords_by_epoch)
    ncols = min(5, len(epochs))
    nrows = int(math.ceil(len(epochs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.5 * nrows), squeeze=False)
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))

    all_coords = np.concatenate(list(coords_by_epoch.values()), axis=0)
    x_pad = 0.05 * max(np.ptp(all_coords[:, 0]), 1e-6)
    y_pad = 0.05 * max(np.ptp(all_coords[:, 1]), 1e-6)
    xlim = (float(all_coords[:, 0].min() - x_pad), float(all_coords[:, 0].max() + x_pad))
    ylim = (float(all_coords[:, 1].min() - y_pad), float(all_coords[:, 1].max() + y_pad))

    for ax, epoch in zip(axes.ravel(), epochs):
        coords = coords_by_epoch[epoch]
        for label, class_name in enumerate(class_names):
            mask = y == label
            ax.scatter(coords[mask, 0], coords[mask, 1], s=5, alpha=0.35, color=colors[label], label=class_name)
        ax.set_title(f"epoch {epoch}")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.2)

    for ax in axes.ravel()[len(epochs) :]:
        ax.axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(class_names))
    fig.suptitle("Latent PCA projection over checkpoints", y=0.98)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_centroid_trajectories(
    coords_by_epoch: dict[int, np.ndarray],
    y: np.ndarray,
    class_names: list[str],
    out: Path,
):
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
    for label, class_name in enumerate(class_names):
        points = []
        for epoch, coords in coords_by_epoch.items():
            centroid = coords[y == label].mean(axis=0)
            points.append(centroid)
        points = np.stack(points)
        ax.plot(points[:, 0], points[:, 1], marker="o", color=colors[label], label=class_name)
        for epoch, point in zip(coords_by_epoch.keys(), points):
            ax.text(point[0], point[1], str(epoch), fontsize=8, color=colors[label])
    ax.set_title("Class centroid trajectories in common PCA space")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_metrics(metrics: pd.DataFrame, history: pd.DataFrame | None, out: Path):
    merged = metrics.copy()
    if history is not None and "epoch" in history:
        history = history.copy()
        history["epoch"] = history["epoch"].astype(int)
        keep = [col for col in ["epoch", "train_acc", "val_acc", "train_loss", "val_loss"] if col in history]
        merged = merged.merge(history[keep], on="epoch", how="left")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes = axes.ravel()
    axes[0].plot(merged["epoch"], merged["effective_rank_entropy"], marker="o")
    axes[0].set_title("Effective rank")
    axes[0].set_ylabel("entropy rank")

    axes[1].plot(merged["epoch"], merged["top1_explained_variance"], marker="o", label="top1")
    axes[1].plot(merged["epoch"], merged["top5_cumulative_explained_variance"], marker="o", label="top5")
    axes[1].set_title("PCA variance concentration")
    axes[1].legend()

    axes[2].plot(merged["epoch"], merged["separation_ratio"], marker="o")
    axes[2].set_title("Class separation")
    axes[2].set_ylabel("between / within MSE")

    axes[3].plot(merged["epoch"], merged["mean_norm"], marker="o", label="mean norm")
    if "val_acc" in merged:
        ax2 = axes[3].twinx()
        ax2.plot(merged["epoch"], merged["val_acc"], marker="s", color="tab:orange", label="val acc")
        ax2.set_ylabel("val_acc")
    axes[3].set_title("Latent norm and validation accuracy")
    axes[3].set_ylabel("latent norm")

    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    cache_dir = Path(args.cache_dir)
    out_dir = run_dir / "figures" / "latent_evolution"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    X, y, class_names, config, sample_meta = load_fixed_sample(
        run_dir=run_dir,
        cache_dir=cache_dir,
        events_per_class=args.events_per_class,
        shard=args.shard,
        seed=args.seed,
    )
    checkpoints = discover_checkpoints(run_dir, args.checkpoint_glob)

    latents_by_epoch = {}
    rows = []
    for epoch, model_path in checkpoints:
        print(f"extracting epoch {epoch}: {model_path}", flush=True)
        Z = extract_latents(model_path, X, device=device, batch_size=args.batch_size)
        latents_by_epoch[epoch] = Z
        rows.append(latent_metrics(epoch, Z, y, class_names))
        if args.save_latents:
            np.savez_compressed(out_dir / f"latents_epoch_{epoch:03d}.npz", latents=Z, labels=y)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "latent_evolution_summary.csv", index=False)
    (out_dir / "latent_evolution_summary.json").write_text(
        json.dumps(rows, indent=2),
        encoding="utf-8",
    )
    (out_dir / "latent_sample_metadata.json").write_text(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "cache_dir": str(cache_dir),
                "events_per_class": int(args.events_per_class),
                "seed": int(args.seed),
                "shard": int(args.shard),
                "class_names": class_names,
                "n_features": int(config.get("n_features", X.shape[-1])),
                "sample_meta": sample_meta,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    Z_all = np.concatenate([latents_by_epoch[epoch] for epoch in latents_by_epoch], axis=0)
    pca = PCA(n_components=2, random_state=args.seed)
    coords_all = pca.fit_transform(Z_all)
    coords_by_epoch = {}
    start = 0
    for epoch, Z in latents_by_epoch.items():
        stop = start + len(Z)
        coords_by_epoch[epoch] = coords_all[start:stop]
        start = stop

    plot_pca_epochs(coords_by_epoch, y, class_names, out_dir / "latent_pca_by_epoch.png")
    plot_centroid_trajectories(coords_by_epoch, y, class_names, out_dir / "latent_centroid_trajectory.png")

    history_path = run_dir / "training_history.csv"
    history = pd.read_csv(history_path) if history_path.exists() else None
    plot_metrics(metrics, history, out_dir / "latent_evolution_metrics.png")

    print(f"saved: {out_dir / 'latent_evolution_summary.csv'}")
    print(f"saved: {out_dir / 'latent_pca_by_epoch.png'}")
    print(f"saved: {out_dir / 'latent_centroid_trajectory.png'}")
    print(f"saved: {out_dir / 'latent_evolution_metrics.png'}")


if __name__ == "__main__":
    main()
