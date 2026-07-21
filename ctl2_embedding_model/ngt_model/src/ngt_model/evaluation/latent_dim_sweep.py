#!/usr/bin/env python3
"""
Latent-dimension sweep for the MHA Collide-2V workflow.

This is a controlled test for latent collapse: keep the data, class granularity,
and model backbone fixed, vary only `latent_dim`, and compare classification,
eval-quality, and effective-dimensionality metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ngt_model.datasets.collide2v_data import (
    Collide2VConfig,
    Collide2VLoader,
    SUPPORTED_CLASS_NAMES,
    augment_rotate_phi,
    build_pid_mapping,
    build_split_indices,
    compute_normalization_constants,
    preprocess_dataset,
)
from ngt_model.evaluation.eval_quality_collide2v import EvalCollide2V
from ngt_model.evaluation.latent_diagnostics import compute_latent_diagnostics
from ngt_model.models.mha_model import LatentOnlyWrapper, MHAEmbeddingFullPrecision


SEED = 42


class TensorDatasetTorch(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32, copy=False))
        self.y = torch.from_numpy(y.astype(np.int64, copy=False))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


@dataclass
class SweepConfig:
    class_names: list[str]
    latent_dims: list[int]
    n_events_per_class: int
    eval_quality_events: int
    epochs: int
    early_stopping_patience: int
    batch_size: int
    lr: float
    weight_decay: float
    ff_dim: int
    num_heads: int
    augmentation_copies: int
    pooling_mode: str


def parse_csv_list(raw: str, cast=str) -> list:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    return [cast(value) for value in values]


def parse_class_names(raw: str) -> list[str]:
    class_names = [value.lower() for value in parse_csv_list(raw, str)]
    unknown = [cls for cls in class_names if cls not in SUPPORTED_CLASS_NAMES]
    if unknown:
        raise ValueError(f"Unsupported classes {unknown}; supported classes are {SUPPORTED_CLASS_NAMES}")
    if len(set(class_names)) != len(class_names):
        raise ValueError(f"class list contains duplicates: {class_names}")
    return class_names


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    loss_sum = 0.0
    correct = 0
    total = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        with torch.set_grad_enabled(is_train):
            out = model(xb)
            loss = F.cross_entropy(out["logits"], yb)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        loss_sum += float(loss.item()) * len(xb)
        correct += int((out["logits"].argmax(dim=1) == yb).sum().item())
        total += len(xb)
    return loss_sum / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def extract_outputs(model, X: np.ndarray, y: np.ndarray, device: torch.device, batch_size: int):
    model.eval()
    latents, logits = [], []
    loader = DataLoader(TensorDatasetTorch(X, y), batch_size=batch_size, shuffle=False, num_workers=0)
    for xb, _ in loader:
        out = model(xb.to(device))
        latents.append(out["latent"].detach().cpu().numpy())
        logits.append(out["logits"].detach().cpu().numpy())
    return np.concatenate(latents, axis=0), np.concatenate(logits, axis=0)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict:
    n_classes = len(class_names)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for true_label, pred_label in zip(y_true.astype(int), y_pred.astype(int)):
        if 0 <= true_label < n_classes and 0 <= pred_label < n_classes:
            cm[true_label, pred_label] += 1

    out = {}
    f1s = []
    for idx, cls_name in enumerate(class_names):
        tp = float(cm[idx, idx])
        fp = float(cm[:, idx].sum() - cm[idx, idx])
        fn = float(cm[idx, :].sum() - cm[idx, idx])
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        out[f"{cls_name}_precision"] = precision
        out[f"{cls_name}_recall"] = recall
        out[f"{cls_name}_f1"] = f1
        f1s.append(f1)
    out["accuracy"] = float(np.trace(cm) / max(cm.sum(), 1))
    out["macro_f1"] = float(np.mean(f1s)) if f1s else 0.0
    return out


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


def setup_mlflow(args, cfg: SweepConfig, device: torch.device):
    _load_mlflow_credentials_from_local_conf()
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError("MLflow logging requested, but the 'mlflow' package is not installed.") from exc

    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment)
    mlflow.start_run(run_name=args.mlflow_run_name or "latent_dim_sweep")
    mlflow.log_params({
        "sweep_type": "latent_dim",
        "class_names": ",".join(cfg.class_names),
        "latent_dims": ",".join(str(x) for x in cfg.latent_dims),
        "n_events_per_class": cfg.n_events_per_class,
        "eval_quality_events": cfg.eval_quality_events,
        "epochs": cfg.epochs,
        "early_stopping_patience": cfg.early_stopping_patience,
        "ff_dim": cfg.ff_dim,
        "num_heads": cfg.num_heads,
        "augmentation_copies": cfg.augmentation_copies,
        "pooling_mode": cfg.pooling_mode,
        "device": str(device),
    })
    return mlflow


def train_one_latent_dim(
    latent_dim: int,
    cfg: SweepConfig,
    data: dict,
    device: torch.device,
) -> tuple[MHAEmbeddingFullPrecision, dict]:
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    n_features = data["n_features"]

    model = MHAEmbeddingFullPrecision(
        n_constituents=data["n_constituents"],
        n_features=n_features,
        latent_dim=latent_dim,
        n_classes=len(cfg.class_names),
        ff_dim=cfg.ff_dim,
        num_heads=cfg.num_heads,
        pooling_mode=cfg.pooling_mode,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    train_loader = DataLoader(
        TensorDatasetTorch(X_train, y_train),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        TensorDatasetTorch(X_val, y_val),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    best_state = None
    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience_left = cfg.early_stopping_patience
    history = []
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, device)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "elapsed_s": time.time() - t0,
        })
        print(
            f"[latent_dim={latent_dim}] epoch {epoch:03d}/{cfg.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}",
            flush=True,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_left = cfg.early_stopping_patience
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_left -= 1
            if cfg.early_stopping_patience > 0 and patience_left <= 0:
                print(f"[latent_dim={latent_dim}] early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, {
        "n_params": sum(p.numel() for p in model.parameters()),
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "history": history,
    }


def prepare_data(cfg: SweepConfig, data_cfg: Collide2VConfig) -> dict:
    loader = Collide2VLoader(data_cfg)
    events_per_class = {cls: cfg.n_events_per_class for cls in cfg.class_names}
    raw_tensor, y, sample_name, _ = loader.load_classes(events_per_class, cache_prefix="latent_dim_sweep")
    label_map = data_cfg.label_map

    train_idx, val_idx, eval_idx = build_split_indices(
        y,
        class_names=cfg.class_names,
        label_map=label_map,
        seed=SEED,
    )
    norm = compute_normalization_constants(raw_tensor[train_idx])
    pid_mapping = build_pid_mapping(raw_tensor[train_idx])
    X_all, pid_mapping = preprocess_dataset(raw_tensor, norm=norm, pid_mapping=pid_mapping, add_pid_ohe=True)

    X_train = X_all[train_idx]
    y_train = y[train_idx]
    if cfg.augmentation_copies > 0:
        blocks = [X_train]
        labels = [y_train]
        for _ in range(cfg.augmentation_copies):
            raw_aug = augment_rotate_phi(raw_tensor[train_idx])
            X_aug, _ = preprocess_dataset(raw_aug, norm=norm, pid_mapping=pid_mapping, add_pid_ohe=True)
            blocks.append(X_aug)
            labels.append(y_train)
        X_train = np.concatenate(blocks, axis=0)
        y_train = np.concatenate(labels, axis=0)

    return {
        "raw_tensor": raw_tensor,
        "y": y,
        "sample_name": sample_name,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "eval_idx": eval_idx,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_all[val_idx],
        "y_val": y[val_idx],
        "X_eval": X_all[eval_idx],
        "y_eval": y[eval_idx],
        "n_features": int(X_all.shape[-1]),
        "n_constituents": data_cfg.n_constituents,
        "pid_mapping": pid_mapping,
    }


def run_sweep(args):
    set_seed(SEED)
    class_names = parse_class_names(args.class_names)
    latent_dims = parse_csv_list(args.latent_dims, int)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    sweep_cfg = SweepConfig(
        class_names=class_names,
        latent_dims=latent_dims,
        n_events_per_class=args.n_events_per_class,
        eval_quality_events=args.eval_quality_events,
        epochs=args.epochs,
        early_stopping_patience=args.early_stopping_patience,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        ff_dim=args.ff_dim,
        num_heads=args.num_heads,
        augmentation_copies=args.augmentation_copies,
        pooling_mode=args.pooling_mode,
    )
    data_cfg_kwargs = dict(
        data_source=args.data_source,
        class_names=class_names,
        n_constituents=args.n_constituents,
        cache_dir=Path(args.output_root) / "cache",
    )
    if args.collide2v_path:
        data_cfg_kwargs["collide2v_path"] = Path(args.collide2v_path)
    data_cfg = Collide2VConfig(**data_cfg_kwargs)

    print(f"device={device}")
    print(f"class_names={class_names}")
    print(f"latent_dims={latent_dims}")
    print(f"n_events_per_class={args.n_events_per_class}")
    data = prepare_data(sweep_cfg, data_cfg)
    print(f"n_features={data['n_features']} n_train={len(data['X_train'])} n_val={len(data['X_val'])} n_eval={len(data['X_eval'])}")

    mlflow_client = setup_mlflow(args, sweep_cfg, device) if args.mlflow_enable else None
    rows = []
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        for step, latent_dim in enumerate(latent_dims):
            model, train_info = train_one_latent_dim(latent_dim, sweep_cfg, data, device)
            Z_eval, logits_eval = extract_outputs(
                model,
                data["X_eval"],
                data["y_eval"],
                device,
                sweep_cfg.batch_size,
            )
            pred_eval = logits_eval.argmax(axis=1)
            cls_metrics = classification_metrics(data["y_eval"], pred_eval, class_names)
            latent_summary, _ = compute_latent_diagnostics(Z_eval)

            eval_scores = {}
            if args.eval_quality_events > 0:
                anomaly_class = args.anomaly_class
                background_classes = parse_class_names(args.background_classes)
                wrapped = LatentOnlyWrapper(model).to(device)
                wrapped.eval()
                ev = EvalCollide2V(
                    wrapped,
                    n_events_per_class=args.eval_quality_events,
                    collide2v_cfg=data_cfg,
                    background_classes=background_classes,
                    anomaly_class=anomaly_class,
                )
                eval_scores = ev.scores

            row = {
                "latent_dim": latent_dim,
                "ff_dim": sweep_cfg.ff_dim,
                "num_heads": sweep_cfg.num_heads,
                "n_events_per_class": sweep_cfg.n_events_per_class,
                "n_params": train_info["n_params"],
                "best_val_loss": train_info["best_val_loss"],
                "best_val_acc": train_info["best_val_acc"],
                **cls_metrics,
                "effective_rank_entropy": latent_summary["effective_rank_entropy"],
                "participation_ratio": latent_summary["participation_ratio"],
                "participation_ratio_norm": latent_summary["participation_ratio_norm"],
                "top1_explained_variance": latent_summary["top1_explained_variance"],
                "top5_cumulative_explained_variance": latent_summary["top5_cumulative_explained_variance"],
                "top10_cumulative_explained_variance": latent_summary["top10_cumulative_explained_variance"],
                "n_components_90pct": latent_summary["n_components_90pct"],
                "n_components_95pct": latent_summary["n_components_95pct"],
                "eval_quality_score": eval_scores.get("score"),
                "eval_quality_align_loss": eval_scores.get("align_loss"),
                "eval_quality_pr": eval_scores.get("pr"),
                "eval_quality_gmm_gap": eval_scores.get("gmm_gap"),
                "eval_quality_mahalanobis": eval_scores.get("mahalanobis"),
            }
            rows.append(row)
            pd.DataFrame(rows).to_csv(args.output, index=False)
            history_path = output_dir / f"latent_dim_{latent_dim}_training_history.csv"
            pd.DataFrame(train_info["history"]).to_csv(history_path, index=False)

            if mlflow_client is not None:
                safe_prefix = f"latent{latent_dim}"
                mlflow_client.log_metrics({
                    f"{safe_prefix}_best_val_loss": float(row["best_val_loss"]),
                    f"{safe_prefix}_best_val_acc": float(row["best_val_acc"]),
                    f"{safe_prefix}_accuracy": float(row["accuracy"]),
                    f"{safe_prefix}_macro_f1": float(row["macro_f1"]),
                    f"{safe_prefix}_effective_rank": float(row["effective_rank_entropy"]),
                    f"{safe_prefix}_participation_ratio": float(row["participation_ratio"]),
                    f"{safe_prefix}_n_components_90pct": float(row["n_components_90pct"]),
                }, step=step)
                for key in ["hh4b_recall", "hh4b_f1", "eval_quality_score"]:
                    if row.get(key) is not None:
                        mlflow_client.log_metric(f"{safe_prefix}_{key}", float(row[key]), step=step)
                mlflow_client.log_artifact(args.output)
                mlflow_client.log_artifact(str(history_path))

            print(f"[latent_dim={latent_dim}] row saved -> {args.output}")
            print(pd.DataFrame([row]).to_string(index=False))
    finally:
        if mlflow_client is not None:
            mlflow_client.end_run()

    return pd.DataFrame(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep latent_dim while holding data/class setup fixed.")
    parser.add_argument("--output", default="ngt_mha_nodes_aligned_outputs/scale_tests/latent_dim_sweep_hh4b.csv")
    parser.add_argument("--data-source", default="collide2v_updated", choices=["collide2v_updated", "samples_json", "hf"])
    parser.add_argument("--class-names", default="minbias,top,qcd,dy,hh4b")
    parser.add_argument("--background-classes", default="minbias,top,qcd,dy")
    parser.add_argument("--anomaly-class", default="hh4b")
    parser.add_argument("--latent-dims", default="16,32,64,128")
    parser.add_argument("--n-events-per-class", type=int, default=100_000)
    parser.add_argument("--eval-quality-events", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ff-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--pooling-mode", default="attention", choices=["mean", "attention"])
    parser.add_argument("--augmentation-copies", type=int, default=1)
    parser.add_argument("--n-constituents", type=int, default=100)
    parser.add_argument("--collide2v-path", default=None)
    parser.add_argument("--output-root", default="ngt_mha_nodes_aligned_outputs")
    parser.add_argument("--device", default=None)
    parser.add_argument("--mlflow-enable", action="store_true")
    parser.add_argument("--mlflow-tracking-uri", default="https://ml.cern.ch/models")
    parser.add_argument("--mlflow-experiment", default="ngt_model")
    parser.add_argument("--mlflow-run-name", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    df = run_sweep(args)
    print(f"\nSaved latent-dim sweep -> {args.output}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
