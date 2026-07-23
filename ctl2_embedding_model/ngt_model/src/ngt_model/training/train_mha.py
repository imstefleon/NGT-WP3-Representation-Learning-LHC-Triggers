#!/usr/bin/env python3
"""
Train the PyTorch MHA embedding model on Collide-2V L1 candidate tensors.

This script is the non-notebook training entry point for the MHA architecture.
It is intentionally close to the MHA notebook defaults, but uses a larger
latent space by default (`latent_dim=128`) and can be launched with `torchrun`
for DistributedDataParallel training.

Examples:
    # Single-GPU quick check
    python train_mha.py --n-events-per-class 1000 --epochs 3 --device cuda

    # Current larger test, latent_dim=128
    python train_mha.py --n-events-per-class 100000 --epochs 50 --device cuda

    # Distributed, if the environment exposes multiple GPUs
    torchrun --nproc_per_node=2 train_mha.py --n-events-per-class 100000 --epochs 50
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from ngt_model.datasets.collide2v_data import (
    CLASS_NAMES,
    SUPPORTED_CLASS_NAMES,
    Collide2VConfig,
    Collide2VLoader,
    augment_rotate_phi,
    build_pid_mapping,
    build_split_indices,
    compute_normalization_constants,
    preprocess_dataset,
)
from ngt_model.models.mha_model import LatentOnlyWrapper, MHAEmbeddingFullPrecision


class TensorDatasetTorch(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32, copy=False))
        self.y = torch.from_numpy(y.astype(np.int64, copy=False))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def parse_args():
    parser = argparse.ArgumentParser(description="Train MHA Collide-2V embedding model.")
    parser.add_argument("--data-source", default="collide2v_updated",
                        choices=["collide2v_updated", "samples_json", "hf"])
    parser.add_argument("--class-names", default=",".join(CLASS_NAMES),
                        help=("Comma-separated class list. Default is the SM baseline "
                              f"{','.join(CLASS_NAMES)}. For HH4B use "
                              "minbias,top,qcd,dy,hh4b."))
    parser.add_argument("--feature-mode", default="l1_only",
                        choices=["l1_only"],
                        help="Only L1 candidate columns are allowed for NGT runs.")
    parser.add_argument("--candidate-selection", default="top_pt")
    parser.add_argument("--collide2v-path", default=None)
    parser.add_argument("--n-constituents", type=int, default=100)
    parser.add_argument("--n-events-per-class", type=int, default=100_000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--ff-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-attention-blocks", type=int, default=2)
    parser.add_argument("--pooling-mode", default="attention", choices=["mean", "attention"])
    parser.add_argument("--classifier-head", default="mlp", choices=["mlp", "linear"])
    parser.add_argument("--latent-var-reg-weight", type=float, default=0.0,
                        help="Weight for latent variance regularization. 0 disables it.")
    parser.add_argument("--latent-var-target", type=float, default=1.0,
                        help="Minimum per-dimension latent std target for variance regularization.")
    parser.add_argument("--augmentation-copies", type=int, default=1)
    parser.add_argument("--class-weight-mode", default="none", choices=["none", "balanced"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="cuda/cpu. In torchrun, local CUDA rank is used.")
    parser.add_argument("--output-root", default="ngt_mha_nodes_aligned_outputs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--eval-quality-events", type=int, default=0,
                        help="If >0, run eval_quality_collide2v on the saved latent-only checkpoint.")
    parser.add_argument("--mlflow-enable", action="store_true",
                        help="Log parameters, metrics, and run artifacts to MLflow.")
    parser.add_argument("--mlflow-tracking-uri", default="https://ml.cern.ch/models")
    parser.add_argument("--mlflow-experiment", default="ngt_model")
    parser.add_argument("--mlflow-run-name", default=None)
    return parser.parse_args()


def parse_class_names(raw: str) -> list[str]:
    class_names = [part.strip().lower() for part in raw.split(",") if part.strip()]
    if not class_names:
        raise ValueError("--class-names must contain at least one class")
    unknown = [cls for cls in class_names if cls not in SUPPORTED_CLASS_NAMES]
    if unknown:
        raise ValueError(f"Unsupported classes {unknown}; supported classes are {SUPPORTED_CLASS_NAMES}")
    if len(set(class_names)) != len(class_names):
        raise ValueError(f"--class-names contains duplicates: {class_names}")
    return class_names


def setup_distributed(args):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if not distributed:
        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        return False, 0, 1, 0, device

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    return True, dist.get_rank(), dist.get_world_size(), local_rank, device


def cleanup_distributed(distributed: bool):
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank: int) -> bool:
    return rank == 0


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_run_dir(args) -> Path:
    run_date = datetime.now().strftime("%Y-%m-%d")
    if args.run_name:
        run_label = args.run_name
    else:
        stamp = datetime.now().strftime("%H%M%S")
        run_label = (
            f"mha_latent{args.latent_dim}_ff{args.ff_dim}_heads{args.num_heads}_"
            f"{args.data_source}_{args.candidate_selection}_n{args.n_events_per_class}_{stamp}"
        )
    project = Path(args.output_root) / "runs" / run_date / run_label
    for subdir in [project, project / "models", project / "figures"]:
        subdir.mkdir(parents=True, exist_ok=True)
    return project


def make_class_weights(
    y_labels: np.ndarray,
    mode: str,
    device: torch.device,
    n_classes: int,
) -> Optional[torch.Tensor]:
    if mode == "none":
        return None
    counts = np.bincount(y_labels, minlength=n_classes).astype(np.float64)
    weights = len(y_labels) / (n_classes * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def metric_reduce(loss_sum, correct, total, device, distributed):
    vals = torch.tensor([loss_sum, correct, total], dtype=torch.float64, device=device)
    if distributed:
        dist.all_reduce(vals, op=dist.ReduceOp.SUM)
    loss = float(vals[0].item() / max(vals[2].item(), 1.0))
    acc = float(vals[1].item() / max(vals[2].item(), 1.0))
    return loss, acc


def latent_variance_regularization(latent: torch.Tensor, target_std: float) -> torch.Tensor:
    if latent.shape[0] <= 1:
        return latent.new_tensor(0.0)
    std = torch.sqrt(latent.var(dim=0, unbiased=False) + 1e-4)
    return torch.mean(F.relu(target_std - std))


def run_epoch(
    model,
    loader,
    device,
    class_weights=None,
    optimizer=None,
    distributed=False,
    latent_var_reg_weight: float = 0.0,
    latent_var_target: float = 1.0,
):
    is_train = optimizer is not None
    model.train(is_train)
    loss_sum = 0.0
    ce_loss_sum = 0.0
    var_reg_sum = 0.0
    correct = 0.0
    total = 0.0

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        with torch.set_grad_enabled(is_train):
            out = model(xb)
            ce_loss = F.cross_entropy(out["logits"], yb, weight=class_weights)
            var_reg = latent_variance_regularization(out["latent"], latent_var_target)
            loss = ce_loss + (latent_var_reg_weight * var_reg if latent_var_reg_weight > 0 else 0.0)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        bs = len(xb)
        loss_sum += float(loss.item()) * bs
        ce_loss_sum += float(ce_loss.item()) * bs
        var_reg_sum += float(var_reg.item()) * bs
        correct += float((out["logits"].argmax(dim=1) == yb).sum().item())
        total += float(bs)

    loss, acc = metric_reduce(loss_sum, correct, total, device, distributed)
    ce_loss, _ = metric_reduce(ce_loss_sum, correct, total, device, distributed)
    var_reg, _ = metric_reduce(var_reg_sum, correct, total, device, distributed)
    return loss, acc, ce_loss, var_reg


@torch.no_grad()
def extract_outputs(model, X: np.ndarray, y: np.ndarray, device: torch.device, batch_size: int):
    model.eval()
    latents, logits = [], []
    loader = DataLoader(TensorDatasetTorch(X, y), batch_size=batch_size, shuffle=False, num_workers=0)
    for xb, _ in loader:
        xb = xb.to(device)
        out = model(xb)
        latents.append(out["latent"].detach().cpu().numpy())
        logits.append(out["logits"].detach().cpu().numpy())
    return np.concatenate(latents, axis=0), np.concatenate(logits, axis=0)


def save_json(path: Path, payload: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def make_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for true_label, pred_label in zip(y_true.astype(int), y_pred.astype(int)):
        if 0 <= true_label < n_classes and 0 <= pred_label < n_classes:
            cm[true_label, pred_label] += 1
    return cm


def make_classification_report(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict:
    n_classes = len(class_names)
    cm = make_confusion_matrix(y_true, y_pred, n_classes)
    report = {}
    precisions, recalls, f1s, supports = [], [], [], []

    for idx, cls_name in enumerate(class_names):
        tp = float(cm[idx, idx])
        fp = float(cm[:, idx].sum() - cm[idx, idx])
        fn = float(cm[idx, :].sum() - cm[idx, idx])
        support = float(cm[idx, :].sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        report[cls_name] = {
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
            "support": support,
        }
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)

    total = float(np.sum(supports))
    accuracy = float(np.trace(cm) / total) if total > 0 else 0.0
    weights = np.array(supports, dtype=np.float64) / total if total > 0 else np.zeros(n_classes)
    report["accuracy"] = {"precision": accuracy, "recall": accuracy, "f1-score": accuracy, "support": total}
    report["macro avg"] = {
        "precision": float(np.mean(precisions)) if precisions else 0.0,
        "recall": float(np.mean(recalls)) if recalls else 0.0,
        "f1-score": float(np.mean(f1s)) if f1s else 0.0,
        "support": total,
    }
    report["weighted avg"] = {
        "precision": float(np.sum(np.array(precisions) * weights)),
        "recall": float(np.sum(np.array(recalls) * weights)),
        "f1-score": float(np.sum(np.array(f1s) * weights)),
        "support": total,
    }
    return report


def _load_mlflow_credentials_from_local_conf():
    """Load local MLflow credentials without requiring PyYAML."""
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


def setup_mlflow(args, project: Path, config: dict):
    _load_mlflow_credentials_from_local_conf()
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError("MLflow logging requested, but the 'mlflow' package is not installed.") from exc

    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment)
    run_name = args.mlflow_run_name or args.run_name or project.name
    mlflow.start_run(run_name=run_name)

    params = {
        "data_source": args.data_source,
        "feature_mode": args.feature_mode,
        "candidate_selection": args.candidate_selection,
        "n_constituents": args.n_constituents,
        "n_events_per_class": args.n_events_per_class,
        "epochs": args.epochs,
        "early_stopping_patience": args.early_stopping_patience,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "latent_dim": args.latent_dim,
        "ff_dim": args.ff_dim,
        "num_heads": args.num_heads,
        "num_attention_blocks": args.num_attention_blocks,
        "pooling_mode": args.pooling_mode,
        "classifier_head": args.classifier_head,
        "latent_var_reg_weight": args.latent_var_reg_weight,
        "latent_var_target": args.latent_var_target,
        "augmentation_copies": args.augmentation_copies,
        "class_weight_mode": args.class_weight_mode,
        "seed": args.seed,
        "n_features": config["n_features"],
        "n_train": config["n_train"],
        "n_train_aug": config["n_train_aug"],
        "n_val": config["n_val"],
        "n_eval": config["n_eval"],
        "distributed": config["distributed"],
        "world_size": config["world_size"],
    }
    mlflow.log_params(params)
    return mlflow


def main():
    args = parse_args()
    class_names = parse_class_names(args.class_names)
    distributed, rank, world_size, local_rank, device = setup_distributed(args)
    set_seed(args.seed + rank)
    mlflow_client = None

    try:
        project = make_run_dir(args) if is_main(rank) else None
        if distributed:
            obj = [str(project) if project is not None else None]
            dist.broadcast_object_list(obj, src=0)
            project = Path(obj[0])

        if is_main(rank):
            print(f"device={device} distributed={distributed} world_size={world_size}")
            print(f"output={project}")
            print(
                f"latent_dim={args.latent_dim} ff_dim={args.ff_dim} "
                f"num_heads={args.num_heads} num_attention_blocks={args.num_attention_blocks}"
            )
            print(f"class_names={class_names}")
            if args.mlflow_enable:
                print(f"mlflow=enabled uri={args.mlflow_tracking_uri} experiment={args.mlflow_experiment}")
            else:
                print("mlflow=disabled (add --mlflow-enable to log this run)")

        cfg_kwargs = dict(
            data_source=args.data_source,
            n_constituents=args.n_constituents,
            feature_mode=args.feature_mode,
            candidate_selection=args.candidate_selection,
            cache_dir=Path(args.output_root) / "cache",
            class_names=class_names,
        )
        if args.collide2v_path:
            cfg_kwargs["collide2v_path"] = Path(args.collide2v_path)
        cfg = Collide2VConfig(**cfg_kwargs)
        label_map = cfg.label_map

        loader = Collide2VLoader(cfg)
        events_per_class = {cls: args.n_events_per_class for cls in class_names}
        raw_tensor, y, sample_name, _ = loader.load_classes(events_per_class, cache_prefix="train_mha")

        train_idx, val_idx, eval_idx = build_split_indices(
            y,
            class_names=class_names,
            label_map=label_map,
            seed=args.seed,
        )
        norm = compute_normalization_constants(raw_tensor[train_idx])
        pid_mapping = build_pid_mapping(raw_tensor[train_idx])
        X_all, pid_mapping = preprocess_dataset(raw_tensor, norm=norm, pid_mapping=pid_mapping, add_pid_ohe=True)

        X_train, y_train = X_all[train_idx], y[train_idx]
        X_val, y_val = X_all[val_idx], y[val_idx]
        X_eval, y_eval = X_all[eval_idx], y[eval_idx]

        if args.augmentation_copies > 0:
            aug_blocks = [X_train]
            for _ in range(args.augmentation_copies):
                raw_aug = augment_rotate_phi(raw_tensor[train_idx])
                X_rot, _ = preprocess_dataset(raw_aug, norm=norm, pid_mapping=pid_mapping, add_pid_ohe=True)
                aug_blocks.append(X_rot)
            X_aug = np.concatenate(aug_blocks, axis=0)
            y_aug = np.tile(y_train, args.augmentation_copies + 1)
        else:
            X_aug = X_train
            y_aug = y_train

        if is_main(rank):
            config = {
                **vars(args),
                "class_names": class_names,
                "label_map": label_map,
                "n_features": int(X_all.shape[-1]),
                "n_train": int(len(X_train)),
                "n_train_aug": int(len(X_aug)),
                "n_val": int(len(X_val)),
                "n_eval": int(len(X_eval)),
                "pid_mapping": {int(k): int(v) for k, v in pid_mapping.items()},
                "norm": norm,
                "distributed": distributed,
                "world_size": world_size,
            }
            save_json(project / "run_config.json", config)
            if args.mlflow_enable:
                mlflow_client = setup_mlflow(args, project, config)

        model = MHAEmbeddingFullPrecision(
            n_constituents=args.n_constituents,
            n_features=X_all.shape[-1],
            latent_dim=args.latent_dim,
            n_classes=len(class_names),
            ff_dim=args.ff_dim,
            num_heads=args.num_heads,
            pooling_mode=args.pooling_mode,
            classifier_head=args.classifier_head,
            num_attention_blocks=args.num_attention_blocks,
        ).to(device)
        n_params = int(sum(p.numel() for p in model.parameters()))
        if is_main(rank):
            config["n_params"] = n_params
            save_json(project / "run_config.json", config)
            if mlflow_client is not None:
                mlflow_client.log_param("n_params", n_params)

        if distributed:
            ddp_kwargs = {"device_ids": [local_rank], "output_device": local_rank} if device.type == "cuda" else {}
            model_for_train = DDP(model, **ddp_kwargs)
        else:
            model_for_train = model

        class_weights = make_class_weights(y_aug, args.class_weight_mode, device, len(class_names))
        optimizer = torch.optim.AdamW(model_for_train.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        train_dataset = TensorDatasetTorch(X_aug, y_aug)
        val_dataset = TensorDatasetTorch(X_val, y_val)
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if distributed else None
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if distributed else None
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

        history = []
        best_val_loss = float("inf")
        best_state = None
        patience_left = args.early_stopping_patience

        for epoch in range(1, args.epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            t0 = time.time()
            train_loss, train_acc, train_ce_loss, train_var_reg = run_epoch(
                model_for_train, train_loader, device, class_weights=class_weights,
                optimizer=optimizer, distributed=distributed,
                latent_var_reg_weight=args.latent_var_reg_weight,
                latent_var_target=args.latent_var_target,
            )
            val_loss, val_acc, val_ce_loss, val_var_reg = run_epoch(
                model_for_train, val_loader, device, class_weights=class_weights,
                optimizer=None, distributed=distributed,
                latent_var_reg_weight=args.latent_var_reg_weight,
                latent_var_target=args.latent_var_target,
            )
            elapsed = time.time() - t0
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "train_ce_loss": train_ce_loss,
                "train_latent_var_reg": train_var_reg,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_ce_loss": val_ce_loss,
                "val_latent_var_reg": val_var_reg,
                "elapsed_s": elapsed,
            }
            history.append(row)
            if is_main(rank):
                print(
                    f"Epoch {epoch:03d}/{args.epochs} "
                    f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                    f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
                    f"var_reg={val_var_reg:.4f} "
                    f"{elapsed:.1f}s",
                    flush=True,
                )
                if mlflow_client is not None:
                    mlflow_client.log_metrics(
                        {
                            "train_loss": train_loss,
                            "train_acc": train_acc,
                            "train_ce_loss": train_ce_loss,
                            "train_latent_var_reg": train_var_reg,
                            "val_loss": val_loss,
                            "val_acc": val_acc,
                            "val_ce_loss": val_ce_loss,
                            "val_latent_var_reg": val_var_reg,
                            "epoch_elapsed_s": elapsed,
                        },
                        step=epoch,
                    )

            improved = val_loss < best_val_loss
            if improved:
                best_val_loss = val_loss
                patience_left = args.early_stopping_patience
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_left -= 1
                if args.early_stopping_patience > 0 and patience_left <= 0:
                    if is_main(rank):
                        print(f"Early stopping at epoch {epoch}; best_val_loss={best_val_loss:.4f}")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        if is_main(rank):
            pd.DataFrame(history).to_csv(project / "training_history.csv", index=False)
            torch.save(model, project / "models" / "mha_nodes_aligned_best.pt")
            torch.save(LatentOnlyWrapper(model), project / "models" / "mha_nodes_aligned_best_for_eval.pt")

            Z_eval, logits_eval = extract_outputs(model, X_eval, y_eval, device, args.batch_size)
            np.save(project / "eval_latents.npy", Z_eval)
            np.save(project / "eval_logits.npy", logits_eval)

            pred_eval = logits_eval.argmax(axis=1)
            report = make_classification_report(y_eval, pred_eval, class_names)
            cm = make_confusion_matrix(y_eval, pred_eval, len(class_names))
            pd.DataFrame(report).T.to_csv(project / "classification_report_eval.csv")
            pd.DataFrame(
                cm,
                index=[f"true_{c}" for c in class_names],
                columns=[f"pred_{c}" for c in class_names],
            ).to_csv(project / "confusion_matrix_eval.csv")
            if mlflow_client is not None:
                mlflow_client.log_metric("best_val_loss", best_val_loss)
                for cls_name in class_names:
                    if cls_name in report:
                        mlflow_client.log_metric(f"eval_{cls_name}_precision", report[cls_name]["precision"])
                        mlflow_client.log_metric(f"eval_{cls_name}_recall", report[cls_name]["recall"])
                        mlflow_client.log_metric(f"eval_{cls_name}_f1", report[cls_name]["f1-score"])
                mlflow_client.log_artifact(str(project / "run_config.json"))
                mlflow_client.log_artifact(str(project / "training_history.csv"))
                mlflow_client.log_artifact(str(project / "classification_report_eval.csv"))
                mlflow_client.log_artifact(str(project / "confusion_matrix_eval.csv"))

            if args.eval_quality_events > 0:
                from ngt_model.evaluation.eval_quality_collide2v import EvalCollide2V

                anomaly_for_eval = "hh4b" if "hh4b" in class_names else "dy"
                background_for_eval = [cls for cls in class_names if cls != anomaly_for_eval]
                ev = EvalCollide2V(
                    str(project / "models" / "mha_nodes_aligned_best_for_eval.pt"),
                    n_events_per_class=args.eval_quality_events,
                    data_source=args.data_source,
                    device=str(device),
                    output_path=str(project / f"eval_quality_scores_n{args.eval_quality_events}.json"),
                    collide2v_cfg=cfg,
                    background_classes=background_for_eval,
                    anomaly_class=anomaly_for_eval,
                )
                print(f"eval_quality_score={ev.score:.6f}")
                if mlflow_client is not None:
                    mlflow_client.log_metric("eval_quality_score", float(ev.score))
                    mlflow_client.log_artifact(str(project / f"eval_quality_scores_n{args.eval_quality_events}.json"))

            print(f"Saved run -> {project}")

        if distributed:
            dist.barrier()

    finally:
        if mlflow_client is not None:
            mlflow_client.end_run()
        cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
