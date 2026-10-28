#!/usr/bin/env python3
"""DDP throughput/VRAM benchmark for the 24h 100M-parameter run.

This benchmark uses synthetic 5-feature L1 tensors to isolate model throughput
and VRAM for the exact architecture before launching the 80M-event data job.
Launch with torchrun on the target 4 x 48 GB node.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from ngt_model.models.mha_model import MHAEmbeddingFullPrecision


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark 100M MHA DDP throughput and VRAM.")
    parser.add_argument("--batch-sizes", default="1,2,4,8,16,24,32,48,64")
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=20)
    parser.add_argument("--d-model", type=int, default=2016)
    parser.add_argument("--ff-dim", type=int, default=8064)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--num-attention-blocks", type=int, default=4)
    parser.add_argument("--n-constituents", type=int, default=100)
    parser.add_argument("--n-features", type=int, default=5)
    parser.add_argument("--n-classes", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--output", default="scaling_study/large_100m_80m/benchmark_100m_ddp_results.csv")
    return parser.parse_args()


def count_params(model: torch.nn.Module) -> int:
    return sum(int(p.numel()) for p in model.parameters() if p.requires_grad)


def setup():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if distributed:
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return distributed, local_rank, world_size, torch.device(f"cuda:{local_rank}")


def cleanup(distributed: bool):
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


def reduce_min(value: int, device: torch.device, distributed: bool) -> int:
    tensor = torch.tensor([value], dtype=torch.int64, device=device)
    if distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
    return int(tensor.item())


def reduce_max_float(value: float, device: torch.device, distributed: bool) -> float:
    tensor = torch.tensor([value], dtype=torch.float64, device=device)
    if distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def benchmark_batch(model, optimizer, batch_size: int, args, device, distributed: bool, world_size: int) -> dict:
    local_ok = 1
    error = ""
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        x = torch.randn(batch_size, args.n_constituents, args.n_features, device=device)
        y = torch.randint(0, args.n_classes, (batch_size,), device=device)
        for _ in range(args.warmup_steps):
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = F.cross_entropy(out["logits"], y)
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(args.measure_steps):
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = F.cross_entropy(out["logits"], y)
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        peak_gib = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        step_time = elapsed / args.measure_steps
        samples_per_sec_per_gpu = batch_size / step_time
    except torch.cuda.OutOfMemoryError as exc:
        local_ok = 0
        error = f"OOM: {exc}"
        torch.cuda.empty_cache()
        peak_gib = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        step_time = float("nan")
        samples_per_sec_per_gpu = 0.0

    global_ok = reduce_min(local_ok, device, distributed)
    peak_gib_global = reduce_max_float(peak_gib, device, distributed)
    step_time_global = reduce_max_float(step_time if step_time == step_time else 0.0, device, distributed)
    per_gpu_global = reduce_max_float(samples_per_sec_per_gpu, device, distributed)
    global_samples_per_sec = per_gpu_global * world_size if global_ok else 0.0
    epoch_seconds = 80_000_000 / global_samples_per_sec if global_samples_per_sec > 0 else float("inf")
    total_hours = epoch_seconds * 400 / 3600 if global_samples_per_sec > 0 else float("inf")
    return {
        "batch_size_per_gpu": batch_size,
        "global_batch_size": batch_size * world_size,
        "ok": bool(global_ok),
        "peak_vram_gib": peak_gib_global,
        "samples_per_sec_per_gpu": per_gpu_global,
        "global_samples_per_sec": global_samples_per_sec,
        "step_time_s": step_time_global,
        "estimated_epoch_hours_80m": epoch_seconds / 3600 if global_samples_per_sec > 0 else float("inf"),
        "estimated_total_hours_400_epochs": total_hours,
        "error": error if not global_ok else "",
    }


def main():
    args = parse_args()
    distributed, local_rank, world_size, device = setup()
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        model = MHAEmbeddingFullPrecision(
            n_constituents=args.n_constituents,
            n_features=args.n_features,
            latent_dim=args.latent_dim,
            n_classes=args.n_classes,
            d_model=args.d_model,
            ff_dim=args.ff_dim,
            num_heads=args.num_heads,
            num_attention_blocks=args.num_attention_blocks,
            pooling_mode="attention",
            classifier_head="linear",
        ).to(device)
        n_params = count_params(model)
        if distributed:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        rows = []
        for batch_size in [int(x) for x in args.batch_sizes.split(",") if x.strip()]:
            row = benchmark_batch(model, optimizer, batch_size, args, device, distributed, world_size)
            row["trainable_params"] = n_params
            rows.append(row)
            if local_rank == 0:
                print(json.dumps(row), flush=True)
            if not row["ok"]:
                break

        if local_rank == 0:
            out = Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            print(f"saved {out}", flush=True)
    finally:
        cleanup(distributed)


if __name__ == "__main__":
    main()
