"""Recover a bounded subset of the original DDP rank holdout from memory-mapped caches."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .core import read_config, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--events-per-class", type=int, default=250)
    parser.add_argument("--rank", type=int, default=0)
    args = parser.parse_args()
    config = read_config(args.run_config)
    if args.output.exists():
        parser.error("Output already exists")
    if not config.get("data_rank_sharded") or config.get("pid_encoding") != "raw":
        parser.error("This helper requires the original rank-sharded, raw-PID 100M run")
    world = config["world_size"]
    if not 0 <= args.rank < world or not 1 <= args.events_per_class <= config["balanced_eval_events_per_class"]:
        parser.error("Invalid rank or events-per-class")
    rng = np.random.default_rng(config["seed"])
    inputs, labels, fit, provenance = [], [], [], {}
    for label, name in enumerate(config["class_names"]):
        if config["label_map"][name] != label:
            parser.error("Unexpected class order/label_map mismatch")
        count = config["loaded_events_per_class"][name]
        pattern = (f"train_mha_{name}_{count}_{config['feature_mode']}_{config['candidate_selection']}_"
                   f"shard{args.rank:04d}of{world:04d}_{config['data_source']}_rawtensor.npy")
        path = args.cache_dir / pattern
        raw = np.load(path, mmap_mode="r", allow_pickle=False)
        if raw.shape != (count, config["n_constituents"], 5):
            parser.error(f"Cache shape mismatch: {path}: {raw.shape}")
        # Matches build_split_indices exactly without concatenating all four 10GB arrays.
        permutation = rng.permutation(count)
        n_eval = config["balanced_eval_events_per_class"]
        n_val = config["balanced_val_events_per_class"]
        if n_eval + n_val >= count:
            parser.error("Invalid original split sizes")
        selected = permutation[:args.events_per_class]
        inputs.append(np.asarray(raw[selected]).copy())
        labels.append(np.full(len(selected), label, dtype=np.int64))
        if name == "minbias":
            fit_indices = permutation[n_eval:n_eval + n_val]
            fit = np.asarray(raw[fit_indices]).copy()
        provenance[name] = {"cache_filename": path.name, "cache_bytes": path.stat().st_size,
                            "eval_local_indices": selected.tolist()}
        del permutation, raw
    args.output.mkdir(parents=True)
    np.save(args.output / "eval_raw.npy", np.concatenate(inputs))
    np.save(args.output / "eval_labels.npy", np.concatenate(labels))
    np.save(args.output / "minbias_fit_raw.npy", fit)
    write_json(args.output / "holdout_provenance.json", dict(
        source_config_sha256=sha256(args.run_config), seed=config["seed"], rank=args.rank,
        class_order=config["class_names"], source="original DDP rank evaluation split",
        minbias_reference_split="original validation split, disjoint from evaluation and training",
        events_per_class=args.events_per_class, caches=provenance,
        preprocessing="raw; use --raw-inputs to apply frozen run normalization"))
    print(f"Prepared {sum(map(len, inputs))} original holdout events", flush=True)


if __name__ == "__main__":
    main()
