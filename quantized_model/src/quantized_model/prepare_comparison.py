"""Prepare a small reproducible numerical comparison cohort from existing 2k caches.

These caches can overlap training. This cohort is not an independent physics test.
"""
import argparse
from pathlib import Path

import numpy as np

from .core import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--events-per-class", type=int, default=128)
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.events_per_class <= 1000:
        parser.error("Choose a new output directory and 1..1000 events per class")
    names = ["minbias", "qcd", "dy", "top"]
    x, y, provenance = [], [], {}
    fit = None
    signals = {}
    for name in names + ["hh4b", "vbfh"]:
        path = args.cache_dir / f"external_higgs_anomaly_{name}_2000_l1_only_top_pt_collide2v_updated_rawtensor.npy"
        raw = np.load(path, allow_pickle=False)
        if raw.shape != (2000, 100, 5):
            parser.error(f"Unexpected cache shape: {path}: {raw.shape}")
        provenance[name] = {"cache_filename": path.name, "sha256": sha256(path)}
        if name in names:
            x.append(raw[:args.events_per_class])
            y.append(np.full(args.events_per_class, names.index(name), dtype=np.int64))
            if name == "minbias":
                fit = raw[1000:1256].copy()
        else:
            signals[name] = raw[:args.events_per_class].copy()
        print(f"Read {name}", flush=True)
    args.output.mkdir(parents=True)
    np.save(args.output / "eval_raw.npy", np.concatenate(x))
    np.save(args.output / "eval_labels.npy", np.concatenate(y))
    np.save(args.output / "minbias_fit_raw.npy", fit)
    for name, array in signals.items():
        np.save(args.output / f"{name}_raw.npy", array)
    write_json(args.output / "cohort_provenance.json", dict(
        kind="numerical_comparison_only", training_overlap="not excluded",
        physics_acceptance=False, class_order=names, eval_cache_rows=[0, args.events_per_class],
        minbias_reference_cache_rows=[1000, 1256], sources=provenance))


if __name__ == "__main__":
    main()
