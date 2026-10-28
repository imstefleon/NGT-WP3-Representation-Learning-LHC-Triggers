"""Run a saved INT8 bundle on a .npy event tensor."""
import argparse
from pathlib import Path

import numpy as np
import torch

from .core import infer, load_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-inputs", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists() or args.output.suffix != ".npz":
        parser.error("Choose a new output file ending in .npz")
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    model, bundle = load_bundle(args.bundle)
    result = infer(model, np.load(args.inputs, mmap_mode="r", allow_pickle=False),
                   bundle["config"], args.batch_size, args.raw_inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **result, class_names=np.array(bundle["config"]["class_names"]))
    print(args.output)


if __name__ == "__main__":
    main()
