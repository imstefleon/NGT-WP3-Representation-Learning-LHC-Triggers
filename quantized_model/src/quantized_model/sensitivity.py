"""Compare candidate layer policies on a small calibration cohort, never a test set."""
import argparse
from pathlib import Path

import numpy as np
import torch

from .core import choose_engine, infer, load_fp32, quantize, read_config, sha256, write_json
from .compare import differences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    choose_engine()
    config = read_config(args.run_config)
    model, _ = load_fp32(args.checkpoint, config)
    all_x = np.load(args.inputs, allow_pickle=False)
    # Evenly spaced calibration sample spanning the supplied class-ordered cohort.
    x = all_x[np.linspace(0, len(all_x) - 1, min(128, len(all_x)), dtype=int)]
    reference = infer(model, x, config, 8, raw=True)
    names = [name for name, layer in model.named_modules() if isinstance(layer, torch.nn.Linear)]
    policies = {
        "post_pool_only": lambda name: name.startswith("post_pool."),
        "post_pool_value_output": lambda name: name.startswith("post_pool.") or name.endswith((".v_proj", ".out_proj")),
        "post_pool_second_only": lambda name: name == "post_pool.2",
    }
    results = {}
    for policy, selected in policies.items():
        excluded = [name for name in names if not selected(name)]
        candidate, layers = quantize(model, excluded)
        output = infer(candidate, x, config, 8, raw=True)
        results[policy] = dict(excluded_layers=excluded, quantized_layers=layers,
                               agreement=float(np.mean(reference["logits"].argmax(1) == output["logits"].argmax(1))),
                               logits=differences(reference["logits"], output["logits"]),
                               latent=differences(reference["latent"], output["latent"]))
        print(policy, results[policy]["agreement"], flush=True)
        del candidate
    write_json(args.output, dict(cohort="Calibration subset; do not interpret as independent test",
                                 input_sha256=sha256(args.inputs), n_events=len(x), policies=results))


if __name__ == "__main__":
    main()
