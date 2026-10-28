"""Convert a real trained checkpoint; no random initialization fallback."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import torch

from .core import FORMAT, choose_engine, load_bundle, load_fp32, quantize, read_config, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engine", default="auto")
    parser.add_argument("--policy", choices=["post_pool_int8", "dense_int8"], default="post_pool_int8",
                        help="Default quantizes only post-pool dense layers; dense_int8 is an experimental policy")
    parser.add_argument("--exclude", nargs="*", default=None,
                        help="Override the policy with explicit Linear layer prefixes kept FP32")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new artifact path")
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    config = read_config(args.run_config)
    engine = choose_engine(args.engine)
    print("Loading trained checkpoint", flush=True)
    model, epoch = load_fp32(args.checkpoint, config)
    count = sum(p.numel() for p in model.parameters())
    if args.exclude is not None:
        excluded = args.exclude
        policy = "custom"
    elif args.policy == "post_pool_int8":
        excluded = [name for name, layer in model.named_modules()
                    if isinstance(layer, torch.nn.Linear) and not name.startswith("post_pool.")]
        policy = args.policy
    else:
        excluded = ["input_proj"]
        policy = args.policy
    model, layers = quantize(model, excluded)
    bundle = dict(format=FORMAT, config=config, engine=engine,
                  excluded_layers=excluded, quantized_layers=layers, quantization_policy=policy,
                  model_state_dict=model.state_dict(), source_epoch=epoch,
                  source_checkpoint_sha256=sha256(args.checkpoint),
                  source_config_sha256=sha256(args.run_config), torch_version=str(torch.__version__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, args.output)
    # Check the delivered state can actually be restored and run, including all-padding events.
    restored, _ = load_bundle(args.output)
    with torch.inference_mode():
        result = restored(torch.zeros(2, config["n_constituents"], config["n_features"]))
        if not all(torch.isfinite(value).all().item() for value in result.values()):
            raise ValueError("Reloaded artifact failed finite-output smoke check")
    manifest = {key: value for key, value in bundle.items() if key != "model_state_dict"}
    manifest.update(created_utc=datetime.now(timezone.utc).isoformat(), n_parameters_fp32=count,
                    quantized_bundle=args.output.name, artifact_sha256=sha256(args.output),
                    artifact_bytes=args.output.stat().st_size,
                    source_checkpoint_bytes=args.checkpoint.stat().st_size,
                    checkpoint_to_bundle_size_ratio=args.checkpoint.stat().st_size / args.output.stat().st_size,
                    precision="per-channel INT8 Linear weights; dynamic activations within Linear; remaining graph FP32",
                    reload_smoke_passed=True, physics_validation="requires held-out comparison",
                    fpga_synthesis_validated=False)
    write_json(args.output.with_suffix(".manifest.json"), manifest)
    print(f"Converted {count:,} parameters, {len(layers)} Linear layers: {args.output}", flush=True)


if __name__ == "__main__":
    main()
