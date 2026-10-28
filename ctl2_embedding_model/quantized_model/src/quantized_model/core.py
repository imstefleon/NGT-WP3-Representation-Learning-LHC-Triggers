"""Checkpoint loading, frozen preprocessing, and portable quantized state bundles."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.ao.quantization import per_channel_dynamic_qconfig, quantize_dynamic

from ngt_model.datasets.collide2v_data import preprocess_dataset
from ngt_model.models.mha_model import MHAEmbeddingFullPrecision

FORMAT = "ngt_dynamic_int8_v1"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_config(path):
    with open(path, encoding="utf-8-sig") as stream:
        config = json.load(stream)
    for key in ("n_constituents", "n_features", "latent_dim", "ff_dim",
                "num_heads", "num_attention_blocks", "pooling_mode", "classifier_head"):
        if key not in config:
            raise ValueError(f"Missing checkpoint architecture field: {key}")
    if not config.get("class_names") or not config.get("norm"):
        raise ValueError("run_config.json must contain class_names and frozen norm")
    return config


def build_model(config):
    return MHAEmbeddingFullPrecision(
        n_constituents=config["n_constituents"], n_features=config["n_features"],
        latent_dim=config["latent_dim"], n_classes=len(config["class_names"]),
        ff_dim=config["ff_dim"], d_model=config.get("d_model"),
        num_heads=config["num_heads"], num_attention_blocks=config["num_attention_blocks"],
        pooling_mode=config["pooling_mode"], classifier_head=config["classifier_head"],
    ).cpu().eval()


def load_fp32(checkpoint, config):
    # Epoch checkpoints contain tensors and primitive metadata. Never unpickle a full model.
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Supply checkpoint_epoch_NNN.pt or a tensor state_dict")
    state = payload.get("model_state_dict", payload.get("state_dict", payload))
    if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError("Expected a tensor-only model state_dict")
    state = {k.removeprefix("module."): v for k, v in state.items()}
    model = build_model(config)
    model.load_state_dict(state, strict=True)
    if not all(torch.isfinite(p).all().item() for p in model.parameters()):
        raise ValueError("Checkpoint contains nonfinite weights")
    if config.get("n_params") and sum(p.numel() for p in model.parameters()) != config["n_params"]:
        raise ValueError("Checkpoint parameter count differs from run_config.json")
    return model, payload.get("epoch")


def choose_engine(requested="auto"):
    supported = torch.backends.quantized.supported_engines
    choices = [requested] if requested != "auto" else ["x86", "fbgemm", "onednn", "qnnpack"]
    for engine in choices:
        if engine in supported and engine != "none":
            torch.backends.quantized.engine = engine
            return engine
    raise ValueError(f"No supported INT8 CPU engine: requested={requested}, available={supported}")


def quantize(model, excluded=("input_proj",)):
    linear_names = {name for name, layer in model.named_modules() if isinstance(layer, nn.Linear)}
    for prefix in excluded:
        if not any(name == prefix or name.startswith(prefix + ".") for name in linear_names):
            raise ValueError(f"Excluded layer does not exist: {prefix}")
    selected = sorted(name for name in linear_names if not any(
        name == prefix or name.startswith(prefix + ".") for prefix in excluded))
    if not selected:
        raise ValueError("No Linear layers selected for quantization")
    spec = {name: per_channel_dynamic_qconfig for name in selected}
    return quantize_dynamic(model, qconfig_spec=spec, dtype=torch.qint8, inplace=False).eval(), selected


def load_bundle(path):
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    if bundle.get("format") != FORMAT:
        raise ValueError("Unsupported quantized bundle format")
    choose_engine(bundle["engine"])
    model, selected = quantize(build_model(bundle["config"]), bundle["excluded_layers"])
    if selected != bundle["quantized_layers"]:
        raise ValueError("Quantized layer topology differs from bundle")
    model.load_state_dict(bundle["model_state_dict"], strict=True)
    return model, bundle


def prepare_inputs(array, config, raw=False):
    x = np.asarray(array, dtype=np.float32)
    if x.ndim != 3 or x.shape[1] != config["n_constituents"]:
        raise ValueError(f"Expected [events,{config['n_constituents']},features], got {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("Inputs contain nonfinite values")
    if raw:
        if x.shape[2] != 5:
            raise ValueError("Raw features must be [pt,eta,phi,dxy,pid]")
        mapping = {int(k): int(v) for k, v in config.get("pid_mapping", {}).items()}
        x, _ = preprocess_dataset(x, norm=config["norm"], pid_mapping=mapping,
                                  add_pid_ohe=config.get("pid_encoding", "onehot") == "onehot")
    if x.shape[2] != config["n_features"]:
        raise ValueError(f"Expected {config['n_features']} preprocessed features, got {x.shape[2]}")
    if not np.isfinite(x).all():
        raise ValueError("Frozen preprocessing produced nonfinite values")
    return np.ascontiguousarray(x)


@torch.inference_mode()
def infer(model, array, config, batch_size=16, raw=False):
    if batch_size < 1 or len(array) < 1:
        raise ValueError("batch_size and input event count must be positive")
    collected = {"latent": [], "logits": []}
    for start in range(0, len(array), batch_size):
        x = prepare_inputs(array[start:start + batch_size], config, raw=raw)
        output = model(torch.from_numpy(x))
        for key in collected:
            value = output[key].cpu().numpy()
            if not np.isfinite(value).all():
                raise ValueError(f"Nonfinite {key} at event {start}")
            collected[key].append(value)
    return {key: np.concatenate(values) for key, values in collected.items()}


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
