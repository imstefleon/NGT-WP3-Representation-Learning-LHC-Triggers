"""Evaluate external Higgs-like samples with anomaly scores from real model outputs.

The script loads a trained Collide-2V NGT run, reuses the repository's shared
L1 loader and preprocessing, extracts event-level latents/logits from the saved
checkpoint, fits background-only PCA whitening and GMM density models, and saves
auditable tables plus figures under each run's figures/external_higgs_anomaly
directory.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture

from ngt_model.datasets.collide2v_data import (
    SUPPORTED_CLASS_NAMES,
    Collide2VConfig,
    Collide2VLoader,
    preprocess_dataset,
)


DEFAULT_SIGNALS = ["hh4b", "hh", "vbfh", "ggh"]
DEFAULT_CHECKPOINT_NAMES = [
    Path("models") / "mha_nodes_aligned_best.pt",
    Path("models") / "mha_nodes_aligned_best_for_eval.pt",
]
PLOT_COLORS = {
    "sm_background": "#7f7f7f",
    "hh4b": "#1f77b4",
    "hh": "#ff7f0e",
    "vbfh": "#2ca02c",
    "ggh": "#9467bd",
}
DEFAULT_OUTPUT_SUBDIR = Path("figures") / "external_higgs_anomaly"
GMM_KWARGS = dict(
    n_components=2,
    covariance_type="full",
    random_state=42,
    max_iter=500,
    reg_covar=1e-5,
    init_params="kmeans",
)


class SkipRun(RuntimeError):
    """Raised when a run is not evaluable because required artifacts are absent."""


@dataclass
class RunDiscovery:
    run_dir: Path
    checkpoint_path: Path
    run_config: dict
    trained_classes: list[str]
    background_classes: list[str]
    output_dir: Path


@dataclass
class RunResult:
    run_dir: str
    status: str
    reason: str = ""
    output_dir: str = ""


def parse_csv_list(raw: str | Iterable[str]) -> list[str]:
    if isinstance(raw, str):
        values = [part.strip().lower() for part in raw.split(",") if part.strip()]
    else:
        values = [str(part).strip().lower() for part in raw if str(part).strip()]
    if not values:
        raise ValueError("CSV list must contain at least one value")
    return values


def unique_ordered(values: Iterable[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict | list) -> None:
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return None if not math.isfinite(value) else value
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def resolve_checkpoint(run_dir: Path, model_arg: str | None = None) -> Path:
    if model_arg:
        model_path = Path(model_arg)
        candidates = unique_ordered([model_path, run_dir / model_path])
    else:
        candidates = [run_dir / name for name in DEFAULT_CHECKPOINT_NAMES]
    for path in candidates:
        if path.exists():
            return path
    raise SkipRun(
        "checkpoint missing; expected one of "
        + ", ".join(str(path) for path in candidates)
    )


def load_run_config(run_dir: Path) -> dict:
    config_path = run_dir / "run_config.json"
    if not config_path.exists():
        raise SkipRun(f"metadata missing: {config_path}")
    config = read_json(config_path)
    if not isinstance(config, dict):
        raise SkipRun(f"metadata is not a JSON object: {config_path}")
    return config


def trained_classes_from_config(config: dict) -> list[str]:
    for key in ("class_names", "classes"):
        raw = config.get(key)
        if raw:
            classes = parse_csv_list(raw)
            unknown = [name for name in classes if name not in SUPPORTED_CLASS_NAMES]
            if unknown:
                raise SkipRun(f"run_config has unsupported trained classes: {unknown}")
            return classes
    raise SkipRun("run_config.json does not define class_names/classes")


def background_classes_from_config(
    config: dict,
    signals: list[str],
    override: str | None,
    exclude: str | None = None,
) -> list[str]:
    excluded = set(parse_csv_list(exclude)) if exclude else set()
    if override:
        classes = parse_csv_list(override)
    else:
        trained = trained_classes_from_config(config)
        signal_set = set(signals)
        classes = [name for name in trained if name not in signal_set]
    classes = [name for name in classes if name not in excluded]
    unknown = [name for name in classes if name not in SUPPORTED_CLASS_NAMES]
    if unknown:
        raise SkipRun(f"unsupported background classes: {unknown}")
    if not classes:
        raise SkipRun("no background classes remain after excluding external signals")
    return classes


def normalize_pid_mapping(raw: dict | None) -> dict[int, int]:
    if not raw:
        raise SkipRun("run_config.json is missing pid_mapping")
    return {int(k): int(v) for k, v in raw.items()}


def validate_run_preprocessing_config(config: dict) -> tuple[dict, dict[int, int]]:
    if config.get("feature_mode", "l1_only") != "l1_only":
        raise SkipRun(f"run feature_mode is not l1_only: {config.get('feature_mode')}")
    norm = config.get("norm")
    if not isinstance(norm, dict):
        raise SkipRun("run_config.json is missing normalization constants under 'norm'")
    required_norm = {"pt_min", "pt_max", "eta_min", "eta_max", "phi_min", "phi_max", "dxy_min", "dxy_max"}
    missing_norm = sorted(required_norm - set(norm))
    if missing_norm:
        raise SkipRun(f"normalization metadata is missing keys: {missing_norm}")
    pid_mapping = normalize_pid_mapping(config.get("pid_mapping"))
    return {key: float(norm[key]) for key in required_norm}, pid_mapping


def expected_input_features(model) -> int | None:
    if hasattr(model, "input_proj") and hasattr(model.input_proj, "in_features"):
        return int(model.input_proj.in_features)
    if hasattr(model, "encoder"):
        return expected_input_features(model.encoder)
    return None


def _strip_module_prefix(state_dict: dict) -> dict:
    if not any(str(key).startswith("module.") for key in state_dict):
        return state_dict
    return {str(key).removeprefix("module."): value for key, value in state_dict.items()}


def build_model_from_config(config: dict, device):
    from ngt_model.models.mha_model import MHAEmbeddingFullPrecision

    model = MHAEmbeddingFullPrecision(
        n_constituents=int(config.get("n_constituents", 100)),
        n_features=int(config.get("n_features", 5)),
        latent_dim=int(config.get("latent_dim", 128)),
        n_classes=len(trained_classes_from_config(config)),
        ff_dim=int(config.get("ff_dim", 32)),
        d_model=config.get("d_model"),
        num_heads=int(config.get("num_heads", 2)),
        pooling_mode=config.get("pooling_mode", "mean"),
        classifier_head=config.get("classifier_head", "mlp"),
        num_attention_blocks=int(config.get("num_attention_blocks", 2)),
    )
    model.to(device)
    return model


def load_saved_model(model_path: str, device, run_config: dict | None = None):
    import torch

    try:
        obj = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        obj = torch.load(model_path, map_location=device)
    if isinstance(obj, dict) and "model_state_dict" in obj:
        if run_config is None:
            raise SkipRun(f"checkpoint dict requires run_config.json to reconstruct model: {model_path}")
        model = build_model_from_config(run_config, device)
        state_dict = _strip_module_prefix(obj["model_state_dict"])
        model.load_state_dict(state_dict)
        model.eval()
        return model
    if not isinstance(obj, torch.nn.Module):
        raise SkipRun(f"checkpoint is not a saved torch.nn.Module: {model_path}")
    obj.to(device)
    obj.eval()
    return obj


def discover_run(run_dir: Path, args) -> RunDiscovery:
    run_dir = run_dir.resolve()
    if not run_dir.exists():
        raise SkipRun(f"run directory does not exist: {run_dir}")
    if not run_dir.is_dir():
        raise SkipRun(f"run path is not a directory: {run_dir}")

    run_config = load_run_config(run_dir)
    trained_classes = trained_classes_from_config(run_config)
    background_classes = background_classes_from_config(
        run_config,
        args.signals,
        args.background_classes,
        args.exclude_background_classes,
    )
    checkpoint_path = resolve_checkpoint(run_dir, args.model)
    output_subdir = Path(args.output_subdir)
    output_dir = run_dir / output_subdir
    return RunDiscovery(
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        run_config=run_config,
        trained_classes=trained_classes,
        background_classes=background_classes,
        output_dir=output_dir,
    )


def read_run_list(path: Path) -> list[Path]:
    run_dirs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        run_dirs.append(Path(stripped))
    return run_dirs


def _float_close(value: float | None, target: float, tol: float = 1e-12) -> bool:
    return value is not None and abs(float(value) - target) <= tol


def _required_profile_name(config: dict) -> str | None:
    latent_dim = int(config.get("latent_dim", -1))
    class_names = parse_csv_list(config.get("class_names", [])) if config.get("class_names") else []
    if set(class_names) == {"minbias", "qcd", "dy", "top"}:
        return "fourclass_comparison"
    if _float_close(config.get("supcon_weight"), 0.05) and _float_close(config.get("latent_var_reg_weight"), 0.01):
        if latent_dim in {32, 64, 128}:
            return f"balanced_supcon005_var001_latent{latent_dim}"
    if _float_close(config.get("latent_var_reg_weight"), 0.01) and _float_close(config.get("latent_cov_reg_weight"), 0.001):
        if latent_dim in {64, 128}:
            return f"ce_var001_cov0001_latent{latent_dim}"
    if float(config.get("vicreg_weight", 0.0) or 0.0) > 0.0 and latent_dim in {64, 128}:
        return f"vicreg_latent{latent_dim}"
    return None


def discover_required_run_dirs(root: Path) -> tuple[list[Path], dict[str, list[str]]]:
    found: dict[str, list[Path]] = {}
    failures: list[str] = []
    for config_path in sorted(root.rglob("run_config.json")):
        try:
            config = read_json(config_path)
            profile = _required_profile_name(config)
        except Exception as exc:
            failures.append(f"{config_path}: {exc}")
            continue
        if profile:
            found.setdefault(profile, []).append(config_path.parent)

    required = [
        "balanced_supcon005_var001_latent32",
        "balanced_supcon005_var001_latent64",
        "balanced_supcon005_var001_latent128",
        "ce_var001_cov0001_latent64",
        "ce_var001_cov0001_latent128",
        "vicreg_latent64",
        "vicreg_latent128",
    ]
    selected = []
    missing = []
    for profile in required:
        matches = found.get(profile, [])
        if matches:
            selected.append(sorted(matches)[-1])
        else:
            missing.append(profile)
    if found.get("fourclass_comparison"):
        selected.append(sorted(found["fourclass_comparison"])[-1])
    return unique_ordered(selected), {"missing_required_profiles": missing, "scan_failures": failures}


def prepare_raw_events(discovery: RunDiscovery, args) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    config = discovery.run_config
    context_classes = unique_ordered([*discovery.background_classes, *args.signals])
    unknown = [name for name in context_classes if name not in SUPPORTED_CLASS_NAMES]
    if unknown:
        raise SkipRun(f"unsupported requested classes: {unknown}")

    cfg_kwargs = {
        "data_source": args.data_source or config.get("data_source", "collide2v_updated"),
        "n_constituents": int(args.n_constituents or config.get("n_constituents", 100)),
        "feature_mode": "l1_only",
        "candidate_selection": args.candidate_selection or config.get("candidate_selection", "top_pt"),
        "class_names": context_classes,
    }
    collide2v_path = args.collide2v_path or config.get("collide2v_path")
    if collide2v_path:
        cfg_kwargs["collide2v_path"] = Path(collide2v_path)
    if args.cache_dir:
        cfg_kwargs["cache_dir"] = Path(args.cache_dir)

    cfg = Collide2VConfig(**cfg_kwargs)
    loader = Collide2VLoader(cfg)
    events_per_class = {}
    for cls in context_classes:
        events_per_class[cls] = (
            args.n_background_per_class if cls in discovery.background_classes else args.n_signal_per_class
        )
    raw_tensor, _y, sample_names, _raw_by_class = loader.load_classes(
        events_per_class,
        cache_prefix=args.cache_prefix,
    )
    event_counts = {cls: int(np.sum(sample_names == cls)) for cls in context_classes}
    return raw_tensor, sample_names.astype(str), np.asarray(context_classes, dtype=object), event_counts


def preprocess_for_run(raw_tensor: np.ndarray, discovery: RunDiscovery, model) -> np.ndarray:
    norm, pid_mapping = validate_run_preprocessing_config(discovery.run_config)
    add_pid_ohe = discovery.run_config.get("pid_encoding", "onehot") == "onehot"
    X, _ = preprocess_dataset(raw_tensor, norm=norm, pid_mapping=pid_mapping, add_pid_ohe=add_pid_ohe)
    expected_features = expected_input_features(model)
    configured_features = discovery.run_config.get("n_features")
    if configured_features is not None and int(configured_features) != X.shape[-1]:
        raise SkipRun(
            f"preprocessed feature count {X.shape[-1]} does not match run_config n_features={configured_features}"
        )
    if expected_features is not None and int(expected_features) != X.shape[-1]:
        raise SkipRun(
            f"preprocessed feature count {X.shape[-1]} does not match checkpoint input features={expected_features}"
        )
    return X.astype(np.float32, copy=False)


def extract_latents_logits(
    model,
    X: np.ndarray,
    device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    latents = []
    logits = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[start : start + batch_size]).to(device)
            out = model(batch)
            if not isinstance(out, dict) or "latent" not in out or "logits" not in out:
                raise SkipRun(
                    "checkpoint does not return {'latent', 'logits'}; use the full classifier checkpoint, "
                    "usually models/mha_nodes_aligned_best.pt"
                )
            latents.append(out["latent"].detach().cpu().double().numpy())
            logits.append(out["logits"].detach().cpu().double().numpy())
    return np.concatenate(latents, axis=0), np.concatenate(logits, axis=0)


def softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def split_background_indices(sample_names: np.ndarray, background_classes: list[str], seed: int, fit_fraction: float):
    if not (0.0 < fit_fraction < 1.0):
        raise ValueError("--background-fit-fraction must be in (0, 1)")
    rng = np.random.default_rng(seed)
    fit_parts = []
    eval_parts = []
    for cls in background_classes:
        indices = np.flatnonzero(sample_names == cls)
        if len(indices) < 4:
            raise SkipRun(f"{cls}: need at least 4 background events for fit/eval split, found {len(indices)}")
        perm = rng.permutation(indices)
        n_fit = int(round(fit_fraction * len(perm)))
        n_fit = min(max(n_fit, 2), len(perm) - 2)
        fit_parts.append(perm[:n_fit])
        eval_parts.append(perm[n_fit:])
    return rng.permutation(np.concatenate(fit_parts)), rng.permutation(np.concatenate(eval_parts))


def fit_density_models(
    Z: np.ndarray,
    sample_names: np.ndarray,
    background_classes: list[str],
    seed: int,
    fit_fraction: float,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    bg_fit_idx, bg_eval_idx = split_background_indices(sample_names, background_classes, seed, fit_fraction)
    pca = PCA(whiten=True, random_state=seed)
    pca.fit(Z[bg_fit_idx])
    Z_white = pca.transform(Z)

    gmm_kwargs = dict(GMM_KWARGS)
    gmm_kwargs["random_state"] = seed
    gmm = GaussianMixture(**gmm_kwargs)
    gmm.fit(Z_white[bg_fit_idx])

    lw = LedoitWolf().fit(Z_white[bg_fit_idx])
    minbias_fit_idx = bg_fit_idx[sample_names[bg_fit_idx] == "minbias"]
    if len(minbias_fit_idx) < 4:
        raise SkipRun(f"minbias reference needs at least 4 fit events, found {len(minbias_fit_idx)}")
    minbias_lw = LedoitWolf().fit(Z_white[minbias_fit_idx])
    model_info = {
        "pca": pca,
        "gmm": gmm,
        "ledoit_wolf": lw,
        "minbias_ledoit_wolf": minbias_lw,
        "bg_fit_idx": bg_fit_idx,
        "bg_eval_idx": bg_eval_idx,
        "minbias_fit_idx": minbias_fit_idx,
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.astype(float).tolist(),
    }
    return model_info, Z_white, bg_fit_idx, bg_eval_idx


def mahalanobis_scores(Z_white: np.ndarray, lw: LedoitWolf) -> np.ndarray:
    delta = Z_white - lw.location_
    precision = lw.precision_
    dist2 = np.einsum("ij,jk,ik->i", delta, precision, delta)
    return np.sqrt(np.maximum(dist2, 0.0))


def robust_zscore(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    reference = np.asarray(reference, dtype=float)
    center = float(np.median(reference))
    q25, q75 = np.percentile(reference, [25, 75])
    scale = float(q75 - q25)
    if not np.isfinite(scale) or scale < 1e-9:
        scale = float(np.std(reference))
    if not np.isfinite(scale) or scale < 1e-9:
        scale = 1.0
    return (np.asarray(values, dtype=float) - center) / scale


def score_method_metrics(
    event_scores: pd.DataFrame,
    signals: list[str],
    method_name: str,
    score_column: str,
) -> pd.DataFrame:
    bg_scores = event_scores.loc[event_scores["score_split"] == "background_eval", score_column].to_numpy(dtype=float)
    if len(bg_scores) < 2:
        raise SkipRun(f"held-out background split has fewer than 2 events for {method_name}")
    bg_p95 = float(np.percentile(bg_scores, 95))
    bg_p99 = float(np.percentile(bg_scores, 99))
    bg_p995 = float(np.percentile(bg_scores, 99.5))
    rows = []
    for signal in signals:
        signal_scores = event_scores.loc[event_scores["sample_class"] == signal, score_column].to_numpy(dtype=float)
        if len(signal_scores) == 0:
            raise SkipRun(f"no evaluated events loaded for signal {signal}")
        y_true = np.concatenate([np.zeros(len(bg_scores), dtype=int), np.ones(len(signal_scores), dtype=int)])
        y_score = np.concatenate([bg_scores, signal_scores])
        rows.append(
            {
                "signal": signal,
                "anomaly_method": method_name,
                "score_column": score_column,
                "n_events": int(len(signal_scores)),
                "median_anomaly_score": float(np.median(signal_scores)),
                "mean_anomaly_score": float(np.mean(signal_scores)),
                "p90_anomaly_score": float(np.percentile(signal_scores, 90)),
                "p95_anomaly_score": float(np.percentile(signal_scores, 95)),
                "p99_anomaly_score": float(np.percentile(signal_scores, 99)),
                "auc_vs_background": float(roc_auc_score(y_true, y_score)),
                "eff_at_95_bg_rejection": float(np.mean(signal_scores > bg_p95)),
                "eff_at_99_bg_rejection": float(np.mean(signal_scores > bg_p99)),
                "eff_at_99_5_bg_rejection": float(np.mean(signal_scores > bg_p995)),
                "background_p95_threshold": bg_p95,
                "background_p99_threshold": bg_p99,
                "background_p99_5_threshold": bg_p995,
            }
        )
    return pd.DataFrame(rows)


def build_event_scores(
    discovery: RunDiscovery,
    args,
    sample_names: np.ndarray,
    Z: np.ndarray,
    logits: np.ndarray,
    density_info: dict,
    Z_white: np.ndarray,
    bg_fit_idx: np.ndarray,
    bg_eval_idx: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    probs = softmax_np(logits)
    if probs.shape[1] != len(discovery.trained_classes):
        raise SkipRun(
            f"logit dimension {probs.shape[1]} does not match trained class count "
            f"{len(discovery.trained_classes)}"
        )

    anomaly_score = -density_info["gmm"].score_samples(Z_white)
    mahal = mahalanobis_scores(Z_white, density_info["ledoit_wolf"])
    minbias_distance = mahalanobis_scores(Z_white, density_info["minbias_ledoit_wolf"])
    pred_idx = np.argmax(probs, axis=1)
    pred_class = np.array([discovery.trained_classes[i] for i in pred_idx], dtype=object)

    roles = np.full(len(sample_names), "signal", dtype=object)
    roles[np.isin(sample_names, discovery.background_classes)] = "background"
    split = np.full(len(sample_names), "signal_eval", dtype=object)
    split[bg_fit_idx] = "background_fit"
    split[bg_eval_idx] = "background_eval"

    event_scores = pd.DataFrame(
        {
            "event_index": np.arange(len(sample_names), dtype=int),
            "sample_class": sample_names,
            "sample_role": roles,
            "score_split": split,
            "is_hh4b_trained_reference": (sample_names == "hh4b") & ("hh4b" in discovery.trained_classes),
            "anomaly_score": anomaly_score,
            "gmm_anomaly_score": anomaly_score,
            "mahalanobis_score": mahal,
            "background_mahalanobis_score": mahal,
            "minbias_distance_score": minbias_distance,
            "predicted_class": pred_class,
            "predicted_class_index": pred_idx.astype(int),
        }
    )
    for idx, cls in enumerate(discovery.trained_classes):
        event_scores[f"prob_{cls}"] = probs[:, idx]

    bg_eval_scores = event_scores.loc[event_scores["score_split"] == "background_eval", "anomaly_score"].to_numpy()
    if len(bg_eval_scores) < 2:
        raise SkipRun("held-out background split has fewer than 2 events")
    bg_p95 = float(np.percentile(bg_eval_scores, 95))
    bg_p99 = float(np.percentile(bg_eval_scores, 99))
    bg_p995 = float(np.percentile(bg_eval_scores, 99.5))
    bg_eval_mask = event_scores["score_split"] == "background_eval"
    event_scores["gmm_robust_z"] = robust_zscore(
        event_scores["gmm_anomaly_score"].to_numpy(dtype=float),
        event_scores.loc[bg_eval_mask, "gmm_anomaly_score"].to_numpy(dtype=float),
    )
    event_scores["minbias_distance_robust_z"] = robust_zscore(
        event_scores["minbias_distance_score"].to_numpy(dtype=float),
        event_scores.loc[bg_eval_mask, "minbias_distance_score"].to_numpy(dtype=float),
    )
    event_scores["overall_anomaly_score"] = 0.5 * (
        event_scores["gmm_robust_z"] + event_scores["minbias_distance_robust_z"]
    )

    method_metrics = pd.concat(
        [
            score_method_metrics(event_scores, args.signals, "gmm_nll", "gmm_anomaly_score"),
            score_method_metrics(event_scores, args.signals, "minbias_mahalanobis", "minbias_distance_score"),
            score_method_metrics(event_scores, args.signals, "overall_robust_mean", "overall_anomaly_score"),
        ],
        ignore_index=True,
    )

    response_rows = []
    metrics_rows = []
    for signal in args.signals:
        signal_df = event_scores[event_scores["sample_class"] == signal].copy()
        if signal_df.empty:
            raise SkipRun(f"no evaluated events loaded for signal {signal}")
        signal_scores = signal_df["anomaly_score"].to_numpy(dtype=float)
        y_true = np.concatenate([np.zeros(len(bg_eval_scores), dtype=int), np.ones(len(signal_scores), dtype=int)])
        y_score = np.concatenate([bg_eval_scores, signal_scores])
        auc = float(roc_auc_score(y_true, y_score))

        pred_fracs = (
            signal_df["predicted_class"]
            .value_counts(normalize=True)
            .reindex(discovery.trained_classes, fill_value=0.0)
            .astype(float)
        )
        top_predicted_class = str(pred_fracs.idxmax())
        top_predicted_fraction = float(pred_fracs.max())

        response_row = {"signal": signal, "n_events": int(len(signal_df))}
        response_row.update({cls: float(pred_fracs[cls]) for cls in discovery.trained_classes})
        response_rows.append(response_row)

        metrics_row = {
            "run_name": discovery.run_dir.name,
            "signal": signal,
            "signal_status": "trained_reference" if signal in discovery.trained_classes else "unseen_external",
            "n_events": int(len(signal_df)),
            "median_anomaly_score": float(np.median(signal_scores)),
            "mean_anomaly_score": float(np.mean(signal_scores)),
            "p90_anomaly_score": float(np.percentile(signal_scores, 90)),
            "p95_anomaly_score": float(np.percentile(signal_scores, 95)),
            "median_mahalanobis_score": float(np.median(signal_df["mahalanobis_score"])),
            "mean_mahalanobis_score": float(np.mean(signal_df["mahalanobis_score"])),
            "p90_mahalanobis_score": float(np.percentile(signal_df["mahalanobis_score"], 90)),
            "p95_mahalanobis_score": float(np.percentile(signal_df["mahalanobis_score"], 95)),
            "median_minbias_distance_score": float(np.median(signal_df["minbias_distance_score"])),
            "p90_minbias_distance_score": float(np.percentile(signal_df["minbias_distance_score"], 90)),
            "p95_minbias_distance_score": float(np.percentile(signal_df["minbias_distance_score"], 95)),
            "median_overall_anomaly_score": float(np.median(signal_df["overall_anomaly_score"])),
            "p90_overall_anomaly_score": float(np.percentile(signal_df["overall_anomaly_score"], 90)),
            "p95_overall_anomaly_score": float(np.percentile(signal_df["overall_anomaly_score"], 95)),
            "roc_auc_vs_sm_background": auc,
            "eff_at_95pct_bg": float(np.mean(signal_scores > bg_p95)),
            "eff_at_99pct_bg": float(np.mean(signal_scores > bg_p99)),
            "eff_at_99_5pct_bg": float(np.mean(signal_scores > bg_p995)),
            "background_p95_threshold": bg_p95,
            "background_p99_threshold": bg_p99,
            "background_p99_5_threshold": bg_p995,
            "classifier_top_predicted_class": top_predicted_class,
            "classifier_top_predicted_fraction": top_predicted_fraction,
        }
        for cls in discovery.trained_classes:
            metrics_row[f"fraction_predicted_{cls}"] = float(pred_fracs[cls])
        metrics_rows.append(metrics_row)

    response = pd.DataFrame(response_rows)
    metrics = pd.DataFrame(metrics_rows)
    metadata = {
        "run_dir": str(discovery.run_dir),
        "checkpoint_path": str(discovery.checkpoint_path),
        "output_dir": str(discovery.output_dir),
        "seed": int(args.seed),
        "signals": args.signals,
        "trained_classes": discovery.trained_classes,
        "background_classes": discovery.background_classes,
        "background_fit_fraction": float(args.background_fit_fraction),
        "event_counts": {
            key: int(value)
            for key, value in event_scores.groupby("sample_class").size().to_dict().items()
        },
        "background_fit_events": int(len(bg_fit_idx)),
        "background_eval_events": int(len(bg_eval_idx)),
        "minbias_fit_events": int(len(density_info["minbias_fit_idx"])),
        "gmm_kwargs": {key: _json_default(value) for key, value in dict(GMM_KWARGS, random_state=args.seed).items()},
        "pca_whiten": True,
        "pca_fit_scope": "background_fit_only",
        "density_fit_scope": "background_fit_only",
        "minbias_distance_fit_scope": "minbias_background_fit_only",
        "threshold_scope": "background_eval_only",
        "overall_score_formula": (
            "0.5 * robust_z(gmm_anomaly_score; background_eval) + "
            "0.5 * robust_z(minbias_distance_score; background_eval)"
        ),
        "pca_explained_variance_ratio": density_info["pca_explained_variance_ratio"],
        "hh4b_label_note": (
            "hh4b is a trained reference for this run"
            if "hh4b" in discovery.trained_classes
            else "hh4b is not a trained class for this run"
        ),
    }
    return event_scores, metrics, method_metrics, response, metadata


def save_outputs(
    output_dir: Path,
    event_scores: pd.DataFrame,
    metrics: pd.DataFrame,
    method_metrics: pd.DataFrame,
    response: pd.DataFrame,
    metadata: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    event_scores.to_csv(output_dir / "external_higgs_event_scores.csv", index=False)
    metrics.to_csv(output_dir / "external_higgs_anomaly_metrics.csv", index=False)
    write_json(output_dir / "external_higgs_anomaly_metrics.json", metrics.to_dict(orient="records"))
    method_metrics.to_csv(output_dir / "external_higgs_anomaly_metrics_by_method.csv", index=False)
    write_json(
        output_dir / "external_higgs_anomaly_metrics_by_method.json",
        method_metrics.to_dict(orient="records"),
    )
    response.to_csv(output_dir / "external_higgs_classifier_response.csv", index=False)
    write_json(output_dir / "external_higgs_run_metadata.json", metadata)


def load_saved_results(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    event_scores = pd.read_csv(output_dir / "external_higgs_event_scores.csv")
    metrics = pd.read_csv(output_dir / "external_higgs_anomaly_metrics.csv")
    if "eff_at_99_5pct_bg" not in metrics.columns and "eff_at_99pct_bg" in metrics.columns:
        metrics["eff_at_99_5pct_bg"] = metrics["eff_at_99pct_bg"]
    if "background_p99_5_threshold" not in metrics.columns and "background_p99_threshold" in metrics.columns:
        metrics["background_p99_5_threshold"] = metrics["background_p99_threshold"]
    response = pd.read_csv(output_dir / "external_higgs_classifier_response.csv")
    metadata = read_json(output_dir / "external_higgs_run_metadata.json")
    return event_scores, metrics, response, metadata


def _finite_scores(*arrays: np.ndarray) -> np.ndarray:
    merged = np.concatenate([np.asarray(arr, dtype=float) for arr in arrays if len(arr)])
    merged = merged[np.isfinite(merged)]
    if len(merged) == 0:
        raise RuntimeError("no finite anomaly scores available for plotting")
    return merged


def _signal_label(signal: str, trained_classes: list[str]) -> str:
    return f"{signal} (trained ref)" if signal == "hh4b" and signal in trained_classes else signal


def plot_anomaly_distributions(event_scores: pd.DataFrame, metrics: pd.DataFrame, metadata: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    bg = event_scores[event_scores["score_split"] == "background_eval"]["anomaly_score"].to_numpy(dtype=float)
    signal_arrays = [
        event_scores[event_scores["sample_class"] == signal]["anomaly_score"].to_numpy(dtype=float)
        for signal in metadata["signals"]
    ]
    all_scores = _finite_scores(bg, *signal_arrays)
    bins = np.linspace(float(np.percentile(all_scores, 0.5)), float(np.percentile(all_scores, 99.5)), 55)
    if len(np.unique(bins)) < 2:
        bins = 55

    ax.hist(bg, bins=bins, histtype="step", linewidth=1.8, color=PLOT_COLORS["sm_background"], label="SM background")
    for signal, values in zip(metadata["signals"], signal_arrays):
        label = _signal_label(signal, metadata["trained_classes"])
        ax.hist(values, bins=bins, histtype="step", linewidth=1.8, color=PLOT_COLORS.get(signal), label=label)

    bg_p95 = float(metrics["background_p95_threshold"].iloc[0])
    bg_p99 = float(metrics["background_p99_threshold"].iloc[0])
    ax.axvline(bg_p95, color="black", linestyle="--", linewidth=1.2, label="BG p95")
    ax.axvline(bg_p99, color="black", linestyle=":", linewidth=1.4, label="BG p99")
    ax.set_yscale("log")
    ax.set_xlabel("anomaly score = -log p_GMM(z)")
    ax.set_ylabel("events")
    ax.set_title(f"External Higgs anomaly scores: {Path(metadata['run_dir']).name}")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_score_distributions(
    event_scores: pd.DataFrame,
    metadata: dict,
    score_column: str,
    xlabel: str,
    title: str,
    out_path: Path,
) -> None:
    if score_column not in event_scores.columns:
        print(f"Skipping {out_path.name}: missing column {score_column}")
        return
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    bg = event_scores[event_scores["score_split"] == "background_eval"][score_column].to_numpy(dtype=float)
    signal_arrays = [
        event_scores[event_scores["sample_class"] == signal][score_column].to_numpy(dtype=float)
        for signal in metadata["signals"]
    ]
    all_scores = _finite_scores(bg, *signal_arrays)
    bins = np.linspace(float(np.percentile(all_scores, 0.5)), float(np.percentile(all_scores, 99.5)), 55)
    if len(np.unique(bins)) < 2:
        bins = 55

    ax.hist(bg, bins=bins, histtype="step", linewidth=1.8, color=PLOT_COLORS["sm_background"], label="SM background")
    for signal, values in zip(metadata["signals"], signal_arrays):
        ax.hist(
            values,
            bins=bins,
            histtype="step",
            linewidth=1.8,
            color=PLOT_COLORS.get(signal),
            label=_signal_label(signal, metadata["trained_classes"]),
        )

    bg_p95 = float(np.percentile(bg, 95))
    bg_p99 = float(np.percentile(bg, 99))
    ax.axvline(bg_p95, color="black", linestyle="--", linewidth=1.2, label="BG p95")
    ax.axvline(bg_p99, color="black", linestyle=":", linewidth=1.4, label="BG p99")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("events")
    ax.set_title(f"{title}: {Path(metadata['run_dir']).name}")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_score_distributions_per_signal(
    event_scores: pd.DataFrame,
    metadata: dict,
    score_column: str,
    xlabel: str,
    title: str,
    out_path: Path,
) -> None:
    if score_column not in event_scores.columns:
        print(f"Skipping {out_path.name}: missing column {score_column}")
        return
    bg = event_scores[event_scores["score_split"] == "background_eval"][score_column].to_numpy(dtype=float)
    signal_values = {
        signal: event_scores[event_scores["sample_class"] == signal][score_column].to_numpy(dtype=float)
        for signal in metadata["signals"]
    }
    all_scores = _finite_scores(bg, *signal_values.values())
    bins = np.linspace(float(np.percentile(all_scores, 0.5)), float(np.percentile(all_scores, 99.5)), 55)
    if len(np.unique(bins)) < 2:
        bins = 55

    bg_p95 = float(np.percentile(bg, 95))
    bg_p99 = float(np.percentile(bg, 99))
    n_signals = len(metadata["signals"])
    ncols = 2 if n_signals > 1 else 1
    nrows = int(math.ceil(n_signals / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2 * ncols, 4.3 * nrows), squeeze=False, sharex=True)

    for ax, signal in zip(axes.ravel(), metadata["signals"]):
        ax.hist(bg, bins=bins, histtype="step", linewidth=1.6, color=PLOT_COLORS["sm_background"], label="SM background")
        ax.hist(
            signal_values[signal],
            bins=bins,
            histtype="step",
            linewidth=1.8,
            color=PLOT_COLORS.get(signal),
            label=_signal_label(signal, metadata["trained_classes"]),
        )
        ax.axvline(bg_p95, color="black", linestyle="--", linewidth=1.1, label="BG p95")
        ax.axvline(bg_p99, color="black", linestyle=":", linewidth=1.3, label="BG p99")
        ax.set_yscale("log")
        ax.set_title(f"SM background vs {_signal_label(signal, metadata['trained_classes'])}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("events")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    for ax in axes.ravel()[n_signals:]:
        ax.axis("off")

    fig.suptitle(f"{title} by signal: {Path(metadata['run_dir']).name}", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_minbias_sm_control(
    event_scores: pd.DataFrame,
    metadata: dict,
    score_column: str,
    xlabel: str,
    title: str,
    out_path: Path,
) -> None:
    if score_column not in event_scores.columns:
        print(f"Skipping {out_path.name}: missing column {score_column}")
        return
    bg_df = event_scores[
        (event_scores["sample_role"] == "background")
        & (event_scores["score_split"] == "background_eval")
    ].copy()
    if bg_df.empty:
        print(f"Skipping {out_path.name}: no held-out SM background events")
        return
    if "minbias" not in set(bg_df["sample_class"].astype(str)):
        print(f"Skipping {out_path.name}: no held-out minbias events")
        return

    class_names = [cls for cls in metadata["background_classes"] if cls in set(bg_df["sample_class"].astype(str))]
    values_by_class = [
        bg_df.loc[bg_df["sample_class"] == cls, score_column].to_numpy(dtype=float)
        for cls in class_names
    ]
    all_scores = _finite_scores(*values_by_class)
    bins = np.linspace(float(np.percentile(all_scores, 0.5)), float(np.percentile(all_scores, 99.5)), 55)
    if len(np.unique(bins)) < 2:
        bins = 55

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    cmap = plt.get_cmap("tab20", max(len(class_names), 1))
    for idx, (cls, values) in enumerate(zip(class_names, values_by_class)):
        color = "#1f77b4" if cls == "minbias" else cmap(idx % cmap.N)
        linewidth = 2.2 if cls == "minbias" else 1.5
        alpha = 1.0 if cls == "minbias" else 0.85
        ax.hist(values, bins=bins, histtype="step", linewidth=linewidth, color=color, alpha=alpha, label=cls)

    minbias_values = bg_df.loc[bg_df["sample_class"] == "minbias", score_column].to_numpy(dtype=float)
    minbias_p95 = float(np.percentile(minbias_values, 95))
    minbias_p99 = float(np.percentile(minbias_values, 99))
    ax.axvline(minbias_p95, color="black", linestyle="--", linewidth=1.2, label="MinBias p95")
    ax.axvline(minbias_p99, color="black", linestyle=":", linewidth=1.4, label="MinBias p99")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("events")
    ax.set_title(f"{title}: {Path(metadata['run_dir']).name}")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_anomaly_distributions_per_signal(
    event_scores: pd.DataFrame,
    metrics: pd.DataFrame,
    metadata: dict,
    out_path: Path,
) -> None:
    bg = event_scores[event_scores["score_split"] == "background_eval"]["anomaly_score"].to_numpy(dtype=float)
    signal_values = {
        signal: event_scores[event_scores["sample_class"] == signal]["anomaly_score"].to_numpy(dtype=float)
        for signal in metadata["signals"]
    }
    all_scores = _finite_scores(bg, *signal_values.values())
    bins = np.linspace(float(np.percentile(all_scores, 0.5)), float(np.percentile(all_scores, 99.5)), 55)
    if len(np.unique(bins)) < 2:
        bins = 55

    bg_p95 = float(metrics["background_p95_threshold"].iloc[0])
    bg_p99 = float(metrics["background_p99_threshold"].iloc[0])
    n_signals = len(metadata["signals"])
    ncols = 2 if n_signals > 1 else 1
    nrows = int(math.ceil(n_signals / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2 * ncols, 4.3 * nrows), squeeze=False, sharex=True)

    for ax, signal in zip(axes.ravel(), metadata["signals"]):
        values = signal_values[signal]
        ax.hist(
            bg,
            bins=bins,
            histtype="step",
            linewidth=1.6,
            color=PLOT_COLORS["sm_background"],
            label="SM background",
        )
        ax.hist(
            values,
            bins=bins,
            histtype="step",
            linewidth=1.8,
            color=PLOT_COLORS.get(signal),
            label=_signal_label(signal, metadata["trained_classes"]),
        )
        ax.axvline(bg_p95, color="black", linestyle="--", linewidth=1.1, label="BG p95")
        ax.axvline(bg_p99, color="black", linestyle=":", linewidth=1.3, label="BG p99")
        ax.set_yscale("log")
        ax.set_title(f"SM background vs {_signal_label(signal, metadata['trained_classes'])}")
        ax.set_xlabel("anomaly score = -log p_GMM(z)")
        ax.set_ylabel("events")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    for ax in axes.ravel()[n_signals:]:
        ax.axis("off")

    fig.suptitle(f"External Higgs anomaly scores by signal: {Path(metadata['run_dir']).name}", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_individual_signal_distributions(
    event_scores: pd.DataFrame,
    metrics: pd.DataFrame,
    metadata: dict,
    output_dir: Path,
) -> None:
    bg = event_scores[event_scores["score_split"] == "background_eval"]["anomaly_score"].to_numpy(dtype=float)
    all_signal_values = [
        event_scores[event_scores["sample_class"] == signal]["anomaly_score"].to_numpy(dtype=float)
        for signal in metadata["signals"]
    ]
    all_scores = _finite_scores(bg, *all_signal_values)
    bins = np.linspace(float(np.percentile(all_scores, 0.5)), float(np.percentile(all_scores, 99.5)), 55)
    if len(np.unique(bins)) < 2:
        bins = 55

    bg_p95 = float(metrics["background_p95_threshold"].iloc[0])
    bg_p99 = float(metrics["background_p99_threshold"].iloc[0])
    for signal in metadata["signals"]:
        values = event_scores[event_scores["sample_class"] == signal]["anomaly_score"].to_numpy(dtype=float)
        fig, ax = plt.subplots(figsize=(7.2, 4.3))
        ax.hist(bg, bins=bins, histtype="step", linewidth=1.6, color=PLOT_COLORS["sm_background"], label="SM background")
        ax.hist(values, bins=bins, histtype="step", linewidth=1.8, color=PLOT_COLORS.get(signal), label=_signal_label(signal, metadata["trained_classes"]))
        ax.axvline(bg_p95, color="black", linestyle="--", linewidth=1.1, label="BG p95")
        ax.axvline(bg_p99, color="black", linestyle=":", linewidth=1.3, label="BG p99")
        ax.set_yscale("log")
        ax.set_xlabel("anomaly score = -log p_GMM(z)")
        ax.set_ylabel("events")
        ax.set_title(f"SM background vs {_signal_label(signal, metadata['trained_classes'])}: {Path(metadata['run_dir']).name}")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / f"anomaly_score_distribution_{signal}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def plot_efficiency(metrics: pd.DataFrame, out_path: Path) -> None:
    signals = metrics["signal"].tolist()
    x = np.arange(len(signals))
    width = 0.24
    eff95 = metrics["eff_at_95pct_bg"].to_numpy(dtype=float)
    eff99 = metrics["eff_at_99pct_bg"].to_numpy(dtype=float)
    eff995 = metrics["eff_at_99_5pct_bg"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    bars95 = ax.bar(x - width, eff95, width, label="eff@95%BG", color="#4c78a8")
    bars99 = ax.bar(x, eff99, width, label="eff@99%BG", color="#f58518")
    bars995 = ax.bar(x + width, eff995, width, label="eff@99.5%BG", color="#54a24b")
    for bars in (bars95, bars99, bars995):
        for bar in bars:
            value = float(bar.get_height())
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, min(1.15, max(1.0, float(np.nanmax([eff95, eff99, eff995])) * 1.18)))
    ax.set_xticks(x)
    ax.set_xticklabels(signals)
    ax.set_ylabel("signal fraction above threshold")
    ax.set_title("Signal efficiency at fixed background rejection")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_auc(metrics: pd.DataFrame, out_path: Path) -> None:
    signals = metrics["signal"].tolist()
    auc = metrics["roc_auc_vs_sm_background"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    bars = ax.bar(signals, auc, color=["#4c78a8", "#f58518", "#54a24b", "#b279a2"][: len(signals)])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("ROC AUC vs combined SM background")
    ax.set_title("External Higgs AUC from anomaly scores")
    for bar, value in zip(bars, auc):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_latent_pca(event_scores: pd.DataFrame, metadata: dict, out_path: Path) -> None:
    latent_path = out_path.parent / "external_higgs_latent_pca_points.npz"
    if not latent_path.exists():
        raise RuntimeError(f"missing PCA plotting points: {latent_path}")
    data = np.load(latent_path, allow_pickle=False)
    pc = data["pc"]
    sample_names = data["sample_class"].astype(str)
    score_split = data["score_split"].astype(str)
    explained = data["explained_variance_ratio"]

    rng = np.random.default_rng(int(metadata["seed"]))
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    bg_mask = score_split == "background_eval"
    bg_idx = np.flatnonzero(bg_mask)
    if len(bg_idx) > 5000:
        bg_idx = rng.choice(bg_idx, size=5000, replace=False)
    ax.scatter(pc[bg_idx, 0], pc[bg_idx, 1], s=5, c="#c7c7c7", alpha=0.35, label="SM background")
    for signal in metadata["signals"]:
        sig_idx = np.flatnonzero(sample_names == signal)
        if len(sig_idx) > 2500:
            sig_idx = rng.choice(sig_idx, size=2500, replace=False)
        label = _signal_label(signal, metadata["trained_classes"])
        ax.scatter(pc[sig_idx, 0], pc[sig_idx, 1], s=8, alpha=0.65, color=PLOT_COLORS.get(signal), label=label)
    ax.set_xlabel(f"PC1 ({100 * float(explained[0]):.1f}% var)")
    ax.set_ylabel(f"PC2 ({100 * float(explained[1]):.1f}% var)")
    ax.set_title("Latent PCA projection fitted on background")
    ax.legend(markerscale=2, fontsize=8)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def balanced_class_indices(
    event_scores: pd.DataFrame,
    class_names: list[str],
    max_per_class: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected = []
    sample_class = event_scores["sample_class"].astype(str).to_numpy()
    for cls in class_names:
        cls_idx = np.flatnonzero(sample_class == cls)
        if len(cls_idx) == 0:
            continue
        if len(cls_idx) > max_per_class:
            cls_idx = rng.choice(cls_idx, size=max_per_class, replace=False)
        selected.append(cls_idx)
    if not selected:
        return np.array([], dtype=int)
    return rng.permutation(np.concatenate(selected)).astype(int)


def make_tsne(n_samples: int, perplexity: float, n_iter: int, seed: int) -> TSNE:
    safe_perplexity = min(float(perplexity), max(5.0, float((n_samples - 1) // 3)))
    kwargs = {
        "n_components": 2,
        "perplexity": safe_perplexity,
        "random_state": seed,
        "init": "pca",
        "learning_rate": "auto",
    }
    if "max_iter" in inspect.signature(TSNE).parameters:
        kwargs["max_iter"] = n_iter
    else:
        kwargs["n_iter"] = n_iter
    return TSNE(**kwargs)


def save_latent_tsne_points(
    output_dir: Path,
    event_scores: pd.DataFrame,
    Z: np.ndarray,
    metadata: dict,
    seed: int,
    max_per_class: int,
    perplexity: float,
    n_iter: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tsne_outputs = [
        (
            "background",
            output_dir / "background_latent_tsne_points.npz",
            metadata["background_classes"],
        ),
        (
            "background_plus_signals",
            output_dir / "external_higgs_latent_tsne_points.npz",
            unique_ordered([*metadata["background_classes"], *metadata["signals"]]),
        ),
    ]
    sample_class_all = event_scores["sample_class"].astype(str).to_numpy(dtype="<U32")
    sample_role_all = event_scores["sample_role"].astype(str).to_numpy(dtype="<U32")
    score_split_all = event_scores["score_split"].astype(str).to_numpy(dtype="<U32")

    metadata["tsne"] = {
        "enabled": True,
        "max_events_per_class": int(max_per_class),
        "perplexity_requested": float(perplexity),
        "n_iter": int(n_iter),
        "fit_space": "raw latent vectors",
        "plots": [],
    }

    for scope, out_path, class_names in tsne_outputs:
        idx = balanced_class_indices(event_scores, class_names, max_per_class, seed)
        if len(idx) < 10:
            metadata["tsne"]["plots"].append({"scope": scope, "status": "skipped", "n_events": int(len(idx))})
            continue
        tsne = make_tsne(len(idx), perplexity, n_iter, seed)
        xy = tsne.fit_transform(Z[idx])
        np.savez_compressed(
            out_path,
            tsne=xy.astype(np.float32),
            event_index=idx.astype(np.int64),
            sample_class=sample_class_all[idx],
            sample_role=sample_role_all[idx],
            score_split=score_split_all[idx],
            perplexity=float(tsne.perplexity),
            max_events_per_class=int(max_per_class),
        )
        metadata["tsne"]["plots"].append(
            {
                "scope": scope,
                "status": "completed",
                "n_events": int(len(idx)),
                "perplexity_used": float(tsne.perplexity),
                "path": str(out_path),
            }
        )


def plot_latent_tsne(metadata: dict, points_path: Path, out_path: Path, background_only: bool) -> None:
    if not points_path.exists():
        return
    data = np.load(points_path, allow_pickle=False)
    xy = data["tsne"]
    sample_names = data["sample_class"].astype(str)
    sample_roles = data["sample_role"].astype(str)
    classes = metadata["background_classes"] if background_only else unique_ordered([*metadata["background_classes"], *metadata["signals"]])
    cmap = plt.get_cmap("tab20", max(len(classes), 1))
    class_colors = {cls: cmap(i % cmap.N) for i, cls in enumerate(classes)}
    for signal in metadata["signals"]:
        class_colors[signal] = PLOT_COLORS.get(signal, class_colors.get(signal, "#333333"))

    fig, ax = plt.subplots(figsize=(7.4, 5.8))
    if not background_only:
        bg_mask = sample_roles == "background"
        ax.scatter(
            xy[bg_mask, 0],
            xy[bg_mask, 1],
            s=6,
            c=[class_colors.get(cls, "#bdbdbd") for cls in sample_names[bg_mask]],
            alpha=0.28,
            linewidths=0,
            label="SM background",
        )
        for signal in metadata["signals"]:
            sig_mask = sample_names == signal
            if not np.any(sig_mask):
                continue
            ax.scatter(
                xy[sig_mask, 0],
                xy[sig_mask, 1],
                s=13,
                color=class_colors.get(signal),
                alpha=0.75,
                linewidths=0,
                label=_signal_label(signal, metadata["trained_classes"]),
            )
    else:
        for cls in classes:
            cls_mask = sample_names == cls
            if not np.any(cls_mask):
                continue
            ax.scatter(
                xy[cls_mask, 0],
                xy[cls_mask, 1],
                s=7,
                color=class_colors.get(cls),
                alpha=0.55,
                linewidths=0,
                label=cls,
            )

    perplexity = float(data["perplexity"])
    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    title_scope = "SM background" if background_only else "SM background + external Higgs signals"
    ax.set_title(f"Latent t-SNE: {title_scope} | perplexity={perplexity:g}")
    ax.legend(markerscale=2, fontsize=7, frameon=True, loc="best")
    ax.grid(True, alpha=0.18)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_minbias_vs_anomalies_projection(
    metadata: dict,
    points_path: Path,
    coordinate_key: str,
    out_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    if not points_path.exists():
        print(f"Skipping {out_path.name}: missing projection points {points_path}")
        return
    data = np.load(points_path, allow_pickle=False)
    if coordinate_key not in data:
        print(f"Skipping {out_path.name}: missing coordinate key {coordinate_key}")
        return
    xy = data[coordinate_key]
    sample_names = data["sample_class"].astype(str)
    keep_classes = ["minbias", *metadata["signals"]]
    keep = np.isin(sample_names, keep_classes)
    if not np.any(keep):
        print(f"Skipping {out_path.name}: no MinBias or signal points found")
        return

    fig, ax = plt.subplots(figsize=(7.4, 5.8))
    minbias_mask = sample_names == "minbias"
    if np.any(minbias_mask):
        ax.scatter(
            xy[minbias_mask, 0],
            xy[minbias_mask, 1],
            s=8,
            color="#bdbdbd",
            alpha=0.42,
            linewidths=0,
            label="minbias",
        )
    for signal in metadata["signals"]:
        sig_mask = sample_names == signal
        if not np.any(sig_mask):
            continue
        ax.scatter(
            xy[sig_mask, 0],
            xy[sig_mask, 1],
            s=14,
            color=PLOT_COLORS.get(signal),
            alpha=0.75,
            linewidths=0,
            label=_signal_label(signal, metadata["trained_classes"]),
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}: {Path(metadata['run_dir']).name}")
    ax.legend(markerscale=2, fontsize=8, frameon=True)
    ax.grid(True, alpha=0.18)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_classifier_response(response: pd.DataFrame, metadata: dict, out_path: Path) -> None:
    trained_classes = metadata["trained_classes"]
    matrix = response.set_index("signal")[trained_classes].astype(float)
    fig_w = max(8.0, 0.72 * len(trained_classes))
    fig_h = max(3.8, 0.72 * len(matrix.index))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(matrix.to_numpy(), cmap="Blues", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("row-normalized classifier response")
    ax.set_xticks(np.arange(len(trained_classes)))
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_xticklabels(trained_classes, rotation=45, ha="right")
    ax.set_yticklabels(matrix.index)
    ax.set_xlabel("trained class")
    ax.set_ylabel("external sample")
    ax.set_title("External Higgs classifier response")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = float(matrix.iloc[i, j])
            ax.text(j, i, f"{100 * value:.1f}%", ha="center", va="center", fontsize=8, color="white" if value > 0.55 else "black")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_summary(event_scores: pd.DataFrame, metrics: pd.DataFrame, response: pd.DataFrame, metadata: dict, out_path: Path) -> None:
    trained_classes = metadata["trained_classes"]
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.0))

    bg = event_scores[event_scores["score_split"] == "background_eval"]["anomaly_score"].to_numpy(dtype=float)
    signal_arrays = [
        event_scores[event_scores["sample_class"] == signal]["anomaly_score"].to_numpy(dtype=float)
        for signal in metadata["signals"]
    ]
    all_scores = _finite_scores(bg, *signal_arrays)
    bins = np.linspace(float(np.percentile(all_scores, 0.5)), float(np.percentile(all_scores, 99.5)), 45)
    axes[0, 0].hist(bg, bins=bins, histtype="step", color=PLOT_COLORS["sm_background"], linewidth=1.5, label="SM background")
    for signal, values in zip(metadata["signals"], signal_arrays):
        axes[0, 0].hist(values, bins=bins, histtype="step", color=PLOT_COLORS.get(signal), linewidth=1.5, label=signal)
    axes[0, 0].axvline(float(metrics["background_p95_threshold"].iloc[0]), color="black", linestyle="--", linewidth=1.0)
    axes[0, 0].axvline(float(metrics["background_p99_threshold"].iloc[0]), color="black", linestyle=":", linewidth=1.2)
    axes[0, 0].axvline(float(metrics["background_p99_5_threshold"].iloc[0]), color="black", linestyle="-.", linewidth=1.2)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Anomaly score distributions")
    axes[0, 0].set_xlabel("-log p_GMM(z)")
    axes[0, 0].set_ylabel("events")
    axes[0, 0].legend(fontsize=7)

    x = np.arange(len(metrics))
    width = 0.24
    axes[0, 1].bar(x - width, metrics["eff_at_95pct_bg"], width, label="eff@95%BG", color="#4c78a8")
    axes[0, 1].bar(x, metrics["eff_at_99pct_bg"], width, label="eff@99%BG", color="#f58518")
    axes[0, 1].bar(x + width, metrics["eff_at_99_5pct_bg"], width, label="eff@99.5%BG", color="#54a24b")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(metrics["signal"])
    axes[0, 1].set_ylim(0, 1.05)
    axes[0, 1].set_title("Fixed-background efficiency")
    axes[0, 1].legend(fontsize=8)

    auc = metrics["roc_auc_vs_sm_background"].to_numpy(dtype=float)
    axes[1, 0].bar(metrics["signal"], auc, color=["#4c78a8", "#f58518", "#54a24b", "#b279a2"][: len(metrics)])
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].set_title("AUC vs SM background")
    axes[1, 0].set_ylabel("ROC AUC")
    for idx, value in enumerate(auc):
        axes[1, 0].text(idx, value + 0.015, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    matrix = response.set_index("signal")[trained_classes].astype(float)
    im = axes[1, 1].imshow(matrix.to_numpy(), cmap="Blues", vmin=0, vmax=1)
    axes[1, 1].set_xticks(np.arange(len(trained_classes)))
    axes[1, 1].set_yticks(np.arange(len(matrix.index)))
    axes[1, 1].set_xticklabels(trained_classes, rotation=45, ha="right", fontsize=7)
    axes[1, 1].set_yticklabels(matrix.index)
    axes[1, 1].set_title("Classifier response")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = float(matrix.iloc[i, j])
            if value >= 0.01:
                axes[1, 1].text(j, i, f"{100 * value:.0f}", ha="center", va="center", fontsize=7, color="white" if value > 0.55 else "black")
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

    for ax in axes.ravel():
        ax.grid(True, alpha=0.18)
    fig.suptitle(f"External Higgs anomaly summary: {Path(metadata['run_dir']).name}", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_latent_pca_points(output_dir: Path, event_scores: pd.DataFrame, Z: np.ndarray, bg_fit_idx: np.ndarray, seed: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pca = PCA(n_components=2, random_state=seed)
    pca.fit(Z[bg_fit_idx])
    pc = pca.transform(Z)
    sample_class = event_scores["sample_class"].astype(str).to_numpy(dtype="<U32")
    score_split = event_scores["score_split"].astype(str).to_numpy(dtype="<U32")
    np.savez_compressed(
        output_dir / "external_higgs_latent_pca_points.npz",
        pc=pc.astype(np.float32),
        sample_class=sample_class,
        score_split=score_split,
        explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float64),
    )


def generate_plots_from_saved(output_dir: Path) -> None:
    event_scores, metrics, response, metadata = load_saved_results(output_dir)
    plot_anomaly_distributions(event_scores, metrics, metadata, output_dir / "anomaly_score_distributions.png")
    plot_anomaly_distributions_per_signal(event_scores, metrics, metadata, output_dir / "anomaly_score_distributions_per_signal.png")
    plot_score_distributions(
        event_scores,
        metadata,
        "minbias_distance_score",
        "distance from MinBias reference (Mahalanobis)",
        "MinBias-distance scores",
        output_dir / "minbias_distance_distributions.png",
    )
    plot_score_distributions_per_signal(
        event_scores,
        metadata,
        "minbias_distance_score",
        "distance from MinBias reference (Mahalanobis)",
        "MinBias-distance scores",
        output_dir / "minbias_distance_distributions_per_signal.png",
    )
    plot_score_distributions(
        event_scores,
        metadata,
        "overall_anomaly_score",
        "overall anomaly score",
        "Combined anomaly scores",
        output_dir / "overall_anomaly_score_distributions.png",
    )
    plot_minbias_sm_control(
        event_scores,
        metadata,
        "gmm_anomaly_score",
        "GMM anomaly score = -log p_GMM(z)",
        "MinBias control vs SM background (GMM)",
        output_dir / "minbias_control_gmm.png",
    )
    plot_minbias_sm_control(
        event_scores,
        metadata,
        "minbias_distance_score",
        "distance from MinBias reference (Mahalanobis)",
        "MinBias control vs SM background (MinBias distance)",
        output_dir / "minbias_control_minbias_distance.png",
    )
    plot_minbias_sm_control(
        event_scores,
        metadata,
        "overall_anomaly_score",
        "overall anomaly score",
        "MinBias control vs SM background (overall)",
        output_dir / "minbias_control_overall.png",
    )
    plot_individual_signal_distributions(event_scores, metrics, metadata, output_dir)
    plot_efficiency(metrics, output_dir / "signal_efficiency_fixed_background.png")
    plot_auc(metrics, output_dir / "external_higgs_auc.png")
    plot_latent_pca(event_scores, metadata, output_dir / "external_higgs_latent_pca.png")
    plot_latent_tsne(
        metadata,
        output_dir / "background_latent_tsne_points.npz",
        output_dir / "background_latent_tsne.png",
        background_only=True,
    )
    plot_latent_tsne(
        metadata,
        output_dir / "external_higgs_latent_tsne_points.npz",
        output_dir / "external_higgs_latent_tsne.png",
        background_only=False,
    )
    pca_points = output_dir / "external_higgs_latent_pca_points.npz"
    if pca_points.exists():
        pca_data = np.load(pca_points, allow_pickle=False)
        explained = pca_data["explained_variance_ratio"] if "explained_variance_ratio" in pca_data else None
        pc1 = f"PC1 ({100 * float(explained[0]):.1f}% var)" if explained is not None and len(explained) > 0 else "PC1"
        pc2 = f"PC2 ({100 * float(explained[1]):.1f}% var)" if explained is not None and len(explained) > 1 else "PC2"
    else:
        pc1, pc2 = "PC1", "PC2"
    plot_minbias_vs_anomalies_projection(
        metadata,
        pca_points,
        "pc",
        output_dir / "minbias_vs_anomalies_latent_pca.png",
        "Latent PCA: MinBias vs injected anomalies",
        pc1,
        pc2,
    )
    plot_minbias_vs_anomalies_projection(
        metadata,
        output_dir / "external_higgs_latent_tsne_points.npz",
        "tsne",
        output_dir / "minbias_vs_anomalies_latent_tsne.png",
        "Latent t-SNE: MinBias vs injected anomalies",
        "t-SNE dimension 1",
        "t-SNE dimension 2",
    )
    plot_classifier_response(response, metadata, output_dir / "external_higgs_classifier_response.png")
    plot_summary(event_scores, metrics, response, metadata, output_dir / "external_higgs_summary.png")


def validate_saved_outputs(output_dir: Path) -> list[str]:
    _event_scores, metrics, response, metadata = load_saved_results(output_dir)
    issues = []
    if not metrics["roc_auc_vs_sm_background"].between(0, 1).all():
        issues.append("AUC outside [0, 1]")
    if not metrics["eff_at_95pct_bg"].between(0, 1).all():
        issues.append("eff_at_95pct_bg outside [0, 1]")
    if not metrics["eff_at_99pct_bg"].between(0, 1).all():
        issues.append("eff_at_99pct_bg outside [0, 1]")
    if not metrics["eff_at_99_5pct_bg"].between(0, 1).all():
        issues.append("eff_at_99_5pct_bg outside [0, 1]")
    if not (metrics["background_p99_threshold"] >= metrics["background_p95_threshold"]).all():
        issues.append("background p99 threshold is lower than p95")
    if not (metrics["background_p99_5_threshold"] >= metrics["background_p99_threshold"]).all():
        issues.append("background p99.5 threshold is lower than p99")
    trained_classes = metadata["trained_classes"]
    row_sums = response[trained_classes].astype(float).sum(axis=1).to_numpy()
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        issues.append(f"classifier response rows do not sum to 1: {row_sums.tolist()}")
    return issues


def evaluate_one_run(run_dir: Path, args) -> RunResult:
    try:
        discovery = discover_run(run_dir, args)
        if args.dry_run:
            print(f"[dry-run] {discovery.run_dir}")
            print(f"  checkpoint: {discovery.checkpoint_path}")
            print(f"  trained_classes: {discovery.trained_classes}")
            print(f"  background_classes: {discovery.background_classes}")
            print(f"  output_dir: {discovery.output_dir}")
            validate_run_preprocessing_config(discovery.run_config)
            return RunResult(str(discovery.run_dir), "completed", "metadata discovery only", str(discovery.output_dir))
        if args.plots_only:
            missing = [
                name
                for name in [
                    "external_higgs_event_scores.csv",
                    "external_higgs_anomaly_metrics.csv",
                    "external_higgs_classifier_response.csv",
                    "external_higgs_run_metadata.json",
                ]
                if not (discovery.output_dir / name).exists()
            ]
            if missing:
                raise SkipRun(f"plots-only inputs missing in {discovery.output_dir}: {missing}")
            generate_plots_from_saved(discovery.output_dir)
            print(f"[plots-only] regenerated plots -> {discovery.output_dir}")
            return RunResult(str(discovery.run_dir), "completed", "plots regenerated from saved results", str(discovery.output_dir))

        import torch

        device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        print("=" * 90)
        print(f"External Higgs anomaly evaluation: {discovery.run_dir}")
        print(f"checkpoint: {discovery.checkpoint_path}")
        print(f"device: {device}")
        print(f"trained_classes: {discovery.trained_classes}")
        print(f"background_classes: {discovery.background_classes}")
        print(f"signals: {args.signals}")

        model = load_saved_model(str(discovery.checkpoint_path), device, discovery.run_config)
        raw_tensor, sample_names, _context_classes, event_counts = prepare_raw_events(discovery, args)
        print(f"loaded events: {event_counts}")
        X = preprocess_for_run(raw_tensor, discovery, model)
        Z, logits = extract_latents_logits(model, X, device, args.batch_size)
        density_info, Z_white, bg_fit_idx, bg_eval_idx = fit_density_models(
            Z,
            sample_names,
            discovery.background_classes,
            args.seed,
            args.background_fit_fraction,
        )
        event_scores, metrics, method_metrics, response, metadata = build_event_scores(
            discovery,
            args,
            sample_names,
            Z,
            logits,
            density_info,
            Z_white,
            bg_fit_idx,
            bg_eval_idx,
        )
        save_latent_pca_points(discovery.output_dir, event_scores, Z, bg_fit_idx, args.seed)
        if args.skip_tsne:
            metadata["tsne"] = {"enabled": False}
        else:
            save_latent_tsne_points(
                discovery.output_dir,
                event_scores,
                Z,
                metadata,
                args.seed,
                args.tsne_max_events_per_class,
                args.tsne_perplexity,
                args.tsne_n_iter,
            )
        metadata["event_counts"].update(event_counts)
        save_outputs(discovery.output_dir, event_scores, metrics, method_metrics, response, metadata)
        generate_plots_from_saved(discovery.output_dir)
        issues = validate_saved_outputs(discovery.output_dir)
        if issues:
            raise RuntimeError("; ".join(issues))

        print(f"outputs -> {discovery.output_dir}")
        return RunResult(str(discovery.run_dir), "completed", "", str(discovery.output_dir))
    except SkipRun as exc:
        print(f"SKIPPED {run_dir}: {exc}")
        if args.raise_errors:
            raise
        return RunResult(str(run_dir), "skipped", str(exc))
    except Exception as exc:
        if args.raise_errors:
            raise
        reason = f"{type(exc).__name__}: {exc}"
        print(f"FAILED {run_dir}: {reason}")
        if args.verbose_errors:
            traceback.print_exc()
        return RunResult(str(run_dir), "failed", reason)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate external Higgs-like signals with background-only latent anomaly scores."
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--run-dir", type=Path, help="Single trained run directory containing run_config.json.")
    target.add_argument("--run-list", type=Path, help="Text file with one run directory per line.")
    target.add_argument("--discover-required-runs", action="store_true", help="Scan --run-root for requested model families.")
    parser.add_argument("--run-root", type=Path, default=Path("ngt_mha_nodes_aligned_outputs") / "runs")
    parser.add_argument("--model", default=None, help="Override checkpoint path for every run.")
    parser.add_argument("--signals", default=",".join(DEFAULT_SIGNALS), help="Comma-separated external signals.")
    parser.add_argument("--background-classes", default=None, help="Override comma-separated background class list.")
    parser.add_argument(
        "--exclude-background-classes",
        default=None,
        help="Comma-separated trained classes to remove from the default background list.",
    )
    parser.add_argument(
        "--output-subdir",
        default=str(DEFAULT_OUTPUT_SUBDIR),
        help="Run-relative output directory. Use a different value for non-Higgs evaluations to avoid overwriting.",
    )
    parser.add_argument("--n-background-per-class", type=int, default=2000)
    parser.add_argument("--n-signal-per-class", type=int, default=2000)
    parser.add_argument("--background-fit-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', or 'cuda'.")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--data-source", default=None, choices=["collide2v_updated", "samples_json", "hf"])
    parser.add_argument("--collide2v-path", default=None, help="Override Collide-2V parquet root.")
    parser.add_argument("--cache-dir", default=None, help="Override raw tensor cache directory.")
    parser.add_argument("--cache-prefix", default="external_higgs_anomaly")
    parser.add_argument("--candidate-selection", default=None)
    parser.add_argument("--n-constituents", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Validate run metadata/checkpoint discovery without loading data.")
    parser.add_argument("--plots-only", action="store_true", help="Regenerate figures from saved external Higgs CSV/NPZ outputs.")
    parser.add_argument("--skip-tsne", action="store_true", help="Skip t-SNE point generation during evaluation.")
    parser.add_argument("--tsne-max-events-per-class", type=int, default=600)
    parser.add_argument("--tsne-perplexity", type=float, default=40.0)
    parser.add_argument("--tsne-n-iter", type=int, default=1000)
    parser.add_argument("--raise-errors", action="store_true")
    parser.add_argument("--verbose-errors", action="store_true")
    return parser


def resolve_run_dirs(args) -> tuple[list[Path], dict]:
    if args.run_dir:
        return [args.run_dir], {}
    if args.run_list:
        return read_run_list(args.run_list), {}
    run_dirs, discovery_notes = discover_required_run_dirs(args.run_root)
    return run_dirs, discovery_notes


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    args.signals = parse_csv_list(args.signals)
    unknown_signals = [name for name in args.signals if name not in SUPPORTED_CLASS_NAMES]
    if unknown_signals:
        parser.error(f"unsupported signals: {unknown_signals}; supported classes are {SUPPORTED_CLASS_NAMES}")

    run_dirs, discovery_notes = resolve_run_dirs(args)
    if discovery_notes:
        print("Required-run discovery notes:")
        print(json.dumps(discovery_notes, indent=2))
    if not run_dirs:
        print("No run directories selected.")
        return 1

    results = [evaluate_one_run(run_dir, args) for run_dir in run_dirs]
    completed = [row for row in results if row.status == "completed"]
    skipped = [row for row in results if row.status == "skipped"]
    failed = [row for row in results if row.status == "failed"]

    print("\nBatch summary")
    print("=" * 50)
    print(f"completed: {len(completed)}")
    print(f"skipped:   {len(skipped)}")
    print(f"failed:    {len(failed)}")
    for row in skipped + failed:
        print(f"  {row.status.upper()} {row.run_dir}: {row.reason}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
