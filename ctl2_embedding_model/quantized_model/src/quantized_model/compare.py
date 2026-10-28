"""Compare FP32 and INT8 on the identical, explicitly supplied held-out events."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import torch
from sklearn.covariance import LedoitWolf
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from .core import infer, load_bundle, load_fp32, sha256, write_json


def differences(reference, candidate):
    a, b = reference.astype(np.float64), candidate.astype(np.float64)
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    cosine = np.divide((a * b).sum(axis=1), denominator,
                       out=np.zeros(len(a)), where=denominator > 0)
    return dict(max_absolute_error=float(np.max(np.abs(a - b))),
                rmse=float(np.sqrt(np.mean((a - b) ** 2))),
                relative_l2_error=float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-12)),
                mean_cosine_similarity=float(cosine.mean()))


def anomaly_metrics(background, signal):
    y = np.r_[np.zeros(len(background)), np.ones(len(signal))]
    result = {"roc_auc": float(roc_auc_score(y, np.r_[background, signal])),
              "n_background": len(background), "n_signal": len(signal)}
    for rejection in (95, 99, 99.5):
        threshold = float(np.percentile(background, rejection))
        efficiency = float(np.mean(signal > threshold))
        result[str(rejection)] = dict(background_threshold=threshold,
                                     background_acceptance=float(np.mean(background > threshold)),
                                     signal_efficiency=efficiency,
                                     signal_binomial_standard_error=float(np.sqrt(
                                         efficiency * (1 - efficiency) / len(signal))))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--raw-inputs", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cohort-description", default="Caller-supplied events; independence from training not verified")
    parser.add_argument("--minbias-fit-inputs", type=Path,
                        help="Independent MinBias reference split; never use evaluation events")
    parser.add_argument("--signal", action="append", default=[], metavar="NAME=RAW_OR_PREPROCESSED_NPY")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output directory already exists; choose a new comparison path")
    if args.signal and not args.minbias_fit_inputs:
        parser.error("--signal requires --minbias-fit-inputs")
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    quantized, bundle = load_bundle(args.bundle)
    if sha256(args.checkpoint) != bundle["source_checkpoint_sha256"]:
        parser.error("FP32 checkpoint does not match the bundle provenance")
    config = bundle["config"]
    fp32, _ = load_fp32(args.checkpoint, config)
    x = np.load(args.inputs, mmap_mode="r", allow_pickle=False)
    if len(x) < 1:
        parser.error("No evaluation events")
    output = {}
    timing = {}
    for name, model in (("fp32", fp32), ("int8", quantized)):
        infer(model, x[:min(len(x), args.batch_size)], config, args.batch_size, args.raw_inputs)
        start = time.perf_counter()
        output[name] = infer(model, x, config, args.batch_size, args.raw_inputs)
        elapsed = time.perf_counter() - start
        timing[name] = dict(seconds=elapsed, events_per_second=len(x) / elapsed)
        print(f"{name}: {len(x)} events in {elapsed:.2f}s", flush=True)
    predicted = {key: value["logits"].argmax(axis=1) for key, value in output.items()}
    report = dict(n_events=len(x), source_epoch=bundle["source_epoch"],
                  checkpoint_sha256=bundle["source_checkpoint_sha256"], bundle_sha256=sha256(args.bundle),
                  inputs_sha256=sha256(args.inputs), raw_inputs=args.raw_inputs,
                  cohort_description=args.cohort_description,
                  torch_version=str(torch.__version__), engine=bundle["engine"],
                  batch_size=args.batch_size, threads=args.threads,
                  precision=bundle["format"], timing=timing,
                  prediction_agreement=float(np.mean(predicted["fp32"] == predicted["int8"])),
                  latent=differences(output["fp32"]["latent"], output["int8"]["latent"]),
                  logits=differences(output["fp32"]["logits"], output["int8"]["logits"]),
                  status="measured_comparison_not_physics_acceptance", fpga_synthesis_validated=False)
    if args.labels:
        y = np.load(args.labels, allow_pickle=False)
        if y.shape != (len(x),) or not np.issubdtype(y.dtype, np.integer):
            parser.error("Labels must be a one-dimensional integer array aligned with inputs")
        if np.any(y < 0) or np.any(y >= len(config["class_names"])):
            parser.error("Labels fall outside the trained class order")
        report["labels_sha256"] = sha256(args.labels)
        report["classification"] = {key: dict(
            accuracy=float(accuracy_score(y, pred)),
            macro_f1=float(f1_score(y, pred, labels=list(range(len(config["class_names"]))),
                                    average="macro", zero_division=0))) for key, pred in predicted.items()}
    if args.signal:
        fit_x = np.load(args.minbias_fit_inputs, mmap_mode="r", allow_pickle=False)
        if len(fit_x) < 2:
            parser.error("MinBias reference requires at least two independent events")
        fitted = {name: LedoitWolf().fit(infer(model, fit_x, config, args.batch_size,
                                              args.raw_inputs)["latent"])
                  for name, model in (("fp32", fp32), ("int8", quantized))}
        report["anomaly"] = {"method": "minbias_mahalanobis",
                             "fit_inputs_sha256": sha256(args.minbias_fit_inputs),
                             "threshold_policy": "independently recalibrated from the same held-out background events",
                             "uncertainty_note": "Signal binomial errors exclude threshold uncertainty; 99.5% tails require large background samples.",
                             "signals": {}}
        for item in args.signal:
            name, path = item.split("=", 1)
            if name in config["class_names"] or name in report["anomaly"]["signals"]:
                parser.error("Signal names must be unique and external to the trained classes")
            signal_x = np.load(path, mmap_mode="r", allow_pickle=False)
            signal_outputs = {key: infer(model, signal_x, config, args.batch_size, args.raw_inputs)["latent"]
                              for key, model in (("fp32", fp32), ("int8", quantized))}
            metrics = {key: anomaly_metrics(fitted[key].mahalanobis(output[key]["latent"]),
                                            fitted[key].mahalanobis(signal_outputs[key]))
                       for key in fitted}
            # A frozen FP32 detector exposes latent-coordinate changes separately from detector refitting.
            metrics["int8_with_frozen_fp32_detector"] = anomaly_metrics(
                fitted["fp32"].mahalanobis(output["int8"]["latent"]),
                fitted["fp32"].mahalanobis(signal_outputs["int8"]))
            metrics["signal_inputs_sha256"] = sha256(path)
            report["anomaly"]["signals"][name] = metrics
    args.output.mkdir(parents=True)
    for name, values in output.items():
        for key, value in values.items():
            np.save(args.output / f"{name}_{key}.npy", value)
    write_json(args.output / "comparison.json", report)
    print(f"Prediction agreement: {report['prediction_agreement']:.4f}", flush=True)


if __name__ == "__main__":
    main()
