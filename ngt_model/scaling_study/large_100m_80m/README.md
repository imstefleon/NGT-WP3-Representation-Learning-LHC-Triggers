# 100M-Parameter / 80M-Event DDP Study

This folder preserves the benchmark, launch/resume scripts, early configs and
Kubeflow templates for the large MHA run.

## Delivered run status

The [selected run](../../final_run/README.md) is the 2026-08-24 CE + InfoNCE run
with five recorded epochs. Epoch 4 is delivered by validation-loss selection.
Its authoritative [run configuration](../../final_run/run_config.json) uses
`d_model=2016`, `ff_dim=8064`, raw PID, attention pooling, a linear classifier,
`lr=1e-5`, no augmentation, SupCon weight 0.05 and seed 42.

The A/B CE + var/cov JSON files and the original `prepared_not_launched` config
below are retained planning artifacts. They are not metadata for this selected
checkpoint. Checkpoints and history live in `ngt_model/final_run/`; large caches
remain on CERN storage.

Run shell launchers from this checkout; they resolve the current repository
root. The editable [Mermaid architecture](../../../docs/diagrams/ngt_100m_architecture.mmd)
and [quantization package](../../../quantized_model/README.md) describe the delivery.

## Target

| item | value |
|---|---:|
| GPUs | 4 x 48 GB |
| training mode | PyTorch DDP via `torchrun` or Kubeflow `PyTorchJob` |
| total events | 80M |
| classes | `minbias,qcd,dy,top` |
| events/class | 20M |
| events/rank | 20M total, 5M/class |
| raw input | 100 candidates x 5 fp32 features |
| external signals | `hh4b,hh,vbfh,ggh` |
| required HH4B metric | efficiency at 99.5% background rejection |

## Repository Audit

- `train_mha.py` already supported `torchrun`/DDP, but the original loader materialized the full requested dataset on every rank and only then used `DistributedSampler`.
- `collide2v_data.py` now supports deterministic event sharding through `event_stride` and `event_offset`; DDP ranks can load disjoint caches such as `shard0000of0004`.
- `train_mha.py` now exposes `--data-loading-mode rank_sharded_materialized` so each DDP rank loads only its local shard.
- Normalization constants and PID mapping are synchronized across ranks before preprocessing, so all ranks train with consistent feature transforms.
- `external_higgs_anomaly.py` now writes `eff_at_99_5pct_bg` and `background_p99_5_threshold`, plus method-level `eff_at_99_5_bg_rejection`.

## Selected 100M Configuration

The 24h large-run configuration uses the requested transformer-style ratio:

```text
ff_dim = 4 * d_model
num_heads = 16
num_attention_blocks = 4
```

The closest valid point to 100M trainable parameters is:

| d_model | ff_dim | heads | attention blocks | trainable params |
|---:|---:|---:|---:|---:|
| 2016 | 8064 | 16 | 4 | 99,889,413 |

`d_model=2016` is divisible by 16, so each attention head has dimension 126. `ff_dim=8064` is the post-pooling dense width and satisfies the requested `4 x d_model` ratio.

## Memory Estimate

Raw event tensor:

```text
100 candidates x 5 fp32 = 2,000 bytes/event
80M events = 160 GB decimal = 149 GiB raw
20M events/rank = 40 GB decimal = 37 GiB raw
```

The current preprocessing keeps raw tensors and preprocessed tensors in host memory. With PID one-hot enabled, the model input is wider than the raw 5 features, so per-rank host memory can be substantially above the 37 GiB raw floor. Use nodes with large host RAM and keep `--augmentation-copies 0` for the full 80M run.

GPU memory is dominated by the 100M-param model plus attention activations. The benchmark found:

| batch/GPU | global batch | peak VRAM | global samples/s | 400-epoch estimate |
|---:|---:|---:|---:|---:|
| 1024 | 4096 | 23.04 GiB | 5,263.37 | 1,688.82 h |
| 1536 | 6144 | 33.80 GiB | 5,267.58 | 1,687.47 h |
| 2048 | 8192 | OOM | - | - |

`1536` per GPU is the largest safe tested batch. `1024` per GPU is the safer long-run choice because it leaves much more VRAM headroom with almost identical throughput.

## Original A/B Run Plan

The exact physics comparison axis for the two 80M runs was not specified, so the large run configs are reproducible replicas of the same recommended point using different seeds. This avoids inventing a new hyperparameter comparison.

| run | seed | objective | augmentation | status |
|---|---:|---|---:|---|
| A | 42 | CE + var/cov | 0 | ready template |
| B | 43 | CE + var/cov | 0 | ready template |

Rationale from the 1M/class evidence: stable runs used `lr=3e-5` and `augmentation_copies=0`; CE+var/cov kept AUC competitive and improved effective latent dimensionality versus CE-only.

## Required Validation Before Full Launch

1. Run single-process smoke at 1k/class.
2. Run 4-rank DDP smoke at 1k/class with `rank_sharded_materialized`.
3. Run the augmentation/no-augmentation smoke pair at 1k/class only.
4. Confirm each rank writes a distinct shard cache.
5. Confirm `run_config.json` has `data_rank_sharded=true`, `world_size=4`, and `loaded_events_per_class=250` for a 1k/class smoke.
6. Run external Higgs evaluation and confirm the output CSV includes `eff_at_99_5pct_bg`.
7. Only then replace `--n-events-per-class 1000` with `20000000`.

## Files

- `configs/run_100m_80m_A.json`
- `configs/run_100m_80m_B.json`
- `commands/smoke_tests.sh`
- `commands/launch_torchrun_100m_80m.sh`
- `commands/evaluate_external_higgs.sh`
- `commands/run_one_24h_benchmark.sh`
- `commands/run_one_24h_smoke_1k.sh`
- `commands/launch_one_24h_100m.sh`
- `commands/benchmark_100m_ddp.py`
- `kubeflow/pytorchjob_run_100m_80m_A.yaml`
- `kubeflow/pytorchjob_run_100m_80m_B.yaml`

For the single 24h test, use `ONE_24H_RUNBOOK.md`. It contains the exact `d_model=2016`, `ff_dim=8064`, benchmark gate, and launch command. The benchmark now shows that the requested 400-epoch / 80M-event run is not physically compatible with a 24h allocation: the best measured estimate is about 70.3 days before real data-loading overhead.

Maciej's decision is to run anyway with checkpoints saved every epoch, then inspect progress around epochs 5, 10, and 15 if the allocation reaches them. The launch script now passes `--checkpoint-every-epochs 1`.

The Kubeflow manifests follow CERN's documented PyTorch distributed-training pattern: `PyTorchJob`, `Master`/`Worker` replicas, persistent storage, and launch through `kubectl apply`.
