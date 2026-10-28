# One 24h 100M / 80M Runbook

Do not launch the full run until the 4-GPU benchmark has completed on the target node.

## Architecture

The previous MHA model used `ff_dim` as the attention/model dimension. To satisfy the requested transformer-style ratio, the model now supports optional `d_model`; when `d_model` is set, attention runs at `d_model` and the post-pool feed-forward block uses `ff_dim`.

Backward compatibility is preserved: if `--d-model` is omitted, old runs keep the previous behavior.

Target constraints:

```text
ff_dim = 4 * d_model
num_heads = 16
num_attention_blocks = 4
n_features = 5
classifier_head = linear
latent_dim = 128
```

Closest valid point to 100M trainable parameters:

| d_model | ff_dim | trainable params | distance from 100M |
|---:|---:|---:|---:|
| 2016 | 8064 | 99,889,413 | -110,587 |

`d_model=2016` is divisible by 16.

## Data

Command settings:

```text
--data-loading-mode rank_sharded_materialized
--ddp-timeout-minutes 2880
--pid-encoding raw
--n-events-per-class 20000000
--augmentation-copies 0
```

With 4 classes and 4 DDP ranks:

```text
80M loaded events total
20M loaded events/rank
5M events/class/rank
~79.968M effective training events after the tiny internal val/eval holdout
```

Rank sharding uses deterministic valid-event modulo sharding:

```text
valid_global_event_index % WORLD_SIZE == rank
```

So ranks receive disjoint shards and every cache is named with `shardXXXXofYYYY`.

The large launch uses a 48h DDP process-group timeout because some ranks can finish cached classes much earlier than others and then wait at the first distributed normalization collective. The default NCCL watchdog timeout is too short for the initial rank-sharded cache build.

Host RAM estimate per rank:

```text
raw tensor floor: 20M * 100 * 5 * 4 bytes = 40 GB decimal = 37 GiB
preprocessed 5-feature tensor: another ~37 GiB
split copies and temporary concatenation can push practical per-rank host RAM above 120-180 GiB
```

On a single 4-GPU node, budget at least ~700 GiB host RAM. More is safer. This estimate is why `augmentation_copies=0` is mandatory.

## 24h Feasibility Gate

400 epochs over 80M events implies:

```text
400 * 80M = 32B event presentations
32B / 24h = 370,370 global samples/sec required
```

If the benchmark reports global throughput below ~370k samples/sec, the full 400-epoch job cannot finish in 24 hours. Do not silently reduce epochs or data size; report the predicted runtime.

## Benchmark First

Run a real-data 1k/class smoke before the synthetic throughput sweep:

```bash
export PYTHONPATH=$PWD/ngt_model/src:$PYTHONPATH
export MPLBACKEND=Agg
SMOKE_BATCH_SIZE=1 bash ngt_model/scaling_study/large_100m_80m/commands/run_one_24h_smoke_1k.sh
```

This checks:

- 4-GPU DDP starts correctly;
- `d_model=2016`, `ff_dim=8064`, 16 heads, 4 blocks instantiate correctly;
- raw 5-feature preprocessing works;
- rank-sharded loading creates disjoint 250-event/class/rank shards from the 1k/class global request;
- `augmentation_copies=0`;
- one epoch can complete and write a run directory.

From `ctl2_embedding_model` on the 4 x 48 GB node:

```bash
export PYTHONPATH=$PWD/ngt_model/src:$PYTHONPATH
export MPLBACKEND=Agg
bash ngt_model/scaling_study/large_100m_80m/commands/run_one_24h_benchmark.sh
```

Optional candidate list:

```bash
BATCH_SIZES=1,2,4,8,16,24,32 bash ngt_model/scaling_study/large_100m_80m/commands/run_one_24h_benchmark.sh
```

Benchmark output:

```text
ngt_model/scaling_study/large_100m_80m/benchmark_100m_ddp_results.csv
```

Required columns:

```text
peak_vram_gib
samples_per_sec_per_gpu
global_samples_per_sec
step_time_s
estimated_epoch_hours_80m
estimated_total_hours_400_epochs
```

Choose the largest batch that:

- does not OOM on any rank;
- keeps peak VRAM comfortably below 48 GiB;
- improves global throughput versus smaller batches;
- does not push estimated runtime further from the 24h window.

Current benchmark result:

| batch/GPU | global batch | peak VRAM | global samples/s | epoch time over 80M | 400-epoch runtime |
|---:|---:|---:|---:|---:|---:|
| 1024 | 4096 | 23.04 GiB | 5,263.37 | 4.22 h | 1,688.82 h |
| 1536 | 6144 | 33.80 GiB | 5,267.58 | 4.22 h | 1,687.47 h |
| 2048 | 8192 | OOM | - | - | - |

`1536` per GPU is the largest safe tested batch. `1024` per GPU is the safer long-run batch because it keeps about 25 GiB of VRAM headroom per 48 GB GPU while losing only about 0.08% throughput.

With the best measured throughput, the 400-epoch / 80M-event run needs about 70.3 days before real data-loading overhead.

Maciej's decision is to launch anyway under the 24h allocation, save checkpoints every epoch, and inspect progress after about 5, 10, and 15 epochs if the allocation allows the job to reach them. If the loss/metrics are already saturated, the job can be killed early.

## Full Launch Command

Only after benchmark selection and only if a longer allocation is available:

```bash
bash ngt_model/scaling_study/large_100m_80m/commands/launch_one_24h_100m.sh 1536
```

For a safer memory margin with nearly identical throughput:

```bash
bash ngt_model/scaling_study/large_100m_80m/commands/launch_one_24h_100m.sh 1024
```

## Recovery After Var/Cov Collapse

The first cached-data training attempt reached epoch 2 but the latent covariance regularizer exploded:

```text
epoch 1: val_loss=1.3053, val_acc=0.3376, cov_reg=2.4580
epoch 2: val_loss=178,722,949.5040, val_acc=0.2500, cov_reg=178,721,837,580.2880
```

This indicates numerical instability in the `CE + var/cov` objective at the 100M scale. The caches were already being reused, so a relaunch should not repeat the parquet read.

Maciej clarified that this test should match the original objective family: remove all regularizers and use `CE + InfoNCE` only. In this codebase, InfoNCE is exposed through the supervised contrastive loss flags:

```text
--supcon-weight 0.05
--supcon-temperature 0.1
```

Recommended clean relaunch:

```bash
bash ngt_model/scaling_study/large_100m_80m/commands/launch_one_24h_100m_ce_infonce.sh 1024
```

This keeps the same 100M architecture and 80M cached dataset, but uses:

```text
CE + InfoNCE/SupCon
supcon_weight = 0.05
supcon_temperature = 0.1
lr = 1e-5
grad_clip_norm = 0.5
batch/GPU = 1024 recommended
no latent var/cov regularization
no VICReg
```

If a CE-only checkpoint is intentionally used as a warm start, launch:

```bash
bash ngt_model/scaling_study/large_100m_80m/commands/resume_one_24h_100m_ce_infonce_from_checkpoint.sh 1024 CHECKPOINT_PATH START_EPOCH
```

For a pure CE+InfoNCE test, prefer the clean relaunch rather than resuming from a CE-only checkpoint.

Older CE-only recovery launch, kept only as a fallback baseline:

```bash
bash ngt_model/scaling_study/large_100m_80m/commands/launch_one_24h_100m_safe_ce.sh 1024
```

The full command uses:

```text
400 epochs
checkpoint every epoch
cosine LR schedule
lr = 3e-5
CE + var/cov
rank-sharded data
raw 5-feature PID representation
```

## Post-Training Evaluation

After training finishes:

```bash
bash ngt_model/scaling_study/large_100m_80m/commands/evaluate_external_higgs.sh RUN_DIR
```

The evaluation writes VBFH and HH4B anomaly metrics, including:

```text
eff_at_95pct_bg
eff_at_99pct_bg
eff_at_99_5pct_bg
```

plus latent t-SNE/PCA, anomaly-score distributions, and classifier-response outputs.
