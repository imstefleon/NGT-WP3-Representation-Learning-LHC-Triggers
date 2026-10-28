# 100M DDP Benchmark Analysis

Input CSV:

```text
scaling_study/large_100m_80m/benchmark_100m_ddp_results.csv
```

## Benchmark Result

Architecture under test:

```text
d_model=2016
ff_dim=8064
num_heads=16
num_attention_blocks=4
trainable_params=99,889,413
```

Best safe tested batch:

| batch/GPU | global batch | peak VRAM | per-GPU samples/s | global samples/s | step time | epoch time over 80M | 400-epoch runtime |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1536 | 6144 | 33.80 GiB | 1,316.90 | 5,267.58 | 1.166 s | 4.22 h | 1,687.47 h |

The next tested point, `2048` per GPU, failed with CUDA OOM after reaching about `43.02 GiB`, so `1536` per GPU is the largest safe tested batch.

Throughput is already saturated by `1024` per GPU:

| batch/GPU | peak VRAM | global samples/s | 400-epoch runtime |
|---:|---:|---:|---:|
| 1024 | 23.04 GiB | 5,263.37 | 1,688.82 h |
| 1536 | 33.80 GiB | 5,267.58 | 1,687.47 h |

So `1536` is the largest safe benchmarked batch, but `1024` is the lower-risk batch with essentially the same throughput.

## 24h Gate

The requested full run is:

```text
400 epochs * 80M events = 32B event presentations
```

To finish in 24 hours, required global throughput is:

```text
32B / 24h = 370,370 global samples/s
```

The best measured throughput is:

```text
5,267.58 global samples/s
```

Required speedup:

```text
370,370 / 5,267.58 = 70.3x
```

Conclusion:

> With the current measured throughput, the 400-epoch / 80M-event run cannot finish in 24 hours. The measured estimate is about 70.3 days, and real data loading/checkpoint overhead can only increase that.

## Practical Implication

At the best tested throughput, 24 hours covers only:

```text
~5.69 epochs over 80M events
```

Equivalently, a 24h budget covers:

```text
~455.9M event presentations total
```

If 400 epochs must be kept inside 24h at this throughput, the dataset would need to be about:

```text
~1.14M events total
~285k events/class for 4 classes
```

That is a different experiment and should not be substituted silently for the requested 80M-event run.

## Batch-Size Recommendation

For the exact "largest safe per-GPU batch" requested by the benchmark, use:

```text
batch_size_per_gpu=1536
effective_global_batch_size=6144
```

For the most stable long run, prefer:

```text
batch_size_per_gpu=1024
effective_global_batch_size=4096
```

because it leaves about `25 GiB` of VRAM headroom per 48 GB GPU while losing only `0.08%` throughput versus `1536`.

## Launch Gate

The benchmark says the full 400-epoch / 80M-event job will not finish under a 24h allocation.

Maciej's decision is to launch anyway, save checkpoints every epoch, and inspect progress after about 5, 10, and 15 epochs if the allocation reaches them. If the model is already saturated, kill the job early.

If a longer allocation is approved and the goal is maximum benchmarked throughput:

```bash
bash scaling_study/large_100m_80m/commands/launch_one_24h_100m.sh 1536
```

If a longer allocation is approved and the goal is safer memory headroom:

```bash
bash scaling_study/large_100m_80m/commands/launch_one_24h_100m.sh 1024
```
