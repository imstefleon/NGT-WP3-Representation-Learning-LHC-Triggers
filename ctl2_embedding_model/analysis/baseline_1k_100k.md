# 2026-07-01 Baseline Runs [HF Collide-2V]: 1k and 100k Events Per Class

Notebook:

`NGT_Linformer_NodesAligned_FullPrecision_Collide2V_L1features.ipynb`

## Context

These runs validated the full-precision Linformer workflow aligned with `ctl2_model`, using four Collide-2V classes:

- `minbias`
- `top`
- `qcd`
- `dy`

Dataset source:

```text
HF Collide-2V / parquet dataset path used by the original notebook loader
```

The goal was to verify data loading, per-event tensor construction, training-split-only normalization, training-split-only PID mapping, supervised training, latent embedding extraction, and Mahalanobis anomaly scoring with MinBias as the normal class.

## 1k Smoke Test

Configuration:

```text
N_EVENTS_PER_CLASS = 1,000
Total events = 4,000
Split = 70/15/15
Train = 2,800
Val = 600
Eval = 600
```

Model:

```text
ff_dim = 32
latent_dim = 12
num_heads = 2
proj_k = 8
pooling = mean pooling
USE_AUGMENTATION = False
```

This run was a local sanity check. Its purpose was to confirm that the full pipeline worked end to end, not to establish final performance.

### Classification Results

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| top | 0.84 | 0.93 | 0.88 |
| qcd | 0.75 | 0.73 | 0.74 |
| minbias | 0.45 | 0.70 | 0.55 |
| dy | 0.46 | 0.17 | 0.25 |

Global accuracy:

```text
0.63
```

Main finding:

```text
113/150 DY events were classified as MinBias
DY -> MinBias = 75.3%
```

Interpretation:

- `top` and `qcd` were already reasonably separable.
- The dominant confusion appeared between `dy` and `minbias`.
- The DY-MinBias problem was already visible with a small dataset.

### Mahalanobis

Normal class:

```text
MinBias
```

| Class | Mean | Std | Median |
|---|---:|---:|---:|
| minbias | -38.7 | 12.8 | -42.7 |
| top | 3.66M | 1.0M | 3.91M |
| qcd | 786k | 1.4M | 3,994 |
| dy | 28k | 212k | -42.6 |

Reading:

- `top` is very far from MinBias.
- `qcd` and `dy` have high-score tails, but their median can remain close to MinBias.
- The median `dy` score is almost identical to MinBias, indicating overlap in the latent space.

## 100k Production Run

Run:

```text
ngt_linformer_nodes_aligned_outputs/runs/2026-07-01/n100000_142733/
```

Configuration:

```text
N_EVENTS_PER_CLASS = 100,000
Total events = 400,000
Split = 70/15/15
Train = 280,000
Val = 60,000
Eval = 60,000
Eval per class = 15,000
```

Model:

```text
ff_dim = 32
latent_dim = 12
num_heads = 2
proj_k = 8
pooling = mean pooling
USE_AUGMENTATION = False
```

This run tested whether the DY-MinBias confusion observed in the smoke test was only a low-statistics artifact.

### Classification Results

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| top | 0.93 | 0.95 | 0.94 |
| qcd | 0.83 | 0.83 | 0.83 |
| minbias | 0.50 | 0.86 | 0.63 |
| dy | 0.67 | 0.16 | 0.26 |

Global accuracy:

```text
0.70
```

Main finding:

```text
11,674/15,000 DY events were classified as MinBias
DY -> MinBias = 77.8%
```

Interpretation:

- Scaling from 1k to 100k events per class improved `top`, `qcd`, and global accuracy.
- DY-MinBias confusion did not improve.
- The problem was not simply caused by limited statistics.

### Mahalanobis

| Class | Mean | Std | Median |
|---|---:|---:|---:|
| minbias | -17.6 | 30.2 | -21.6 |
| top | 48,659 | 36,827 | 41,902 |
| qcd | 1,463 | 6,818 | 48.9 |
| dy | 152 | 1,192 | -21.7 |

Reading:

- `top` is clearly separated from MinBias.
- `qcd` is partially separated.
- `dy` has a median almost identical to MinBias.
- A typical DY event still lands in the MinBias-like latent region.

## 1k vs 100k Comparison

| Metric | 1k/class | 100k/class |
|---|---:|---:|
| Total events | 4,000 | 400,000 |
| Eval events | 600 | 60,000 |
| Global accuracy | 0.63 | 0.70 |
| F1 top | 0.88 | 0.94 |
| F1 qcd | 0.74 | 0.83 |
| F1 minbias | 0.55 | 0.63 |
| F1 dy | 0.25 | 0.26 |
| DY -> MinBias | 75.3% | 77.8% |

## Conclusion

The baseline runs show that:

- The pipeline works end to end.
- `top` and `qcd` are learned reasonably well.
- DY-MinBias separability is the main bottleneck.
- Increasing statistics from 1k to 100k does not solve DY-MinBias confusion.
- The learned latent space does not separate typical DY events from MinBias.
