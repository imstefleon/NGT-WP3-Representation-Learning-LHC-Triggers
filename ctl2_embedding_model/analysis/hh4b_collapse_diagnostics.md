# 2026-07-21 HH4B Latent-Collapse Diagnostics

## Purpose

This analysis studies whether the low effective dimensionality observed in the MHA latent representation is stable, seed-dependent, or affected by increasing the dataset size.

The diagnostic work focused on three controlled checks:

- repeated-seed stability at fixed `latent_dim=32`,
- dataset-size scaling at fixed architecture and fixed label granularity,
- classifier-head Fisher diagnostics to compare latent compression with classifier-head sensitivity.

All runs used the updated Collide-2V L1-only inputs.

## Common Configuration

```text
data_source: collide2v_updated
classes: minbias, top, qcd, dy, hh4b
background for eval_quality: minbias + top + qcd + dy
anomaly for eval_quality: hh4b
candidate_selection: top_pt
feature_mode: l1_only
n_constituents: 100
latent_dim: 32
ff_dim: 128
num_heads: 8
pooling_mode: attention
augmentation_copies: 1
loss: cross-entropy
MLflow experiment: ngt_model
```

## Repeated-Seed Stability Test

The repeated-seed test used `100k` events/class and `latent_dim=32`, with seeds `42`, `43`, and `44`.

| seed | accuracy | macro F1 | DY F1 | HH4B F1 | effective rank | participation ratio | eval-quality |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.605 | 0.596 | 0.396 | 0.713 | 5.04 | 3.22 | 0.214 |
| 43 | 0.595 | 0.589 | 0.425 | 0.728 | 4.69 | 2.79 | 0.203 |
| 44 | 0.598 | 0.592 | 0.405 | 0.722 | 4.66 | 2.83 | 0.216 |

The result is reasonably stable across seeds. Accuracy, macro F1, HH4B F1, and eval-quality vary only mildly. This suggests that the observed `latent_dim=32` behavior is not a single-seed artifact.

The latent effective dimensionality is also stable: the entropy effective rank stays around `4.7-5.0`, and the participation ratio stays around `2.8-3.2`.

Classifier-head Fisher was also evaluated for the seed study. The `seed42` Fisher point is taken from the equivalent `100k`, `latent_dim=32`, seed-42 dataset-size run, which has the same controlled setup.

| seed | latent effective rank | latent participation ratio | head Fisher effective rank | head Fisher participation ratio |
|---:|---:|---:|---:|---:|
| 42 | 5.04 | 3.22 | 115.27 | 45.03 |
| 43 | 4.69 | 2.79 | 55.49 | 15.44 |
| 44 | 4.66 | 2.83 | 22.46 | 6.27 |

The classification metrics and latent PCA dimensionality are stable across seeds. The classifier-head Fisher values vary more strongly, but they remain larger than the PCA effective dimensionality in all evaluated seeds. This means the classifier head is sensitive to more parameter directions than the latent covariance spectrum suggests, even when the latent representation itself remains compressed.

## Dataset-Size Scaling Test

The dataset-size sweep fixed the architecture and label granularity while varying only the number of events per class.

| events/class | total events | samples/parameter | accuracy | macro F1 | DY F1 | HH4B F1 | effective rank | participation ratio | eval-quality |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100k | 500k | 2.69 | 0.605 | 0.596 | 0.396 | 0.713 | 5.04 | 3.22 | 0.214 |
| 200k | 1.00M | 5.39 | 0.568 | 0.559 | 0.494 | 0.711 | 3.37 | 2.43 | 0.208 |
| 300k | 1.50M | 8.08 | 0.582 | 0.558 | 0.230 | 0.732 | 2.88 | 1.90 | 0.202 |

Increasing the dataset size from `100k` to `300k` events/class did not improve the representation quality monotonically.

The `100k` run remained the best-balanced point in terms of global classification metrics and latent effective dimensionality. The `300k` run gave the highest HH4B F1, but it also showed the strongest latent compression and the weakest DY F1.

## Latent-Collapse Behavior

The latent effective dimensionality decreased as the dataset size increased:

| events/class | effective rank | participation ratio | components for 90% variance | components for 95% variance |
|---:|---:|---:|---:|---:|
| 100k | 5.04 | 3.22 | 5 | 8 |
| 200k | 3.37 | 2.43 | 4 | 4 |
| 300k | 2.88 | 1.90 | 3 | 4 |

This is an important observation: adding more data did not unlock more latent dimensions. Instead, the representation became more compressed under the same supervised cross-entropy objective.

This supports the interpretation that the bottleneck is not only dataset size. The classification objective and/or architecture may be encouraging a compact class-separation solution.

## Class-Level Behavior

HH4B performance was stable and slightly improved at the largest dataset size:

| events/class | HH4B precision | HH4B recall | HH4B F1 |
|---:|---:|---:|---:|
| 100k | 0.741 | 0.687 | 0.713 |
| 200k | 0.744 | 0.681 | 0.711 |
| 300k | 0.756 | 0.710 | 0.732 |

The dominant HH4B confusion mode remained HH4B-to-Top:

| events/class | HH4B eval events | HH4B predicted as Top |
|---:|---:|---:|
| 100k | 15,000 | 4,193 |
| 200k | 30,000 | 8,430 |
| 300k | 45,000 | 11,393 |

This is physically plausible because HH4B and Top both produce busier L1 topologies than MinBias or DY.

DY remained unstable and strongly affected by MinBias confusion:

| events/class | DY precision | DY recall | DY F1 | DY predicted as MinBias |
|---:|---:|---:|---:|---:|
| 100k | 0.580 | 0.301 | 0.396 | 8,594 / 15,000 |
| 200k | 0.442 | 0.561 | 0.494 | 6,053 / 30,000 |
| 300k | 0.539 | 0.146 | 0.230 | 31,475 / 45,000 |

The `300k` run improved HH4B but substantially degraded DY recall, indicating that more data alone does not resolve the DY-MinBias overlap.

## Classifier-Head Fisher Diagnostic

The classifier-head Fisher diagnostic estimates how broadly the classification loss uses the classifier-head parameters. It is a lightweight Fisher/Hessian-style approximation on the classifier head, not on the full model.

| events/class | latent effective rank | latent participation ratio | head Fisher effective rank | head Fisher participation ratio |
|---:|---:|---:|---:|---:|
| 100k | 5.04 | 3.22 | 115.27 | 45.03 |
| 200k | 3.37 | 2.43 | 67.32 | 26.08 |
| 300k | 2.88 | 1.90 | 39.62 | 16.09 |

The classifier head uses more effective degrees of freedom than the latent PCA spectrum suggests. However, the head Fisher effective rank also decreases as dataset size increases.

This indicates that the collapse is not only a visualization artifact from PCA. The latent representation is geometrically compressed, while the supervised classifier head still spreads sensitivity across more parameters than the latent covariance would imply.

The dataset-size trend and the seed trend together suggest that the latent collapse is robust at the representation level, while the classifier-head Fisher is more sensitive to the trained solution. This makes the head Fisher useful as a diagnostic, but it should not be interpreted alone without the PCA/effective-rank and class-level metrics.

## Training Behavior

| events/class | epochs run | best epoch | best validation loss | best validation accuracy |
|---:|---:|---:|---:|---:|
| 100k | 30 | 27 | 0.816 | 0.606 |
| 200k | 17 | 7 | 0.859 | 0.568 |
| 300k | 25 | 15 | 0.847 | 0.582 |

There is no clear sign that the larger datasets simply need more epochs under the current early-stopping setup. The training and validation metrics remain close, so the issue does not look like a standard overfitting gap.

## Main Findings

- The `100k`, `latent_dim=32` result is stable across seeds.
- Increasing dataset size from `100k` to `300k` events/class does not improve global performance monotonically.
- HH4B F1 is stable and slightly better at `300k`, but HH4B remains strongly confused with Top.
- DY remains unstable and continues to overlap significantly with MinBias.
- Latent effective dimensionality decreases as dataset size increases.
- Head Fisher confirms that the classifier head uses more degrees of freedom than latent PCA suggests, but this also decreases with larger datasets.
- Head Fisher varies more across seeds than the classification metrics, so it should be treated as a complementary diagnostic rather than a standalone quality metric.
- More data alone does not resolve the latent-collapse issue under the current MHA + cross-entropy setup.

## Current Status and Remaining Work

The diagnostic block is now complete enough to support the conclusion that simply increasing data or nominal latent capacity is not solving the latent-collapse behavior.

Completed checks:

- repeated-seed stability at `100k` events/class,
- dataset-size scaling at `100k`, `200k`, and `300k` events/class,
- Chinchilla-style samples/parameter summary,
- latent PCA/effective-dimensionality diagnostics,
- eval-quality comparison,
- classifier-head Fisher diagnostics for dataset-size runs and seed variations,
- unified summary table for completed runs.

Remaining work before changing the model objective:

- decide whether a heavier Hessian/Fisher approximation is still needed, or whether the current Head Fisher is sufficient as a first-order diagnostic,
- use the current results as the baseline for the next controlled objective experiment,
- test anti-collapse latent regularization, starting with cross-entropy plus variance/covariance penalties,
- compare attention pooling against mean pooling under the same `100k`, `latent_dim=32` setup.

## Generated Artifacts

Key summary tables:

```text
ngt_mha_nodes_aligned_outputs/scale_tests/unified_experiment_summary_complete_only.csv
ngt_mha_nodes_aligned_outputs/scale_tests/seed_sweep_hh4b_100k_latent32.csv
ngt_mha_nodes_aligned_outputs/scale_tests/seed_sweep_hh4b_100k_latent32_mean_std.csv
ngt_mha_nodes_aligned_outputs/scale_tests/dataset_size_sweep_hh4b_latent32_100k.csv
ngt_mha_nodes_aligned_outputs/scale_tests/dataset_size_sweep_hh4b_latent32_200k.csv
ngt_mha_nodes_aligned_outputs/scale_tests/dataset_size_sweep_hh4b_latent32_300k.csv
ngt_mha_nodes_aligned_outputs/scale_tests/hh4b_collapse_diagnostics_runs_only.csv
ngt_mha_nodes_aligned_outputs/scale_tests/chinchilla_style_dataset_size_summary.csv
```

Presentation figures:

```text
ngt_mha_nodes_aligned_outputs/scale_tests/figures/dataset_size_scaling_summary.png
ngt_mha_nodes_aligned_outputs/scale_tests/figures/seed_stability_summary.png
ngt_mha_nodes_aligned_outputs/scale_tests/figures/latent_vs_head_fisher_dimensionality.png
```

Main run folders:

```text
ngt_mha_nodes_aligned_outputs/runs/2026-07-21/dataset_size_sweep_hh4b_latent32_100k_n100000
ngt_mha_nodes_aligned_outputs/runs/2026-07-21/dataset_size_sweep_hh4b_latent32_200k_n200000
ngt_mha_nodes_aligned_outputs/runs/2026-07-21/dataset_size_sweep_hh4b_latent32_300k_n300000
ngt_mha_nodes_aligned_outputs/runs/2026-07-21/seed_sweep_hh4b_100k_latent32_seed42
ngt_mha_nodes_aligned_outputs/runs/2026-07-21/seed_sweep_hh4b_100k_latent32_seed43
ngt_mha_nodes_aligned_outputs/runs/2026-07-21/seed_sweep_hh4b_100k_latent32_seed44
```
