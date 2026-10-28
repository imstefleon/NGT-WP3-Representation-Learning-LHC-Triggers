# Main 4-Class Scaling Runs

This directory contains the focused `4class` scaling follow-up after the broader class-diversity and probe-ablation studies.

## Layout

| path | content |
|---|---|
| `4class/50k` | 50k/class baselines and no-augmentation checks |
| `4class/100k` | 100k/class no-augmentation checks |
| `4class/150k` | 150k/class width, depth, learning-rate, and no-augmentation variants |
| `4class/200k` | 200k/class width, depth, augmentation, and learning-rate variants |
| `4class/500k` | 500k/class failed/collapsed pilot evidence |
| `4class/1m` | 1M/class pilots, including failed default-LR attempts and stable low-LR no-augmentation runs |
| `logs` | copied or legacy launch/evaluation logs |

## Current Interpretation

Use the full-dimensional `minbias_mahalanobis` score as the primary anomaly metric. PCA/t-SNE and effective-rank diagnostics are supporting checks.

The direct 200k `ff512_blocks2` augmentation comparison is:

| run | augmentation | lr | mean AUC | eff@95%BG | eff@99%BG | effective rank |
|---|---:|---:|---:|---:|---:|---:|
| `scaling_4class_200k_ff512_blocks2_linear_seed42` | 1 | 0.001 | 0.7869 | 0.1630 | 0.0535 | 3.8477 |
| `scaling_4class_200k_ff512_blocks2_aug0_seed42` | 0 | 0.001 | 0.7760 | 0.1906 | 0.0774 | 2.7627 |

At 1M/class, the usable points are the low-learning-rate no-augmentation runs:

| run | ff_dim | objective | mean AUC | eff@95%BG | eff@99%BG | effective rank |
|---|---:|---|---:|---:|---:|---:|
| `scaling_4class_1m_medium_ce_only_lr3e5_noaug_seed42` | 256 | CE-only | 0.8425 | 0.3890 | 0.1797 | 1.7688 |
| `scaling_4class_1m_medium_varcov_lr3e5_noaug_seed42` | 256 | CE + var/cov | 0.8408 | 0.3872 | 0.2240 | 3.6619 |
| `scaling_4class_1m_ff512_blocks2_ce_only_lr3e5_noaug_seed42` | 512 | CE-only | 0.8421 | 0.3590 | 0.1716 | 2.2661 |
| `scaling_4class_1m_ff512_blocks2_varcov_lr3e5_noaug_seed42` | 512 | CE + var/cov | 0.8445 | 0.3705 | 0.1736 | 6.3331 |

The default-LR 1M attempts with augmentation or without low-LR tuning are not valid anomaly-analysis points. They are retained as stability evidence.

## Summary File

The compact table for the current focused scaling points is:

```text
../summaries/main_4class_scaling_minbias_summary.csv
```
