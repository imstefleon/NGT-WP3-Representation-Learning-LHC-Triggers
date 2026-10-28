# 1M and 500k Scaling Pilot Analysis

This note records the 500k and 1M/class follow-up tests for the 4-class scaling study. The focus is training stability and MinBias-vs-injected-Higgs anomaly separation. No conclusion below uses missing metrics; unavailable values are marked explicitly.

## Context

The previous 200k study suggested that the most promising stable candidate was:

```text
classes = minbias, qcd, dy, top
latent_dim = 128
ff_dim = 512
attention_blocks = 2
heads = 8
classifier_head = linear
```

However, first 1M attempts exposed two separate issues:

- The original 1M command with augmentation was killed before training because the notebook cgroup memory limit was about 144 GB, even though the node reported much larger total memory.
- After removing augmentation to fit in memory, the default learning-rate runs were unstable and collapsed.

The later 1M pilots therefore tested lower learning rate (`lr=3e-5`) and compared CE-only versus CE + latent variance/covariance regularization.

## 500k Pilot

### Run

```text
run_name = scaling_4class_500k_ff512_blocks2_linear_seed42
events_per_class = 500k
latent_dim = 128
ff_dim = 512
attention_blocks = 2
heads = 8
loss_mode = ce
latent_var_reg_weight = 0.01
latent_cov_reg_weight = 0.001
augmentation_copies = 1
```

### Observation

The run collapsed to random 4-class behavior. The saved evidence records:

```text
train_acc approximately 0.25
val_acc approximately 0.25
train/val loss approximately log(4) = 1.386
var_reg approximately 0.99
cov_reg approximately 0
```

Evidence path:

```text
outputs/scaling_study/main_4class_scaling/4class/500k/ff512_blocks2/runs/2026-08-16/scaling_4class_500k_ff512_blocks2_linear_seed42/failed_run_evidence/
```

### Interpretation

This run is not a valid anomaly-analysis point. It should be treated as a failed/collapsed scaling point. The failure is useful because it shows that simply increasing statistics and keeping the 200k `ff512_blocks2` recipe is not enough; training stability must be controlled.

## 1M Failed/Unstable Attempts

| run | lr | augmentation | status | evidence |
|---|---:|---:|---|---|
| `ff512_blocks2_ce_only_noaug` | default | 0 | collapsed | final `val_acc=0.25`, loss near `log(4)` |
| `ff512_blocks2_ce_varcov` | default | 1 | collapsed | early stopping at epoch 26, final `val_acc=0.25` |
| `ff512_blocks2_ce_varcov_noaug` | default | 0 | unstable/collapsed | epoch 1 reached `val_acc=0.5657`, then collapsed to `0.25` |

Key point:

> The 1M failure was not only a memory issue. Once memory was avoided by disabling augmentation, the default learning-rate recipe still collapsed. Lowering the learning rate was required for stable 1M training.

## 1M Stable Low-LR Runs

All stable 1M runs used:

```text
classes = minbias, qcd, dy, top
events_per_class = 1M
latent_dim = 128
attention_blocks = 2
heads = 8
classifier_head = linear
augmentation_copies = 0
lr = 3e-5
```

### Training Summary

| run | ff_dim | objective | final epoch | final train acc | final val acc | final val loss | status |
|---|---:|---|---:|---:|---:|---:|---|
| `medium_ce_only_lr3e5_noaug` | 256 | CE-only | 50 | 0.6046 | 0.6041 | 0.7758 | stable |
| `medium_varcov_lr3e5_noaug` | 256 | CE + var/cov | 50 | 0.6018 | 0.6022 | 0.7791 | stable |
| `ff512_blocks2_ce_only_lr3e5_noaug` | 512 | CE-only | 50 | 0.6185 | 0.6128 | 0.7629 | stable |
| `ff512_blocks2_varcov_lr3e5_noaug` | 512 | CE + var/cov | 50 | 0.6180 | 0.6127 | 0.7607 | stable |

Interpretation:

- Lowering the learning rate to `3e-5` stabilized all four 1M configurations.
- Increasing `ff_dim` from 256 to 512 improves classifier validation accuracy slightly at 1M.
- CE-only and CE + var/cov have very similar validation accuracy at matched `ff_dim`.
- These are training-quality observations only; the model choice must still be based on the MinBias-vs-anomaly metrics.

## Latent-Dimensionality Diagnostics

| run | effective rank | participation ratio | top1 explained variance | PCs for 90% | PCs for 95% |
|---|---:|---:|---:|---:|---:|
| `medium_ce_only_lr3e5_noaug` | 1.7688 | 1.3422 | 0.8563 | 2 | 2 |
| `medium_varcov_lr3e5_noaug` | 3.6619 | 2.8005 | 0.5028 | 3 | 4 |
| `ff512_blocks2_ce_only_lr3e5_noaug` | 2.2661 | 1.5349 | 0.7969 | 2 | 3 |
| `ff512_blocks2_varcov_lr3e5_noaug` | 6.3331 | 4.1270 | 0.4074 | 6 | 8 |

Interpretation:

- The latent remains low-dimensional relative to the nominal `latent_dim=128`.
- Var/cov regularization increases effective dimensionality in both medium and `ff512` models.
- The largest latent spread is obtained by `ff512_blocks2_varcov_lr3e5_noaug`.
- Effective rank alone is not the anomaly objective; it should be read together with MinBias-distance AUC and efficiencies.

## 1M Anomaly Metrics Available Locally

The external anomaly evaluation completed for all four low-LR no-augmentation runs. The local CSVs are now readable for both `medium` and `ff512_blocks2`.

### Medium CE-only

Path:

```text
outputs/scaling_study/main_4class_scaling/4class/1m/medium_ce_only_lr3e5_noaug/runs/2026-08-16/scaling_4class_1m_medium_ce_only_lr3e5_noaug_seed42/figures/external_higgs_anomaly/external_higgs_anomaly_metrics_by_method.csv
```

| signal | AUC vs SM background | eff@95%BG | eff@99%BG |
|---|---:|---:|---:|
| hh4b | 0.8967 | 0.3970 | 0.1215 |
| hh | 0.8894 | 0.4675 | 0.2295 |
| vbfh | 0.8312 | 0.3530 | 0.1775 |
| ggh | 0.7527 | 0.3385 | 0.1905 |
| mean | 0.8425 | 0.3890 | 0.1797 |

### Medium CE + var/cov

Path:

```text
outputs/scaling_study/main_4class_scaling/4class/1m/medium_varcov_lr3e5_noaug/runs/2026-08-16/scaling_4class_1m_medium_varcov_lr3e5_noaug_seed42/figures/external_higgs_anomaly/external_higgs_anomaly_metrics_by_method.csv
```

| signal | AUC vs SM background | eff@95%BG | eff@99%BG |
|---|---:|---:|---:|
| hh4b | 0.8942 | 0.3860 | 0.1485 |
| hh | 0.8897 | 0.4690 | 0.2880 |
| vbfh | 0.8252 | 0.3520 | 0.2220 |
| ggh | 0.7542 | 0.3420 | 0.2375 |
| mean | 0.8408 | 0.3872 | 0.2240 |

Interpretation for the medium model:

- CE-only and CE + var/cov have nearly identical mean AUC.
- CE + var/cov gives better mean `eff@99%BG` in the available medium comparison.
- Var/cov also increases effective latent dimensionality from `1.7688` to `3.6619`.
- This supports keeping the regularized objective as a candidate, but it does not prove it is better until the `ff512` anomaly metrics are read and compared.

### ff512 CE-only

Path:

```text
outputs/scaling_study/main_4class_scaling/4class/1m/ff512_blocks2_ce_only_lr3e5_noaug/runs/2026-08-16/scaling_4class_1m_ff512_blocks2_ce_only_lr3e5_noaug_seed42/figures/external_higgs_anomaly/external_higgs_anomaly_metrics_by_method.csv
```

| signal | AUC vs SM background | eff@95%BG | eff@99%BG |
|---|---:|---:|---:|
| hh4b | 0.8883 | 0.3685 | 0.1210 |
| hh | 0.8814 | 0.4160 | 0.1955 |
| vbfh | 0.8359 | 0.3255 | 0.1710 |
| ggh | 0.7631 | 0.3260 | 0.1990 |
| mean | 0.8421 | 0.3590 | 0.1716 |

### ff512 CE + var/cov

Path:

```text
outputs/scaling_study/main_4class_scaling/4class/1m/ff512_blocks2_varcov_lr3e5_noaug/runs/2026-08-16/scaling_4class_1m_ff512_blocks2_varcov_lr3e5_noaug_seed42/figures/external_higgs_anomaly/external_higgs_anomaly_metrics_by_method.csv
```

| signal | AUC vs SM background | eff@95%BG | eff@99%BG |
|---|---:|---:|---:|
| hh4b | 0.8933 | 0.3820 | 0.1005 |
| hh | 0.8868 | 0.4380 | 0.2035 |
| vbfh | 0.8372 | 0.3355 | 0.1825 |
| ggh | 0.7607 | 0.3265 | 0.2080 |
| mean | 0.8445 | 0.3705 | 0.1736 |

## Consolidated 1M Low-LR Comparison

| run | ff_dim | objective | mean AUC | eff@95%BG | eff@99%BG | effective rank | PCA dims 95% |
|---|---:|---|---:|---:|---:|---:|---:|
| `medium_ce_only_lr3e5_noaug` | 256 | CE-only | 0.8425 | 0.3890 | 0.1797 | 1.7688 | 2 |
| `medium_varcov_lr3e5_noaug` | 256 | CE + var/cov | 0.8408 | 0.3872 | 0.2240 | 3.6619 | 4 |
| `ff512_blocks2_ce_only_lr3e5_noaug` | 512 | CE-only | 0.8421 | 0.3590 | 0.1716 | 2.2661 | 3 |
| `ff512_blocks2_varcov_lr3e5_noaug` | 512 | CE + var/cov | 0.8445 | 0.3705 | 0.1736 | 6.3331 | 8 |

Interpretation after reading the `ff512` CSVs:

- All four stable 1M low-LR runs are close in mean AUC.
- The best mean AUC is `ff512_blocks2_varcov_lr3e5_noaug` at `0.8445`, but the margin over the medium runs is small.
- The best mean `eff@99%BG` is `medium_varcov_lr3e5_noaug` at `0.2240`.
- Var/cov regularization consistently increases effective latent dimensionality at both widths.
- The `ff512` model does not clearly dominate the medium model on anomaly separation, despite slightly better classifier validation accuracy.

## Classifier Response to External Signals

Across the four stable 1M low-LR runs, external Higgs-like samples are mostly assigned to non-MinBias classes:

- `hh4b` is mostly predicted as `top` (`~0.91-0.94` top fraction).
- `hh` is mostly predicted as `top` (`~0.75-0.84` top fraction).
- `vbfh` and `ggh` are more often split toward `qcd`/`top`.

This is consistent with earlier 4-class behavior: the coarse class setup can push Higgs-like signals into broad non-MinBias regions. Therefore classifier response is useful context, but it should not be interpreted alone as anomaly-detection performance.

## Updated Conclusions

1. The 500k `ff512_blocks2` run with the previous recipe collapsed and should not be used for anomaly conclusions.

2. The initial 1M failures were caused by both resource and optimization issues:
   - memory pressure came from materializing the full dataset and augmentation in RAM;
   - training instability came from the default learning rate.

3. A lower learning rate (`3e-5`) stabilizes 1M/class training for both medium and `ff512` models, even without augmentation.

4. At 1M, increasing `ff_dim` from 256 to 512 improves validation accuracy slightly, but it does not clearly dominate the medium model on the MinBias-distance anomaly metrics.

5. Var/cov regularization increases latent effective dimensionality at both tested widths. It improves medium-model `eff@99%BG`, while leaving mean AUC similar to CE-only.

6. The recommended next step is to choose the 40M pilot configuration based on MinBias-vs-anomaly AUC and fixed-background efficiencies, not classifier accuracy alone. The safest candidates are the low-LR no-augmentation medium CE+var/cov model for high-rejection efficiency, or the low-LR no-augmentation ff512 CE+var/cov model if latent spread is prioritized.

## Pending Items

- If 40M training uses the same script, the data loader should be changed to streaming/chunked loading; the current pipeline materializes too much data in RAM for large-scale training with augmentation.
- For 40M, keep `augmentation_copies=0` unless the loader is changed; augmentation was a memory multiplier in the failed 1M path.
- Keep `lr=3e-5` as the starting point for 40M pilots; default `lr=0.001` caused unstable/collapsed 1M behavior.
