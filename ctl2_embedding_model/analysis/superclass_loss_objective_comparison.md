# Superclass Loss-Objective Comparison

## Scope

This note summarizes the six loss-objective runs performed on the expanded Standard Model superclass setup with HH4B as the anomaly candidate.

All runs use the updated Collide-2V L1-only inputs:

```text
[pt, eta, phi, D0/dxy, PID]
```

The common training setup was:

```text
classes = minbias, qcd, dy, top, wjets, ww, wz, ttw, ttz, hh4b
n_events_per_class = 100000
latent_dim = 64
ff_dim = 128
num_heads = 8
num_attention_blocks = 2
pooling_mode = mean
classifier_head = linear
latent_var_reg_weight = 0.01
latent_var_target = 1.0
MLflow tracking = enabled
```

The goal was to test whether changing the training objective can reduce latent collapse or improve the representation for anomaly-detection-style evaluation, without changing the architecture.

## Artifact Validation Status

The six run folders contain saved latent arrays, logits, regenerated training histories, Mahalanobis summaries, latent effective-dimensionality diagnostics, and Head-Fisher files. However, the original `run_config.json`, `classification_report_eval.csv`, and `confusion_matrix_eval.csv` files are not present in these six folders.

Therefore:

- Classification metrics in this note come from the unified summary table and recovered log-derived values.
- Latent effective rank and participation ratio are supported by the saved `latent_effective_dim_summary.json` files.
- Head-Fisher was recomputed for all six runs using the intended 10-class superclass label set:

```text
minbias, qcd, dy, top, wjets, ww, wz, ttw, ttz, hh4b
```

The Head-Fisher values in this note are therefore valid for the 10-class setup.

## Runs Compared

| Run | Loss objective | Main change |
|---|---|---|
| `sm_superclasses_labelsmooth005_var001_100k_latent64` | label-smoothed CE + variance | Softens supervised labels with smoothing = 0.05 |
| `sm_superclasses_focal2_var001_100k_latent64` | focal loss + variance | Emphasizes harder examples with gamma = 2.0 |
| `sm_superclasses_ce_supcon005_var001_100k_latent64` | CE + SupCon + variance | Adds supervised contrastive loss with weight = 0.05 |
| `sm_superclasses_ce_var001_cov0001_100k_latent64` | CE + variance + covariance | Adds latent covariance regularization with weight = 0.001 |
| `sm_superclasses_ce_supcon010_var001_100k_latent64` | CE + SupCon + variance | Stronger supervised contrastive loss with weight = 0.10 |
| `sm_superclasses_ce_var001_cov0005_100k_latent64` | CE + variance + covariance | Stronger covariance regularization with weight = 0.005 |

## Summary Metrics

| Run | Accuracy | Macro F1 | DY F1 | HH4B F1 | Eval-quality |
|---|---:|---:|---:|---:|---:|
| label smoothing 0.05 + var 0.01 | **0.4747** | **0.4677** | 0.4794 | **0.5613** | 0.000039 |
| focal gamma 2.0 + var 0.01 | 0.4600 | 0.4543 | 0.4333 | 0.5567 | 0.1476 |
| CE + SupCon 0.05 + var 0.01 | 0.4602 | 0.4560 | **0.5189** | 0.5612 | 0.2207 |
| CE + var 0.01 + cov 0.001 | 0.4556 | 0.4468 | 0.3974 | 0.5566 | **0.2294** |
| CE + SupCon 0.10 + var 0.01 | 0.4662 | 0.4580 | 0.4174 | 0.5609 | 0.1937 |
| CE + var 0.01 + cov 0.005 | 0.4716 | 0.4644 | 0.4981 | 0.5599 | 0.1318 |

## Latent and Head-Fisher Diagnostics

| Run | Latent effective rank | Latent participation ratio | Head-Fisher effective rank | Head-Fisher participation ratio |
|---|---:|---:|---:|---:|
| label smoothing 0.05 + var 0.01 | 11.17 | 8.32 | 79.63 | 41.61 |
| focal gamma 2.0 + var 0.01 | 8.08 | 5.28 | 51.63 | 27.32 |
| CE + SupCon 0.05 + var 0.01 | 2.84 | 1.82 | 23.87 | 13.07 |
| CE + var 0.01 + cov 0.001 | **29.12** | **20.19** | **445.55** | **302.15** |
| CE + SupCon 0.10 + var 0.01 | 3.57 | 2.27 | 22.14 | 14.50 |
| CE + var 0.01 + cov 0.005 | 27.47 | 16.42 | 254.24 | 125.83 |

The validated latent and Head-Fisher diagnostics are available for all six runs. Both diagnostics point to the covariance-regularized runs as the strongest dimensional activation, especially `cov_weight=0.001`.

## Interpretation

### Classification performance

Label smoothing gave the best supervised classification metrics:

```text
accuracy = 0.4747
macro F1 = 0.4677
HH4B F1 = 0.5613
```

This suggests that softening the target labels helps the classifier generalize slightly better in the expanded 10-class setup. It also increases the latent effective rank compared with the previous CrossEntropy baseline. However, its eval-quality score is nearly zero, which means that the anomaly-quality metric does not view this latent geometry as useful for HH4B-vs-SM anomaly separation.

### Anomaly-detection representation quality

The best eval-quality score came from:

```text
sm_superclasses_ce_var001_cov0001_100k_latent64
eval_quality = 0.2294
```

This supports the idea that covariance/decorrelation regularization is useful for anomaly-detection-style representation learning. It improves the latent-space quality metric more than label smoothing or focal loss, even though the classification metrics are lower.

This same run also has the strongest dimensional activation:

```text
effective rank = 29.12
participation ratio = 20.19
head-Fisher effective rank = 445.55
head-Fisher participation ratio = 302.15
```

This is the clearest evidence from the current validated diagnostics that covariance regularization can directly fight latent collapse.

### Best compromise

The most balanced run is:

```text
sm_superclasses_ce_supcon005_var001_100k_latent64
```

It has:

```text
DY F1 = 0.5189
HH4B F1 = 0.5612
eval_quality = 0.2207
```

This run does not win in global accuracy, but it gives the best DY F1 and preserves HH4B classification while improving the anomaly-quality score. Its latent effective rank remains low, so its benefit seems to come more from class-aware separation than from globally activating many latent dimensions.

### Focal loss

Focal loss did not clearly improve the model:

```text
accuracy = 0.4600
macro F1 = 0.4543
HH4B F1 = 0.5567
eval_quality = 0.1476
```

This is not surprising because the training setup is class-balanced. Focal loss is usually more useful when easy examples dominate or when class imbalance is severe.

### Stronger regularization was not better

Increasing SupCon weight from 0.05 to 0.10 reduced eval-quality:

```text
SupCon 0.05: eval_quality = 0.2207
SupCon 0.10: eval_quality = 0.1937
```

Increasing covariance weight from 0.001 to 0.005 also reduced eval-quality:

```text
cov 0.001: eval_quality = 0.2294
cov 0.005: eval_quality = 0.1318
```

Both covariance runs activated many more dimensions than the baseline, but the weaker covariance penalty was better:

```text
cov 0.001: effective rank = 29.12, head-Fisher effective rank = 445.55
cov 0.005: effective rank = 27.47, head-Fisher effective rank = 254.24
```

This suggests that simply forcing more dimensions to be active is not sufficient; the active dimensions must preserve physically useful separation. The best result is a moderate covariance penalty, not the strongest one.

## Main Findings

- Changing the objective does affect the latent representation.
- Label smoothing is best for supervised classification.
- CE + variance + covariance with `cov_weight=0.001` is best for anomaly-detection-style eval-quality.
- CE + SupCon with `supcon_weight=0.05` is the best compromise between HH4B performance and latent quality.
- Focal loss is not the strongest candidate in this balanced training setup.
- Stronger SupCon or covariance weights do not improve the result and can over-regularize the latent space.
- Covariance regularization is the clearest objective-level intervention against latent collapse in both latent and Head-Fisher diagnostics.
- Label smoothing improves supervised classification but does not improve the anomaly-quality score.
- SupCon improves DY F1, but does not strongly activate the global latent dimensionality.
- More active latent dimensions alone are not enough; the dimensions must encode useful physics separation.

## Working Conclusion

The current evidence supports the interpretation that latent collapse is partly objective-driven. CrossEntropy alone is not enough to preserve a rich latent geometry. Adding variance, covariance, or contrastive terms can change the representation, but there is a trade-off between classification metrics and anomaly-detection-style latent quality.

For the current superclass HH4B setup:

```text
Best classification candidate: label smoothing + variance
Best anomaly-quality candidate: CE + variance + covariance, cov_weight=0.001
Best balanced candidate: CE + SupCon 0.05 + variance
```
