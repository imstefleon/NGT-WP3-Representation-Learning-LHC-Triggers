# Granular Label Study, Latent Sweep, and Best Configuration

## Scope

This note summarizes the study performed after the latent-collapse analysis. The goal was to test Maciej's hypothesis that the latent representation may be collapsing close to the number of supervised classes, and that adding more granular physics labels could activate more latent dimensions.

All runs use the updated Collide-2V L1-only inputs:

```text
[pt, eta, phi, D0/dxy, PID]
```

The main changes studied were:

- Adding more granular physics labels for top and diboson final states.
- Reusing the best configuration found from the architecture and variance-regularization studies.
- Comparing nominal latent sizes `32`, `64`, and `128` under the granular-label setup.

## Dataset Granularity Added

The class loader was extended to support granular process labels from updated Collide-2V:

```text
minbias
qcd
dy
tt_hadronic
tt_leptonic
tt_semileptonic
ww_hadronic
ww_leptonic
ww_semileptonic
hh4b
```

The available parquet files were confirmed for each class:

| Class | Parquet files |
|---|---:|
| minbias | 10,500 |
| qcd | 16,014 |
| dy | 2,348 |
| tt_hadronic | 2,730 |
| tt_leptonic | 2,730 |
| tt_semileptonic | 2,730 |
| ww_hadronic | 326 |
| ww_leptonic | 326 |
| ww_semileptonic | 326 |
| hh4b | 210 |

This made it possible to test whether a richer label space would encourage the model to use more latent directions.

## Best Configuration Used

The granular runs reused the strongest configuration found in the previous collapse studies:

```text
latent_dim = 32 / 64 / 128
ff_dim = 128
num_heads = 8
num_attention_blocks = 2
pooling_mode = mean
classifier_head = linear
latent_var_reg_weight = 0.01
latent_var_target = 1.0
n_events_per_class = 100000
epochs = 100
early_stopping_patience = 10
```

This configuration was selected because the earlier 5-class study showed that `latent_var_reg_weight=0.01` gave the best overall trade-off between classification performance and latent dimensional usage:

| Run | Accuracy | Macro F1 | DY F1 | HH4B F1 | Effective rank | Participation ratio | Eval-quality |
|---|---:|---:|---:|---:|---:|---:|---:|
| CrossEntropy baseline | 0.6187 | 0.6123 | 0.4429 | 0.7134 | 4.03 | 2.85 | 0.1798 |
| `var_reg=0.01`, 100 epochs | 0.6244 | 0.6196 | 0.4734 | 0.7166 | 6.55 | 4.29 | 0.1961 |
| `var_reg=0.03` | 0.6126 | 0.6036 | 0.4163 | 0.7105 | 5.46 | 3.44 | 0.1890 |

The selected configuration improves the effective latent rank relative to pure CrossEntropy while preserving or slightly improving classification.

## Granular Latent-Dimension Sweep

The granular-label experiment was run with `latent_dim = 32, 64, 128`.

| latent_dim | Accuracy | Macro F1 | DY F1 | HH4B F1 | Eval-quality |
|---:|---:|---:|---:|---:|---:|
| 32 | 0.4668 | 0.4575 | 0.4571 | 0.5562 | 0.1697 |
| 64 | 0.4561 | 0.4465 | 0.4407 | 0.5513 | 0.1828 |
| 128 | 0.4642 | 0.4553 | 0.4980 | 0.5504 | 0.0008 |

With the updated `latent_dim=64` result, the best classification balance is now `latent_dim=32`. The `latent_dim=64` run has slightly lower classification metrics but a better eval-quality score than `latent_dim=32`.

The 128-dimensional run does not improve classification, despite having much more nominal latent capacity. Its eval-quality score is also unstable and nearly zero, suggesting that the larger latent space is not automatically organized into a better anomaly-separation geometry.

## Latent Dimensionality

The PCA/effective-rank diagnostics show that adding granular labels activates more latent directions than the previous 5-class setup.

| Setup | latent_dim | Effective rank | Participation ratio | PCs for 90% variance | PCs for 95% variance |
|---|---:|---:|---:|---:|---:|
| 5-class best config | 32 | 6.55 | 4.29 | 6 | 12 |
| granular labels | 32 | 10.49 | 7.13 | 9 | 14 |
| granular labels | 64 | 12.10 | 6.77 | 14 | 21 |
| granular labels | 128 | 16.62 | 9.65 | 17 | 24 |

All three granular runs support the hypothesis that more granular labels activate more latent directions than the earlier 5-class setup.

## Head-Fisher Diagnostics

The classifier-head Fisher diagnostics also show that the granular setup activates substantially more directions than the earlier 5-class setup.

| Setup | latent_dim | Head-Fisher effective rank | Head-Fisher participation ratio |
|---|---:|---:|---:|
| 5-class best config | 32 | 21.04 | 13.57 |
| granular labels | 32 | 44.99 | 23.53 |
| granular labels | 64 | 66.48 | 25.01 |
| granular labels | 128 | 82.26 | 32.63 |

This is an important result: even when classification accuracy is not high, the classifier becomes sensitive to many more latent directions once the labels are more granular.

## Per-Class Behavior

The granular task is significantly harder than the previous 5-class task. Accuracy drops from approximately `0.62` in the best 5-class setup to approximately `0.46-0.47` in the 10-class setup.

The most difficult categories are the granular final states, especially the WW and tt submodes:

- `ww_hadronic`, `ww_semileptonic`, and `tt_semileptonic` have low F1 scores.
- `hh4b` remains visible, but its F1 drops from approximately `0.72` in the 5-class setup to approximately `0.55-0.56` in the granular setup.
- `dy` is best in the `latent_dim=128` granular run, while the `latent_dim=64` run is lower than both `32` and `128` for DY F1.

This behavior is physically plausible because several granular categories have similar busy L1 topologies and may not be cleanly separable using only the current L1 candidate representation.

## Main Findings

1. Adding granular physics labels does activate more of the latent space.

The 5-class best configuration used an effective rank of about `6.55`. With granular labels, the effective rank rises to `10.49` at `latent_dim=32`, `12.10` at `latent_dim=64`, and `16.62` at `latent_dim=128`.

2. The classifier head also uses more dimensions.

Head-Fisher effective rank increases from about `21` in the 5-class setup to about `45` for granular `latent_dim=32`, about `66` for granular `latent_dim=64`, and about `82` for granular `latent_dim=128`.

3. More active latent dimensions do not automatically improve classification.

The 10-class granular task has much lower accuracy and macro F1 than the original 5-class setup. This suggests that the model is using more representational capacity, but the L1-only information may still be insufficient to cleanly separate all final states.

4. There is no single dominant granular configuration yet.

The `latent_dim=32` run has the best classification balance among the three granular runs. The `latent_dim=64` run has the best eval-quality score and activates more latent/Fisher structure than `32`, but its classification metrics are lower. The `latent_dim=128` run activates the most dimensions but does not improve classification.

5. `latent_dim=128` activates more dimensions but does not improve the task.

The 128-dimensional model has the largest latent and Fisher effective ranks, but it has lower classification metrics and an unstable eval-quality score. This supports the idea that nominal latent capacity alone is not the bottleneck.

## Summary

The granular-label study supports the hypothesis that the latent collapse is partly driven by the supervised label structure. When the number of physics labels increases from 5 to 10, the model uses more latent directions and the classifier-head Fisher spectrum becomes broader.

However, the extra activated dimensions do not directly translate into better classification. The granular final-state task is substantially harder at L1 level, especially for WW and tt subcategories. The most important result is not that a single latent size is clearly best, but that label granularity changes the geometry of the representation and reduces the earlier low-dimensional collapse.
