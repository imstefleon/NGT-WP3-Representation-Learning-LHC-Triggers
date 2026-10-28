# 2026-07-16 HH4B Latent-Dimension Sweep

## Scope

This note summarizes the MHA latent-dimension sweep performed with the updated Collide-2V L1-only dataset. The goal was to test whether increasing the nominal latent dimension improves HH4B separation against the full SM background, and whether it reduces the observed latent-space collapse.

The sweep uses the same class granularity and sample size for all configurations:

- Dataset source: updated Collide-2V production parquets.
- Input mode: L1-only candidate features.
- Candidate selection: top-pT.
- Classes: MinBias, Top, QCD, DY, HH4B.
- Background for anomaly evaluation: MinBias + Top + QCD + DY.
- Anomaly class: HH4B.
- Events per class: 100k.
- Fixed architecture settings: `ff_dim=128`, `num_heads=8`.
- Swept parameter: `latent_dim = 16, 32, 64, 128`.

The two partial sweep outputs were merged into:

`ngt_mha_nodes_aligned_outputs/scale_tests/latent_dim_sweep_hh4b_100k_combined.csv`

## Combined Results

| latent_dim | params | val_acc | eval_acc | macro F1 | HH4B F1 | DY F1 | effective rank | participation ratio | PCs for 90% var. | eval_quality |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 181,526 | 0.602 | 0.600 | 0.599 | 0.708 | 0.456 | 4.43 | 3.09 | 4 | 0.169 |
| 32 | 185,638 | 0.609 | 0.608 | 0.598 | 0.717 | 0.400 | 5.44 | 3.36 | 5 | 0.193 |
| 64 | 193,862 | 0.598 | 0.598 | 0.580 | 0.719 | 0.315 | 4.87 | 2.88 | 5 | 0.00004 |
| 128 | 210,310 | 0.585 | 0.582 | 0.575 | 0.694 | 0.365 | 4.74 | 2.98 | 5 | 0.176 |

## Classification Behavior

The best overall validation and evaluation accuracy occurred at `latent_dim=32`:

- best validation accuracy: 0.609;
- evaluation accuracy: 0.608;
- macro F1: 0.598;
- HH4B F1: 0.717.

The highest HH4B F1 was obtained at `latent_dim=64` with HH4B F1 = 0.719, but this did not improve the global metrics and produced an unstable eval-quality result.

Increasing to `latent_dim=128` did not improve classification. Both global accuracy and HH4B F1 decreased relative to the smaller latent dimensions. This suggests that the current model does not benefit from simply increasing the nominal latent-space size.

## Latent-Space Dimensionality

The latent diagnostics show that the effective dimensionality remains low across the full sweep:

- effective rank stays between approximately 4.4 and 5.4;
- participation ratio stays between approximately 2.9 and 3.4;
- only 4 to 5 principal components are needed to explain 90% of the latent variance;
- only 6 to 8 principal components are needed to explain 95% of the latent variance.

This means that even when the nominal latent dimension is increased from 16 to 128, the model continues to use only a small number of active latent directions.

## Eval-Quality Interpretation

The eval-quality score is not monotonic with latent dimension:

- `latent_dim=32` gives the best composite score among the stable configurations.
- `latent_dim=64` has the best HH4B F1 but a near-zero composite score because the GMM gap becomes negative in this run.
- `latent_dim=128` recovers a reasonable composite score, but classification and HH4B F1 are worse than at `latent_dim=32`.

This confirms that eval-quality and supervised classification metrics are related but not identical. The sweep should be interpreted with both views: classification performance and latent geometry.

## Samples-to-Parameters Context

Each run used 100k events per class, corresponding to 500k total events before train/validation/evaluation splitting. The approximate total-events-to-parameters ratio decreases as the latent dimension increases:

| latent_dim | params | total events / params |
|---:|---:|---:|
| 16 | 181,526 | 2.75 |
| 32 | 185,638 | 2.69 |
| 64 | 193,862 | 2.58 |
| 128 | 210,310 | 2.38 |

This is not a direct application of Chinchilla scaling laws, but it gives a useful reference for tracking how data and parameter count are being scaled in the event-embedding setup.

## Main Conclusion

The latent-dimension sweep does not support increasing the nominal latent dimension beyond 32 for the current setup. The model already compresses the representation into roughly 4 to 6 effective latent directions, so the bottleneck is not simply the configured latent size.

For this configuration, `latent_dim=32` is the best balanced point: it gives the strongest overall accuracy, the best stable eval-quality score, and one of the best HH4B F1 scores while using fewer parameters than the 64 and 128 dimensional models.
