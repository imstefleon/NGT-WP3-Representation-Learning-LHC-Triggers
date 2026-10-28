# Parameter/Sample Power-Law Sweep

This folder prepares the scaling-law study requested after the 1M CE+var/cov result.

Goal:

> Measure signal efficiency at fixed background rejection as a function of model parameters and samples, then test whether a power-law exponent is visible.

Primary metric:

```text
eff_at_99_5pct_bg for HH4B and VBFH
```

Secondary metrics:

```text
AUC, eff_at_95pct_bg, eff_at_99pct_bg, effective rank, PCA dims 95%
```

## Scaling Rule

Maciej's suggested grid scales parameters and samples together:

| target N_params | target N_samples |
|---:|---:|
| 1e5 | 1e4 |
| 5e5 | 5e4 |
| 2.5e6 | 2.5e5 |
| 1e7 | 1e6 |

This keeps `N_samples / N_params ~= 0.1`.

For the current 4-class balanced setup, `N_samples` is interpreted as total samples across:

```text
minbias,qcd,dy,top
```

The training command therefore uses:

```text
--n-events-per-class = N_samples / 4
```

This differs from earlier project shorthand like "1M/class"; the point here is a compact scaling-law sweep.

The Chinchilla analogy is used only qualitatively: scale model size and data together. The literal language-model ratio of roughly 20 tokens per parameter is not directly transferable to event-level L1 data.

## Model Grid

To keep the sweep interpretable, the architecture keeps `latent_dim=128`, `num_attention_blocks=2`, `num_heads=8`, `pooling=attention`, and `classifier_head=linear`; it varies mainly `ff_dim` to reach the parameter targets.

| point | target params | actual params | total samples | events/class | ff_dim | param error |
|---|---:|---:|---:|---:|---:|---:|
| `plaw_p1e5_n1e4` | 100,000 | 94,717 | 10,000 | 2,500 | 88 | -5.283% |
| `plaw_p5e5_n5e4` | 500,000 | 485,077 | 50,000 | 12,500 | 208 | -2.985% |
| `plaw_p2p5e6_n2p5e5` | 2,500,000 | 2,489,445 | 250,000 | 62,500 | 480 | -0.422% |
| `plaw_p1e7_n1e6` | 10,000,000 | 9,979,757 | 1,000,000 | 250,000 | 968 | -0.202% |

## Objective

Use the best current high-background-rejection recipe from the 1M comparison:

```text
CE + latent variance/covariance regularization
augmentation_copies = 0
lr = 3e-5
```

The 1M comparison showed that `ff256 CE+var/cov` is better than CE-only at the harder HH4B/VBFH 99% and 99.5% fixed-background rejection points, even when CE-only can have slightly higher AUC.

## Files

- `power_law_manifest.csv`
- `power_law_manifest.json`
- `commands/run_power_law_sweep.sh`
- `commands/collect_and_fit_power_law.py`

The command script does not run automatically. Launch points sequentially or through the batch system after checking available GPU memory for the larger `ff968` point.
