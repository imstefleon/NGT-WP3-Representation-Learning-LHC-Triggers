# 2026-08-14 Scaling Axes Analysis

## Purpose

This note separates the current scaling study into the two axes requested by Maciej:

1. **Axis 1: class diversity / granularity**
2. **Axis 2: sample size + encoder capacity**

The main physics-use-case metric is **MinBias-vs-injected-anomaly separation**, computed in the full latent space. PCA/t-SNE plots are qualitative diagnostics only.

Injected anomaly probes:

```text
hh4b, hh, vbfh, ggh
```

These signals are **not** supervised training labels in the current `4class`, `10plus_sm`, or `granular_sm13` scaling studies.

## Main Metric

The primary score used here is the **MinBias Mahalanobis distance** in latent space:

> MinBias is treated as the in-distribution reference. Larger distance from the MinBias latent distribution is interpreted as more anomalous.

Main reported quantities:

- mean AUC vs SM background, averaged over `hh4b`, `hh`, `vbfh`, `ggh`;
- signal efficiency at 95% background rejection;
- signal efficiency at 99% background rejection.

Supporting diagnostics:

- GMM negative log likelihood;
- combined robust anomaly score;
- latent effective rank / PCA spectrum;
- classifier response;
- PCA/t-SNE qualitative plots.

## Executive Summary

Current evidence:

- More class diversity/granularity does **not** automatically improve MinBias-vs-Higgs-like anomaly separation.
- The `4class` setup remains the strongest baseline for the current injected Higgs-like probes.
- Increasing samples from 50k to 100k does not show a clear improvement for any class-diversity setup.
- At 150k, the stable `medium` encoder learns normally and gives usable anomaly metrics.
- A depth-only increase (`medium_depth3`) still learns and gives the strongest MinBias-distance separation among the completed 150k tests.
- A width/latent-size increase (`wide2blocks`) collapses to chance accuracy, suggesting that the instability is driven mainly by the wider latent/FF dimensions rather than by depth alone.
- At 200k, the stable `medium` run completed successfully.
- The 200k `ff512_blocks2` run also completed and improves the MinBias-distance anomaly metric relative to `medium`, while using fewer effective latent dimensions.
- The 200k `medium_depth3` and `ff512_depth3` configurations collapse, so depth=3 is not stable at 200k with the current recipe.

## Axis 1: Class Diversity / Granularity

### Configurations

#### 4class

```text
minbias, qcd, dy, top
```

#### 10plus_sm

```text
minbias, qcd, dy,
tt_hadronic, tt_leptonic, tt_semileptonic,
wjets, ww, wz, ttw, ttz
```

#### granular_sm13

```text
minbias, qcd, dy,
tt_hadronic, tt_leptonic, tt_semileptonic,
wjets,
ww_hadronic, ww_leptonic, ww_semileptonic,
wz, ttw, ttz
```

The intended `granular_sm15` configuration with `dy_ee`, `dy_mumu`, and `dy_tautau` could not be run with `collide2v_updated`, because those DY subchannels were not matched in the updated EOS/parquet source. They exist in the HuggingFace dataset definitions, but not in the current updated parquet source used for the main study.

### Axis 1 Results

All rows use the `linear` probe and the `minbias_mahalanobis` score.

| class setup | samples/class | mean AUC | eff@95%BG | eff@99%BG | interpretation |
|---|---:|---:|---:|---:|---|
| 4class | 50k | 0.7836 | 0.1716 | 0.0338 | strongest apparent separation |
| 4class | 100k | 0.7830 | 0.1666 | 0.0474 | stable |
| 10plus_sm | 50k | 0.5890 | 0.0480 | 0.0064 | lower separation |
| 10plus_sm | 100k | 0.5744 | 0.0377 | 0.0100 | slightly lower |
| granular_sm13 | 50k | 0.5606 | 0.0436 | 0.0080 | lower separation |
| granular_sm13 | 100k | 0.5538 | 0.0389 | 0.0056 | no improvement |

### Axis 1 Interpretation

The class-diversity hypothesis was:

> More SM process granularity may produce a richer SM manifold and improve anomaly separation.

The current results do **not** support that hypothesis for the present Higgs-like probes. The most granular available setup, `granular_sm13`, remains below both `4class` and `10plus_sm` in mean MinBias-distance AUC.

This does not mean that `4class` is universally better for anomaly detection. A likely interpretation is:

> The coarse `4class` setup can push injected Higgs-like events into broad non-MinBias/top-like regions, which increases apparent distance from MinBias. More granular SM setups may learn a broader and more realistic SM manifold, where the same injected probes are less isolated.

Therefore:

> Increasing class diversity changes the latent manifold, but does not automatically improve the current MinBias-vs-Higgs-like anomaly metric.

### Axis 1 Plots

Recommended plot for presentation / Maciej update:

```text
analysis/figures/scaling_study/class_granularity_minbias_auc_summary.png
```

Useful run-level plots:

```text
outputs/scaling_study/granular_sm13/100k/medium/runs/2026-08-13/scaling_granular_sm13_100k_medium_linear_seed42/figures/external_higgs_anomaly/minbias_distance_distributions.png
outputs/scaling_study/granular_sm13/100k/medium/runs/2026-08-13/scaling_granular_sm13_100k_medium_linear_seed42/figures/external_higgs_anomaly/minbias_vs_anomalies_latent_tsne.png
outputs/scaling_study/granular_sm13/100k/medium/runs/2026-08-13/scaling_granular_sm13_100k_medium_linear_seed42/figures/external_higgs_anomaly/external_higgs_anomaly_metrics_by_method.csv
```

## Axis 2: Sample Size + Encoder Capacity

### Scaling Schedule

The class setup is fixed to `4class` and the classifier probe is kept simple/fixed as `linear`.

| samples/class | model size | latent_dim | ff_dim | attention blocks | heads | status |
|---:|---|---:|---:|---:|---:|---|
| 50k | small | 64 | 128 | 2 | 8 | completed |
| 100k | medium | 128 | 256 | 2 | 8 | completed |
| 150k | medium | 128 | 256 | 2 | 8 | completed, valid |
| 150k | medium_depth3 | 128 | 256 | 3 | 8 | completed, valid ablation |
| 150k | wide2blocks | 192 | 384 | 2 | 8 | collapsed |
| 150k | large | 192 | 384 | 3 | 8 | collapsed / unstable |
| 200k | medium | 128 | 256 | 2 | 8 | completed, valid |
| 200k | ff512_blocks2 | 128 | 512 | 2 | 8 | completed, valid |
| 200k | medium_depth3 | 128 | 256 | 3 | 8 | collapsed |
| 200k | ff512_depth3 | 128 | 512 | 3 | 8 | collapsed / terminated |
| 200k | XL | 256 | 512 | 4 | 8 | earlier attempt collapsed / not used |

The original large schedule increased multiple capacity knobs at once:

```text
medium: latent_dim=128, ff_dim=256, attention_blocks=2
large:  latent_dim=192, ff_dim=384, attention_blocks=3
```

The 150k ablations were designed to isolate whether the instability came from extra depth or from the wider latent/FF dimensions.

## Axis 2: 150k Results

### Training Summary

| run | latent_dim | ff_dim | blocks | best val loss | best val acc | final val acc | status |
|---|---:|---:|---:|---:|---:|---:|---|
| 150k medium | 128 | 256 | 2 | 0.7086 | 0.6576 | 0.6580 | valid |
| 150k medium_depth3 | 128 | 256 | 3 | 0.8005 | 0.5905 | 0.5647 | valid but lower classifier quality |
| 150k wide2blocks | 192 | 384 | 2 | 0.8316 | 0.5601 | 0.2500 | collapsed after early epochs |

The `wide2blocks` run starts with non-trivial validation accuracy in epoch 1, but then collapses to chance-level accuracy:

```text
final train_acc ~= 0.25
final val_acc = 0.25
final loss ~= log(4)
```

This indicates that the instability is not caused by sample size alone.

### Latent Diagnostics

| run | latent_dim | total latent variance | effective rank | participation ratio | PCs for 90% | PCs for 95% |
|---|---:|---:|---:|---:|---:|---:|
| 150k medium | 128 | 434.49 | 11.64 | 6.12 | 14 | 23 |
| 150k medium_depth3 | 128 | 164.22 | 2.53 | 1.95 | 3 | 3 |
| 150k wide2blocks | 192 | 167.11 | 1.53 | 1.24 | 2 | 2 |

Interpretation:

- `150k medium` uses a broader latent geometry and is the cleanest scaling point.
- `150k medium_depth3` is much more compressed, but it still produces strong MinBias-vs-anomaly separation.
- `150k wide2blocks` is highly compressed and training-collapsed; it should not be used for external anomaly conclusions.

This again supports the earlier observation:

> Higher or lower effective dimensionality is not by itself the final physics metric. It is a diagnostic that must be interpreted together with full-latent anomaly scores.

### 150k External Anomaly Metrics

All rows use the `minbias_mahalanobis` score, averaged over `hh4b`, `hh`, `vbfh`, and `ggh`.

| run | mean AUC | eff@95%BG | eff@99%BG | interpretation |
|---|---:|---:|---:|---|
| 150k medium | 0.7683 | 0.1512 | 0.0282 | stable, usable scaling point |
| 150k medium_depth3 | 0.7963 | 0.3245 | 0.2190 | strongest 150k separation, but lower classifier quality and lower latent rank |
| 150k wide2blocks | n/a | n/a | n/a | not evaluated because training collapsed |

Per-signal MinBias-distance results:

| run | signal | AUC | eff@95%BG | eff@99%BG |
|---|---|---:|---:|---:|
| 150k medium | hh4b | 0.7832 | 0.0695 | 0.0065 |
| 150k medium | hh | 0.7795 | 0.1055 | 0.0195 |
| 150k medium | vbfh | 0.7806 | 0.2025 | 0.0450 |
| 150k medium | ggh | 0.7300 | 0.2275 | 0.0420 |
| 150k medium_depth3 | hh4b | 0.8659 | 0.3045 | 0.0870 |
| 150k medium_depth3 | hh | 0.8574 | 0.3950 | 0.2935 |
| 150k medium_depth3 | vbfh | 0.7824 | 0.2895 | 0.2320 |
| 150k medium_depth3 | ggh | 0.6795 | 0.3090 | 0.2635 |

Supporting method comparison:

| run | method | mean AUC | eff@95%BG | eff@99%BG |
|---|---|---:|---:|---:|
| 150k medium | GMM NLL | 0.7538 | 0.1413 | 0.0296 |
| 150k medium | MinBias Mahalanobis | 0.7683 | 0.1512 | 0.0282 |
| 150k medium | combined robust score | 0.7588 | 0.1410 | 0.0273 |
| 150k medium_depth3 | GMM NLL | 0.7883 | 0.3320 | 0.2120 |
| 150k medium_depth3 | MinBias Mahalanobis | 0.7963 | 0.3245 | 0.2190 |
| 150k medium_depth3 | combined robust score | 0.7928 | 0.3312 | 0.2142 |

## Axis 2 Interpretation

The 150k tests answer a specific stability question:

> Does the collapse come from adding depth, increasing width/latent size, or the full large configuration?

Evidence:

- `medium_depth3` keeps `latent_dim=128`, `ff_dim=256`, and increases only depth from 2 to 3 blocks. It still learns and produces complete anomaly metrics.
- `wide2blocks` increases `latent_dim=192`, `ff_dim=384`, but keeps depth at 2 blocks. It collapses to chance-level accuracy.
- The full `large` setup also collapsed.

Conclusion:

> The 150k instability appears to be driven mainly by the wider latent/FF dimensions, or their interaction with the current CE + latent regularization setup, rather than by the third attention block alone.

This means the 150k result should not be interpreted as "more samples are bad." Instead:

> 150k medium is a valid scaling point; width/capacity scaling needs additional stability tuning before it can be used for scientific conclusions.

## Axis 2: 200k Results

### 200k Architecture Comparison

The 200k runs were used to decide whether the 1M/class pilot should keep the conservative `medium` architecture or increase capacity through `ff_dim`.

| run | latent_dim | ff_dim | blocks | training status | interpretation |
|---|---:|---:|---:|---|---|
| 200k medium | 128 | 256 | 2 | completed, stable | safest baseline |
| 200k ff512_blocks2 | 128 | 512 | 2 | completed, stable | better anomaly score than medium |
| 200k medium_depth3 | 128 | 256 | 3 | collapsed | depth=3 not stable at 200k |
| 200k ff512_depth3 | 128 | 512 | 3 | collapsed / terminated | depth+FF scaling unstable |

### 200k Latent and Anomaly Metrics

All anomaly rows use `minbias_mahalanobis`, averaged over `hh4b`, `hh`, `vbfh`, and `ggh`.

| run | effective rank | participation ratio | PCs 95% | mean AUC | eff@95%BG | eff@99%BG |
|---|---:|---:|---:|---:|---:|---:|
| 200k medium | 8.94 | 5.63 | 13 | 0.7531 | 0.1141 | 0.0390 |
| 200k ff512_blocks2 | 3.85 | 2.98 | 4 | 0.7869 | 0.1630 | 0.0535 |

Supporting method comparison:

| run | method | mean AUC | eff@95%BG | eff@99%BG |
|---|---|---:|---:|---:|
| 200k medium | MinBias Mahalanobis | 0.7531 | 0.1141 | 0.0390 |
| 200k medium | GMM NLL | 0.7491 | 0.1143 | 0.0394 |
| 200k medium | combined robust score | 0.7513 | 0.1140 | 0.0394 |
| 200k ff512_blocks2 | MinBias Mahalanobis | 0.7869 | 0.1630 | 0.0535 |
| 200k ff512_blocks2 | GMM NLL | 0.7775 | 0.1421 | 0.0451 |
| 200k ff512_blocks2 | combined robust score | 0.7836 | 0.1673 | 0.0540 |

Interpretation:

> Increasing `ff_dim` from 256 to 512 while keeping `latent_dim=128` and `attention_blocks=2` improves the MinBias-vs-anomaly metric at 200k. However, it also produces a more compressed latent representation.

This is another example where effective latent dimensionality and anomaly separation are not equivalent:

```text
200k medium:       effective_rank=8.94, AUC=0.7531
200k ff512_blocks2 effective_rank=3.85, AUC=0.7869
```

The depth results are less favorable:

> Both 200k depth=3 configurations collapse under the current recipe. Therefore depth should not be increased for the 1M pilot unless additional stability tuning is introduced.

## Relation to Earlier Collapse Studies

This is consistent with previous analysis:

- `analysis/hh4b_latent_dim_sweep.md` showed that increasing nominal `latent_dim` alone did not reliably improve effective latent dimensionality or performance.
- `analysis/hh4b_collapse_diagnostics.md` showed that increasing dataset size from 100k to 300k did not remove latent collapse under the MHA + CE setup; effective rank decreased with more data.

The new 150k results extend this:

> Scaling the encoder is not just a resource problem. The training recipe and stability need to be tuned before using larger/wider models for conclusions.

## 1M Pilot Recommendation

The current best candidate for the weekend 1M/class pilot is:

```text
class_setup = 4class
events_per_class = 1M
latent_dim = 128
ff_dim = 512
attention_blocks = 2
heads = 8
probe = linear
```

Reasoning:

- `latent_dim=128` is stable and avoids the wider latent configuration that collapsed.
- `attention_blocks=2` is stable at 200k, while depth=3 collapsed.
- `ff_dim=512` improves the 200k MinBias-distance metrics over `ff_dim=256`.
- The probe remains simple/fixed (`linear`), as requested.

Recommended objective comparison for the 1M pilot:

| run | objective | purpose |
|---|---|---|
| 1M ff512_blocks2 CE+var/cov | CE + latent variance/covariance regularization | direct continuation of current anomaly-metric setup |
| 1M ff512_blocks2 CE-only | CE only | control to test whether var/cov helps or contributes to collapse |

This keeps architecture fixed and varies only the objective/regularization.

## Recommended Presentation Story

### Slide: Scaling Axes

Introduce the two axes before showing results:

- Axis 1: class diversity / granularity
  - `4class`
  - `10plus_sm`
  - `granular_sm13`
- Axis 2: sample size + encoder capacity
  - 50k, 100k, 150k, 200k events/class
  - small, medium, ff512_blocks2, depth3/large attempts

### Slide: Scaling Diversity

Main message:

> More granular SM supervision did not improve the MinBias-vs-Higgs-like anomaly metric for the current probes.

Use:

```text
analysis/figures/scaling_study/class_granularity_minbias_auc_summary.png
```

### Slide: Scaling Sample Size & Model Size

Main message:

> 150k and 200k show that stable capacity scaling is selective: increasing depth is unstable at 200k, while increasing `ff_dim` to 512 with depth fixed improves the anomaly metric.

Use:

```text
outputs/scaling_study/main_4class_scaling/4class/150k/medium/runs/2026-08-14/scaling_4class_150k_medium_linear_seed42_rerun2/figures/external_higgs_anomaly/minbias_distance_distributions.png
outputs/scaling_study/main_4class_scaling/4class/150k/medium_depth3/runs/2026-08-14/scaling_4class_150k_medium_depth3_linear_seed42/figures/external_higgs_anomaly/minbias_distance_distributions.png
```

For a compact result table, use:

| setup | mean AUC | eff@95%BG | eff@99%BG | note |
|---|---:|---:|---:|---|
| 100k medium | 0.7830 | 0.1666 | 0.0474 | previous stable baseline |
| 150k medium | 0.7683 | 0.1512 | 0.0282 | valid, similar/slightly lower |
| 150k medium_depth3 | 0.7963 | 0.3245 | 0.2190 | stronger separation, lower classifier/latent-rank diagnostics |
| 200k medium | 0.7531 | 0.1141 | 0.0390 | stable baseline at 200k |
| 200k ff512_blocks2 | 0.7869 | 0.1630 | 0.0535 | best stable 200k candidate |
| 150k wide2blocks | n/a | n/a | n/a | collapsed |
| 200k medium_depth3 | n/a | n/a | n/a | collapsed |

## Caveats

- Results are specific to `hh4b`, `hh`, `vbfh`, and `ggh` injected probes.
- Stronger MinBias-distance separation in `4class` may be partly due to coarse non-MinBias/top-like assignment.
- `medium_depth3` gives strong anomaly scores at 150k but collapses at 200k; it should not be selected for the 1M pilot without additional stability tuning.
- `wide2blocks` and earlier large/XL runs should not be interpreted as evidence that more capacity hurts physics performance. They show a training-stability failure.
- `ff512_blocks2` improves 200k anomaly metrics but compresses the latent rank, so it should be evaluated with both anomaly metrics and latent diagnostics after the 1M pilot.
- This remains an offline representation/anomaly-detection study, not an online HLT/FPGA throughput result.

## Short Conclusion

The class-diversity axis is now reasonably clear:

> More granular SM labels do not automatically improve MinBias-vs-Higgs-like anomaly separation for the current probes.

The sample-size/capacity axis is partially answered:

> 150k and 200k show that increasing capacity must be controlled. Depth=3 is unstable at 200k, while increasing `ff_dim` to 512 with `latent_dim=128` and two attention blocks improves the anomaly metric. The recommended 1M pilot is therefore `4class`, `latent_dim=128`, `ff_dim=512`, `attention_blocks=2`, `heads=8`, `linear` probe, with CE+var/cov and CE-only as objective controls.
