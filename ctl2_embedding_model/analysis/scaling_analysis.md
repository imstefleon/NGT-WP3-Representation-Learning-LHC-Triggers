# Scaling Analysis: MinBias-vs-Anomaly Latent Separation

## Executive Summary

This analysis summarizes the current scaling and probe-ablation studies for the NGT representation-learning model. The latest guidance from Maciej is to prioritize the physics use case:

> How well do injected Higgs-like anomalies separate from MinBias, and does that separation improve with data diversity, sample size, and encoder capacity?

The main quantitative metric is therefore the full-dimensional MinBias-distance anomaly score, not the visual appearance of PCA/t-SNE alone. PCA and t-SNE remain useful qualitative diagnostics of the latent manifold.

The current evidence suggests:

- Increasing class diversity does not automatically improve MinBias-vs-anomaly separation.
- The `4class` setup currently gives the strongest apparent separation from MinBias.
- The classifier probe affects latent geometry, but higher effective latent dimensionality does not necessarily improve anomaly separation.
- The first copied `4class` 150k/200k large/XL runs under `main_4class_scaling/logs` were degenerate and remain failed evidence only.
- Subsequent focused `4class` runs under `main_4class_scaling/4class` produced usable 150k/200k results, including a direct 200k `ff512_blocks2` augmentation-vs-no-augmentation check.
- The stable 1M/class runs all require `lr=3e-5` and `augmentation_copies=0`; their MinBias-distance mean AUC is about `0.841-0.845`, with best available `eff@99%BG` from the `ff256` CE+var/cov run.
- The current scaling evidence supports using low-LR, no-augmentation 4class training for large samples, with CE+var/cov kept as a candidate because it increases effective latent dimensionality without hurting AUC.

Important caveat:

> The stronger `4class` separation may partly reflect coarse class assignment, where injected Higgs-like samples are mapped into a broad non-MinBias/top-like region. This should not be overinterpreted as proof of a generally better exotic-anomaly representation.

## Scope

This note covers:

- probe ablation: `linear`, `mlp`, `mlp2`;
- class-diversity comparison: `4class` vs `10plus_sm`;
- matched 50k and 100k studies;
- current decision to focus on 4class scaling;
- focused 150k/200k/1M 4class scaling;
- augmentation vs no-augmentation checks;
- optional granular cross-checks.

Signals evaluated as injected anomalies:

- `hh4b`
- `hh`
- `vbfh`
- `ggh`

These signals are evaluation probes, not supervised training classes in the background-representation experiments discussed here.

## Main Evaluation Criterion

The primary anomaly score is the full-dimensional distance from the MinBias latent distribution.

Primary metrics:

- mean AUC vs SM background;
- signal efficiency at 95% background rejection;
- signal efficiency at 99% background rejection.
- signal efficiency at 99.5% background rejection for HH4B/VBFH large-scale checks.

Supporting diagnostics:

- GMM negative log likelihood;
- combined robust anomaly score;
- PCA/t-SNE latent visualizations;
- effective latent rank;
- PCA explained-variance spectrum;
- classifier-response matrices.

Interpretation rule:

> PCA/t-SNE show a 2D projection of the latent geometry. The MinBias-distance score is computed in the latent space and should be treated as the quantitative comparison.

## Experimental Configurations

### 4class

```text
minbias,qcd,dy,top
```

### 10plus_sm

```text
minbias,qcd,dy,
tt_hadronic,tt_leptonic,tt_semileptonic,
wjets,ww,wz,ttw,ttz
```

### granular_sm13 cross-check

```text
minbias,qcd,dy,
tt_hadronic,tt_leptonic,tt_semileptonic,
wjets,
ww_hadronic,ww_leptonic,ww_semileptonic,
wz,ttw,ttz
```

### granular_sm15 caveat

The intended `granular_sm15` configuration included:

```text
dy_ee,dy_mumu,dy_tautau
```

Those DY subchannels exist in the HuggingFace loader definitions, but the `collide2v_updated` EOS parquet source did not match files for `dy_ee`. The run failed with:

```text
dy_ee: no updated Collide-2V parquet files matched
```

Therefore, `granular_sm15` is not currently a valid `collide2v_updated` run. It can only be tested as a separate HuggingFace-data-source cross-check.

## Completed Runs

### Probe ablation at 50k/class

Completed:

- `4class 50k ff128 linear`
- `4class 50k ff128 mlp`
- `4class 50k ff128 mlp2`
- `10plus_sm 50k ff128 linear`
- `10plus_sm 50k ff128 mlp`
- `10plus_sm 50k ff128 mlp2`

### Matched 100k/class comparison

Completed:

- `4class 100k ff256 linear`
- `4class 100k ff256 mlp`
- `4class 100k ff256 mlp2`
- `10plus_sm 100k ff256 linear`
- `10plus_sm 100k ff256 mlp`

Pending / diagnostics incomplete:

- `10plus_sm 100k ff256 mlp2`

### Focused scaling after Maciej feedback

Completed training/diagnostics:

- `4class 150k large linear`
- `4class 200k XL linear`
- `4class 150k ff256/ff384/depth/noaug variants`
- `4class 200k ff256/ff512/depth/aug0/noaug variants`
- `4class 1M noaug lr3e-5 ff256 and ff512 variants`
- `granular_sm13 50k ff128 linear`
- `granular_sm13 100k ff256 linear`

External-anomaly evaluation status:

- `granular_sm13 50k ff128 linear`: completed with MinBias-distance, GMM, combined-score metrics and plots.
- `granular_sm13 100k ff256 linear`: completed with MinBias-distance, GMM, combined-score metrics and plots.
- `4class 150k large linear`: external-anomaly evaluation failed during GMM fitting because the whitened latent matrix contained NaNs.
- `4class 200k XL linear`: external-anomaly evaluation failed with the same NaN/GMM failure.
- `4class 1M ff512_blocks2 CE+var/cov` with augmentation and default LR: completed to a degenerate zero-variance latent and is not a valid anomaly-analysis point.

## Primary Results

Mean values are averaged over `hh4b`, `hh`, `vbfh`, and `ggh`.

| class config | samples/class | probe | mean AUC | eff@95%BG | eff@99%BG | effective rank | PCA dims 95% |
|---|---:|---|---:|---:|---:|---:|---:|
| 4class | 50k | linear | 0.7836 | 0.1716 | 0.0338 | 9.49 | 14 |
| 4class | 50k | mlp | 0.7603 | 0.1893 | 0.0619 | 10.56 | 14 |
| 4class | 50k | mlp2 | 0.7471 | 0.1530 | 0.0255 | 13.81 | 20 |
| 4class | 100k | linear | 0.7830 | 0.1666 | 0.0474 | 8.30 | 10 |
| 4class | 100k | mlp | 0.6463 | 0.1239 | 0.0299 | 12.80 | 19 |
| 4class | 100k | mlp2 | 0.6299 | 0.2280 | 0.1011 | 4.40 | 5 |
| 10plus_sm | 50k | linear | 0.5890 | 0.0480 | 0.0064 | 14.41 | 20 |
| 10plus_sm | 50k | mlp | 0.5682 | 0.0382 | 0.0070 | 20.50 | 28 |
| 10plus_sm | 50k | mlp2 | 0.5311 | 0.0340 | 0.0051 | 17.35 | 25 |
| 10plus_sm | 100k | linear | 0.5744 | 0.0377 | 0.0100 | 17.17 | 28 |
| 10plus_sm | 100k | mlp | 0.4565 | 0.0190 | 0.0025 | 12.79 | 14 |
| granular_sm13 | 50k | linear | 0.5606 | 0.0436 | 0.0080 | 16.58 | 22 |
| granular_sm13 | 100k | linear | 0.5538 | 0.0389 | 0.0056 | 13.66 | 25 |

The `granular_sm13` rows use the `minbias_mahalanobis` score, averaged over `hh4b`, `hh`, `vbfh`, and `ggh`.

Focused `4class` scaling results use the same `minbias_mahalanobis` score and are averaged over `hh4b`, `hh`, `vbfh`, and `ggh`.

| run | samples/class | ff_dim | blocks | aug copies | lr | objective | mean AUC | eff@95%BG | eff@99%BG | effective rank | PCA dims 95% |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| `50k ff128 noaug` | 50k | 128 | 2 | 0 | 1e-3 | CE + var/cov | 0.7986 | 0.2236 | 0.0734 | 5.10 | 6 |
| `100k ff256 noaug` | 100k | 256 | 2 | 0 | 1e-3 | CE + var/cov | 0.7934 | 0.1720 | 0.0503 | 5.31 | 6 |
| `150k ff256 aug` | 150k | 256 | 2 | 1 | 1e-3 | CE + var/cov | 0.7683 | 0.1512 | 0.0282 | 11.64 | 23 |
| `150k wide2blocks aug` | 150k | 384 | 2 | 1 | 1e-3 | CE + var/cov | 0.8349 | 0.3679 | 0.1475 | 1.53 | 2 |
| `150k ff256 noaug lr3e-5` | 150k | 256 | 2 | 0 | 3e-5 | CE + var/cov | 0.8367 | 0.3494 | 0.1688 | 1.27 | 1 |
| `200k ff256 aug` | 200k | 256 | 2 | 1 | 1e-3 | CE + var/cov | 0.7531 | 0.1141 | 0.0390 | 8.94 | 13 |
| `200k ff256 depth3 aug` | 200k | 256 | 3 | 1 | 1e-3 | CE + var/cov | 0.8275 | 0.3197 | 0.0910 | 1.45 | 2 |
| `200k ff512 blocks2 aug` | 200k | 512 | 2 | 1 | 1e-3 | CE + var/cov | 0.7869 | 0.1630 | 0.0535 | 3.85 | 4 |
| `200k ff512 blocks2 aug0` | 200k | 512 | 2 | 0 | 1e-3 | CE + var/cov | 0.7760 | 0.1906 | 0.0774 | 2.76 | 4 |
| `200k ff512 blocks2 noaug lr3e-5` | 200k | 512 | 2 | 0 | 3e-5 | CE + var/cov | 0.8272 | 0.3405 | 0.1270 | 1.91 | 3 |
| `1M ff256 CE-only noaug lr3e-5` | 1M | 256 | 2 | 0 | 3e-5 | CE-only | 0.8425 | 0.3890 | 0.1797 | 1.77 | 2 |
| `1M ff256 var/cov noaug lr3e-5` | 1M | 256 | 2 | 0 | 3e-5 | CE + var/cov | 0.8408 | 0.3872 | 0.2240 | 3.66 | 4 |
| `1M ff512 CE-only noaug lr3e-5` | 1M | 512 | 2 | 0 | 3e-5 | CE-only | 0.8421 | 0.3590 | 0.1716 | 2.27 | 3 |
| `1M ff512 var/cov noaug lr3e-5` | 1M | 512 | 2 | 0 | 3e-5 | CE + var/cov | 0.8445 | 0.3705 | 0.1736 | 6.33 | 8 |

For HH4B and VBFH specifically, recomputing the MinBias-distance threshold at 99.5% background rejection from `external_higgs_event_scores.csv` gives:

| run | signal | AUC | eff@95%BG | eff@99%BG | eff@99.5%BG |
|---|---|---:|---:|---:|---:|
| `1M ff256 CE-only noaug lr3e-5` | HH4B | 0.8967 | 0.3970 | 0.1215 | 0.0825 |
| `1M ff256 CE+var/cov noaug lr3e-5` | HH4B | 0.8942 | 0.3860 | 0.1485 | 0.1055 |
| `1M ff512 CE-only noaug lr3e-5` | HH4B | 0.8883 | 0.3685 | 0.1210 | 0.0630 |
| `1M ff512 CE+var/cov noaug lr3e-5` | HH4B | 0.8933 | 0.3820 | 0.1005 | 0.0520 |
| `1M ff256 CE-only noaug lr3e-5` | VBFH | 0.8312 | 0.3530 | 0.1775 | 0.1275 |
| `1M ff256 CE+var/cov noaug lr3e-5` | VBFH | 0.8252 | 0.3520 | 0.2220 | 0.1845 |
| `1M ff512 CE-only noaug lr3e-5` | VBFH | 0.8359 | 0.3255 | 0.1710 | 0.1120 |
| `1M ff512 CE+var/cov noaug lr3e-5` | VBFH | 0.8372 | 0.3355 | 0.1825 | 0.1295 |

Conclusion for the high-background-rejection operating point:

> `1M ff256 CE+var/cov noaug lr3e-5` is the best practical 1M point for HH4B and VBFH at 99.5% background rejection. CE-only has slightly higher HH4B AUC for `ff256`, but CE+var/cov gives better fixed-threshold signal efficiency in the harder 99% and 99.5% background-rejection regimes.

Generated plots and summary table:

```text
analysis/figures/scaling_1m_vbf_hh4b_995/
```

Per-signal `granular_sm13 50k` MinBias-distance results:

| signal | AUC | eff@95%BG | eff@99%BG |
|---|---:|---:|---:|
| hh4b | 0.5456 | 0.0330 | 0.0045 |
| hh | 0.5424 | 0.0265 | 0.0055 |
| vbfh | 0.5995 | 0.0545 | 0.0065 |
| ggh | 0.5551 | 0.0605 | 0.0155 |

Per-signal `granular_sm13 100k` MinBias-distance results:

| signal | AUC | eff@95%BG | eff@99%BG |
|---|---:|---:|---:|
| hh4b | 0.5179 | 0.0180 | 0.0025 |
| hh | 0.5345 | 0.0130 | 0.0015 |
| vbfh | 0.5864 | 0.0365 | 0.0065 |
| ggh | 0.5765 | 0.0880 | 0.0120 |

Supporting anomaly scores for `granular_sm13 50k`:

| method | mean AUC | eff@95%BG | eff@99%BG |
|---|---:|---:|---:|
| GMM NLL | 0.5692 | 0.0404 | 0.0074 |
| MinBias Mahalanobis | 0.5606 | 0.0436 | 0.0080 |
| combined robust score | 0.5680 | 0.0438 | 0.0076 |

Supporting anomaly scores for `granular_sm13 100k`:

| method | mean AUC | eff@95%BG | eff@99%BG |
|---|---:|---:|---:|
| GMM NLL | 0.5445 | 0.0389 | 0.0061 |
| MinBias Mahalanobis | 0.5538 | 0.0389 | 0.0056 |
| combined robust score | 0.5574 | 0.0381 | 0.0061 |

## Focused Scaling Diagnostics

The first copied large/XL `4class` runs finished, but should still be treated as failed/degenerate for the anomaly-separation study.

| class config | samples/class | model size | latent_dim | ff_dim | blocks | probe | status | diagnostic issue |
|---|---:|---|---:|---:|---:|---|---|---|
| granular_sm13 | 50k | ff128 | 64 | 128 | 2 | linear | usable | full external metrics available |
| granular_sm13 | 100k | ff256 | 128 | 256 | 2 | linear | usable | full external metrics available |
| 4class | 150k | large | 192 | 384 | 3 | linear | not usable yet | zero latent variance; chance-level Fisher accuracy; GMM NaNs |
| 4class | 200k | XL | 256 | 512 | 4 | linear | not usable yet | zero latent variance; chance-level Fisher accuracy; GMM NaNs |

Diagnostics from the 150k/200k runs:

| run | latent variance total | latent participation ratio | Fisher mean loss | Fisher accuracy | external evaluation |
|---|---:|---:|---:|---:|---|
| 4class 150k large linear | 0.0 | 0.0 | 1.3863 | 0.25 | failed: NaNs in GMM input |
| 4class 200k XL linear | 0.0 | 0.0 | 1.3863 | 0.25 | failed: NaNs in GMM input |

Interpretation:

> The 150k/200k models currently look degenerate for this analysis. `loss ~= log(4)`, `accuracy ~= 0.25`, and zero latent variance indicate that the checkpoint used by the diagnostics is not learning a useful class-dependent representation. The resulting NaNs in PCA/GMM are downstream symptoms, not physics conclusions.

These failed copied runs should not be used to claim that more samples or larger encoder capacity worsens anomaly detection. They only show that the current large-run setup/checkpoint needed investigation.

The later focused runs under `main_4class_scaling/4class` are the usable scaling evidence. The cleanest direct augmentation check is the 200k `ff512_blocks2` pair:

| run | augmentation copies | lr | mean MinBias AUC | eff@95%BG | eff@99%BG | interpretation |
|---|---:|---:|---:|---:|---:|---|
| `200k ff512_blocks2 linear` | 1 | 1e-3 | 0.7869 | 0.1630 | 0.0535 | usable, but weaker high-rejection efficiency |
| `200k ff512_blocks2 aug0` | 0 | 1e-3 | 0.7760 | 0.1906 | 0.0774 | similar AUC, better fixed-background efficiencies |

At 1M/class, the augmentation/default-LR path is not usable:

| run | augmentation copies | lr | status | issue |
|---|---:|---:|---|---|
| `1M ff512_blocks2_ce_varcov` | 1 | 1e-3 | failed | zero latent variance; no valid anomaly point |
| `1M ff512_blocks2_ce_only_noaug` | 0 | 1e-3 | failed | random/chance 4-class behavior |
| `1M ff512_blocks2_ce_varcov_noaug` | 0 | 1e-3 | unstable | early nontrivial accuracy followed by collapse |
| `1M low-LR noaug variants` | 0 | 3e-5 | usable | stable training and complete anomaly metrics |

The usable 1M comparison is therefore no-augmentation plus low learning rate. CE-only and CE+var/cov are close in mean AUC; var/cov improves effective latent rank, and for the `ff256` model improves `eff@99%BG`.

## Analysis Questions

### 1. Does the classifier probe affect latent geometry?

Yes. The probe changes the effective latent dimensionality.

Examples:

- `4class 50k`
  - `linear`: effective rank 9.49
  - `mlp`: effective rank 10.56
  - `mlp2`: effective rank 13.81

- `10plus_sm 50k`
  - `linear`: effective rank 14.41
  - `mlp`: effective rank 20.50
  - `mlp2`: effective rank 17.35

This supports the idea that the classifier head can influence latent collapse.

However, the direction is not fully stable. At `4class 100k`, `mlp2` has an unusually low effective rank:

- `linear`: effective rank 8.30
- `mlp`: effective rank 12.80
- `mlp2`: effective rank 4.40

Conclusion:

> Probe architecture affects latent geometry, but it is not a reliable standalone indicator of anomaly sensitivity.

### 2. Does reducing latent collapse improve anomaly separation?

Not consistently.

Examples:

- `10plus_sm 50k mlp` has higher effective rank than `10plus_sm 50k linear`, but lower MinBias-distance AUC:
  - `mlp`: rank 20.50, AUC 0.5682
  - `linear`: rank 14.41, AUC 0.5890

- `10plus_sm 100k mlp` is worse than `10plus_sm 100k linear`:
  - `mlp`: AUC 0.4565, eff@95%BG 0.0190
  - `linear`: AUC 0.5744, eff@95%BG 0.0377

- `4class 100k mlp` has higher effective rank than `4class 100k linear`, but worse MinBias-distance AUC:
  - `mlp`: rank 12.80, AUC 0.6463
  - `linear`: rank 8.30, AUC 0.7830

Conclusion:

> A richer or less collapsed latent space does not automatically produce better MinBias-vs-anomaly separation.

### 3. Does increasing class diversity improve anomaly separation?

Not in the completed tests.

Matched comparisons:

| comparison | 4class AUC | 10plus_sm AUC |
|---|---:|---:|
| 50k linear | 0.7836 | 0.5890 |
| 50k mlp | 0.7603 | 0.5682 |
| 100k linear | 0.7830 | 0.5744 |
| 100k mlp | 0.6463 | 0.4565 |

The `4class` setup gives larger apparent separation from MinBias than `10plus_sm`.

The new `granular_sm13 50k linear` cross-check is consistent with this trend:

| comparison | mean MinBias-distance AUC |
|---|---:|
| 4class 50k linear | 0.7836 |
| 10plus_sm 50k linear | 0.5890 |
| granular_sm13 50k linear | 0.5606 |
| granular_sm13 100k linear | 0.5538 |

Caveat:

> This does not prove that `4class` is a generally better anomaly-detection model. Because the labels are coarse, injected Higgs-like samples can be mapped into a broad non-MinBias/top-like region, increasing distance from MinBias.

Possible interpretation:

> `10plus_sm` may learn a broader and more realistic SM manifold. In that manifold, injected Higgs-like probes are less artificially isolated from MinBias, reducing apparent anomaly separation.

Conclusion:

> Increasing class diversity changes the latent representation, but does not automatically improve MinBias-vs-anomaly separation for the current Higgs-like probes.

The `granular_sm13` results strengthen this statement: making the SM labels more detailed did not improve the MinBias-vs-Higgs-like separation at either 50k/class or 100k/class. This is still a cross-check, not a complete diversity study, but it supports Maciej's decision to prioritize the simpler 4class scaling path for the current use case.

### 4. Does increasing statistics improve separation?

The original matched 50k-to-100k comparisons did not improve. The later tuned 150k/200k/1M 4class runs do show stronger fixed-background-efficiency performance when the run is stable.

| setup | 50k AUC | 100k AUC | interpretation |
|---|---:|---:|---|
| 4class linear | 0.7836 | 0.7830 | stable |
| 4class mlp | 0.7603 | 0.6463 | worse |
| 10plus_sm linear | 0.5890 | 0.5744 | slightly worse |
| 10plus_sm mlp | 0.5682 | 0.4565 | worse |

Conclusion:

> More statistics alone did not improve the first 50k-to-100k matched comparisons, but the later tuned 150k/200k/1M 4class scaling runs do improve the fixed-background-efficiency regime when training is stable.

The key difference is not only sample count. The usable larger runs combine focused 4class labels, a fixed linear probe, controlled width/depth, and either no augmentation or lower learning rate. The 1M results are the strongest evidence so far that scaling can help once optimization is stabilized.

## Updated Interpretation After Maciej Feedback

Maciej's current working interpretation:

> For the current Higgs-like anomaly probes, the physics needed for MinBias separation may already be captured by the coarse classes WJets/DY/QCD/Top. More classes do not necessarily help, and there is not enough time to fully study the class-diversity axis now.

Operational decision:

- Focus on `4class`.
- Keep probe simple/fixed.
- Scale samples and encoder parameters.
- Treat 10plus/granular as cross-checks.

This does not invalidate the class-diversity study. Instead, it turns it into evidence supporting a narrower scaling plan.

## Active Scaling Plan

Main path:

| run | classes | samples/class | model size | probe | status |
|---|---|---:|---|---|---|
| `4class_50k_ff128_linear` | 4 SM | 50k | ff128 | linear | completed |
| `4class_100k_ff256_linear` | 4 SM | 100k | ff256 | linear | completed |
| `4class_150k_ff256/ff384/depth_linear` | 4 SM | 150k | tuned ff256/ff384 | linear | completed, usable |
| `4class_200k_ff256/ff512/depth_linear` | 4 SM | 200k | tuned ff256/ff512 | linear | completed, usable |
| `4class_1m_low_lr_noaug_linear` | 4 SM | 1M | ff256/ff512 | linear | completed, usable |
| `4class_80m_100m_ddp_noaug_varcov` | 4 SM | 20M | ~100M MHA | mlp | prepared, not launched |

Requested next sweep after the 1M CE+var/cov result:

| point | target params | actual params | total samples | events/class | ff_dim |
|---|---:|---:|---:|---:|---:|
| `plaw_p1e5_n1e4` | 100,000 | 94,717 | 10,000 | 2,500 | 88 |
| `plaw_p5e5_n5e4` | 500,000 | 485,077 | 50,000 | 12,500 | 208 |
| `plaw_p2p5e6_n2p5e5` | 2,500,000 | 2,489,445 | 250,000 | 62,500 | 480 |
| `plaw_p1e7_n1e6` | 10,000,000 | 9,979,757 | 1,000,000 | 250,000 | 968 |

This sweep is organized under `scaling_study/power_law_param_sample/`. It tests fixed-background efficiency as a function of model size and sample count scaled in tandem, following the qualitative Chinchilla-style idea of scaling data and parameters together. In this event-level setup, the literal language-model token/parameter ratio is not used; the requested grid keeps `N_samples / N_params ~= 0.1`.

Model-capacity schedule:

| samples/class | model size | latent_dim | ff_dim | attention blocks | heads |
|---:|---|---:|---:|---:|---:|
| 50k | ff128 | 64 | 128 | 2 | 8 |
| 100k | ff256 | 128 | 256 | 2 | 8 |
| 150k | tuned ff256/ff384 | 128 | 256/384 | 2/3 | 8 |
| 200k | tuned ff256/ff512 | 128 | 256/512 | 2/3 | 8 |
| 1M | ff256/ff512 low-LR noaug | 128 | 256/512 | 2 | 8 |
| 20M | ~100M DDP target | 256 | 2304 | 4 | 16 |

## 100M-Parameter / 80M-Event Preparation

The large-run preparation is now organized under:

```text
ctl2_embedding_model/scaling_study/large_100m_80m/
```

Prepared target:

| item | value |
|---|---:|
| total events | 80M |
| events/class | 20M |
| GPUs | 4 x 48 GB |
| events/rank | 20M total, 5M/class |
| model | MHA, `latent_dim=256`, `ff_dim=2304`, 4 attention blocks, 16 heads |
| estimated trainable params | 99,454,725 |
| loading mode | `rank_sharded_materialized` |
| objective | CE + variance/covariance |
| augmentation | 0 |
| learning rate | `3e-5` |

The key implementation change is rank-sharded loading. Each DDP rank now loads a deterministic shard of valid events before materialization, instead of every rank loading the full requested dataset and relying on `DistributedSampler` afterward. Normalization constants and PID mappings are synchronized across ranks before preprocessing.

This does not make the run cheap. The raw 80M-event tensor is about 149 GiB total, or about 37 GiB per rank before preprocessing. Since the current preprocessing keeps raw and preprocessed tensors resident, and PID one-hot expands the effective input width, the full run requires large host RAM and must stay at `augmentation_copies=0` unless memory is reworked further.

Two large run configs are prepared as reproducible seed replicas:

| run | seed | config |
|---|---:|---|
| A | 42 | `scaling_study/large_100m_80m/configs/run_100m_80m_A.json` |
| B | 43 | `scaling_study/large_100m_80m/configs/run_100m_80m_B.json` |

The exact second physics/hyperparameter comparison axis was not specified, so no new hyperparameter difference was invented. The augmentation/no-augmentation comparison remains a small smoke test only; the 1M evidence indicates that default-LR augmented runs can become unstable, while the stable large-sample recipe is low-LR and no augmentation.

External Higgs evaluation now records:

- `eff_at_95pct_bg`
- `eff_at_99pct_bg`
- `eff_at_99_5pct_bg`

Use `eff_at_99_5pct_bg` as the HH4B 99.5% background-rejection metric.

## Required Diagnostics for Each Scaling Run

For every completed scaling run, generate:

- `minbias_vs_anomalies_latent_pca.png`
- `minbias_vs_anomalies_latent_tsne.png`
- `minbias_distance_distributions.png`
- `minbias_distance_distributions_per_signal.png`
- `external_higgs_anomaly_metrics_by_method.csv`
- `latent_effective_dim_summary.json`
- `latent_effective_dim_spectrum.png`
- classifier-response summary/matrix

The final scaling table should include:

- class configuration;
- samples per class;
- model size;
- parameter count;
- latent dimension;
- attention depth;
- anomaly method;
- signal;
- AUC;
- eff@95%BG;
- eff@99%BG;
- eff@99.5%BG;
- effective rank;
- PCA dims 90/95/99.

Additional checks are now required before accepting a large scaling run:

- confirm nonzero latent variance before fitting PCA/GMM;
- confirm validation accuracy is above chance;
- confirm `eval_latents.npy` is finite (`NaN`/`Inf` free);
- if zero variance appears, inspect checkpoint loading, learning-rate stability, and whether the saved `best` checkpoint is from a non-learning epoch.

## Presentation Notes

Recommended story:

> PCA/t-SNE show the latent structure, but the physics-use-case metric is full-dimensional separation from MinBias. The current evidence suggests that more class diversity and less latent collapse do not automatically improve anomaly sensitivity. Therefore the main next step is to scale the 4class baseline in samples and encoder capacity with a fixed/simple classifier head.

Recommended plots:

1. `4class 100k linear` MinBias-vs-anomalies t-SNE.
2. `10plus_sm 100k linear` MinBias-vs-anomalies t-SNE.
3. `4class 100k linear` MinBias-distance distribution.
4. `10plus_sm 100k linear` MinBias-distance distribution.
5. Compact MinBias-distance metric table.
6. Optional latent effective-rank plot to show that rank and anomaly separation are not equivalent.

For the current presentation, do not use the copied 150k/200k large/XL zero-variance plots as positive scaling results. Use the focused `main_4class_scaling/4class` runs for the updated scaling story:

- `4class 50k/100k linear` as the stable baseline;
- `10plus_sm 50k/100k linear` for class-diversity comparison;
- `granular_sm13 50k linear` as the extra granularity cross-check;
- `200k ff512_blocks2` with `augmentation_copies=1` vs `augmentation_copies=0` as the direct augmentation check;
- the four `1M noaug lr3e-5` rows as the strongest current scaling evidence;
- a caveat slide stating that the first copied 150k/200k large-scaling attempts and default-LR 1M augmented attempts produced degenerate or unstable outputs.

## Caveats

- The current conclusions are specific to HH4B/HH/VBFH/ggH injected probes.
- 4class separation may be inflated by coarse class assignment into non-MinBias/top-like regions.
- `10plus_sm` may be more physically realistic even if its MinBias-distance score is lower.
- PCA/t-SNE are qualitative and should not be used as quantitative evidence alone.
- `granular_sm15` is not currently valid for `collide2v_updated` because the DY subchannels do not match EOS parquet files.
- `granular_sm13 100k` now has complete external-anomaly metrics; it does not show an improvement over `granular_sm13 50k`.
- The first `4class 150k/200k` large-scaling runs completed but produced zero latent variance in diagnostics; they are not valid anomaly-separation evidence yet.
- The valid 1M points are no-augmentation low-LR runs; augmented/default-LR 1M attempts are stability evidence, not physics evidence.
- The current study is offline representation analysis, not an online trigger-throughput or FPGA deployment result.

## Short Conclusion

The evidence so far supports a focused scaling strategy:

> Use the 4class model as the practical baseline, keep the classifier probe simple, and test whether increasing samples plus encoder capacity improves full-dimensional MinBias-vs-anomaly separation.

The broad class-diversity and probe-ablation studies remain useful cross-checks, but they should not block the main 4class scaling path.

Update after the latest runs:

> The 50k and 100k `granular_sm13` cross-checks further support the statement that more class granularity does not automatically improve the MinBias-vs-Higgs-like anomaly metric. The usable 150k/200k/1M `4class` follow-ups show that scaling can improve the MinBias-vs-Higgs-like metric when the run is stable. For 1M/class, the stable recipe is `augmentation_copies=0` with `lr=3e-5`; CE+var/cov remains attractive because it raises effective latent rank without reducing mean AUC.
