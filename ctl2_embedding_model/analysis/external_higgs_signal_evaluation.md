# External Higgs Signal Evaluation

This note summarizes the first evaluation of external Higgs-related signal groups using two selected 10-class MHA models and one 4-class comparison model:

- **Balanced model:** CE + SupCon 0.05 + latent variance regularization, `latent_dim=128`.
- **Anomaly-oriented model:** CE + latent variance + covariance regularization, `latent_dim=128`.
- **4-class comparison model:** CE + SupCon 0.05 + latent variance regularization, `latent_dim=128`, trained only on `minbias`, `qcd`, `dy`, and `top`.

The background used in the anomaly evaluation is the full SM superclass set: `minbias`, `qcd`, `dy`, `top`, `wjets`, `ww`, `wz`, `ttw`, and `ttz`. The evaluated Higgs-related signal groups are `hh4b`, `hh`, `vbfh`, and `ggh`, using only L1 candidate features from updated Collide-2V.

Important distinction:

- `hh4b` is part of the trained label space, so its classifier response can be interpreted as a real `HH4B` recall/tagging rate.
- `hh`, `vbfh`, and `ggh` are external/unseen signal groups. For these samples, the classifier output is not a confusion matrix or accuracy. It is a classifier-response map showing which trained SM/HH4B class each unseen event is mapped to.
- In the 4-class model, `hh4b` is also external because the classifier has no `hh4b` output node. Therefore, no `HH4B` tag rate or `mean_hh4b_probability` is defined for that comparison.

## Summary Table

| model                                       | signal   |    score |   mahalanobis |      gmm_gap |   participation_ratio |   align_loss |   n_eval_events | json_path                                                                                                                                    |
|:--------------------------------------------|:---------|---------:|--------------:|-------------:|----------------------:|-------------:|----------------:|:---------------------------------------------------------------------------------------------------------------------------------------------|
| Anomaly-oriented: CE + var 0.01 + cov 0.001 | hh4b     | 0.003646 |  8600         |    3.156e+10 |                 1     |       0.8061 |            3000 | ngt_mha_nodes_aligned_outputs\runs\2026-08-05\sm_superclasses_ce_var001_cov0001_100k_latent128\eval_quality_external_hh4b_n2000.json         |
| Balanced: CE + SupCon 0.05 + var 0.01       | hh4b     | 0.3064   |     2.589     | 7936         |               124.7   |       0.8252 |            3000 | ngt_mha_nodes_aligned_outputs\runs\2026-08-04\sm_superclasses_balanced_supcon005_var001_100k_latent128\eval_quality_external_hh4b_n2000.json |
| Anomaly-oriented: CE + var 0.01 + cov 0.001 | hh       | 0.2616   |     1.498     |  791.9       |               115.2   |       0.8263 |            3000 | ngt_mha_nodes_aligned_outputs\runs\2026-08-05\sm_superclasses_ce_var001_cov0001_100k_latent128\eval_quality_external_hh_n2000.json           |
| Balanced: CE + SupCon 0.05 + var 0.01       | hh       | 0.2562   |     1.423     |    2.191e+04 |               124.7   |       0.8446 |            3000 | ngt_mha_nodes_aligned_outputs\runs\2026-08-04\sm_superclasses_balanced_supcon005_var001_100k_latent128\eval_quality_external_hh_n2000.json   |
| Anomaly-oriented: CE + var 0.01 + cov 0.001 | vbfh     | 0.2227   |     1.973     | 4953         |                49.54  |       0.8368 |            3000 | ngt_mha_nodes_aligned_outputs\runs\2026-08-05\sm_superclasses_ce_var001_cov0001_100k_latent128\eval_quality_external_vbfh_n2000.json         |
| Balanced: CE + SupCon 0.05 + var 0.01       | vbfh     | 0.1086   |     6.922e+04 |    1.523e+12 |                 1.117 |       0.8492 |            3000 | ngt_mha_nodes_aligned_outputs\runs\2026-08-04\sm_superclasses_balanced_supcon005_var001_100k_latent128\eval_quality_external_vbfh_n2000.json |
| Anomaly-oriented: CE + var 0.01 + cov 0.001 | ggh      | 0.08293  |     9.064e+05 |    1.175e+14 |                 1.037 |       0.8368 |            3000 | ngt_mha_nodes_aligned_outputs\runs\2026-08-05\sm_superclasses_ce_var001_cov0001_100k_latent128\eval_quality_external_ggh_n2000.json          |
| Balanced: CE + SupCon 0.05 + var 0.01       | ggh      | 0.002119 |     2.613e+04 |    3.211e+11 |                 1     |       0.8508 |            3000 | ngt_mha_nodes_aligned_outputs\runs\2026-08-04\sm_superclasses_balanced_supcon005_var001_100k_latent128\eval_quality_external_ggh_n2000.json  |

## Classifier Response Summary

These values are generated from the classifier logits and are row-normalized by evaluated signal sample. For unseen signals, the key quantity is the `HH4B tag rate`: the fraction of external events mapped to the trained `hh4b` output node.

| Model | Signal | Top mapped class | Top fraction | HH4B tag rate | Mean HH4B probability |
|:--|:--|:--|--:|--:|--:|
| Balanced: CE + SupCon 0.05 + var 0.01 | hh4b | hh4b | 0.6205 | 0.6205 | 0.4033 |
| Balanced: CE + SupCon 0.05 + var 0.01 | hh | ww | 0.2560 | 0.1725 | 0.1329 |
| Balanced: CE + SupCon 0.05 + var 0.01 | vbfh | wz | 0.4135 | 0.0115 | 0.0294 |
| Balanced: CE + SupCon 0.05 + var 0.01 | ggh | wz | 0.2885 | 0.0100 | 0.0243 |
| Anomaly-oriented: CE + var 0.01 + cov 0.001 | hh4b | hh4b | 0.5730 | 0.5730 | 0.3811 |
| Anomaly-oriented: CE + var 0.01 + cov 0.001 | hh | top | 0.2595 | 0.1520 | 0.1303 |
| Anomaly-oriented: CE + var 0.01 + cov 0.001 | vbfh | wz | 0.3980 | 0.0180 | 0.0379 |
| Anomaly-oriented: CE + var 0.01 + cov 0.001 | ggh | wz | 0.2790 | 0.0125 | 0.0275 |

## 4-Class Comparison

The 4-class comparison model was trained with the same balanced objective family, `CE + SupCon 0.05 + latent variance regularization`, but only using `minbias`, `qcd`, `dy`, and `top` as output labels. This makes it useful for checking how external Higgs-like samples are projected into a smaller SM label space.

Because this model has no `hh4b` output node, the `hh4b`, `hh`, `vbfh`, and `ggh` samples cannot be assigned an `HH4B` tag rate. The table below reports the dominant trained class for each external signal.

| Model | Signal | Top mapped class | Top fraction | HH4B tag rate | Mean HH4B probability |
|:--|:--|:--|--:|--:|--:|
| 4-class: CE + SupCon 0.05 + var 0.01 | hh4b | top | 0.9400 | N/A | N/A |
| 4-class: CE + SupCon 0.05 + var 0.01 | hh | top | 0.8350 | N/A | N/A |
| 4-class: CE + SupCon 0.05 + var 0.01 | vbfh | top | 0.4850 | N/A | N/A |
| 4-class: CE + SupCon 0.05 + var 0.01 | ggh | qcd | 0.4125 | N/A | N/A |

The 4-class model maps `hh4b`, `hh`, and `vbfh` mostly to `top`, while `ggh` is mostly mapped to `qcd`. This suggests that, without a trained `hh4b` node, the model mostly interprets Higgs-related activity through broad SM-like topologies rather than as a dedicated Higgs-like class.

### 4-Class Latent Diagnostics

The 4-class comparison model also has latent and classifier-head Fisher diagnostics:

| diagnostic | value |
|:--|--:|
| latent_dim | 128 |
| latent effective rank | 2.6762 |
| latent participation ratio | 2.1270 |
| top-1 explained variance | 0.5986 |
| PCs for 90% variance | 2 |
| PCs for 95% variance | 3 |
| head-Fisher effective rank | 5.2610 |
| head-Fisher participation ratio | 3.4761 |
| head-Fisher dims for 90% mass | 4 |
| head-Fisher dims for 95% mass | 9 |

This confirms that the 4-class model is highly compressed in latent space despite `latent_dim=128`: only **2 principal components explain about 90%** of the latent variance, and only **3 components explain about 95%**. This is consistent with the earlier observation that coarse label spaces tend to activate fewer latent directions.

## VICReg-Style Objective Check

Two additional 10-class models were evaluated as a first VICReg-style objective check:

- **VICReg-style 64:** VICReg-style regularization + latent variance regularization, `latent_dim=64`.
- **VICReg-style 128:** VICReg-style regularization + latent variance regularization, `latent_dim=128`.

These runs use the same 10-class superclass setup as the balanced and anomaly-oriented models: `minbias`, `qcd`, `dy`, `top`, `wjets`, `ww`, `wz`, `ttw`, `ttz`, and `hh4b`.

### Classifier Response on External Higgs Signals

| Model | Signal | Top mapped class | Top fraction | HH4B tag rate | Mean HH4B probability |
|:--|:--|:--|--:|--:|--:|
| VICReg-style + var 0.01, latent_dim=64 | hh4b | hh4b | 0.6385 | 0.6385 | 0.4173 |
| VICReg-style + var 0.01, latent_dim=64 | hh | ww | 0.2330 | 0.1695 | 0.1363 |
| VICReg-style + var 0.01, latent_dim=64 | vbfh | wz | 0.4460 | 0.0155 | 0.0316 |
| VICReg-style + var 0.01, latent_dim=64 | ggh | wz | 0.3250 | 0.0095 | 0.0238 |
| VICReg-style + var 0.01, latent_dim=128 | hh4b | hh4b | 0.5915 | 0.5915 | 0.3835 |
| VICReg-style + var 0.01, latent_dim=128 | hh | ww | 0.2485 | 0.1505 | 0.1205 |
| VICReg-style + var 0.01, latent_dim=128 | vbfh | wz | 0.4555 | 0.0125 | 0.0283 |
| VICReg-style + var 0.01, latent_dim=128 | ggh | wz | 0.3255 | 0.0100 | 0.0227 |

The VICReg-style `latent_dim=64` run gives the strongest trained-`hh4b` classifier response among the evaluated models in this report: `HH4B` tag rate **0.6385** and mean `HH4B` probability **0.4173**. However, this improvement is mostly visible for the trained `hh4b` sample. The unseen `hh`, `vbfh`, and `ggh` samples are still not dominantly mapped to `HH4B`.

### HH4B Anomaly-Quality Diagnostic

| Model | latent_dim | eval-quality score | Mahalanobis | GMM gap | participation ratio | n_params |
|:--|--:|--:|--:|--:|--:|--:|
| VICReg-style + var 0.01 | 64 | 0.1098 | 2.4526 | 57.3004 | 62.1534 | 177226 |
| VICReg-style + var 0.01 | 128 | 0.1702 | 2.5409 | 188397.7335 | 120.7989 | 186122 |

The `latent_dim=128` VICReg-style model improves the composite eval-quality score relative to the `latent_dim=64` VICReg-style model (**0.1702** vs **0.1098**) and has a much larger GMM gap. At the same time, its classifier `HH4B` tag rate is lower than the 64-dimensional version. This is another example where classifier response and latent-geometry metrics do not fully agree.

### Latent and Head-Fisher Diagnostics

| Model | latent_dim | latent effective rank | latent participation ratio | PCs for 90% variance | PCs for 95% variance | Head-Fisher effective rank | Head-Fisher participation ratio |
|:--|--:|--:|--:|--:|--:|--:|--:|
| VICReg-style + var 0.01 | 64 | 7.2298 | 4.4192 | 8 | 10 | 61.0454 | 27.3493 |
| VICReg-style + var 0.01 | 128 | 6.3132 | 3.1948 | 8 | 11 | 89.2152 | 48.7884 |

The latent PCA diagnostics still show compression: even with `latent_dim=128`, only **8 components explain 90%** of the latent variance. The classifier-head Fisher diagnostic is more active than the PCA view, especially for `latent_dim=128`, where the Head-Fisher effective rank reaches **89.2152**. This suggests that the classifier head uses more parameter directions than the global latent variance alone suggests, but the latent representation itself is still concentrated in a relatively small number of dominant components.

## Key Observations

- The largest composite eval-quality score in this comparison is **0.3064** for **HH4B** with the **Balanced: CE + SupCon 0.05 + var 0.01** model.
- The largest Mahalanobis distance is **9.064e+05** for **GGH** with the **Anomaly-oriented: CE + var 0.01 + cov 0.001** model.
- The balanced SupCon+variance model gives more stable composite scores across the external Higgs signals.
- The covariance-regularized model can produce very large GMM gaps for some signals, but the composite score may remain small when the participation-ratio sub-score is poor. This should be interpreted as a geometry diagnostic, not as a standalone physics-performance conclusion.
- For classifier response, both models correctly map the trained `hh4b` sample mostly to `hh4b`, with a stronger `HH4B` tag rate in the balanced model: **0.6205** vs **0.5730**.
- The external `hh` sample is partially mapped to `HH4B`, but not dominantly: **0.1725** for the balanced model and **0.1520** for the anomaly-oriented model. Its largest mapped class is `ww` for the balanced model and `top` for the anomaly-oriented model.
- `vbfh` and `ggh` are not strongly mapped to `HH4B`; both models mainly map them to `wz`, with low `HH4B` tag rates near **1-2%**.
- In the 4-class comparison, external Higgs-like samples are absorbed mainly by `top` or `qcd`, confirming that the 4-class classifier cannot provide a direct `HH4B` tag. It is still useful as a background-only projection baseline.
- The 4-class latent diagnostics show stronger compression than the 10-class granular/superclass models: the 128-dimensional latent space effectively uses only a few dominant directions.
- The VICReg-style `latent_dim=64` run improves the classifier response for trained `hh4b` relative to the balanced and anomaly-oriented 10-class models, reaching an `HH4B` tag rate of **0.6385**.
- The VICReg-style `latent_dim=128` run improves the eval-quality score relative to the VICReg-style 64-dimensional run, but does not improve the classifier `HH4B` tag rate.
- These results are evaluation-only tests: `hh`, `vbfh`, and `ggh` were not added as training labels in these two runs.

## Interpretation

The balanced SupCon+variance model is the stronger candidate if the immediate goal is to preserve a stable `HH4B` classifier response while keeping reasonable external-signal behavior. It gives the best composite score on trained `HH4B` and the highest `HH4B` tag rate for `hh4b` and `hh`.

The covariance-regularized model remains useful as an anomaly-geometry comparison. It can produce very large Mahalanobis/GMM separations for some external Higgs samples, especially `ggh`, but those large values do not always translate into a stronger classifier mapping to `HH4B`.

The most important physics takeaway is that external Higgs samples do not all behave like the trained `HH4B` class at L1. `HH` has some overlap with `HH4B`, while `VBFH` and `GGH` are mostly mapped to electroweak-like background classes, especially `WZ`. This supports keeping both views in the presentation: anomaly-score geometry and classifier-response mapping.

The VICReg-style check adds one more useful observation: it can strengthen the trained-`HH4B` classifier response, especially at `latent_dim=64`, but it does not automatically make all external Higgs processes look `HH4B`-like. For anomaly detection, this means the next comparison should keep both metrics visible: the classifier-response map and the latent anomaly score.

## Figures

- `analysis/figures/external_higgs_signal_eval/external_higgs_eval_quality_score.png`
- `analysis/figures/external_higgs_signal_eval/external_higgs_mahalanobis.png`
- `analysis/figures/external_higgs_signal_eval/external_higgs_gmm_gap_log.png`
- `analysis/figures/external_higgs_signal_eval/external_higgs_participation_ratio.png`
- `analysis/figures/external_higgs_signal_eval/external_higgs_eval_quality_subscores_heatmap.png`
- `ngt_mha_nodes_aligned_outputs/runs/2026-08-04/sm_superclasses_balanced_supcon005_var001_100k_latent128/figures/external_classifier_response/classifier_response_matrix.png`
- `ngt_mha_nodes_aligned_outputs/runs/2026-08-04/sm_superclasses_balanced_supcon005_var001_100k_latent128/figures/external_classifier_response/hh4b_tag_rate_by_signal.png`
- `ngt_mha_nodes_aligned_outputs/runs/2026-08-05/sm_superclasses_ce_var001_cov0001_100k_latent128/figures/external_classifier_response/classifier_response_matrix.png`
- `ngt_mha_nodes_aligned_outputs/runs/2026-08-05/sm_superclasses_ce_var001_cov0001_100k_latent128/figures/external_classifier_response/hh4b_tag_rate_by_signal.png`
- `ngt_mha_nodes_aligned_outputs/runs/2026-08-06/fourclass_balanced_supcon005_var001_100k_latent128/figures/external_classifier_response/classifier_response_matrix.png`
- `ngt_mha_nodes_aligned_outputs/runs/2026-08-06/fourclass_balanced_supcon005_var001_100k_latent128/figures/external_classifier_response/classifier_mean_probability_matrix.png`
- `ngt_mha_nodes_aligned_outputs/runs/2026-08-06/fourclass_balanced_supcon005_var001_100k_latent128/figures/latent_effective_dim_spectrum.png`
- `ngt_mha_nodes_aligned_outputs/runs/2026-08-06/fourclass_balanced_supcon005_var001_100k_latent128/figures/head_fisher_diag.png`
- `ngt_mha_nodes_aligned_outputs/runs/2026-08-06/sm_superclasses_vicreg001_var001_100k_latent64/figures/external_classifier_response/classifier_response_matrix.png`
- `ngt_mha_nodes_aligned_outputs/runs/2026-08-06/sm_superclasses_vicreg001_var001_100k_latent64/figures/external_classifier_response/hh4b_tag_rate_by_signal.png`
- `ngt_mha_nodes_aligned_outputs/runs/2026-08-06/sm_superclasses_vicreg001_var001_100k_latent64/figures/latent_effective_dim_spectrum.png`
- `ngt_mha_nodes_aligned_outputs/runs/2026-08-06/sm_superclasses_vicreg001_var001_100k_latent64/figures/head_fisher_diag.png`
- `ngt_mha_nodes_aligned_outputs/runs/2026-08-06/sm_superclasses_vicreg001_var001_100k_latent128/figures/external_classifier_response/classifier_response_matrix.png`
- `ngt_mha_nodes_aligned_outputs/runs/2026-08-06/sm_superclasses_vicreg001_var001_100k_latent128/figures/external_classifier_response/hh4b_tag_rate_by_signal.png`
- `ngt_mha_nodes_aligned_outputs/runs/2026-08-06/sm_superclasses_vicreg001_var001_100k_latent128/figures/latent_effective_dim_spectrum.png`
- `ngt_mha_nodes_aligned_outputs/runs/2026-08-06/sm_superclasses_vicreg001_var001_100k_latent128/figures/head_fisher_diag.png`

## Data Products

- `analysis/external_higgs_signal_eval_summary.csv`
- `analysis/external_higgs_classifier_response_summary.csv`
- `ngt_mha_nodes_aligned_outputs/runs/2026-08-04/sm_superclasses_balanced_supcon005_var001_100k_latent128/figures/external_classifier_response/classifier_response_summary.csv`
- `ngt_mha_nodes_aligned_outputs/runs/2026-08-05/sm_superclasses_ce_var001_cov0001_100k_latent128/figures/external_classifier_response/classifier_response_summary.csv`
- `ngt_mha_nodes_aligned_outputs/runs/2026-08-06/fourclass_balanced_supcon005_var001_100k_latent128/figures/external_classifier_response/classifier_response_summary.csv`
- `ngt_mha_nodes_aligned_outputs/runs/2026-08-06/fourclass_balanced_supcon005_var001_100k_latent128/latent_effective_dim_summary.json`
- `ngt_mha_nodes_aligned_outputs/runs/2026-08-06/fourclass_balanced_supcon005_var001_100k_latent128/head_fisher_summary.json`
- `ngt_mha_nodes_aligned_outputs/runs/2026-08-06/sm_superclasses_vicreg001_var001_100k_latent64/figures/external_classifier_response/classifier_response_summary.csv`
- `ngt_mha_nodes_aligned_outputs/runs/2026-08-06/sm_superclasses_vicreg001_var001_100k_latent64/eval_quality_scores_n2000.json`
- `ngt_mha_nodes_aligned_outputs/runs/2026-08-06/sm_superclasses_vicreg001_var001_100k_latent64/latent_effective_dim_summary.json`
- `ngt_mha_nodes_aligned_outputs/runs/2026-08-06/sm_superclasses_vicreg001_var001_100k_latent64/head_fisher_summary.json`
- `ngt_mha_nodes_aligned_outputs/runs/2026-08-06/sm_superclasses_vicreg001_var001_100k_latent128/figures/external_classifier_response/classifier_response_summary.csv`
- `ngt_mha_nodes_aligned_outputs/runs/2026-08-06/sm_superclasses_vicreg001_var001_100k_latent128/eval_quality_scores_n2000.json`
- `ngt_mha_nodes_aligned_outputs/runs/2026-08-06/sm_superclasses_vicreg001_var001_100k_latent128/latent_effective_dim_summary.json`
- `ngt_mha_nodes_aligned_outputs/runs/2026-08-06/sm_superclasses_vicreg001_var001_100k_latent128/head_fisher_summary.json`
