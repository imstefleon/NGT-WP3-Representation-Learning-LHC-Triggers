# Quantization Readiness and Anomaly Scores

This note summarizes the quantization outlook for the current `ngt_model` workflow, records the original `eval_quality` anomaly-detection scores generated for the three runs produced on 2026-08-06, and adds the external Higgs-like plus MinBias-vs-SM anomaly evaluations across the completed balanced, covariance-regularized, VICReg-style, and four-class comparison runs.

## Quantization Readiness

The current `ngt_model` is still a full-precision PyTorch workflow. It is not quantized yet. This does not mean that the model is impossible to quantize; it means the current implementation is not yet in a quantization-aware or FPGA/HLS-ready form.

This is an important distinction from the earlier Linformer path in `ctl2_model`, which was already written with HGQ/Keras quantized layers such as `QDense`, `QAdd`, `QSum`, and `QLinformerAttention`. The current MHA model is a PyTorch implementation using standard `nn.Linear` layers and full scaled dot-product attention. Those operations can be quantized in principle, but they need an explicit quantization/export path and hardware validation.

### Quantization-friendly parts

- The raw event representation is fixed-size: `100 x 5` L1-only candidates per event. The model itself receives the frozen preprocessed tensor, typically `100 x 24` after kinematic normalization plus PID one-hot encoding.
- The model uses standard dense projections: input projection, Q/K/V projections, post-pool dense layers, latent projection, and classifier logits.
- The current preferred deployment-oriented setup uses mean pooling and a linear classifier head, which is simpler than attention pooling or an MLP head.
- Training-only losses such as SupCon, VICReg-style terms, latent variance, covariance regularization, and eval-quality diagnostics are not part of the FPGA inference graph.

### Quantization and FPGA deployment challenges

- Full multi-head attention computes an `N x N` attention matrix per head. With `N = 100`, this is significantly heavier than the earlier Linformer-style low-rank attention, where the sequence dimension was projected to `proj_k` before the attention computation.
- The attention `softmax` is the main quantization/synthesis challenge. It is not unquantizable, but it requires an approximation or a toolchain-supported implementation, and its numerical behavior must be validated after quantization.
- Attention pooling also uses a learned score and `softmax`; mean pooling avoids this second softmax and is safer for a first quantized baseline.
- The previous Linformer implementation was closer to a quantization prototype because it already used HGQ quantized layers. The current MHA path would require either porting the architecture to a quantization-aware framework or building a controlled PyTorch-to-quantized-export flow.
- The L1 preprocessing must be frozen exactly: feature order, normalization constants, PID encoding, candidate sorting, masking, and padding.
- The exported model should be inference-only: input tensor -> latent embedding -> classifier logits, without MLflow, evaluation scripts, diagnostics, or training losses.
- A future quantization validation step should compare full precision vs quantized inference on the same saved evaluation tensors.

### What is not a blocker

- Dense layers, additions, ReLU activations, mean pooling, and fixed-size tensor reshapes are standard quantization targets.
- The latent projection and classifier head are straightforward compared with the attention block.
- The anomaly-evaluation code does not need to be implemented in FPGA. PCA, GMM, ROC AUC, plotting, and MLflow are offline diagnostics, not trigger-inference logic.
- SupCon, VICReg, variance regularization, and covariance regularization are training losses only. They affect the learned weights but do not add inference-time operations.

### Practical recommendation

For the first quantization prototype, the safest version of the current PyTorch MHA family is:

- L1-only inputs,
- top-pT candidate selection,
- frozen preprocessing from raw `100 x 5` to the model's preprocessed feature tensor,
- MHA encoder with mean pooling,
- linear classifier head,
- no diagnostic/eval code in the exported inference graph.

However, if FPGA deployment is the near-term priority, the Linformer-style architecture remains the lower-risk baseline because it was already formulated with quantized HGQ layers and reduces the attention sequence cost through low-rank projection. The MHA models are useful physics/representation baselines, but they need a separate quantization study before claiming FPGA readiness.

### Message for Maciej and Sabrina

The current results do not show an unquantizable model. They show that the strongest current anomaly studies are full-precision PyTorch MHA models, whereas the older Linformer path was already closer to an HGQ/hls4ml-style quantization workflow. The main technical risk is the full attention block, especially the `N x N` attention matrix and softmax, not the dense layers or the anomaly-evaluation machinery.

A reasonable next step is to keep the MHA results as the physics benchmark, then run a focused quantization comparison:

- quantized or HGQ Linformer baseline;
- full-precision MHA baseline;
- quantized/approximated MHA prototype if feasible;
- identical evaluation tensors before and after quantization;
- compare classification metrics, latent geometry, and anomaly scores.

This would let us say whether the performance gain from full MHA justifies the additional FPGA complexity, or whether the Linformer-style low-rank attention is the better deployment candidate.

## 2026-08-06 Anomaly-Detection Scores

The scores below were read directly from each run's `eval_quality_scores_n2000.json` file and exported to:

`ngt_mha_nodes_aligned_outputs/scale_tests/anomaly_detection_scores_2026_08_06.csv`

| Run | Benchmark | latent_dim | Eval-quality score | PR norm | GMM anomaly gap | Mahalanobis distance |
|---|---:|---:|---:|---:|---:|---:|
| `fourclass_balanced_supcon005_var001_100k_latent128` | DY vs MinBias/QCD/Top | 128 | 0.1115 | 0.0086 | 82978.1896 | 43.9379 |
| `sm_superclasses_vicreg001_var001_100k_latent64` | HH4B vs 9-class SM background | 64 | 0.1098 | 0.9711 | 57.3004 | 2.4526 |
| `sm_superclasses_vicreg001_var001_100k_latent128` | HH4B vs 9-class SM background | 128 | 0.1702 | 0.9437 | 188397.7335 | 2.5409 |

## External Higgs Anomaly Evaluation

The external Higgs-like anomaly evaluation was run with:

- external samples: `hh4b`, `hh`, `vbfh`, `ggh`;
- `2,000` evaluated events per external sample;
- `2,000` events per background class;
- PCA whitening and GMM density fit only on a background fit split;
- anomaly score defined as `-log p_GMM(z)`;
- background thresholds computed on a held-out background split;
- outputs saved under each run's `figures/external_higgs_anomaly/` directory.

For 10-class runs, the background was:

`minbias, qcd, dy, top, wjets, ww, wz, ttw, ttz`

with `9,000` background-fit events and `9,000` held-out background-evaluation events. In those runs, `hh4b` is a trained reference class but is excluded from the SM background density fit. The `hh`, `vbfh`, and `ggh` samples are unseen external signals.

For the four-class comparison run, the background was:

`minbias, qcd, dy, top`

with `4,000` background-fit events and `4,000` held-out background-evaluation events. In that model all four Higgs-like samples are external with respect to the classifier label space.

The tables below were read directly from `external_higgs_anomaly_metrics.csv`; no values were estimated from plots.

### ROC AUC vs Combined SM Background

| Model family | Run | latent_dim | hh4b | hh | vbfh | ggh |
|---|---|---:|---:|---:|---:|---:|
| Balanced CE + SupCon 0.05 + var 0.01 | `sm_superclasses_balanced_supcon005_var001_100k_latent32` | 32 | 0.436 | 0.433 | 0.644 | 0.682 |
| Balanced CE + SupCon 0.05 + var 0.01 | `sm_superclasses_balanced_supcon005_var001_100k_latent64` | 64 | 0.579 | 0.571 | 0.649 | 0.635 |
| Balanced CE + SupCon 0.05 + var 0.01 | `sm_superclasses_balanced_supcon005_var001_100k_latent128` | 128 | 0.586 | 0.568 | 0.660 | 0.655 |
| CE + var 0.01 + cov 0.001 | `sm_superclasses_ce_var001_cov0001_100k_latent128` | 128 | 0.585 | 0.571 | 0.668 | 0.650 |
| VICReg-style | `sm_superclasses_vicreg001_var001_100k_latent64` | 64 | 0.599 | 0.558 | 0.654 | 0.645 |
| VICReg-style | `sm_superclasses_vicreg001_var001_100k_latent128` | 128 | 0.589 | 0.562 | 0.656 | 0.641 |
| Four-class comparison | `fourclass_balanced_supcon005_var001_100k_latent128` | 128 | 0.718 | 0.719 | 0.766 | 0.754 |

### Signal Efficiency at Fixed Background Rejection

`eff@95%BG` is the fraction of signal events above the 95th percentile of the held-out background anomaly-score distribution.

| Model family | latent_dim | hh4b | hh | vbfh | ggh |
|---|---:|---:|---:|---:|---:|
| Balanced CE + SupCon 0.05 + var 0.01 | 32 | 0.032 | 0.048 | 0.065 | 0.069 |
| Balanced CE + SupCon 0.05 + var 0.01 | 64 | 0.024 | 0.036 | 0.058 | 0.071 |
| Balanced CE + SupCon 0.05 + var 0.01 | 128 | 0.017 | 0.017 | 0.055 | 0.079 |
| CE + var 0.01 + cov 0.001 | 128 | 0.026 | 0.014 | 0.043 | 0.086 |
| VICReg-style | 64 | 0.031 | 0.017 | 0.057 | 0.078 |
| VICReg-style | 128 | 0.027 | 0.013 | 0.054 | 0.076 |
| Four-class comparison | 128 | 0.097 | 0.082 | 0.156 | 0.197 |

`eff@99%BG` is the fraction of signal events above the 99th percentile of the held-out background anomaly-score distribution.

| Model family | latent_dim | hh4b | hh | vbfh | ggh |
|---|---:|---:|---:|---:|---:|
| Balanced CE + SupCon 0.05 + var 0.01 | 32 | 0.001 | 0.005 | 0.012 | 0.019 |
| Balanced CE + SupCon 0.05 + var 0.01 | 64 | 0.002 | 0.004 | 0.005 | 0.023 |
| Balanced CE + SupCon 0.05 + var 0.01 | 128 | 0.003 | 0.004 | 0.010 | 0.020 |
| CE + var 0.01 + cov 0.001 | 128 | 0.007 | 0.002 | 0.010 | 0.019 |
| VICReg-style | 64 | 0.003 | 0.001 | 0.006 | 0.017 |
| VICReg-style | 128 | 0.003 | 0.000 | 0.012 | 0.017 |
| Four-class comparison | 128 | 0.025 | 0.022 | 0.053 | 0.061 |

### Classifier-Response Diagnostics

The classifier response is not a confusion matrix for unseen external signals. It only shows which trained class receives the largest fraction of predictions.

| Model family | latent_dim | hh4b top response | hh top response | vbfh top response | ggh top response |
|---|---:|---|---|---|---|
| Balanced CE + SupCon 0.05 + var 0.01 | 32 | `hh4b` (64.0%) | `ww` (33.6%) | `wz` (36.6%) | `wjets` (26.0%) |
| Balanced CE + SupCon 0.05 + var 0.01 | 64 | `hh4b` (64.3%) | `ww` (28.3%) | `wz` (37.8%) | `wjets` (26.1%) |
| Balanced CE + SupCon 0.05 + var 0.01 | 128 | `hh4b` (64.7%) | `ww` (33.2%) | `wz` (37.0%) | `wjets` (25.4%) |
| CE + var 0.01 + cov 0.001 | 128 | `hh4b` (61.4%) | `ww` (30.9%) | `wz` (37.8%) | `wjets` (27.1%) |
| VICReg-style | 64 | `hh4b` (65.6%) | `ww` (30.8%) | `wz` (44.0%) | `wjets` (26.2%) |
| VICReg-style | 128 | `hh4b` (62.2%) | `ww` (32.4%) | `wz` (43.0%) | `wjets` (27.1%) |
| Four-class comparison | 128 | `top` (91.6%) | `top` (81.4%) | `top` (43.0%) | `qcd` (40.9%) |

All classifier-response rows sum to 1.0 in the saved CSV files.

### External Higgs Interpretation

Across the 10-class models, `vbfh` and `ggh` are consistently more separable from the combined SM background than `hh4b` and inclusive `hh` by ROC AUC. The best 10-class AUCs are:

- `hh4b`: 0.599 from the VICReg-style `latent_dim=64` run;
- `hh`: 0.571 from the balanced `latent_dim=64` run;
- `vbfh`: 0.668 from the CE + var + cov `latent_dim=128` run;
- `ggh`: 0.682 from the balanced `latent_dim=32` run.

At fixed 95% background rejection, the 10-class efficiencies are small, typically a few percent. `ggh` is the most efficient signal in most runs, reaching 0.086 in the CE + var + cov `latent_dim=128` run and 0.079 in the balanced `latent_dim=128` run. At 99% background rejection, the 10-class efficiencies drop below 0.023 for all signals.

The four-class comparison run gives higher AUC and efficiency for all four Higgs-like samples, but it is not directly comparable to the 10-class models because the background density is fit only on `minbias`, `qcd`, `dy`, and `top`. The classifier-response matrix also shows that this model maps most `hh4b` and `hh` events to `top`, indicating a different learned label geometry rather than a direct improvement over the 10-class setup.

The missing `CE + var 0.01 + cov 0.001`, `latent_dim=64` run was not included because the available directory lacks `run_config.json`. Without that metadata, the trained class order, PID mapping, and normalization constants cannot be validated.

## MinBias vs SM Background Anomaly Evaluation

The MinBias control evaluation was run with `minbias` treated as the scored sample and the remaining SM classes used for the background density fit. For the 10-class models, the background was:

`qcd, dy, top, wjets, ww, wz, ttw, ttz`

The trained `hh4b` reference class was explicitly excluded from the background. Each 10-class run used `8,000` background-fit events and `8,000` held-out background-evaluation events, plus `2,000` scored MinBias events. The four-class comparison run used:

`qcd, dy, top`

with `3,000` background-fit events, `3,000` held-out background-evaluation events, and `2,000` scored MinBias events. Outputs were saved under each run's `figures/minbias_vs_sm_background/` directory.

The table below was read directly from the saved `external_higgs_anomaly_metrics.csv` files in the MinBias output directories.

| Model family | Run | latent_dim | AUC | eff@95%BG | eff@99%BG | median score | p95 score | classifier top response |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Balanced CE + SupCon 0.05 + var 0.01 | `sm_superclasses_balanced_supcon005_var001_100k_latent32` | 32 | 0.326 | 0.097 | 0.034 | 11.687 | 84.980 | `minbias` (78.4%) |
| Balanced CE + SupCon 0.05 + var 0.01 | `sm_superclasses_balanced_supcon005_var001_100k_latent64` | 64 | 0.376 | 0.080 | 0.028 | 59.272 | 350.036 | `minbias` (68.4%) |
| Balanced CE + SupCon 0.05 + var 0.01 | `sm_superclasses_balanced_supcon005_var001_100k_latent128` | 128 | 0.358 | 0.055 | 0.014 | 123.979 | 415.286 | `minbias` (75.8%) |
| CE + var 0.01 + cov 0.001 | `sm_superclasses_ce_var001_cov0001_100k_latent128` | 128 | 0.314 | 0.069 | 0.039 | -193.537 | 363.734 | `minbias` (55.2%) |
| VICReg-style | `sm_superclasses_vicreg001_var001_100k_latent64` | 64 | 0.348 | 0.085 | 0.037 | -31.876 | 204.192 | `minbias` (79.9%) |
| VICReg-style | `sm_superclasses_vicreg001_var001_100k_latent128` | 128 | 0.366 | 0.069 | 0.029 | 123.290 | 729.467 | `minbias` (62.5%) |
| Four-class comparison | `fourclass_balanced_supcon005_var001_100k_latent128` | 128 | 0.398 | 0.087 | 0.034 | 121.190 | 1689.079 | `minbias` (74.1%) |

### MinBias Interpretation

All MinBias-vs-SM AUC values are below 0.5 with the convention that higher anomaly score means more anomalous. This indicates that MinBias is not being isolated as an anomalous tail relative to the other SM classes; it generally receives lower anomaly scores than the comparison background. This is qualitatively consistent with MinBias acting as an in-distribution control rather than a Higgs-like external signal.

The fixed-threshold efficiencies are modest: `eff@95%BG` ranges from 0.055 to 0.097 in the 10-class runs, and `eff@99%BG` stays below 0.039. The classifier response is also internally consistent: MinBias is the top predicted class in every run, with top-response fractions from 55.2% to 79.9% in the 10-class models and 74.1% in the four-class comparison. All saved classifier-response rows sum to 1.0.

The balanced `latent_dim=64` run has the highest 10-class MinBias-vs-SM AUC at 0.376, but because all AUCs remain below 0.5 this should not be interpreted as anomaly-detection improvement. Instead, the MinBias control confirms that the anomaly-score direction is primarily separating Higgs-like or structured signal samples more than the MinBias reference sample.

## Eval-Quality Interpretation

The 4-class run should not be directly compared with the two 10-class superclass runs because it uses a different anomaly benchmark: DY is treated as the anomaly, not HH4B.

For the HH4B benchmark, the VICReg-style `latent_dim=128` run has the strongest composite eval-quality score among the two 2026-08-06 superclass runs. However, the Mahalanobis distance is similar for `latent_dim=64` and `latent_dim=128`, while the GMM anomaly gap changes dramatically. This suggests that the composite score is sensitive to the GMM component and should be interpreted together with classifier response, latent PCA, and per-class metrics.

For quantization planning, these scores are diagnostic only. They help select candidate checkpoints, but they do not imply that the current full-precision model is already FPGA-ready.
