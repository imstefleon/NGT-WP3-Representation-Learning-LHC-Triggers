# July 6, 2026 Top-pT Eta Fix Run Analysis

## Context

Sabrina pointed out that the raw `eta` distribution looked unphysical, with a strong peak around `eta = [-5, -4]` and a gap around `[-4, -2.5]`.

The source of the problem was not the plotting configuration. The input tensor builder was taking the first 100 L1 candidates from the parquet arrays:

```python
event[:n, ...] = arrays[:100]
```

The parquet candidate arrays are not sorted by `pT`. They are ordered in a way that biases the first candidates toward a detector/eta region. This made the first-100 truncation create an artificial eta structure.

The fix was to select the top 100 valid L1 candidates by descending `pT` before padding/truncation:

```python
valid_idx = np.nonzero(valid)[0]
order = valid_idx[np.argsort(-pt_all[valid_idx], kind="stable")]
selected_idx = order[:N_CONSTITUENTS]
```

The cache name now includes `top_pt`, so corrected runs do not reuse the old biased caches.

## Eta Validation

The corrected 100k caches now show a physical eta coverage across `[-5, 5]`. The medians are close to zero and the previous empty gap is gone.

| Class | eta p10 | eta p50 | eta p90 | frac [-5,-4] | frac [-4,-2.5] | frac [-2.5,2.5] | frac [2.5,4] | frac [4,5] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MinBias | -4.270 | -0.026 | 4.254 | 0.139 | 0.235 | 0.256 | 0.234 | 0.136 |
| Top | -3.848 | 0.009 | 3.854 | 0.085 | 0.169 | 0.491 | 0.170 | 0.085 |
| QCD | -4.172 | -0.011 | 4.168 | 0.122 | 0.223 | 0.311 | 0.222 | 0.121 |
| DY | -4.254 | 0.001 | 4.258 | 0.136 | 0.235 | 0.259 | 0.234 | 0.136 |

Conclusion: the eta gap was an artifact of candidate selection, and the `top_pt` fix removed it.

## Runs Performed Today

All runs below used L1-only features and the corrected top-pT candidate selection.

| Run | Events per class | Model | Augmentation | Pooling | Notes |
|---|---:|---|---|---|---|
| `baseline_no_aug_minbias1000_top1000_qcd1000_dy1000_100440` | 1,000 | baseline | no | mean | Quick validation run. The name does not include `top_pt`, but it used the corrected `top_pt` caches. |
| `baseline_no_aug_top_pt_minbias100000_top100000_qcd100000_dy100000_101653` | 100,000 | baseline | no | mean | Corrected 100k baseline. |
| `dy_minbias_wide_attn_aug_top_pt_minbias100000_top100000_qcd100000_dy100000_124857` | 100,000 | wider model | yes | attention | Wider Linformer with attention pooling and one augmentation copy. |

## Classification Results

| Run | Accuracy | Macro F1 | MinBias F1 | Top F1 | QCD F1 | DY Precision | DY Recall | DY F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1k corrected baseline | 0.625 | 0.551 | 0.629 | 0.879 | 0.657 | 0.333 | 0.020 | 0.038 |
| 100k corrected baseline | 0.665 | 0.607 | 0.625 | 0.920 | 0.774 | 0.628 | 0.060 | 0.109 |
| 100k wide attention + augmentation | 0.668 | 0.606 | 0.625 | 0.928 | 0.776 | 0.734 | 0.051 | 0.095 |

## DY and MinBias Confusion

| Run | DY -> MinBias | DY -> DY | MinBias -> DY |
|---|---:|---:|---:|
| 1k corrected baseline | 122 / 150 = 81.3% | 3 / 150 = 2.0% | 3 / 150 = 2.0% |
| 100k corrected baseline | 13,098 / 15,000 = 87.3% | 894 / 15,000 = 6.0% | 248 / 15,000 = 1.7% |
| 100k wide attention + augmentation | 13,234 / 15,000 = 88.2% | 765 / 15,000 = 5.1% | 64 / 15,000 = 0.4% |

Main observation: increasing statistics from 1k to 100k improved the overall stability and global accuracy, but DY is still mostly classified as MinBias. The wide attention + augmentation run slightly improves global accuracy and Top/QCD performance, but it does not improve DY recall.

## Mahalanobis Summary

The Mahalanobis score still shows strong overlap between MinBias and DY in the latent space.

| Run | MinBias p50 | DY p50 | QCD p50 | Top p50 |
|---|---:|---:|---:|---:|
| 1k corrected baseline | -45.86 | -46.01 | 288.31 | 3,079,375 |
| 100k corrected baseline | -23.50 | -23.33 | 65.12 | 41,136.5 |
| 100k wide attention + augmentation | -45.99 | -45.83 | 54.18 | 88,023.4 |

Interpretation: Top is very well separated, QCD is partially separated, but DY and MinBias remain almost coincident in the learned latent space.

## Training Behavior

| Run | Best epoch | Best validation loss | Best validation accuracy | Final validation accuracy |
|---|---:|---:|---:|---:|
| 1k corrected baseline | 30 | 0.750 | 0.630 | 0.630 |
| 100k corrected baseline | 29 | 0.643 | 0.664 | 0.663 |
| 100k wide attention + augmentation | 4 | 0.643 | 0.667 | 0.652 |

The wide attention + augmentation model reaches its best validation loss very early, then validation quality degrades while training accuracy continues increasing. This suggests overfitting or that augmentation/attention is not directly addressing the DY-MinBias separation.

## DY-MinBias Event-Level Diagnostic

The auxiliary DY vs MinBias classifier using engineered L1 event features remains weak:

| Metric | Value |
|---|---:|
| Accuracy | 0.536 |
| MinBias F1 | 0.643 |
| DY F1 | 0.337 |
| DY recall | 0.236 |

Confusion:

| True class | Pred MinBias | Pred DY |
|---|---:|---:|
| MinBias | 25,088 | 4,912 |
| DY | 22,920 | 7,080 |

Most useful engineered features:

1. `max_pt_gamma`
2. `sum_pt`
3. `n_lep`
4. `max_pt`
5. `sum_pt_gamma`
6. `n_ch45`
7. `pt_p90`

The feature importances are small, which supports the conclusion that DY and MinBias are difficult to separate with the current L1-only representation and current objective.

## Conclusions

1. The eta gap was caused by first-100 candidate truncation and was fixed by selecting the top 100 candidates by `pT`.
2. The corrected `eta` distributions now look physically reasonable.
3. The corrected 100k baseline is the new reliable reference point.
4. DY remains the main failure mode: most DY events are still predicted as MinBias.
5. Attention pooling and augmentation did not solve the DY-MinBias overlap. It slightly improved global accuracy but reduced DY recall from 6.0% to 5.1%.
6. The latent-space Mahalanobis scores confirm that DY and MinBias remain highly overlapping.

## Next Steps and Open Questions

### 1. Targeted DY-MinBias Strategy

The current results suggest that simply increasing the global model capacity is not enough to solve the DY-MinBias overlap. A more targeted strategy could be tested, for example:

- Apply class weighting or focal loss to give more importance to DY misclassification.
- Add a dedicated binary DY-vs-MinBias classification head.
- Oversample DY events or construct balanced DY-MinBias hard-example batches.
- Use DY recall, DY F1, and DY -> MinBias confusion as explicit model-selection metrics, instead of relying only on global validation accuracy.

**Question:** Should the next run prioritize improving DY recall, even if this slightly reduces MinBias purity or global accuracy?

### 2. Revisit L1-Only Discriminating Features

The current L1-only representation may not contain enough discriminating information for DY vs MinBias, or the model may not be using it effectively. Features worth reviewing include:

- Lepton-like candidate counts.
- High-pT charged candidates.
- Photon-like activity.
- Event-level pT tails.
- Charge/PID composition of the leading L1 candidates.

**Question:** Are there additional L1-level quantities that should be included to better capture the DY topology?

### 3. Training and Model-Selection Strategy

The wide attention + augmentation run reached its best validation loss around epoch 4, while later epochs did not improve DY separation. This suggests that early stopping or a DY-focused validation criterion may be useful.

- Add early stopping based on validation loss.
- Track DY recall and DY -> MinBias confusion during validation.
- Save the best checkpoint using a DY-focused metric, not only validation loss.

**Question:** Should I select the best model based on DY recall/F1 rather than overall validation loss?
