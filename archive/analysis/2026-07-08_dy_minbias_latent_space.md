# 2026-07-08 DY-MinBias Latent Space Analysis [HF Collide-2V]

Notebook:

`NGT_Linformer_NodesAligned_FullPrecision_Collide2V_L1features.ipynb`

Run:

```text
ngt_linformer_nodes_aligned_outputs/runs/2026-07-08/
dy_minbias_wide_attn_aug_top_pt_minbias100000_top100000_qcd100000_dy100000_155902/
```

## Purpose

This run was used to study whether the DY-MinBias confusion is only a final classifier problem or whether the overlap already exists in the learned latent space.

The run uses:

- L1-only input features.
- Corrected `top_pt` L1 candidate selection.
- 100,000 events per class.
- Wide Linformer configuration.
- Attention pooling.
- Augmentation enabled.
- Gen-level information only for DY final-state diagnostics, not as model input.

Dataset source:

```text
HF Collide-2V / parquet dataset path used by the original notebook loader
```

## Global Classification Result

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| MinBias | 0.472 | 0.927 | 0.625 | 15,000 |
| Top | 0.919 | 0.938 | 0.928 | 15,000 |
| QCD | 0.799 | 0.754 | 0.776 | 15,000 |
| DY | 0.734 | 0.051 | 0.095 | 15,000 |

Overall:

| Metric | Value |
|---|---:|
| Accuracy | 0.668 |
| Macro F1 | 0.606 |

The model still identifies Top and QCD much better than DY. DY precision is high only because the model predicts DY rarely. The DY recall remains very low.

## Confusion Matrix

| True class | Pred MinBias | Pred Top | Pred QCD | Pred DY |
|---|---:|---:|---:|---:|
| MinBias | 13,906 | 0 | 1,030 | 64 |
| Top | 2 | 14,066 | 861 | 71 |
| QCD | 2,342 | 1,202 | 11,314 | 142 |
| DY | 13,234 | 42 | 959 | 765 |

Main DY-MinBias result:

```text
DY -> MinBias = 13,234 / 15,000 = 88.2%
DY -> DY      =    765 / 15,000 =  5.1%
```

This confirms that the dominant failure mode is still DY being classified as MinBias.

## DY Final-State Split

Following the suggestion to go back to the original dataset, DY was split using gen-level hard-process lepton information. This information was used only to label the diagnostic subgroups.

Loaded DY final states:

| DY final state | Loaded events |
|---|---:|
| `ee` | 45,700 |
| `mumu` | 45,655 |
| `tautau` | 8,645 |

Evaluation behavior:

| DY final state | Eval support | Recall as DY | Confused as MinBias | Confused as QCD | Confused as Top |
|---|---:|---:|---:|---:|---:|
| `DY->ee` | 6,853 | 0.048 | 0.894 | 0.056 | 0.002 |
| `DY->mumu` | 6,859 | 0.058 | 0.888 | 0.052 | 0.002 |
| `DY->tautau` | 1,288 | 0.030 | 0.791 | 0.171 | 0.009 |

Interpretation:

- The network struggles with all DY final states.
- `DY->ee` and `DY->mumu` are mostly classified as MinBias.
- `DY->tautau` is also poorly recovered as DY, but has a larger confusion with QCD.
- The issue is not isolated to a single DY final state.

## Latent PCA

The first PCA components of the latent embeddings explain most of the variance:

| Component | Explained variance |
|---|---:|
| PC1 | 60.4% |
| PC2 | 28.4% |
| PC3 | 10.2% |

The focused PCA plot shows that MinBias, `DY->ee`, `DY->mumu`, and `DY->tautau` occupy almost the same latent region.

Main observation:

```text
The DY-MinBias overlap is already present in the latent space.
```

## Latent Centroid Distances

The closest latent centroids are:

| Group A | Group B | Distance |
|---|---|---:|
| `DY->ee` | `DY->mumu` | 0.103 |
| `DY->ee` | MinBias | 0.298 |
| `DY->mumu` | MinBias | 0.391 |
| `DY->ee` | `DY->tautau` | 0.640 |
| `DY->tautau` | MinBias | 0.657 |

This shows that:

- `DY->ee` and `DY->mumu` are almost identical in the latent space.
- Both are closer to MinBias than to QCD or Top.
- `DY->tautau` is somewhat different, but still not well separated as a DY population.

## Linear Separability Test

A simple linear classifier was trained on top of the frozen latent embeddings to test whether the information needed to separate DY from MinBias is present in the latent space.

| Task | AUC | Balanced accuracy |
|---|---:|---:|
| DY all vs MinBias | 0.543 | 0.533 |
| `DY->ee` vs MinBias | 0.557 | 0.536 |
| `DY->mumu` vs MinBias | 0.547 | 0.532 |
| `DY->tautau` vs MinBias | 0.591 | 0.580 |
| QCD vs MinBias | 0.938 | 0.884 |
| Top vs MinBias | 1.000 | 0.999 |

Interpretation:

- Top and QCD are separable from MinBias using the learned embeddings.
- DY is not linearly separable from MinBias in the learned embeddings.
- This suggests that the current latent representation does not encode enough discriminating information for DY-MinBias.

## Local Neighborhood Mixing

The k-nearest-neighbor study also shows strong local mixing between MinBias and DY.

| Group | Same-group neighbors | MinBias neighbors | QCD neighbors | Any-DY neighbors |
|---|---:|---:|---:|---:|
| MinBias | 0.253 | 0.253 | 0.076 | 0.671 |
| `DY->ee` | 0.246 | 0.241 | 0.073 | 0.681 |
| `DY->mumu` | 0.255 | 0.239 | 0.070 | 0.690 |
| `DY->tautau` | 0.186 | 0.226 | 0.140 | 0.623 |
| QCD | 0.555 | 0.076 | 0.555 | 0.263 |
| Top | 0.878 | 0.000 | 0.106 | 0.016 |

This confirms that DY and MinBias are locally interleaved in the latent space, while Top forms a much cleaner region.

## Main Conclusion

The DY-MinBias problem is not only a final classifier head issue. The overlap is already present in the latent embeddings.

The evidence is consistent across:

- The confusion matrix.
- DY final-state recall.
- PCA visualization.
- Centroid distances.
- Linear separability tests.
- Local kNN neighborhood mixing.

The current model learns useful structure for Top and QCD, but it does not learn a latent representation that separates DY from MinBias using the current L1-only inputs and training objective.

## Files Generated

Additional latent-separation outputs were saved in:

```text
ngt_linformer_nodes_aligned_outputs/runs/2026-07-08/
dy_minbias_wide_attn_aug_top_pt_minbias100000_top100000_qcd100000_dy100000_155902/
latent_separation_analysis/
```

Key files:

- `latent_pca_minbias_dy_pc1_pc2.png`
- `latent_pca_minbias_dy_pc1_pc3.png`
- `latent_pca_minbias_dy_pc2_pc3.png`
- `latent_linear_separability.csv`
- `latent_knn_neighborhood_mixing.csv`
- `latent_normalized_centroid_distances.csv`
- `latent_group_prediction_summary.csv`
