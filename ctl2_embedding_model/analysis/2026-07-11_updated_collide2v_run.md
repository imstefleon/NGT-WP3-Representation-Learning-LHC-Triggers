# 2026-07-11 Updated Collide-2V Run Analysis [Updated Collide-2V]

## Scope

This note summarizes the updated Collide-2V baseline run produced before future model changes. The goal was to validate the new updated Collide-2V loader, run the same DY-MinBias focused configuration with L1-only inputs, and document the current separation behavior.

Run directory:

`ngt_linformer_nodes_aligned_outputs/runs/2026-07-11/collide2v_updated_dy_minbias_wide_attn_aug_top_pt_minbias100000_top100000_qcd100000_dy100000_143554`

## Dataset And Configuration

- Dataset source: updated Collide-2V production parquet files.
- Notebook data source: `DATA_SOURCE = 'collide2v_updated'`.
- Events: 100,000 per class before train/validation/evaluation splitting.
- Classes: MinBias, Top, QCD, DY.
- Input type: L1-only PUPPI particle candidates.
- Selected branches: `L1T_PUPPIPart_PT`, `L1T_PUPPIPart_Eta`, `L1T_PUPPIPart_Phi`, `L1T_PUPPIPart_D0`, `L1T_PUPPIPart_PID`.
- Candidate policy: top-pT candidate selection with 100 candidates per event.
- Model preset: DY-MinBias wide attention with augmentation.
- Model shape: full-precision PyTorch Linformer-style encoder with attention pooling, `latent_dim = 24`, `ff_dim = 64`, `num_heads = 4`, and `proj_k = 16`.

## Training Behavior

The model was trained for 30 epochs. The best validation loss occurred around epoch 8:

- Best validation loss: approximately 0.8074.
- Validation accuracy at best validation loss: approximately 0.5903.
- Best validation accuracy: approximately 0.5943 around epoch 28.
- Final training accuracy: approximately 0.637.

The training accuracy continued to increase while validation accuracy stayed close to 0.59. This suggests that the current setup reaches a separation plateau and begins to show mild overfitting without improving DY-MinBias separation.

## Multiclass Evaluation

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| MinBias | 0.433 | 0.733 | 0.544 | 15,000 |
| Top | 0.908 | 0.936 | 0.921 | 15,000 |
| QCD | 0.553 | 0.571 | 0.562 | 15,000 |
| DY | 0.527 | 0.128 | 0.206 | 15,000 |

Overall evaluation accuracy was approximately 0.592.

Top remains the easiest class to separate. QCD has moderate separation. DY remains the weakest class, mainly because most DY events are assigned to MinBias or QCD.

## Confusion Matrix Interpretation

| True class | Pred MinBias | Pred Top | Pred QCD | Pred DY |
|---|---:|---:|---:|---:|
| MinBias | 10,995 | 21 | 2,921 | 1,063 |
| Top | 4 | 14,033 | 941 | 22 |
| QCD | 4,508 | 1,287 | 8,569 | 636 |
| DY | 9,888 | 121 | 3,071 | 1,920 |

Main observations:

- DY recall is 12.8 percent.
- 65.9 percent of true DY events are classified as MinBias.
- 20.5 percent of true DY events are classified as QCD.
- Only 12.8 percent of true DY events remain in the DY class.
- Top is highly stable, with 93.6 percent recall.

The main failure mode is still DY being absorbed by the MinBias-like region of the latent space.

## Mahalanobis Anomaly Score

The Mahalanobis score was computed with respect to the MinBias latent-space reference. The score distributions show strong overlap between MinBias and DY:

| Class | Median | P95 | P99 |
|---|---:|---:|---:|
| MinBias | -28.65 | -8.43 | 56.09 |
| DY | -28.39 | 38.79 | 2870.88 |
| QCD | -24.75 | 2637.24 | 9779.71 |
| Top | 7806.50 | 37304.41 | 63694.97 |

The MinBias and DY medians are almost identical. This indicates that the current latent representation does not place most DY events far from the MinBias reference. Top is clearly separated by this score, while QCD has a broader partially separated tail.

## DY-MinBias Event-Level Diagnostic

A separate DY-vs-MinBias diagnostic classifier was trained from event-level summary features.

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| MinBias | 0.541 | 0.802 | 0.647 | 30,000 |
| DY | 0.618 | 0.320 | 0.422 | 30,000 |

Binary accuracy was approximately 0.561. In this diagnostic, 68.0 percent of DY events were still assigned to MinBias.

The strongest event-level feature was `mean_abs_dxy`, followed by high-pT and lepton-like summary quantities. This supports the interpretation that some L1-level differences exist, but they are not yet strong enough to create clean DY-MinBias separation in the current embedding.

## Input Distribution Diagnostic

The updated Collide-2V run uses `L1T_PUPPIPart_D0` as the L1 displacement-like input. Its distribution is concentrated near zero with rare large outliers. This behavior is consistent with the previous updated Collide-2V/HF diagnostics and differs from the ROOT `L1PuppiCands_dxy` shape observed in earlier comparisons.

This means the updated Collide-2V run should be treated as a separate safe baseline, not as a direct one-to-one replacement for the previous ROOT-based diagnostic behavior.

## Conclusion

This run validates the updated Collide-2V loader and produces a reproducible 100k-per-class L1-only baseline before future model changes. The updated Collide-2V baseline keeps Top well separated, gives moderate QCD separation, and confirms that DY-MinBias overlap remains the main limitation. The latent-space and Mahalanobis diagnostics both show that most DY events are still close to MinBias in the current embedding.
