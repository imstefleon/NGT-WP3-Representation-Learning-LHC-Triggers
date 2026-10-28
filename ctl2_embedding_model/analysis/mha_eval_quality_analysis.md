# 2026-07-13 MHA Run And Eval-Quality Analysis [Updated Collide-2V]

## Scope

This note summarizes the first updated Collide-2V MHA run and the first test of Maciej's latent-quality evaluator adapted to the DY-MinBias study.

Run directory:

`ngt_mha_nodes_aligned_outputs/runs/2026-07-13/collide2v_updated_dy_minbias_wide_attn_aug_top_pt_minbias100000_top100000_qcd100000_dy100000_121558`

Eval-quality output:

`eval_quality_scores_n500.json`

## Dataset And Model Configuration

- Dataset source: updated Collide-2V production parquet files.
- Classes: MinBias, Top, QCD, DY.
- Events: 100,000 per class before splitting.
- Input type: L1-only PUPPI candidates.
- Selected L1 branches: `L1T_PUPPIPart_PT`, `L1T_PUPPIPart_Eta`, `L1T_PUPPIPart_Phi`, `L1T_PUPPIPart_D0`, `L1T_PUPPIPart_PID`.
- Candidate policy: top-pT selection with 100 candidates per event.
- Architecture: full-precision PyTorch MHA encoder.
- Attention blocks: two standard multi-head attention blocks.
- Latent dimension: 24.
- Hidden dimension: 64.
- Heads: 4.
- Pooling: attention pooling.
- Trainable parameters: 48,733.

This run replaces the Linformer low-rank attention approximation with standard full multi-head attention while keeping the rest of the training setup comparable.

## Training Behavior

The model was trained for 30 epochs. The best validation loss and best validation accuracy both occurred at the final epoch:

| Epoch | Train loss | Train acc | Val loss | Val acc |
|---:|---:|---:|---:|---:|
| 30 | 0.7299 | 0.6353 | 0.7384 | 0.6340 |

The training and validation curves stayed close to each other:

- Final train accuracy: 0.6353.
- Final validation accuracy: 0.6340.
- Final train loss: 0.7299.
- Final validation loss: 0.7384.

This does not show strong overfitting. The validation performance was still improving slowly at epoch 30, so the model appears closer to slow convergence or plateauing than to overtraining.

## Multiclass Evaluation

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| MinBias | 0.472 | 0.653 | 0.548 | 15,000 |
| Top | 0.924 | 0.937 | 0.931 | 15,000 |
| QCD | 0.637 | 0.549 | 0.590 | 15,000 |
| DY | 0.517 | 0.382 | 0.440 | 15,000 |

Overall evaluation accuracy was 0.6306 and macro F1 was 0.6271.

Compared with the previous updated Collide-2V Linformer baseline, MHA improves the main DY metric:

| Model | Accuracy | Macro F1 | DY recall | DY F1 |
|---|---:|---:|---:|---:|
| Linformer baseline | 0.592 | 0.558 | 0.128 | 0.206 |
| MHA | 0.631 | 0.627 | 0.382 | 0.440 |

The MHA architecture substantially improves DY recovery, suggesting that full candidate-to-candidate attention captures information that was partly lost or compressed by the Linformer projection.

## Confusion Matrix Interpretation

| True class | Pred MinBias | Pred Top | Pred QCD | Pred DY |
|---|---:|---:|---:|---:|
| MinBias | 9,801 | 6 | 1,861 | 3,332 |
| Top | 8 | 14,059 | 902 | 31 |
| QCD | 3,693 | 1,078 | 8,237 | 1,992 |
| DY | 7,269 | 66 | 1,929 | 5,736 |

Main observations:

- DY recall increased to 38.2 percent.
- DY misclassified as MinBias decreased from 65.9 percent in the Linformer baseline to 48.5 percent in MHA.
- MinBias misclassified as DY increased to 22.2 percent.
- Top remains highly stable, with 93.7 percent recall.

MHA shifts the decision boundary in the desired direction for DY, but it also makes the MinBias-DY boundary less pure on the MinBias side.

## Mahalanobis Evaluation

The Mahalanobis score was computed with respect to the MinBias latent-space reference.

| Class | Median | P95 | P99 |
|---|---:|---:|---:|
| MinBias | 1.88 | 27.36 | 118.58 |
| DY | 2.17 | 156.51 | 709.41 |
| QCD | 5.58 | 372.52 | 1047.07 |
| Top | 1023.90 | 3204.48 | 4739.50 |

Top remains clearly separated from MinBias. DY has a higher tail than MinBias, but the median DY score is still very close to the MinBias median. This means the typical DY event remains close to the MinBias reference, while a subset of DY events is more separable in the tail.

## Eval-Quality Test

Maciej's evaluator was adapted to the Collide-2V DY-MinBias setting:

- Background/reference class: MinBias.
- Anomaly/signal-like class: DY.
- Test size: 500 events per class loaded, with 300 events used in the evaluator split.
- Latent dimension: 24.
- PID classes observed: 20.

The composite score was:

| Metric | Value | Normalized score |
|---|---:|---:|
| Alignment loss | 0.5312 | 0.4688 |
| Participation ratio | 1.5667 | 0.0246 |
| GMM anomaly gap | 475.1470 | 0.9503 |
| Mahalanobis distance | 7.3486 | 0.1470 |
| Composite score | 0.2004 | - |

The evaluator shows two different effects:

- The GMM anomaly gap is high, meaning DY looks unlikely under a MinBias-fitted density model in the whitened latent space.
- The participation ratio is very low, meaning that the 24-dimensional latent space is effectively using only a small number of directions.

This explains why the composite score remains limited despite the strong GMM term. The geometric mean penalizes weak sub-scores, and the participation-ratio term is currently the main bottleneck.

## Interpretation

The MHA run is a meaningful improvement over the previous Linformer baseline for DY recovery. It improves global accuracy, macro F1, DY recall, and DY F1. However, the DY-MinBias separation is still not clean: many DY events remain close to MinBias, and more MinBias events are now classified as DY.

The eval-quality result adds an important latent-space interpretation. The model has learned some density-level separation between DY and MinBias, but the latent representation appears geometrically compressed. The low participation ratio suggests that increasing latent dimensionality alone should be evaluated carefully: a larger latent space is useful only if the model actually uses more independent latent directions.

## Conclusion

MHA is a better architecture candidate than the Linformer baseline for the current DY-MinBias problem, but the latent geometry still needs improvement. The most important discovery from `eval_quality` is that the current MHA model separates DY from MinBias in some density-sensitive metrics while still using the latent space inefficiently. This makes the composite score a useful complement to classification accuracy, because it exposes latent-space collapse that is not visible from the multiclass report alone.
