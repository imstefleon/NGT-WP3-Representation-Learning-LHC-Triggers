# MHA L1-Only 100k Latent-128 Eval-Quality Run

## Run Configuration

This run tested the reorganized `ngt_model` PyTorch MHA workflow using the updated Collide-2V production dataset with L1-only candidate features.

| Item | Value |
|---|---|
| Run name | `mha_l1_only_100k_latent128` |
| Output path | `ngt_mha_nodes_aligned_outputs/runs/2026-07-14/mha_l1_only_100k_latent128` |
| Dataset source | `collide2v_updated` |
| Feature mode | `l1_only` |
| Candidate selection | `top_pt` |
| Classes | `minbias`, `top`, `qcd`, `dy` |
| Events per class | 100,000 |
| Constituents per event | 100 |
| Input features after PID one-hot | 24 |
| Model | MHA embedding model |
| Latent dimension | 128 |
| FF dimension | 128 |
| Attention heads | 8 |
| Pooling | attention pooling |
| Augmentation | 1 phi-rotation copy |
| MLflow experiment | `ngt_model` |

The split stored in `run_config.json` was:

| Split | Events |
|---|---:|
| Train | 280,000 |
| Train after augmentation | 560,000 |
| Validation | 60,000 |
| Evaluation | 60,000 |

## Training Behavior

Training was stable and stopped through early stopping at epoch 28. The best validation loss was reached at epoch 20:

| Metric | Value |
|---|---:|
| Best epoch | 20 |
| Best validation loss | 0.7658 |
| Validation accuracy at best epoch | 0.6141 |
| Final epoch | 28 |
| Final train loss | 0.7681 |
| Final train accuracy | 0.6132 |
| Final validation loss | 0.7699 |
| Final validation accuracy | 0.6140 |
| Total training time | ~72.9 minutes |

The final train and validation accuracies are very close. This indicates that the model was not strongly overfitting the training sample. Instead, the performance appears limited by the current representation and class overlap, especially for DY and MinBias.

## Evaluation Classification Results

The evaluation set contained 15,000 events per class.

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| MinBias | 0.4552 | 0.7545 | 0.5678 | 15,000 |
| Top | 0.9173 | 0.9380 | 0.9275 | 15,000 |
| QCD | 0.5617 | 0.6033 | 0.5818 | 15,000 |
| DY | 0.6346 | 0.1562 | 0.2507 | 15,000 |
| Overall accuracy |  |  | 0.6130 | 60,000 |
| Macro average | 0.6422 | 0.6130 | 0.5820 | 60,000 |

Top is clearly the best separated class. QCD and MinBias remain partially mixed. DY is the most problematic class: precision is moderate, but recall is very low.

## Confusion Matrix

| True class | Pred MinBias | Pred Top | Pred QCD | Pred DY |
|---|---:|---:|---:|---:|
| MinBias | 11,317 | 7 | 3,006 | 670 |
| Top | 5 | 14,070 | 890 | 35 |
| QCD | 4,105 | 1,202 | 9,049 | 644 |
| DY | 9,433 | 60 | 3,164 | 2,343 |

The main failure mode remains DY being classified as MinBias:

| DY destination | Events | Fraction of true DY |
|---|---:|---:|
| Predicted as MinBias | 9,433 | 62.9% |
| Predicted as QCD | 3,164 | 21.1% |
| Correctly predicted as DY | 2,343 | 15.6% |
| Predicted as Top | 60 | 0.4% |

This confirms that, even after moving to the updated Collide-2V dataset, L1-only features, 100k events per class, and a larger 128-dimensional latent space, DY-MinBias confusion is still the dominant bottleneck.

## Eval-Quality Results

Maciej's latent-quality evaluator was run with 2,000 events per class. It used MinBias as the background class and DY as the anomaly class.

| Metric | Value |
|---|---:|
| Composite score | 0.1828 |
| Alignment loss | 0.7735 |
| Participation ratio | 1.6258 |
| Participation ratio normalized | 0.0127 |
| GMM anomaly gap | 1.2204e16 |
| Mahalanobis distance | 1.6360e7 |
| Latent dimension | 128 |
| PID classes | 20 |
| Model parameters | 210,181 |

Normalized sub-scores:

| Sub-score | Value |
|---|---:|
| Alignment loss norm | 0.2265 |
| Participation ratio norm | 0.0049 |
| GMM gap norm | 1.0000 |
| Mahalanobis distance norm | 1.0000 |

The anomaly-oriented metrics are high: both the GMM gap and Mahalanobis sub-score saturate at 1.0. This means DY can appear far from MinBias under some anomaly-distance views of the latent space.

However, the participation ratio is very low: only about 1.6 effective dimensions are being used out of a 128-dimensional latent space. This suggests that the representation is still geometrically narrow or partially collapsed. The model is not yet using the full latent capacity in a balanced way.

## Interpretation

The 100k L1-only MHA run is technically successful: it trains stably, logs to MLflow, produces evaluation artifacts, and runs the eval-quality pipeline. Compared with the earlier smoke test, the composite eval-quality score is much more meaningful.

The physics/classification result is mixed. Top is well separated, QCD is moderately separated, and MinBias has high recall but low precision. DY remains difficult: most true DY events are absorbed into MinBias, which is consistent with the previously observed DY-MinBias overlap.

The eval-quality result gives a complementary view. It suggests that DY and MinBias can be separated along certain anomaly-sensitive directions, but the global latent geometry is weak because the effective dimensionality is very low. In other words, increasing `latent_dim` to 128 helped create capacity, but the model is not yet using that capacity broadly.

## Artifacts

Key files produced by the run:

- `run_config.json`
- `training_history.csv`
- `classification_report_eval.csv`
- `confusion_matrix_eval.csv`
- `eval_latents.npy`
- `eval_logits.npy`
- `eval_quality_scores_n2000.json`
- `models/mha_nodes_aligned_best.pt`
- `models/mha_nodes_aligned_best_for_eval.pt`
