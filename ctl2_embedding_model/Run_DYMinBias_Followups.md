# Follow-up Runs: DY-MinBias, Attention, and 300k MinBias L1

Notebook:

[`NGT_Linformer_NodesAligned_FullPrecision_Collide2V_L1features.ipynb`](NGT_Linformer_NodesAligned_FullPrecision_Collide2V_L1features.ipynb)

Context:

After the 1k and 100k baseline runs, the main problem was the confusion between:

```text
DY <-> MinBias
```

This document summarizes the follow-up runs focused on understanding and
addressing that issue.

## 1. DY vs MinBias Diagnostic With Aggregated Features

Before changing the model, a binary diagnostic was built using event-level
aggregated features:

- `n_lep`
- `n_mu`
- `n_gamma`
- `sum_pt`
- `max_pt`
- `sum_pt_lep`
- `mean_abs_eta`
- `mean_abs_dxy`
- `pt_p75`
- `pt_p90`
- `max_ossf_mass`

A simple `HistGradientBoostingClassifier` was trained for:

```text
MinBias vs DY
```

Result:

```text
accuracy ~= 0.57
ROC AUC ~= 0.61
```

Interpretation:

- Even with simple aggregated variables, DY and MinBias are only weakly
  separable.
- This suggests that the issue is not only the final classifier layer.
- Explicit discriminating information may be missing from the current L1
  representation.

## 2. Wide Attention + Phi Augmentation Run

Run:

```text
ngt_linformer_nodes_aligned_outputs/runs/2026-07-02/dy_minbias_wide_attn_aug_n100000_143843/
```

Motivation:

Test whether more model capacity, attention pooling, and data augmentation help
recover DY.

Configuration:

```text
N_EVENTS_PER_CLASS = 100,000
Total events = 400,000
Eval per class = 15,000

latent_dim = 24
ff_dim = 64
num_heads = 4
proj_k = 16
pooling = attention pooling
USE_AUGMENTATION = True
augmentation = phi rotation
```

### Classification Results

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| minbias | 0.509 | 0.476 | 0.492 |
| top | 0.932 | 0.958 | 0.945 |
| qcd | 0.822 | 0.840 | 0.831 |
| dy | 0.550 | 0.558 | 0.554 |

Global accuracy:

```text
0.708
```

DY-MinBias confusion:

```text
DY -> MinBias = 5,670/15,000 = 37.8%
MinBias -> DY = 6,653/15,000 = 44.4%
```

Compared with the 100k baseline:

```text
DY recall: 0.16 -> 0.56
DY F1:     0.26 -> 0.55
```

Interpretation:

- The model no longer collapses almost all DY events into MinBias.
- However, the decision boundary shifts, and many MinBias events are now
  classified as DY.
- The run improves DY recovery, but it does not produce a clean separation.

### Mahalanobis

| Class | Median |
|---|---:|
| minbias | -43.66 |
| dy | -43.84 |
| qcd | 47.7 |
| top | 133,039 |

Reading:

- `top` remains well separated.
- `qcd` is partially separated.
- The median `dy` score is still almost identical to MinBias.
- The latent space still does not clearly separate DY and MinBias.

## 3. 300k MinBias L1 Mean No Aug Run

Run:

```text
ngt_linformer_nodes_aligned_outputs/runs/2026-07-03/minbias300k_l1_mean_no_aug_minbias300000_top100000_qcd100000_dy100000_142736/
```

Motivation:

Test whether a larger MinBias sample helps the model learn the structure of the
normal class and improves separation from DY.

Configuration:

```text
MinBias = 300,000 events
top     = 100,000 events
qcd     = 100,000 events
dy      = 100,000 events

FEATURE_MODE = l1_only
FullReco fallback = disabled
pooling = mean pooling
USE_AUGMENTATION = False
attention pooling = No
class weights = balanced
val/eval = balanced by class
```

Generated caches:

```text
train_minbias_300000_l1_only_rawtensor.npy
train_top_100000_l1_only_rawtensor.npy
train_qcd_100000_l1_only_rawtensor.npy
train_dy_100000_l1_only_rawtensor.npy
```

This run was strictly L1-only.

### Classification Results

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| minbias | 0.515 | 0.328 | 0.401 |
| top | 0.927 | 0.955 | 0.941 |
| qcd | 0.810 | 0.844 | 0.827 |
| dy | 0.534 | 0.690 | 0.602 |

Global accuracy:

```text
0.704
```

DY-MinBias confusion:

```text
DY -> MinBias = 3,623/15,000 = 24.2%
MinBias -> DY = 8,741/15,000 = 58.3%
```

Interpretation:

- DY improves even more than in `wide_attn_aug`.
- MinBias performance gets significantly worse.
- The model classifies many MinBias events as DY.
- Increasing MinBias to 300k did not make the MinBias class cleaner.

### Mahalanobis

| Class | Median |
|---|---:|
| minbias | -32.28 |
| dy | -32.44 |
| qcd | 88.59 |
| top | 42,982 |

Reading:

- `top` remains separated.
- `qcd` is partially separated.
- `dy` still has a median almost identical to MinBias.
- The latent space still does not clearly separate DY-MinBias.

## 4. Follow-up Run Comparison

| Metric | wide_attn_aug | 300k MinBias L1 mean |
|---|---:|---:|
| Accuracy | 0.708 | 0.704 |
| F1 MinBias | 0.492 | 0.401 |
| Recall MinBias | 0.476 | 0.328 |
| F1 DY | 0.554 | 0.602 |
| Recall DY | 0.558 | 0.690 |
| DY -> MinBias | 37.8% | 24.2% |
| MinBias -> DY | 44.4% | 58.3% |
| F1 top | 0.945 | 0.941 |
| F1 qcd | 0.831 | 0.827 |

## 5. Follow-up Conclusion

The follow-up runs show that:

- Increasing capacity + attention pooling + augmentation improves DY, but causes
  more MinBias -> DY confusion.
- Using 300k MinBias in L1-only mode improves DY recall even further, but
  strongly degrades MinBias performance.
- In both cases, Mahalanobis shows that the median DY score remains almost the
  same as MinBias.
- Therefore, the problem is not solved: the decision boundary has mostly shifted.

Main conclusion:

```text
DY and MinBias remain strongly overlapping with the current L1 features.
```

## 6. Recommended Next Steps

It does not seem sufficient to keep increasing only the number of MinBias events.

Recommended next tests:

1. L1-only run with explicit aggregated features:
   - `n_lep`
   - `n_gamma`
   - `sum_pt`
   - leading `pt`
   - OS/SF pairs
   - approximate dilepton mass
   - isolation-like variables

2. Dedicated binary run:

```text
DY vs MinBias
```

3. Compare 300k MinBias with and without class weights:

```text
300k MinBias + balanced class weights
300k MinBias without class weights
```

4. Consider an auxiliary loss or contrastive loss to explicitly separate DY and
   MinBias in the latent space.

5. Check whether DY should be split into subcategories:
   - DY with visible L1 leptons
   - DY without visible L1 leptons
   - low-activity DY
   - high-pt DY
