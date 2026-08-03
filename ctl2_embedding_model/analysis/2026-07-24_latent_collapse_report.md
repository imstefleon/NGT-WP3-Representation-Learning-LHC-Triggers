# Week 5 Latent Collapse Report

## Scope

This report summarizes the Week 2 latent-collapse study for the L1-only MHA workflow in `ngt_model`.
All runs use updated Collide-2V L1 candidate inputs only:

```text
[pt, eta, phi, D0/dxy, PID]
```

The main benchmark uses:

```text
classes = minbias, top, qcd, dy, hh4b
background = minbias + top + qcd + dy
anomaly = hh4b
```

The central question was whether the low effective dimensionality of the latent space is mainly caused by architecture choices, dataset size, model capacity, or the supervised `CrossEntropy` objective.

## Baseline Setup

The most useful supervised baseline for the later ablations was:

```text
n_events_per_class = 100000
latent_dim = 32
ff_dim = 128
num_heads = 8
num_attention_blocks = 2
pooling_mode = mean
classifier_head = linear
loss = CrossEntropy
```

This setup is close to the CTL2 particle-level architecture: stacked attention blocks with residual connections, global/mean event pooling, a dense post-pool stack, and a fixed-size latent embedding.

## Diagnostics Used

Four diagnostics were used to study latent collapse:

- `latent_diagnostics.py`: PCA/eigenvalue spectrum, effective rank, participation ratio.
- `latent_fisher.py`: empirical Fisher sensitivity by latent coordinate.
- `head_fisher.py`: empirical Fisher sensitivity in the classifier head.
- `eval_quality_collide2v.py`: composite representation-quality score based on latent geometry and anomaly-separation behavior.

These metrics are complementary. Classification metrics measure task performance, while PCA/Fisher/eval-quality measure representation geometry and dimensional usage.

## Main Findings

### Latent Capacity

The latent-dimension sweep tested `latent_dim = 16, 32, 64, 128`.

Increasing nominal latent size did not reliably improve classification, HH4B separation, or effective dimensionality. The model typically used only a small number of effective latent directions, even when the nominal latent dimension was much larger.

This suggested that nominal latent capacity was not the main bottleneck.

### Seed Stability

Repeated seeds at `100k` events/class and `latent_dim=32` showed stable behavior:

```text
seed 42: accuracy=0.605240, macro_f1=0.595673, hh4b_f1=0.712664, eval_quality=0.214096
seed 43: accuracy=0.594987, macro_f1=0.589351, hh4b_f1=0.728350, eval_quality=0.202553
seed 44: accuracy=0.598187, macro_f1=0.592067, hh4b_f1=0.722235, eval_quality=0.216044
```

The collapse and classification behavior are therefore not just a single-seed artifact.

### Dataset Size

The dataset-size sweep tested `100k`, `200k`, and `300k` events/class at fixed architecture.

```text
100k: accuracy=0.605240, macro_f1=0.595673, hh4b_f1=0.712664, eval_quality=0.214096
200k: accuracy=0.567760, macro_f1=0.558563, hh4b_f1=0.710829, eval_quality=0.207927
300k: accuracy=0.581640, macro_f1=0.558366, hh4b_f1=0.732116, eval_quality=0.202068
```

More data did not remove the low-dimensional latent behavior. HH4B remained visible, but the representation did not become substantially higher-dimensional.

### Pooling

Mean pooling was compared against learned attention pooling.

```text
attention pooling: accuracy=0.605240, macro_f1=0.595673, hh4b_f1=0.712664, eval_quality=0.214096
mean pooling:      accuracy=0.612333, macro_f1=0.609658, hh4b_f1=0.721938, eval_quality=0.208120
```

Mean pooling gave slightly better supervised classification and is more aligned with the CTL2 reference architecture using global average pooling. Pooling affects the latent geometry, but does not fully solve collapse.

### Classifier Head

The classifier head was tested as an MLP versus a single linear layer.

```text
attention + linear: accuracy=0.611453, macro_f1=0.610902, hh4b_f1=0.723462, eval_quality=0.213784
mean + linear:      accuracy=0.618747, macro_f1=0.612317, hh4b_f1=0.713425, eval_quality=0.179823
```

The linear head performed similarly to the MLP head. This suggests that the classifier head is not hiding most of the complexity; the latent representation is already close to linearly separable for the trained labels.

### Attention Blocks

The number of attention blocks was varied at fixed `mean` pooling and `linear` classifier head.

```text
1 block:  accuracy=0.608800, macro_f1=0.609290, hh4b_f1=0.723340, eval_quality=0.212952
2 blocks: accuracy=0.618747, macro_f1=0.612317, hh4b_f1=0.713425, eval_quality=0.179823
4 blocks: accuracy=0.553840, macro_f1=0.553144, hh4b_f1=0.704966, eval_quality=0.228505
```

The original two-block baseline remains reasonable for supervised classification. Four blocks worsened classification, even though eval-quality increased. More attention depth therefore does not directly solve latent collapse.

### Internal Width (`ff_dim`)

The internal feed-forward/attention width was varied.

```text
ff_dim=64:  accuracy=0.603027, macro_f1=0.599107, dy_f1=0.444709, hh4b_f1=0.716744, effective_rank=2.952715, eval_quality=0.212936
ff_dim=128: accuracy=0.618747, macro_f1=0.612317, dy_f1=0.442925, hh4b_f1=0.713425, effective_rank=4.029164, eval_quality=0.179823
ff_dim=256: accuracy=0.620467, macro_f1=0.610173, dy_f1=0.420136, hh4b_f1=0.733486, effective_rank=4.598284, eval_quality=0.123382
```

Increasing `ff_dim` increased latent effective rank and improved HH4B F1, especially at `ff_dim=256`. However, eval-quality decreased. More internal capacity helps some classification behavior but does not remove the collapse.

### Number of Attention Heads

The number of MHA heads was varied.

```text
heads=4:  accuracy=0.610280, macro_f1=0.601085, hh4b_f1=0.714305, eval_quality=0.197135
heads=8:  accuracy=0.618747, macro_f1=0.612317, hh4b_f1=0.713425, eval_quality=0.179823
heads=16: accuracy=0.615547, macro_f1=0.609296, hh4b_f1=0.708573, eval_quality=0.190954
```

Changing the number of heads did not provide a clear gain. The original `8`-head setup remains a good supervised baseline.

## CrossEntropy + Latent Variance Regularization

After architecture ablations, an objective-level test was added:

```text
loss = CrossEntropy + lambda * variance_penalty
```

The variance penalty discourages latent dimensions from having low batch-wise standard deviation. The goal is to keep more latent dimensions active.

Fixed setup:

```text
n_events_per_class = 100000
latent_dim = 32
ff_dim = 128
num_heads = 8
num_attention_blocks = 2
pooling_mode = mean
classifier_head = linear
```

Results:

```text
lambda=0.00:
accuracy=0.618747
macro_f1=0.612317
dy_f1=0.442925
hh4b_f1=0.713425
effective_rank=4.029164
participation_ratio=2.852609
head_fisher_effective_rank=15.451691
head_fisher_participation_ratio=11.016638
eval_quality=0.179823

lambda=0.01:
accuracy=0.615747
macro_f1=0.610463
dy_f1=0.453261
hh4b_f1=0.728051
effective_rank=5.537607
participation_ratio=3.544085
head_fisher_effective_rank=19.880772
head_fisher_participation_ratio=11.689489
eval_quality=0.205883

lambda=0.05:
accuracy=0.613907
macro_f1=0.606641
dy_f1=0.420173
hh4b_f1=0.715509
effective_rank=5.274620
participation_ratio=3.448192
head_fisher_effective_rank=17.402694
head_fisher_participation_ratio=11.162409
eval_quality=0.199430
```

The `lambda=0.01` run gave the best balance:

- Effective rank increased from `4.03` to `5.54`.
- Participation ratio increased from `2.85` to `3.54`.
- Head-Fisher effective rank increased from `15.45` to `19.88`.
- HH4B F1 increased from `0.713` to `0.728`.
- DY F1 increased from `0.443` to `0.453`.
- Eval-quality increased from `0.180` to `0.206`.
- Accuracy decreased only slightly from `0.619` to `0.616`.

This is the clearest indication so far that directly regularizing the latent representation is more effective for the collapse problem than only changing architecture capacity.

## Interpretation

The architecture ablations changed model behavior but did not remove latent compression. In contrast, a small objective-level variance regularizer improved both representation diagnostics and HH4B separation.

This supports the hypothesis that the low effective dimensionality is driven largely by the supervised objective:

```text
CrossEntropy only needs enough directions to separate the known training labels.
It does not require the latent space to preserve all event-level information.
```

The model is not broken. Under the current objective, it learns a compressed representation that is sufficient for the supervised classification task.

## Evidence Files

Key curated outputs:

```text
ngt_mha_nodes_aligned_outputs/scale_tests/unified_experiment_summary_complete_only.csv
ngt_mha_nodes_aligned_outputs/scale_tests/hh4b_collapse_diagnostics_runs_only.csv
ngt_mha_nodes_aligned_outputs/scale_tests/chinchilla_style_dataset_size_summary.csv
ngt_mha_nodes_aligned_outputs/scale_tests/figures/dataset_size_scaling_summary.png
ngt_mha_nodes_aligned_outputs/scale_tests/figures/seed_stability_summary.png
ngt_mha_nodes_aligned_outputs/scale_tests/figures/latent_vs_head_fisher_dimensionality.png
```

## Conclusion

The Week 2 analysis found that latent collapse is stable across seeds and is not solved by increasing latent dimension, dataset size, attention depth, internal width, number of heads, or classifier-head complexity alone.

The most promising result is `CrossEntropy + latent variance regularization` with `lambda=0.01`, which improves effective dimensionality, Fisher dimensionality, HH4B F1, DY F1, and eval-quality with only a small accuracy cost.

This motivates moving next toward more principled objective-level representation learning tests, such as VICReg/gVICReg, Barlow Twins, BLUE-style decorrelation, or related variance/covariance objectives.
