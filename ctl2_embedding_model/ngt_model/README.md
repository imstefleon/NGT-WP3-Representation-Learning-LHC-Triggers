# NGT Model

This folder contains the PyTorch-based NGT model code extracted from the notebooks.

The goal is to keep the new MHA / eval-quality workflow organized similarly to
`ctl2_model`, while avoiding changes to the original Kedro project.

## Layout

```text
ngt_model/
|-- conf/
|   |-- base/
|   `-- local/
|-- data/
|   |-- 01_raw/
|   |-- 02_loaded/
|   |-- 03_preprocessed/
|   |-- 04_models/
|   |-- 05_validation/
|   `-- 06_compile/
|-- src/ngt_model/
|   |-- augmentation/
|   |-- datasets/
|   |-- evaluation/
|   |-- models/
|   |-- pipelines/
|   |-- synthesis/
|   |-- training/
|   `-- utils/
`-- tests/
```

## Current Modules

- `src/ngt_model/datasets/collide2v_data.py`: Collide-2V loading, top-pT candidate selection, preprocessing, PID one-hot encoding, splits, and augmentation helpers.
- `src/ngt_model/models/mha_model.py`: standard MHA embedding model and latent-only wrapper.
- `src/ngt_model/training/train_mha.py`: scriptable PyTorch training entry point, defaulting to `latent_dim=128`.
- `src/ngt_model/evaluation/eval_quality.py`: Maciej's original latent-quality evaluator.
- `src/ngt_model/evaluation/eval_quality_collide2v.py`: Collide-2V adaptation of the evaluator with configurable SM background and anomaly classes.
- `src/ngt_model/evaluation/latent_diagnostics.py`: covariance/PCA effective-dimensionality diagnostics for saved `eval_latents.npy` files.
- `src/ngt_model/evaluation/latent_dim_sweep.py`: fixed-data latent-dimension sweep for testing latent collapse.
- `src/ngt_model/evaluation/scale_test.py`: small-to-larger MHA capacity/data sweep for eval-quality testing.
- `src/ngt_model/evaluation/latent_fisher.py`: empirical Fisher diagnostic over latent coordinates.
- `src/ngt_model/evaluation/head_fisher.py`: empirical Fisher diagnostic over classifier-head parameters.
- `src/ngt_model/evaluation/seed_sweep.py`: repeated-seed stability sweep.
- `src/ngt_model/evaluation/dataset_size_sweep.py`: fixed-architecture dataset-size sweep.
- `src/ngt_model/evaluation/summary_table.py`: unified summary table builder for completed runs.

## Running from the Repository Root

Because the NGT code now lives inside `ngt_model/src`, run scripts with the
package path available:

```bash
PYTHONPATH=ngt_model/src python -m ngt_model.training.train_mha --help
```

On Windows PowerShell:

```powershell
$env:PYTHONPATH = "ngt_model/src"
python -m ngt_model.training.train_mha --help
```

## Dependencies

Install the NGT workflow dependencies in the active NGT/Kubeflow Python
environment before using MLflow tracking:

```bash
python -m pip install --user -r ngt_model/requirements.txt
python -c "import mlflow; print(mlflow.__version__)"
```

## MLflow Tracking

To log a new training run to CERN MLflow, pass `--mlflow-enable`:

```bash
PYTHONPATH=ngt_model/src python -m ngt_model.training.train_mha \
  --data-source collide2v_updated \
  --n-events-per-class 1000 \
  --epochs 1 \
  --latent-dim 128 \
  --mlflow-enable \
  --mlflow-experiment ngt_model \
  --run-name smoke_test_ngt_model
```

To backfill a run that already finished without MLflow tracking:

```bash
PYTHONPATH=ngt_model/src python -m ngt_model.evaluation.log_run_to_mlflow \
  ngt_mha_nodes_aligned_outputs/runs/YYYY-MM-DD/RUN_NAME \
  --mlflow-experiment ngt_model
```

## Eval-Quality Scaling Sweep

To reproduce the small-to-reasonable eval-quality test requested by Maciej,
run:

```bash
PYTHONPATH=ngt_model/src python -m ngt_model.evaluation.scale_test \
  --grid scaling \
  --epochs 5 \
  --eval-events-per-class 2000 \
  --data-source collide2v_updated \
  --mlflow-enable \
  --mlflow-experiment ngt_model \
  --mlflow-run-name scaling_eval_quality \
  --output ngt_mha_nodes_aligned_outputs/scale_tests/scaling_eval_quality.csv
```

The `scaling` grid trains increasingly larger configurations:

| Tier | Events/class | latent_dim | ff_dim | heads |
|---|---:|---:|---:|---:|
| `tiny_1k` | 1,000 | 24 | 64 | 4 |
| `small_10k` | 10,000 | 64 | 128 | 4 |
| `medium_50k` | 50,000 | 128 | 128 | 8 |
| `reasonable_100k` | 100,000 | 128 | 128 | 8 |

## HH4B Anomaly Benchmark

The next anomaly benchmark follows the setup clarified with Sabrina:

```text
background = minbias + top + qcd + dy
anomaly    = hh4b
```

Small smoke test:

```bash
PYTHONPATH=ngt_model/src python -m ngt_model.training.train_mha \
  --data-source collide2v_updated \
  --class-names minbias,top,qcd,dy,hh4b \
  --n-events-per-class 1000 \
  --epochs 1 \
  --latent-dim 128 \
  --eval-quality-events 500 \
  --run-name smoke_hh4b_full_sm
```

Eval-quality on an existing HH4B-capable checkpoint:

```bash
PYTHONPATH=ngt_model/src python -m ngt_model.evaluation.eval_quality_collide2v \
  ngt_mha_nodes_aligned_outputs/runs/YYYY-MM-DD/RUN_NAME/models/mha_nodes_aligned_best_for_eval.pt \
  --background-classes minbias,top,qcd,dy \
  --anomaly-class hh4b \
  --n-events-per-class 2000 \
  --output ngt_mha_nodes_aligned_outputs/runs/YYYY-MM-DD/RUN_NAME/eval_quality_hh4b_full_sm.json
```

## HH4B With Granular DY Final States

If the updated Collide-2V dataset contains separate DY final-state samples, the
loader can also train with granular DY labels:

```text
dy_ee, dy_mumu, dy_tautau
```

Run a small availability smoke test first:

```bash
PYTHONPATH=ngt_model/src python -m ngt_model.training.train_mha \
  --data-source collide2v_updated \
  --class-names minbias,top,qcd,dy_ee,dy_mumu,dy_tautau,hh4b \
  --n-events-per-class 1000 \
  --epochs 1 \
  --latent-dim 32 \
  --run-name smoke_hh4b_dy_granular_1k
```

If all classes load correctly, run the 100k/class benchmark:

```bash
PYTHONPATH=ngt_model/src python -m ngt_model.training.train_mha \
  --data-source collide2v_updated \
  --class-names minbias,top,qcd,dy_ee,dy_mumu,dy_tautau,hh4b \
  --n-events-per-class 100000 \
  --epochs 30 \
  --latent-dim 32 \
  --ff-dim 128 \
  --num-heads 8 \
  --eval-quality-events 2000 \
  --mlflow-enable \
  --mlflow-experiment ngt_model \
  --mlflow-run-name hh4b_full_sm_dy_granular_100k_latent32 \
  --run-name hh4b_full_sm_dy_granular_100k_latent32
```

Important: these labels are only valid if the source dataset provides separate
DY final-state samples or file paths. The code does not infer DY final state
from L1-only candidates.

## Architecture Ablations

The Week 2 collapse study keeps the HH4B full-SM benchmark fixed and changes one model component at a time:

```text
baseline = 100k events/class, latent_dim=32, ff_dim=128,
           num_heads=8, num_attention_blocks=2,
           pooling=mean, classifier_head=linear
```

The tested axes are:

```text
pooling_mode: mean vs attention
classifier_head: linear vs mlp
num_attention_blocks: 1, 2, 4
ff_dim: 64, 128, 256
num_heads: 4, 8, 16
```

Example:

```bash
PYTHONPATH=ngt_model/src python -m ngt_model.training.train_mha \
  --data-source collide2v_updated \
  --class-names minbias,top,qcd,dy,hh4b \
  --n-events-per-class 100000 \
  --epochs 30 \
  --latent-dim 32 \
  --ff-dim 128 \
  --num-heads 8 \
  --num-attention-blocks 2 \
  --pooling-mode mean \
  --classifier-head linear \
  --eval-quality-events 2000 \
  --mlflow-enable \
  --mlflow-experiment ngt_model \
  --run-name ablation_baseline_blocks2_mean_linear_100k_latent32
```

## CrossEntropy + Latent Variance Regularization

The first objective-level anti-collapse test adds an optional variance penalty:

```text
loss = CrossEntropy + lambda * variance_penalty
```

This is controlled by:

```text
--latent-var-reg-weight
--latent-var-target
```

Example:

```bash
PYTHONPATH=ngt_model/src python -m ngt_model.training.train_mha \
  --data-source collide2v_updated \
  --class-names minbias,top,qcd,dy,hh4b \
  --n-events-per-class 100000 \
  --epochs 30 \
  --latent-dim 32 \
  --ff-dim 128 \
  --num-heads 8 \
  --num-attention-blocks 2 \
  --pooling-mode mean \
  --classifier-head linear \
  --latent-var-reg-weight 0.01 \
  --latent-var-target 1.0 \
  --eval-quality-events 2000 \
  --mlflow-enable \
  --mlflow-experiment ngt_model \
  --run-name ablation_varreg001_blocks2_mean_linear_100k_latent32
```

In the Week 2 tests, `latent_var_reg_weight=0.01` improved effective rank, participation ratio, head-Fisher effective rank, HH4B F1, DY F1, and eval-quality with only a small accuracy cost.

## Latent Effective-Dimensionality Diagnostics

To quantify latent-space collapse on any completed run:

```bash
PYTHONPATH=ngt_model/src python -m ngt_model.evaluation.latent_diagnostics \
  ngt_mha_nodes_aligned_outputs/runs/YYYY-MM-DD/RUN_NAME
```

This writes:

```text
latent_effective_dim_summary.json
latent_effective_dim_spectrum.csv
figures/latent_effective_dim_spectrum.png
```

## Latent-Dimension Sweep

To test whether `latent_dim=128` is actually needed, keep the HH4B full-SM setup fixed and vary only `latent_dim`:

```bash
PYTHONPATH=ngt_model/src python -m ngt_model.evaluation.latent_dim_sweep \
  --class-names minbias,top,qcd,dy,hh4b \
  --background-classes minbias,top,qcd,dy \
  --anomaly-class hh4b \
  --latent-dims 16,32,64,128 \
  --n-events-per-class 100000 \
  --epochs 30 \
  --eval-quality-events 2000 \
  --mlflow-enable \
  --mlflow-experiment ngt_model \
  --mlflow-run-name latent_dim_sweep_hh4b_100k \
  --output ngt_mha_nodes_aligned_outputs/scale_tests/latent_dim_sweep_hh4b_100k.csv
```

The output CSV compares classification metrics, HH4B recall/F1, eval-quality, effective rank, participation ratio, and PCA cumulative explained variance for each latent dimension.

## Latent Empirical-Fisher Diagnostic

To test whether the latent dimensions are actually used by the classifier, compute the empirical Fisher diagonal in latent space:

```bash
PYTHONPATH=ngt_model/src python -m ngt_model.evaluation.latent_fisher \
  ngt_mha_nodes_aligned_outputs/runs/YYYY-MM-DD/RUN_NAME \
  --n-events-per-class 2000
```

For the HH4B full-SM 100k run:

```bash
PYTHONPATH=ngt_model/src python -m ngt_model.evaluation.latent_fisher \
  ngt_mha_nodes_aligned_outputs/runs/2026-07-15/hh4b_full_sm_100k_latent128 \
  --n-events-per-class 2000
```

This writes:

```text
latent_fisher_summary.json
latent_fisher_by_dim.csv
figures/latent_fisher_diag.png
```

The key quantity is:

```text
F_j = mean((d loss / d latent_j)^2)
```

Large Fisher mass means that a latent dimension has strong influence on the classification loss. If only a few dimensions contain most of the Fisher mass, the classifier is using a low-dimensional subspace even if the nominal latent dimension is large.

## Classifier-Head Fisher Diagnostic

To test whether the classifier head is hiding complexity, run:

```bash
PYTHONPATH=ngt_model/src python -m ngt_model.evaluation.head_fisher \
  ngt_mha_nodes_aligned_outputs/runs/YYYY-MM-DD/RUN_NAME \
  --n-events-per-class 2000 \
  --device cuda
```

This writes:

```text
head_fisher_summary.json
head_fisher_by_parameter.csv
head_fisher_spectrum.csv
figures/head_fisher_diag.png
```

If the linear classifier head performs similarly to the MLP head, and the head-Fisher dimensionality is not collapsing to a single parameter, then the latent representation is already close to linearly separable for the supervised labels.
