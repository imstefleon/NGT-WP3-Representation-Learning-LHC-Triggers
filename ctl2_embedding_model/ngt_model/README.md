# NGT Model

**Delivery entry point:** [selected 100M run](final_run/README.md), including the
epoch-4 weights, frozen five-feature/raw-PID configuration and five-epoch history.
The separate [quantization package](../quantized_model/README.md) preserves this
checkpoint and supplies a validated numerical CPU prototype. See the
[delivery guide](../../DELIVERY.md) for exact status. The generic configurations
and examples below describe the broader research workflow, not a substitute for
the selected run's `run_config.json`.


This folder contains the modular PyTorch workflow used for the current NGT
MHA studies on L1-only Collide-2V event embeddings.

The code was extracted from the original notebooks so that training,
evaluation, MLflow logging, and latent-collapse diagnostics can be run from the
terminal in a reproducible way.

## Purpose

The current workflow studies whether an L1-only event embedding can separate
Standard Model processes and expose HH4B-like events as an anomaly-like signal.

Current focus:

- train MHA-based event encoders on updated Collide-2V L1 candidate features;
- compare coarse labels, granular final-state labels, and broader SM
  superclasses;
- diagnose latent collapse using PCA/effective rank, latent Fisher, and
  classifier-head Fisher;
- compare loss-objective variants such as CrossEntropy, label smoothing, focal
  loss, SupCon, latent variance regularization, and covariance regularization;
- evaluate injected Higgs-like samples with classifier response, GMM anomaly
  score, MinBias-distance score, robust combined score, and fixed-background
  efficiencies through 99.5% background rejection;
- generate and collect scaling-study manifests for focused 4-class studies;
- prepare rank-sharded DDP scaling runs for the 100M-parameter / 80M-event study;
- track training runs and summary metrics in CERN MLflow.

## Layout

```text
ngt_model/
|-- conf/
|   |-- base/
|   `-- local/                 # local credentials; ignored by git
|-- src/ngt_model/
|   |-- datasets/
|   |   `-- collide2v_data.py   # updated Collide-2V L1 loading/preprocessing
|   |-- evaluation/
|   |   |-- eval_quality.py
|   |   |-- eval_quality_collide2v.py
|   |   |-- external_classifier_response.py
|   |   |-- external_higgs_anomaly.py
|   |   |-- latent_diagnostics.py
|   |   |-- latent_fisher.py
|   |   |-- head_fisher.py
|   |   |-- latent_dim_sweep.py
|   |   |-- seed_sweep.py
|   |   |-- dataset_size_sweep.py
|   |   |-- scale_test.py
|   |   |-- summary_table.py
|   |   `-- log_run_to_mlflow.py
|   |-- models/
|   |   `-- mha_model.py
|   |-- pipelines/
|   |   |-- scaling_study_manifest.py
|   |   `-- scaling_study_collect_results.py
|   `-- training/
|       `-- train_mha.py
`-- README.md
```

Run outputs are written outside this package, mainly under:

```text
ngt_mha_nodes_aligned_outputs/
|-- cache/
|-- runs/
`-- scale_tests/

scaling_study/
|-- manifests/
|-- summaries/
|-- probe_ablation/
|-- granular_sm13/
|-- main_4class_scaling/
`-- large_100m_80m/
```

## Environment Setup

From the repository root:

```bash
cd /eos/home-i03/e/emorenol/ctl2_embedding_model
export PYTHONPATH=$PWD/ngt_model/src:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/bin/miniconda3/lib:$LD_LIBRARY_PATH
```

Quick import check:

```bash
python -c "import ngt_model; print('ngt_model OK')"
python -m ngt_model.training.train_mha --help
```

## Supported Classes

Current updated Collide-2V class names include:

```text
minbias, qcd, dy, top, hh4b,
tt_hadronic, tt_leptonic, tt_semileptonic,
ww_hadronic, ww_leptonic, ww_semileptonic,
wjets, ww, wz, ttw, ttz,
dy_ee, dy_mumu, dy_tautau
```

Important: the current training representation is L1-only. Granular labels are
assigned from the sample/file category, not inferred from L1 features.

## Current Baseline

The controlled MHA baseline uses:

```text
data_source          = collide2v_updated
feature_mode         = l1_only
candidate_selection  = top_pt
n_constituents       = 100
features             = pT, eta, phi, D0, PID
attention blocks     = 2
ff_dim               = 128
num_heads            = 8
pooling              = mean
classifier head      = linear
latent_var_weight    = 0.01
```

For the focused anomaly-oriented scaling path:

```text
classes              = minbias, qcd, dy, top
classifier head      = linear
primary score        = minbias_mahalanobis
stable 1M recipe     = augmentation_copies=0, lr=3e-5
large DDP loading    = --data-loading-mode rank_sharded_materialized
```

For the broad SM superclass setup:

```text
classes = minbias, qcd, dy, top, wjets, ww, wz, ttw, ttz, hh4b
```

For the granular final-state setup:

```text
classes = minbias, qcd, dy,
          tt_hadronic, tt_leptonic, tt_semileptonic,
          ww_hadronic, ww_leptonic, ww_semileptonic,
          hh4b
```

## Expected Run Outputs

Completed training runs should contain:

```text
run_config.json
training_history.csv
models/mha_nodes_aligned_best.pt
models/mha_nodes_aligned_best_for_eval.pt
eval_latents.npy
eval_logits.npy
eval_labels.npy
classification_report_eval.csv
confusion_matrix_eval.csv
eval_quality_scores_n*.json
```

External-anomaly evaluations add:

```text
figures/external_higgs_anomaly/external_higgs_anomaly_metrics_by_method.csv
figures/external_higgs_anomaly/external_higgs_event_scores.csv
figures/external_higgs_anomaly/minbias_distance_distributions.png
figures/external_higgs_anomaly/minbias_distance_distributions_per_signal.png
figures/external_higgs_anomaly/background_latent_tsne.png
figures/external_higgs_anomaly/external_higgs_latent_tsne.png
figures/external_classifier_response/classifier_response_summary.json
```

The external-anomaly metric tables include 95%, 99%, and 99.5%
fixed-background-rejection efficiencies. Use `eff_at_99_5pct_bg` for the HH4B
99.5% background-rejection check.

Use `classification_report_eval.csv` for per-class precision, recall, and F1.
Use `confusion_matrix_eval.csv` for confusion-matrix heatmaps. If a run does
not contain `eval_labels.npy`, `classification_report_eval.csv`, or
`confusion_matrix_eval.csv`, do not treat generated per-class plots as evidence.

## MLflow Tracking

The project scripts can load MLflow credentials from:

```text
ngt_model/conf/local/credentials.yml
```

This file is local-only and must not be committed. Manual Python snippets that
call `mlflow` directly do not automatically load this file; the loader is called
inside the NGT training/evaluation scripts.

Check that the local credentials are visible to the project scripts:

```bash
python - <<'PY'
import os
from ngt_model.training.train_mha import _load_mlflow_credentials_from_local_conf
print("before_user:", bool(os.environ.get("MLFLOW_TRACKING_USERNAME")))
print("before_password:", bool(os.environ.get("MLFLOW_TRACKING_PASSWORD")))
_load_mlflow_credentials_from_local_conf()
print("after_user:", bool(os.environ.get("MLFLOW_TRACKING_USERNAME")))
print("after_password:", bool(os.environ.get("MLFLOW_TRACKING_PASSWORD")))
PY
```

## Diagnostics

Run latent effective-dimensionality diagnostics:

```bash
python -m ngt_model.evaluation.latent_diagnostics \
  ngt_mha_nodes_aligned_outputs/runs/YYYY-MM-DD/RUN_NAME
```

This writes:

```text
latent_effective_dim_summary.json
latent_effective_dim_spectrum.csv
figures/latent_effective_dim_spectrum.png
```

Run latent Fisher diagnostics:

```bash
python -m ngt_model.evaluation.latent_fisher \
  ngt_mha_nodes_aligned_outputs/runs/YYYY-MM-DD/RUN_NAME \
  --n-events-per-class 2000 \
  --device cuda
```

Run classifier-head Fisher diagnostics:

```bash
python -m ngt_model.evaluation.head_fisher \
  ngt_mha_nodes_aligned_outputs/runs/YYYY-MM-DD/RUN_NAME \
  --n-events-per-class 2000 \
  --device cuda
```

Run external Higgs-like anomaly diagnostics:

```bash
python -m ngt_model.evaluation.external_higgs_anomaly \
  --run-dir ngt_mha_nodes_aligned_outputs/runs/YYYY-MM-DD/RUN_NAME \
  --signals hh4b,hh,vbfh,ggh \
  --n-background-per-class 2000 \
  --n-signal-per-class 2000 \
  --device cuda
```

Run external classifier-response diagnostics:

```bash
python -m ngt_model.evaluation.external_classifier_response \
  ngt_mha_nodes_aligned_outputs/runs/YYYY-MM-DD/RUN_NAME \
  --external-classes hh4b,hh,vbfh,ggh \
  --n-events-per-class 2000 \
  --device cuda
```

Rebuild the complete-only summary table:

```bash
python -m ngt_model.evaluation.summary_table \
  --complete-only \
  --run-root ngt_mha_nodes_aligned_outputs/runs \
  --output ngt_mha_nodes_aligned_outputs/scale_tests/unified_experiment_summary_complete_only.csv
```

Generate scaling-study manifests:

```bash
python -m ngt_model.pipelines.scaling_study_manifest \
  --manifest-dir scaling_study/manifests \
  --output-root outputs/scaling_study \
  --class-configurations 4class \
  --samples-per-class 50000,100000,150000,200000
```

Collect scaling-study results:

```bash
python -m ngt_model.pipelines.scaling_study_collect_results \
  --manifest scaling_study/manifests/scaling_study_manifest.csv \
  --output scaling_study/summaries/master_results_table.csv
```
