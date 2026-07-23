# openlab_ss26

Repository for the CERN Openlab Summer Student 2026 project:

**NGT WP3 Representation Learning for LHC Triggers**

This project studies L1-only machine-learning representations of simulated LHC collision events for Level-1 trigger and anomaly-detection workflows. The current focus is to learn compact latent spaces where Standard Model processes and candidate anomaly signals such as HH4B can be compared, while keeping the input representation compatible with L1 trigger information.

Project notes and meeting context are tracked in the CERN CodiMD page:

- <https://codimd.web.cern.ch/s/oEeoJz7QJ>

## Project Status

Current safe baseline:

- Updated Collide-2V production parquet workflow implemented.
- L1-only inputs from `L1T_PUPPIPart_*` branches.
- Linformer notebook baseline preserved for historical DY-MinBias studies.
- PyTorch MHA workflow added under `ctl2_embedding_model/ngt_model/`.
- HH4B-vs-full-SM benchmark added with `minbias,top,qcd,dy` as background.
- Latent-collapse diagnostics added: PCA/effective rank, latent Fisher, classifier-head Fisher, and eval-quality.
- Supervised architecture ablations completed for pooling, classifier head, attention blocks, `ff_dim`, and `num_heads`.
- CrossEntropy plus small latent variance regularization was added as the first objective-level anti-collapse test.
- Notebook paths were updated so notebooks can live inside their workflow output folders.

Main current finding:

- Latent collapse is stable across seeds and is not solved by increasing nominal latent dimension, dataset size, attention depth, or number of heads alone.
- Mean pooling and linear classifier heads are good supervised baselines.
- `ff_dim=256` improves HH4B F1 but does not remove low effective dimensionality.
- A small latent variance regularizer (`lambda=0.01`) improves effective rank, participation ratio, head-Fisher dimensionality, HH4B F1, DY F1, and eval-quality with only a small accuracy cost.

## Repository Layout

```text
.
|-- README.md
`-- ctl2_embedding_model/
    |-- README.md
    |-- analysis/
    |-- ctl2_model/
    |-- diagnostic_notebooks/
    |-- diagnostics/
    |-- ngt_linformer_nodes_aligned_outputs/
    |   |-- NGT_Linformer_NodesAligned_FullPrecision_Collide2V_L1features.ipynb
    |   `-- runs/
    |-- ngt_mha_nodes_aligned_outputs/
    |   |-- NGT_MHA_NodesAligned_FullPrecision_Collide2V_L1features.ipynb
    |   |-- runs/
    |   `-- scale_tests/
    `-- ngt_model/
        |-- conf/
        |-- data/
        |-- src/ngt_model/
        `-- tests/
```

## Main Workflows

| Workflow | Location | Purpose |
|---|---|---|
| Linformer notebook | `ctl2_embedding_model/ngt_linformer_nodes_aligned_outputs/NGT_Linformer_NodesAligned_FullPrecision_Collide2V_L1features.ipynb` | Historical baseline and DY-MinBias studies |
| MHA notebook | `ctl2_embedding_model/ngt_mha_nodes_aligned_outputs/NGT_MHA_NodesAligned_FullPrecision_Collide2V_L1features.ipynb` | MHA architecture comparison |
| Scriptable MHA package | `ctl2_embedding_model/ngt_model/` | PyTorch training, MLflow logging, and eval-quality tests |

The notebooks now resolve the project root automatically. This avoids broken relative paths after moving the notebooks into their workflow folders.

## Current Model Flow

```mermaid
flowchart TD
    A["Updated Collide-2V parquet samples"] --> B["L1-only candidate tensor<br/>100 candidates x 5 features"]
    B --> C["Input projection<br/>Linear + ReLU"]
    C --> D["Stacked MHA attention blocks<br/>Q/K/V message passing + residuals"]
    D --> E["Event pooling<br/>mean or learned attention pooling"]
    E --> F["Post-pool dense stack<br/>Linear + ReLU + Linear + ReLU"]
    F --> G["Latent embedding"]
    G --> H["Classifier head<br/>linear or MLP"]
    G --> I["Latent diagnostics<br/>PCA, effective rank, Fisher, eval-quality"]
    H --> J["Process labels<br/>MinBias, Top, QCD, DY, HH4B"]
```

The current ablation campaign changes one component at a time: pooling, classifier head,
number of attention blocks, `ff_dim`, number of attention heads, and latent variance regularization. This is used to study
whether the low effective dimensionality is caused by architecture choices or by the
cross-entropy training objective.

## L1-Only Inputs

The current updated Collide-2V representation uses a `100 x 5` tensor per event:

```text
[pt, eta, phi, dxy_or_D0, pid]
```

For updated Collide-2V, these are read from:

```text
L1T_PUPPIPart_PT
L1T_PUPPIPart_Eta
L1T_PUPPIPart_Phi
L1T_PUPPIPart_D0
L1T_PUPPIPart_PID
```

Gen-level or FullReco information is not used as model input. It is only used for diagnostics, for example when studying DY final-state subgroups.

## MHA Training

From `ctl2_embedding_model/`:

```bash
export PYTHONPATH=$PWD/ngt_model/src:$PYTHONPATH
python -m ngt_model.training.train_mha \
  --data-source collide2v_updated \
  --n-events-per-class 100000 \
  --epochs 30 \
  --latent-dim 128 \
  --mlflow-enable \
  --mlflow-experiment ngt_model \
  --run-name mha_l1_only_100k_latent128
```

Eval-quality scaling sweep:

```bash
python -m ngt_model.evaluation.scale_test \
  --grid scaling \
  --epochs 5 \
  --eval-events-per-class 2000 \
  --data-source collide2v_updated \
  --mlflow-enable \
  --mlflow-experiment ngt_model \
  --mlflow-run-name scaling_eval_quality \
  --output ngt_mha_nodes_aligned_outputs/scale_tests/scaling_eval_quality.csv
```

## Analysis Notes

English analysis notes are stored in:

```text
ctl2_embedding_model/analysis/
```

Recent analysis files:

- `2026-07-11_updated_collide2v_run.md`
- `2026-07-13_mha_eval_quality_analysis.md`
- `2026-07-14_mha_l1_100k_eval_quality.md`
- `2026-07-21_hh4b_collapse_diagnostics.md`
- `2026-07-23_week2_latent_collapse_report.md`

## Safe Output Policy

The repository keeps notebooks, analysis notes, selected CSV/JSON run summaries, and small curated evidence files.

The repository intentionally does not track:

- local credentials,
- generated caches,
- model checkpoints,
- large `.npy` tensors,
- diagnostic outputs that can be regenerated.

MLflow credentials should stay only in local ignored files such as:

```text
ctl2_embedding_model/ngt_model/conf/local/credentials.yml
ctl2_embedding_model/ctl2_model/conf/local/credentials.yml
```
