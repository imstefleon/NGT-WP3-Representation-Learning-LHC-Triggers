# openlab_ss26

Repository for the CERN Openlab Summer Student 2026 project:

**NGT WP3 Representation Learning for LHC Triggers**

This project studies L1-only machine-learning representations of simulated LHC collision events for Level-1 trigger and anomaly-detection workflows. The current focus is to learn compact latent spaces where physics processes such as MinBias, Top, QCD, and Drell-Yan can be compared, while keeping the input representation compatible with L1 trigger information.

Project notes and meeting context are tracked in the CERN CodiMD page:

- <https://codimd.web.cern.ch/s/oEeoJz7QJ>

## Project Status

Current safe baseline:

- Updated Collide-2V production parquet workflow implemented.
- L1-only inputs from `L1T_PUPPIPart_*` branches.
- Linformer notebook baseline preserved for historical DY-MinBias studies.
- PyTorch MHA workflow added under `ctl2_embedding_model/ngt_model/`.
- `latent_dim=128` MHA run with 100k events per class completed and logged to MLflow.
- Eval-quality scaling tests added to compare representation quality from small to larger configurations.
- Notebook paths were updated so notebooks can live inside their workflow output folders.

Main current finding:

- Top is consistently well separated.
- QCD has moderate separation.
- DY still overlaps strongly with MinBias.
- Increasing model capacity improves eval-quality, but DY-MinBias separability is still the main limitation.

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
