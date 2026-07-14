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
- `src/ngt_model/evaluation/eval_quality_collide2v.py`: Collide-2V DY-vs-MinBias adaptation of the evaluator.
- `src/ngt_model/evaluation/scale_test.py`: small-to-larger MHA capacity/data sweep for eval-quality testing.

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
