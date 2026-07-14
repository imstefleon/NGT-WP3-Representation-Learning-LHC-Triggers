# Configuration

Configuration files for the NGT PyTorch workflow.

`base/parameters.yml` contains default MHA training settings. Local overrides can
be placed under `local/` and should not contain secrets or large paths that are
not portable.

MLflow credentials are expected in `local/credentials.yml`, which is ignored by
Git. The default MLflow target and experiment name can be documented in
`local/mlflow.yml`.
