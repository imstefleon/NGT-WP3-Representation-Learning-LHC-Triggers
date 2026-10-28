# NGT configuration

`base/parameters.yml` preserves generic research defaults. The executable trainer
is configured through its command-line arguments; this reference YAML does not
replace the selected checkpoint's metadata.

For the delivered 100M model, use
[final_run/run_config.json](../final_run/run_config.json). It fixes architecture,
raw PID encoding, normalization, class order, data sharding and training settings.
The same source configuration is copied into the quantization package.

`local/mlflow.yml` records the tracking endpoint. User credentials belong in the
ignored `local/credentials.yml`; the trainer can also use MLflow environment
variables. Supply credentials in your own environment when logging a run.
