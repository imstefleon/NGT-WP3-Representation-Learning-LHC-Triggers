# CTL2 Kedro configuration

| File | Purpose |
|---|---|
| [base/parameters.yml](base/parameters.yml) | Shared data, model, objective, validation and synthesis parameters |
| [base/catalog.yml](base/catalog.yml) | Kedro dataset definitions and data-tier outputs |
| `local/catalog.yml` | Existing CERN-specific catalog overrides; adapt paths for your environment |
| `local/mlflow.yml` | Tracking endpoint and experiment configuration |
| [local/credentials.example.yml](local/credentials.example.yml) | Credential structure with placeholders |

Run Kedro from `ctl2_model/` so relative data paths resolve in this project.
Review both the shared catalog and local overrides before loading data.

For authenticated MLflow use, create your local configuration:

```bash
cp conf/local/credentials.example.yml conf/local/credentials.yml
```

Populate that ignored file with your own account values. Shared configuration
and example files are tracked; live credentials are excluded from the delivery.
The [CTL2 README](../README.md) explains the required framework environment.
