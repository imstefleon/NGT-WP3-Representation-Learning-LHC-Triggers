# MHA Nodes-Aligned Outputs

This folder stores the PyTorch Multi-Head Attention workflow outputs for the
updated Collide-2V L1-only studies.

## Contents

```text
ngt_mha_nodes_aligned_outputs/
|-- NGT_MHA_NodesAligned_FullPrecision_Collide2V_L1features.ipynb
|-- runs/          # Curated run summaries, metrics, diagnostics, and figures
`-- scale_tests/   # Aggregated sweep tables and comparison figures
```

Large regenerated artifacts are intentionally not tracked in Git:

- raw tensor caches,
- `.npy` latent/logit arrays,
- model checkpoints under `runs/**/models/`,
- transient `.log` files.

The committed outputs are the lightweight evidence needed for analysis:

- `classification_report_eval.csv`,
- `confusion_matrix_eval.csv`,
- `training_history.csv`,
- `run_config.json`,
- eval-quality summaries,
- latent/Fisher diagnostic summaries,
- selected presentation-ready figures.

## Current Study Focus

The MHA workflow is used to study whether the latent representation collapses
under supervised CrossEntropy training and whether architecture or objective
changes can recover useful latent dimensionality.

Current diagnostics include:

- PCA/effective-rank analysis of `eval_latents`,
- latent Fisher diagnostics,
- classifier-head Fisher diagnostics,
- eval-quality scores,
- repeated-seed stability tests,
- dataset-size scaling tests,
- architecture ablations,
- loss-objective comparisons.
