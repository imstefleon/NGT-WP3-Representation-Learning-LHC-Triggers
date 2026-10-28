# Mermaid diagrams

Current diagrams are embedded in the corresponding README files so GitLab can
render them directly. Editable `.mmd` sources are preserved here.

| Source | Meaning | Rendered in |
|---|---|---|
| [repository_overview.mmd](repository_overview.mmd) | The three projects and their relationship | [Root README](../../README.md) |
| [ngt_100m_architecture.mmd](ngt_100m_architecture.mmd) | Selected four-block, raw-PID, attention-pooling MHA graph | [NGT README](../../ngt_model/README.md) |
| [quantization_workflow.mmd](quantization_workflow.mmd) | Checkpoint conversion, selective INT8 and paired comparison | [Quantization README](../../quantized_model/README.md) |
| [ctl2_pipeline.mmd](ctl2_pipeline.mmd) | Separate Keras/HGQ/Kedro and synthesis source workflow | [CTL2 README](../../ctl2_model/README.md) |

Earlier two-block/mean-pooling/broader-class Mermaid diagrams are retained under
`../analysis/` and `../analysis/diagrams/`. They describe historical baselines.
The [historical project overview](../historical/overview_2026_08.md) and
[CTL2 reference](../historical/ctl2_workflow_reference.md) retain their original
Mermaid diagrams as well.
