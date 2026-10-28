# Run Analysis Index

This folder contains the English analysis notes for the main runs performed during the DY-MinBias, HH4B, latent-collapse, and scaling studies.

For the delivered checkpoint and the new selective INT8 measurements, start
with [the delivery guide](../DELIVERY.md), [current diagrams](../diagrams/README.md)
and [quantization evidence](../../quantized_model/evidence/). These notes include
earlier architectures, label spaces and scoring procedures.

The notes are listed by topic. Dates are intentionally omitted from file names.

| File | Scope |
|---|---|
| `baseline_1k_100k.md` | Baseline 1k and 100k runs |
| `dy_minbias_followups.md` | Wide attention, augmentation, and 300k MinBias follow-up runs |
| `top_pt_eta_fix.md` | Top-pT candidate selection fix and corrected runs |
| `dy_minbias_latent_space.md` | DY final-state and latent-space separation analysis |
| `dataset_diagnostics_and_runs.md` | Dataset source diagnostics across HF, ROOT, and updated Collide-2V |
| `updated_collide2v_run.md` | Updated Collide-2V 100k-per-class safe baseline and DY-MinBias behavior |
| `mha_eval_quality_analysis.md` | MHA updated Collide-2V run and first eval-quality latent-space score |
| `mha_l1_100k_eval_quality.md` | MHA L1-only 100k-per-class run with latent_dim=128 and eval-quality analysis |
| `hh4b_latent_dim_sweep.md` | HH4B full-SM latent-dimension sweep and latent-collapse diagnostics |
| `hh4b_collapse_diagnostics.md` | PCA, effective rank, Fisher, repeated seeds, dataset-size sweep, and architecture ablations |
| `latent_collapse_report.md` | Consolidated latent-collapse diagnostic report |
| `best_config_finding_and_latent_sweep.md` | Best-configuration finding, granular Top/WW labels, and latent-dimension sweep |
| `superclass_loss_objective_comparison.md` | Broader SM superclass benchmark and CrossEntropy-based loss-objective comparison |
| `external_higgs_signal_evaluation.md` | External Higgs-like signal evaluation |
| `quantization_readiness_and_anomaly_scores.md` | Quantization readiness and anomaly-score notes |
| `scaling_axes_analysis.md` | Scaling-axis comparison and 150k/200k focused runs |
| `1m_500k_scaling_analysis.md` | 500k/1M scaling pilot analysis |
| `scaling_analysis.md` | Consolidated scaling analysis |
