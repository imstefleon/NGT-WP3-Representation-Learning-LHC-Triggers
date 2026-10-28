# NGT Model Scaling Study

This folder contains curated manifests, compact summaries, launch/resume tools
and benchmark metadata. The broader run directories and raw caches remain in
the original CERN research workspace.

The delivered large-run checkpoint is in [../final_run/](../final_run/README.md).
Its five-epoch CE + InfoNCE history is separate from the earlier CE + var/cov
launch plans below. Quantization is documented in the sibling
[quantized_model project](../../quantized_model/README.md).

## Folder Layout

| Path | Included content |
|---|---|
| `manifests/` | Command grids and run definitions |
| `summaries/` | Compact available result tables |
| `main_4class_scaling/` | Focused four-class interpretation and references |
| `power_law_param_sample/` | Planned joint parameter/sample sweep, commands and compact tables |
| `probe_ablation/` | Matched classifier-probe command manifest |
| `large_100m_80m/` | Large-run configs, benchmark, smoke/launch/resume tools and templates |

Manifests describe planned runs. A row alone does not prove a run completed.
Individual study README files distinguish original plans from retained evidence.

## Class Configurations

| configuration | classes | note |
| --- | --- | --- |
| `4class` | `minbias,qcd,dy,top` | Current balanced SM baseline. |
| `10plus_sm` | `minbias,qcd,dy,tt_hadronic,tt_leptonic,tt_semileptonic,wjets,ww,wz,ttw,ttz` | Background-only 11-class configuration. It avoids training on HH4B/HH/VBFH/ggH so those remain external anomaly probes. |
| `granular_sm15` | `minbias,qcd,dy_ee,dy_mumu,dy_tautau,tt_hadronic,tt_leptonic,tt_semileptonic,wjets,ww_hadronic,ww_leptonic,ww_semileptonic,wz,ttw,ttz` | Most granular SM-background class list currently exposed by the repository class names, assuming all samples are available in the dataset. |
| `legacy_10class_hh4b_ref` | `minbias,qcd,dy,top,wjets,ww,wz,ttw,ttz,hh4b` | Optional legacy/reference configuration only. HH4B is supervised here, so it should not be used as the primary exotic-anomaly baseline. |

`10plus_sm` is not the maximum possible granularity. It is the safer first >10-class background-only setup. For the "as granular as possible" study, use `granular_sm15` after verifying that every class has enough events for the requested balanced sample size.

## Default Scaling Schedule

| samples/class | model size | latent_dim | ff_dim | heads | attention blocks |
| ---: | --- | ---: | ---: | ---: | ---: |
| 50k | `ff128` | 64 | 128 | 8 | 2 |
| 100k | `ff256` | 128 | 256 | 8 | 2 |
| 150k | `large` | 128 | 256 | 8 | 4 |
| 200k | `xl` | 256 | 512 | 8 | 4 |

All generated commands use:

- `--classifier-head mlp2`
- `--loss-mode ce`
- `--pooling-mode mean`
- `--latent-var-reg-weight 0.01`
- `--latent-cov-reg-weight 0.001`
- `--augmentation-copies 1`
- `--mlflow-enable`

The `mlp2` probe is:

`Linear(latent_dim, ff_dim) -> GELU -> Linear(ff_dim, ff_dim) -> GELU -> Linear(ff_dim, n_classes)`

## Files

- `manifests/scaling_study_manifest.csv`: one row per planned run.
- `manifests/scaling_study_manifest.json`: same information in JSON.
- `manifests/scaling_study_commands.sh`: bash commands for training and post-training diagnostics.
- `summaries/probe_ablation_summary.csv`: compact probe-ablation metrics.
- `summaries/main_4class_scaling_minbias_summary.csv`: compact focused 4-class scaling metrics.
- `summaries/master_results_table.csv`: collected manifest results when matching run directories are available.
The collector also creates a missing-run report when executed; regenerated reports remain local.
- `large_100m_80m/README.md`: audit, memory estimates, 100M model table, two-run plan, DDP smoke checks, and CERN/Kubeflow launch templates.
- `power_law_param_sample/README.md`: proposed power-law sweep with `ff88/ff208/ff480/ff968` paired to `10k/50k/250k/1M` total samples.

## First Validation Run

Run only the first block first:

```bash
cd /path/to/openlab_ss26
export PYTHONPATH="${PWD}/ngt_model/src:${PYTHONPATH:-}"
export MPLBACKEND=Agg
PY=python

"${PY}" -m ngt_model.training.train_mha --data-source collide2v_updated --class-names minbias,qcd,dy,top --n-events-per-class 50000 --epochs 50 --early-stopping-patience 10 --batch-size 256 --num-workers 0 --latent-dim 64 --ff-dim 128 --num-heads 8 --num-attention-blocks 2 --pooling-mode mean --classifier-head mlp2 --loss-mode ce --latent-var-reg-weight 0.01 --latent-cov-reg-weight 0.001 --augmentation-copies 1 --class-weight-mode none --seed 42 --output-root outputs/scaling_study/4class/50k/ff128 --run-name scaling_4class_50k_ff128_mlp2_seed42 --eval-quality-events 2000 --mlflow-enable --mlflow-tracking-uri https://ml.cern.ch/models --mlflow-experiment ngt_model_scaling_study --mlflow-run-name scaling_4class_50k_ff128_mlp2_seed42
```

After it finishes, run the diagnostics for the generated run directory:

```bash
RUN_DIR="outputs/scaling_study/4class/50k/ff128/runs/$(date +%F)/scaling_4class_50k_ff128_mlp2_seed42"

"${PY}" -m ngt_model.evaluation.latent_diagnostics "${RUN_DIR}"
"${PY}" -m ngt_model.evaluation.latent_fisher "${RUN_DIR}" --n-events-per-class 2000 --device cpu --batch-size 256
"${PY}" -m ngt_model.evaluation.head_fisher "${RUN_DIR}" --scope classifier_head --n-events-per-class 2000 --device cpu --batch-size 256
"${PY}" -m ngt_model.evaluation.external_classifier_response "${RUN_DIR}" --external-classes hh4b,hh,vbfh,ggh --n-events-per-class 2000 --device cpu --batch-size 256
"${PY}" -m ngt_model.evaluation.external_higgs_anomaly --run-dir "${RUN_DIR}" --signals hh4b,hh,vbfh,ggh --n-background-per-class 2000 --n-signal-per-class 2000 --device cpu --batch-size 256 --tsne-max-events-per-class 600 --tsne-perplexity 40 --tsne-n-iter 1000 --raise-errors
```

## PCA vs t-SNE

PCA and t-SNE are both useful, but they answer different questions.

- PCA is the quantitative collapse diagnostic. Use it for explained variance, effective rank, participation ratio, and number of dimensions needed for 90/95/99% variance.
- t-SNE is the qualitative clustering diagnostic. Use it to see whether Top, DY, MinBias, QCD, and injected Higgs-like samples form visible islands or overlap in local latent neighborhoods.

The anomaly script now writes:

- `background_latent_tsne.png`: SM/background classes only.
- `external_higgs_latent_tsne.png`: SM/background plus HH4B, HH, VBFH, ggH.
- `background_latent_tsne_points.npz`
- `external_higgs_latent_tsne_points.npz`

By default t-SNE uses a balanced subsample of up to 600 events per class with perplexity 40. Use `--skip-tsne` if you want a faster diagnostic pass.

## Probe Non-Linearity Test

Maciej's collapse hypothesis should be tested with matched runs where only the classifier probe changes:

```bash
"${PY}" -m ngt_model.pipelines.scaling_study_manifest \
  --manifest-dir outputs/scaling_study/probe_ablation \
  --output-root outputs/scaling_study/probe_ablation \
  --class-configurations 4class \
  --samples-per-class 50000 \
  --classifier-heads linear,mlp,mlp2
```

This creates three matched commands:

- `linear`: direct linear probe from latent to class logits.
- `mlp`: old one-hidden-layer probe.
- `mlp2`: new two-hidden-layer non-linear probe.

Compare these with:

- PCA explained-variance spectrum.
- `effective_rank_entropy`.
- `participation_ratio`.
- number of PCA dimensions needed for 90/95/99%.
- Fisher diagnostics for latent and classifier head.
- anomaly metrics from GMM, MinBias distance, and overall score.

If `linear` has strong low-dimensional collapse and `mlp2` has higher effective rank / smoother PCA spectrum at similar classifier accuracy, that supports the hypothesis that the linear classifier pressure was shaping the latent too directly.

## Outputs Per Run

The training script writes:

- `run_config.json`
- training history
- checkpoints under `models/`
- evaluation logits/latents/labels
- classification metrics
- parameter counts: total, encoder, classifier/probe
- exact classifier probe architecture

The anomaly evaluation writes:

- `external_higgs_event_scores.csv`
- `external_higgs_anomaly_metrics.csv` for the existing GMM score plots
- `external_higgs_anomaly_metrics_by_method.csv` with one row per signal and method:
  - `gmm_nll`
  - `minbias_mahalanobis`
  - `overall_robust_mean`
- fixed-background efficiencies at 95%, 99%, and 99.5% background rejection
- classifier response CSV
- PCA points and figures
- anomaly-score distributions and fixed-background-efficiency plots

The overall score is:

`0.5 * robust_z(gmm_anomaly_score; background_eval) + 0.5 * robust_z(minbias_distance_score; background_eval)`

## Master Results Table

After one or more runs have finished and diagnostics have been run:

```bash
"${PY}" -m ngt_model.pipelines.scaling_study_collect_results \
  --manifest outputs/scaling_study/manifests/scaling_study_manifest.csv \
  --output outputs/scaling_study/summaries/master_results_table.csv
```

This creates one row per:

`model configuration x anomaly signal x anomaly scoring method`

## HGQ2 Quantization Note

The current attention layer is a PyTorch custom `StandardMHA` in `ngt_model/src/ngt_model/models/mha_model.py`. HGQ2 `QMultiHeadAttention` is a Keras/HGQ2 layer, so it is not a drop-in replacement for this PyTorch module. The safe path is to keep the floating-point PyTorch baseline unchanged, use the separate `quantized_model/` selective INT8 baseline for the delivered checkpoint. A Keras/HGQ2 architecture port would require its own equivalence and synthesis study.

Reference: https://calad0i.github.io/HGQ2/autodoc/hgq.layers.attn.html#hgq.layers.attn.mha.QMultiHeadAttention
