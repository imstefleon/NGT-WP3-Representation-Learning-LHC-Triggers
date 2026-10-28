# L1-Only Event Representations for LHC Trigger Anomaly Detection: Architecture, Scaling, and MinBias-Referenced Scoring in `ngt_model`

**Type:** Read-only technical audit and scientific report
**Scope of audit:** `ctl2_embedding_model/` (workspace root), with primary focus on the `ngt_model/` package and its run outputs
**Audit performed:** read-only; no file in the audited project was created, edited, moved, renamed, or deleted
**Reference used for structure only:** Finol et al., *Contrastive Learning for Interpretable Anomaly Detection at Collider Experiments*, arXiv:2608.13652 — used only to shape section organization and level of methodological detail; no claims, numbers, or wording were copied from it, and none of its methodology (contrastive framework, specific architecture, results) is attributed to this project.

> **Scope note, read this first.** The repository contains **two separate, unrelated ML pipelines**:
> - **`ngt_model/`** — an active PyTorch project (multi-head-attention event encoder + classifier probe + a MinBias-referenced anomaly-scoring pipeline). This is the source of every run directory, CSV, and analysis note this report is built from, and is the subject of this report.
> - **`ctl2_model/`** — a separate Kedro-orchestrated Keras 3 / HGQ quantized **Linformer** pipeline targeting hls4ml/da4ml FPGA firmware synthesis, with its own, different anomaly-score implementations (`ClassifierAnomalyScore`, `KNNAnomalyScore`, `MahalanobisAnomalyScore`, `CombinedAnomalyScore` in `ctl2_model/src/ctl2_model/evaluation/anomaly_score.py`, verified present but not further audited here).
>
> These must not be conflated. Every architectural and numerical claim below is about `ngt_model/`. `ctl2_model/` is mentioned only where it is directly relevant (quantization outlook, Future Work).

---

## Output 1 — Repository Map

```
ctl2_embedding_model/
├── README.md                              Evidence: README.md
├── analysis/                              English analysis notes (prose, derived from run outputs) + curated figures
│   ├── *.md                               17 dated lab-notebook-style notes (see Output 2.7)
│   └── figures/{external_higgs_signal_eval, granular_latent_dim_sweep,
│                latent_dim_sweep_test, scaling_study,
│                superclass_loss_objective_comparison}/    curated PNG/SVG/CSV figures
├── ngt_model/                              ACTIVE SOURCE — subject of this report
│   ├── README.md                           package-level docs (partially stale, see §3 note)
│   ├── conf/{base,local}/                  Hydra/CLI-style config + gitignored MLflow credentials
│   └── src/ngt_model/
│       ├── datasets/collide2v_data.py      loading + preprocessing (Evidence for Output 4)
│       ├── models/mha_model.py             encoder + classifier head (Evidence for Output 3)
│       ├── training/train_mha.py           training entrypoint, losses, MLflow (Evidence for §Objective)
│       ├── evaluation/                     external_higgs_anomaly.py, latent_diagnostics.py,
│       │                                   latent_fisher.py, head_fisher.py, eval_quality*.py,
│       │                                   latent_dim_sweep.py, dataset_size_sweep.py, seed_sweep.py,
│       │                                   scale_test.py, summary_table.py
│       └── pipelines/                      scaling_study_manifest.py, scaling_study_collect_results.py
├── ngt_mha_nodes_aligned_outputs/          MHA run outputs (run_config.json, models/, eval_*.npy, figures/)
│   └── runs/YYYY-MM-DD/RUN_NAME/           incl. the 5-class and 10-class "SM superclass" objective study
├── ngt_linformer_nodes_aligned_outputs/    historical Linformer notebook baseline outputs (DY-MinBias era)
├── scaling_study/                          Maciej-directed class-diversity × sample-size/capacity study
│   ├── README.md                           study design doc (Evidence for Output 2)
│   ├── manifests/                          planned-run manifests (CSV/JSON/sh)
│   ├── summaries/                          consolidated result tables (Output 2, §Results)
│   ├── eval_logs/, probe_ablation/, granular_sm13/, priority2/, main_4class_scaling/
│   └── main_4class_scaling/4class/{50k,100k,150k,200k,500k,1m}/  the main scaling axis
├── handoff_anomaly_var01_cov0001_lat128_minbias/   curated single-run handoff package incl. `src_snapshot/`
├── presentations/                          5 PDF decks (project's own prior presentations)
├── diagnostic_notebooks/                   Jupyter notebooks (schema audits, DY final-state diagnostics)
├── ctl2_model/                             SEPARATE PROJECT — Kedro + Keras/HGQ + FPGA synthesis (not this report's subject)
└── (no top-level `outputs/` directory; no local `mlruns/` — MLflow tracking is remote only)
```

**Evidence:** `README.md`; direct directory listing of the workspace root and the paths above; `scaling_study/README.md`; `handoff_anomaly_var01_cov0001_lat128_minbias/README.md`.

---

## Output 2 — Experiment Inventory

### 2.1 Experiment families, in the order they occurred

| # | Family | Class set | HH4B status | Architecture | Evidence |
|---|---|---|---|---|---|
| A | Linformer notebook baseline | DY-MinBias era, various | n/a | Linformer notebook | `ngt_linformer_nodes_aligned_outputs/`, README §"Linformer notebook baseline preserved" |
| B | Early MHA architecture ablations | 5-class (`minbias,top,qcd,dy,hh4b`) | **trained** (5th class) | MHA, pooling/head/blocks/ff_dim/heads swept | README.md:21-23; `analysis/hh4b_latent_dim_sweep.md`, `analysis/hh4b_collapse_diagnostics.md` |
| C | Best-config + granular-label test | 5-class → Top/WW split into final states | **trained** | MHA, `CE+SupCon0.05+var` at latent_dim 32/64/128 | `analysis/best_config_finding_and_latent_sweep.md` |
| D | Six-objective comparison ("SM superclass") | 10-class (`minbias,qcd,dy,top,wjets,ww,wz,ttw,ttz,hh4b`) | **trained** | MHA, 6 CE-based objective variants | `analysis/superclass_loss_objective_comparison.md`; runs under `ngt_mha_nodes_aligned_outputs/runs/2026-08-04/` and `2026-08-06/` |
| E | External Higgs-like signal evaluation | 4-class vs. 10-class superclass models | HH4B trained in the 10-class models only; HH/VBFH/ggH always evaluation-only | MHA | `analysis/external_higgs_signal_evaluation.md` |
| F | **Scaling study** (Maciej-directed, current main line) | `4class`, `10plus_sm`, `granular_sm13` | **never trained** — HH4B/HH/VBFH/ggH are all injected evaluation-only probes | MHA, capacity/probe/sample-size swept | `scaling_study/README.md`, `analysis/scaling_axes_analysis.md`, `analysis/1m_500k_scaling_analysis.md` |
| — | (reference config, not a primary line) | `legacy_10class_hh4b_ref` (`minbias,qcd,dy,top,wjets,ww,wz,ttw,ttz,hh4b`) | **trained** | — | `scaling_study/README.md:24`, explicitly flagged: *"HH4B is supervised here, so it should not be used as the primary exotic-anomaly baseline."* |

**This HH4B-trained-vs-evaluation-only distinction is the single most important experimental-design fact in the repository** and the report treats it accordingly throughout: families B–D (and the `legacy_10class_hh4b_ref` config) trained HH4B as a supervised class; family F (the current scaling study, and the 4-class arm of family E) treats HH4B exactly like HH/VBFH/ggH — an injected, evaluation-only anomaly probe never seen during training.

### 2.2 Scaling-study class configurations (Axis 1: class diversity)

| config | classes | n_classes | note | Evidence |
|---|---|---:|---|---|
| `4class` | `minbias,qcd,dy,top` | 4 | current balanced SM baseline | `scaling_study/README.md:21` |
| `10plus_sm` | `minbias,qcd,dy,tt_hadronic,tt_leptonic,tt_semileptonic,wjets,ww,wz,ttw,ttz` | 11 | background-only; avoids training on any Higgs-like process | `scaling_study/README.md:22` |
| `granular_sm13` | `minbias,qcd,dy,tt_hadronic,tt_leptonic,tt_semileptonic,wjets,ww_hadronic,ww_leptonic,ww_semileptonic,wz,ttw,ttz` | 13 | most granular setup actually **run** | `analysis/scaling_axes_analysis.md:72-80` |
| `granular_sm15` (intended, not achieved) | adds `dy_ee,dy_mumu,dy_tautau` in place of `dy` | 15 | **failed**: those DY subchannels exist in the HuggingFace loader definitions but were not matched in the `collide2v_updated` EOS/parquet source | `analysis/scaling_axes_analysis.md:82`; `scaling_study/README.md:23,26` |

### 2.3 Scaling-study sample sizes and capacity (Axis 2), as actually executed

Confirmed present as real, non-empty run directories under `scaling_study/main_4class_scaling/4class/`: **50k, 100k, 150k, 200k, 500k, 1m**. (`1M` is written `1m` on disk.)

**Evidence:** direct listing of `scaling_study/main_4class_scaling/4class/` (six subdirectories: `50k/ 100k/ 150k/ 200k/ 500k/ 1m/`).

| samples/class | configs actually run (name → latent_dim/ff_dim/blocks) | outcome | Evidence |
|---:|---|---|---|
| 50k | `small` (64/128/2) | stable | `scaling_study/summaries/main_4class_scaling_minbias_summary.csv` |
| 100k | `medium` (128/256/2) | stable | same |
| 150k | `medium` (128/256/2); `medium_depth3` (128/256/3); `wide2blocks` (192/384/2); `large` (192/384/3) | `medium` and `medium_depth3` stable; `wide2blocks` and `large` **collapsed to chance accuracy** | `analysis/scaling_axes_analysis.md:135-148`; per-run `run_config.json` |
| 200k | `medium` (128/256/2); `ff512_blocks2` (128/512/2, aug on/off, both LRs); `medium_depth3` (128/256/3); `ff512_depth3` (128/512/3) | `medium` and `ff512_blocks2` stable; both `depth3` variants **collapsed** | same |
| 500k | `ff512_blocks2` (128/512/2, augmentation_copies=1) | **collapsed** to chance (train/val acc ≈0.25, loss ≈ln 4) | `analysis/1m_500k_scaling_analysis.md:25-62` |
| 1M | `medium` and `ff512_blocks2`, each with CE-only and CE+var/cov, all at `lr=3e-5, augmentation_copies=0` | all 4 stable; earlier default-LR / augmented 1M attempts collapsed | `analysis/1m_500k_scaling_analysis.md:64-105`; confirmed live via `run_config.json` of `.../1m/medium_ce_only_lr3e5_noaug/runs/2026-08-16/.../run_config.json` (`n_events_per_class=1000000, lr=3e-05, augmentation_copies=0`) |

### 2.4 Probe ablation (Axis 3)

| probe | architecture | tested at | Evidence |
|---|---|---|---|
| `linear` | `Linear(latent_dim, n_classes)` | 4class, 50k | `scaling_study/summaries/probe_ablation_summary.csv`; `mha_model.py` classifier dispatch |
| `mlp` | `Linear(latent,ff)→ReLU→Linear(ff,n_classes)` | 4class, 50k | same |
| `mlp2` | `Linear(latent,ff)→GELU→Linear(ff,ff)→GELU→Linear(ff,n_classes)` | 4class, 50k | same; formula also documented in `scaling_study/README.md:49` |

**Coverage gap, verified:** `scaling_study/summaries/master_results_missing.txt` lists 8 planned-but-never-run `mlp2` directories (4class and 10plus_sm × 50k/100k/150k/200k). The probe ablation is therefore **complete only at 4class/50k**; it was not repeated at other scales or for `10plus_sm`.

### 2.5 Objective / loss-function comparisons

| study | objectives compared | class set | Evidence |
|---|---|---|---|
| Best-config / granular-label sweep | `CE + SupCon0.05 + variance`, granular tt/ww/hh4b final-state labels, `latent_dim` 32/64/128 | 10-class, granular labels (HH4B trained) | `analysis/best_config_finding_and_latent_sweep.md`; confirmed run `granular_tt_ww_hh4b_best_config_100k_latent64_confirm` (`ngt_mha_nodes_aligned_outputs/runs/2026-08-04/`) |
| 10-class six-objective | CE+variance (baseline), +label-smoothing(0.05), +focal(γ=2), +SupCon(0.05), +SupCon(0.10), +covariance(0.001), +covariance(0.005) | 10-class superclass (HH4B trained) | `analysis/superclass_loss_objective_comparison.md`; run dirs under `ngt_mha_nodes_aligned_outputs/runs/2026-07-30/` and `2026-07-31/` |
| 1M scaling pilot | CE-only vs. CE+variance+covariance, at `medium` and `ff512_blocks2` capacity | `4class` (HH4B evaluation-only) | `analysis/1m_500k_scaling_analysis.md`; `scaling_study/summaries/main_4class_scaling_minbias_summary.csv` |

> **Correction after a second, independent evidence pass:** an earlier draft of this table described a "5-class best-config" study. A dedicated exhaustive search of every `analysis/*.md` file and both run trees found **no 5-class equivalent** of the objective/latent-dimension sweeps — `best_config_finding_and_latent_sweep.md` is itself a 10-class (granular-label) study. A "5-class HH4B benchmark" is referenced in prior project presentation material, but this audit could not independently re-confirm it against a source file; **any 5-class-specific number should be treated as unverified** until traced to a file directly. This is corrected here rather than left standing, per the audit's own traceability requirement.

**The six-objective runs have no `run_config.json` or `classification_report_eval.csv`** (both are absent from their run directories); their config fields and metrics below were instead recovered from `ngt_mha_nodes_aligned_outputs/scale_tests/unified_experiment_summary_complete_only.before_2026_08_06_update.csv`, where each row's `run_status` is explicitly marked `complete_from_analysis` with `missing_artifacts=run_config.json;classification_report_eval.csv` — i.e. the repository itself flags these as reconstructed, not original, records:

| run | objective | accuracy | macro F1 | eval-quality |
|---|---|---:|---:|---:|
| `sm_superclasses_labelsmooth005_var001_100k_latent64` | label smoothing 0.05 | 0.4747 | 0.4677 | 0.000039 |
| `sm_superclasses_focal2_var001_100k_latent64` | focal γ=2 | 0.4600 | 0.4543 | 0.1476 |
| `sm_superclasses_ce_supcon005_var001_100k_latent64` | CE + SupCon 0.05 | 0.4602 | 0.4560 | 0.2207 |
| `sm_superclasses_ce_supcon010_var001_100k_latent64` | CE + SupCon 0.10 | 0.4662 | 0.4580 | 0.1937 |
| `sm_superclasses_ce_var001_cov0001_100k_latent64` | CE + covariance 0.001 | 0.4556 | 0.4468 | 0.2294 |
| `sm_superclasses_ce_var001_cov0005_100k_latent64` | CE + covariance 0.005 | 0.4716 | 0.4644 | 0.1318 |

(all at `latent_dim=64, ff_dim=128, latent_var_reg_weight=0.01`). **Interpretation:** accuracy is similar across all six (0.46–0.47), but eval-quality varies by a factor of >50 (0.00004 to 0.229) — objective design changes latent-quality diagnostics far more than it changes classification accuracy, consistent with the capacity/probe findings in §5/§7.2 that classification accuracy and anomaly-relevant latent structure are only loosely coupled.

VICReg-style objective: **implemented in code** (`vicreg_loss()` in `train_mha.py`) and **exercised** in a follow-up run family with complete `run_config.json` (`sm_superclasses_vicreg001_var001_100k_latent{64,128}`, `ngt_mha_nodes_aligned_outputs/runs/2026-08-06/`), but was not part of the six-objective comparison itself and was not part of the later scaling study's objective comparison.

### 2.6 Evaluation-artifact completeness

**Exhaustively verified** (not sampled) across all **48** `run_config.json` directories in the repository (39 under `scaling_study/`, 9 under `ngt_mha_nodes_aligned_outputs/runs/`; zero under the Linformer-era `ngt_linformer_nodes_aligned_outputs/`, which predates `run_config.json` entirely):

- **`eval_quality_scores_n2000.json`-style quality diagnostics are missing from nearly the entire scaling-study run set.** Only 2 of the 12 probe-ablation runs have it (`scaling_4class_50k_small_{linear,mlp}_seed42`); it is absent from every `main_4class_scaling`, `priority2`, and `granular_sm13` run. By contrast, all 9 `ngt_mha_nodes_aligned_outputs` runs (the earlier SM-superclass family) have it, and two also have per-signal `eval_quality_external_{hh4b,hh,vbfh,ggh}_n2000.json` files. This is a genuine, verified completeness gap in the current scaling study, not a sampling artifact.
- `external_higgs_anomaly_metrics.csv`/`_event_scores.csv` are confirmed **missing** for at least: `1m/ff512_blocks2_ce_varcov`, `1m/ff512_blocks2_varcov_lr3e5_noaug` (metrics file only, scores present), `200k/ff512_depth3`, both `logs/4class/150k/large` seeds, `logs/4class/200k/xl`, `priority2/10plus_sm/.../scaling_10plus_sm_100k_medium_mlp2_seed42`, and `granular_tt_ww_hh4b_best_config_100k_latent64_confirm`.
- Two 1M runs are documented in `analysis/1m_500k_scaling_analysis.md` as collapsed/unstable (`ff512_blocks2_ce_varcov`: early-stopped epoch 26 at val_acc=0.25; `ff512_blocks2_ce_varcov_noaug`: reached val_acc=0.5657 at epoch 1, then collapsed to 0.25) but have **no** `failed_run_evidence/` folder — unlike the 500k `ff512_blocks2` run and one 1M `ff512_blocks2_ce_only_noaug` attempt, which *do* have an explicit `failed_run_evidence/README.md` self-documenting the collapse (train/val acc ≈0.25, loss ≈ln 4 ≈1.386, `var_reg≈0.99, cov_reg≈0`).
- **No successful 500k run exists anywhere in the repository** — the only 500k attempt (`main_4class_scaling/4class/500k/ff512_blocks2/`) collapsed; there is no successful-rerun directory for that sample size.
- The curated `scaling_study/summaries/main_4class_scaling_minbias_summary.csv` (18 rows, all `status=usable`, one row directly cross-checked against its source `run_config.json`+CSV and found to match exactly) **silently excludes** `150k/large`, `200k/xl`, and `200k/ff512_depth3` — three configurations that *do* have real run directories with `run_config.json` (the first two under `scaling_study/logs/4class/`) but are not documented as collapsed either. No file in the repository documents why these three are excluded from the summary table; this is flagged in Output 10 as evidence worth requesting directly from the project team rather than inferred.

### 2.7 Analysis notes (project's own chronological lab notebook)

**Evidence:** `analysis/README.md`.

| file | scope |
|---|---|
| `baseline_1k_100k.md` | Baseline 1k and 100k runs |
| `dy_minbias_followups.md` | Wide attention, augmentation, 300k MinBias follow-up |
| `top_pt_eta_fix.md` | Top-pT candidate-selection fix |
| `dy_minbias_latent_space.md` | DY final-state / latent-space separation |
| `dataset_diagnostics_and_runs.md` | Dataset-source diagnostics (HF/ROOT/Collide-2V) |
| `updated_collide2v_run.md` | Updated Collide-2V 100k safe baseline |
| `mha_eval_quality_analysis.md` | First MHA eval-quality latent score |
| `mha_l1_100k_eval_quality.md` | MHA L1-only 100k, latent_dim=128 |
| `hh4b_latent_dim_sweep.md` | HH4B full-SM latent-dim sweep |
| `hh4b_collapse_diagnostics.md` | PCA, effective rank, Fisher, seed/dataset-size sweeps, architecture ablations |
| `latent_collapse_report.md` | Consolidated latent-collapse diagnostics |
| `best_config_finding_and_latent_sweep.md` | Best-config search, granular Top/WW labels, latent-dim sweep |
| `superclass_loss_objective_comparison.md` | 10-class superclass, six-objective comparison |
| `external_higgs_signal_evaluation.md` | External Higgs-like evaluation (4-class vs 10-class) |
| `quantization_readiness_and_anomaly_scores.md` | Quantization outlook + anomaly-score notes |
| `scaling_axes_analysis.md` | Two-axis scaling study (dated 2026-08-14) |
| `1m_500k_scaling_analysis.md` | 500k/1M pilots |
| `scaling_analysis.md` | Consolidated scaling analysis (dated 2026-08-14, superseded in places by `scaling_axes_analysis.md` / `1m_500k_scaling_analysis.md`) |

---

## Output 3 — Model Architecture Summary

**Source of truth:** `ngt_model/src/ngt_model/models/mha_model.py` (verified directly, not inferred).

```
Event (raw, per candidate) → Input projection → [MHA block + residual] × num_attention_blocks
   → Pooling (mean | attention) → Post-pool dense stack → Latent projection (z)
   → Classifier head (linear | mlp | mlp2) → class logits
```

| Component | Implementation | Configurable? | Evidence |
|---|---|---|---|
| Input projection | `nn.Linear(n_features, ff_dim)` + ReLU | `ff_dim` | `mha_model.py:76,137` |
| Attention block | `StandardMHA`: separate Q/K/V/out `nn.Linear`, scaled dot-product, softmax, dropout — standard **full** attention, not low-rank | `num_heads` | `mha_model.py:25-52` |
| Depth | `num_attention_blocks` instances, each with residual `res = attention(res) + res` | yes (2 in all baseline runs; 3–4 tested and found unstable at ≥150k, see §Scaling) | `mha_model.py:77-80,142-143` |
| Pooling | mean (masked average) or attention (learned 2-layer scorer, masked softmax) | `pooling_mode ∈ {mean, attention}`; mean used in every run inspected | `mha_model.py:69,81-89,123-133` |
| Post-pool stack | `Linear(ff,ff)→ReLU→Linear(ff,ff)→ReLU` | fixed 2-layer | `mha_model.py:90-95` |
| Latent projection | `nn.Linear(ff_dim, latent_dim)` | `latent_dim` (32/64/128/192/256 tested) | `mha_model.py:96` |
| Classifier head | `linear`: `Linear(latent_dim,n_classes)`. `mlp`: `Linear→ReLU→Linear`. `mlp2`: `Linear→GELU→Linear→GELU→Linear` | `classifier_head ∈ {linear, mlp, mlp2}` | `mha_model.py:97-121` |
| Output | `{'latent': z, 'logits': logits}`; a `LatentOnlyWrapper` exposes just `z` for evaluation checkpoints | — | `mha_model.py:135-164` |

### Example verified instantiation (10-class superclass, `latent128` handoff run)

| field | value | Evidence |
|---|---|---|
| Attention blocks | 2 | `handoff_anomaly_var01_cov0001_lat128_minbias/README.md` |
| Heads | 8 | same |
| `ff_dim` | 128 | same |
| Pooling | mean | same |
| Classifier | linear, `Linear(128,10)` | same |
| `latent_dim` | 128 | same |
| Parameters | **186,122** | same |
| Raw input tensor | `[batch, 100, 5]` = pt, eta, phi, dxy, pid | same |
| Preprocessed model input | `[batch, 100, 24]` = 4 normalized kinematic features + 20 PID one-hot channels | same |

### Configurations spanning the scaling study

| samples/class | model label | `latent_dim` | `ff_dim` | blocks | heads | probe | Evidence |
|---:|---|---:|---:|---:|---:|---|---|
| 50k | small | 64 | 128 | 2 | 8 | linear | `scaling_study/README.md:32`; confirmed in `run_config.json` |
| 100k | medium | 128 | 256 | 2 | 8 | linear | same |
| 150k | medium | 128 | 256 | 2 | 8 | linear | `analysis/scaling_axes_analysis.md:139` |
| 150k | medium_depth3 | 128 | 256 | **3** | 8 | linear | same |
| 150k | wide2blocks | **192** | **384** | 2 | 8 | linear | same — **collapsed** |
| 150k | large | 192 | 384 | 3 | 8 | linear | same — **collapsed** |
| 200k | medium | 128 | 256 | 2 | 8 | linear | same |
| 200k | ff512_blocks2 | 128 | **512** | 2 | 8 | linear | same |
| 200k | medium_depth3 / ff512_depth3 | 128 | 256/512 | 3 | 8 | linear | same — **both collapsed** |
| 1M | medium / ff512_blocks2 | 128 | 256 / 512 | 2 | 8 | linear | `analysis/1m_500k_scaling_analysis.md` |

**Optimizer / schedule:** `torch.optim.AdamW`, default `lr=1e-3, weight_decay=1e-4`; **no LR scheduler** is used anywhere in `train_mha.py`; the stable 1M recipe instead manually sets `lr=3e-5`. Early stopping on validation loss, default patience 10. Default `epochs=50` (some earlier runs used `epochs=100` with the same patience=10, e.g. the handoff run stopped at epoch 69/100). `batch_size=256` default. Seeded (`seed + rank`) across `random`/`numpy`/`torch`/`torch.cuda`; DDP/`torchrun` supported. **Evidence:** `train_mha.py:150-181,617,736,891-904`.

**Not verifiable from the repository:** wall-clock training time per configuration; GPU hardware used per run (beyond what is visible in the earlier Kubeflow-notebook screenshot context, which is outside this audit's file scope); whether any hyperparameter search (beyond the manually-designed ablations documented above) was performed.

---

## Output 4 — Dataset / Class Summary

**Source of truth:** `ngt_model/src/ngt_model/datasets/collide2v_data.py`.

| Aspect | Value | Evidence |
|---|---|---|
| Data source | `collide2v_updated` (parquet under an EOS production path); loader also supports `samples_json` (ROOT/uproot) and `hf` (HuggingFace parquet), not used for the main scaling study | `collide2v_data.py:85-116` |
| Raw features read | `L1T_PUPPIPart_{PT,Eta,Phi,D0,PID}` | `collide2v_data.py:118-124` |
| Feature order in tensor | `[pt, eta, phi, dxy(=D0), pid]` | `collide2v_data.py:184-207` |
| Candidate selection | top-`pT`, via `argsort(-pt_all)` after filtering `pt>0` and finite | `collide2v_data.py:196-199` |
| Event tensor shape (raw) | `100 × 5`, zero-padded | same |
| Normalization | `pt → log1p → min-max`; `eta → min-max`; `phi → (phi+π)/2π`; `dxy` left raw; padded rows zeroed | `collide2v_data.py:579-609` |
| PID encoding | mapping built from **training data only**, then one-hot appended (→ `4 + n_pid_classes` total input features, e.g. 24 in the verified example) | `collide2v_data.py:612-648` |
| Split | stratified per class, seed-controlled, ~70/15/15 train/val/eval by default | `collide2v_data.py:669-711` |
| Augmentation | optional random global φ rotation, `--augmentation-copies N` | `collide2v_data.py:651-662`; gated via `train_mha.py` |

### Supported class names (superset)

`minbias, top, tt_hadronic, tt_leptonic, tt_semileptonic, qcd, dy, dy_ee, dy_mumu, dy_tautau, wjets, ww (+ subtypes), wz, ttw, ttz, hh4b, hh, vbfh, ggh` — **Evidence:** `collide2v_data.py:30-54` (`SUPPORTED_CLASS_NAMES`/`LABEL_MAP`). Every named class configuration (`4class`, `10plus_sm`, `granular_sm13`, `legacy_10class_hh4b_ref`, the 5-class and 10-class superclass sets) is simply a different `--class-names` selection over this one list — there is no separate per-configuration loader code path.

### Training classes vs. evaluation-only anomaly probes (current/main scaling study, family F)

| role | processes | Evidence |
|---|---|---|
| **Trained** (background, `4class`) | `minbias, qcd, dy, top` | `scaling_study/README.md:21` |
| **Trained** (background, `10plus_sm`) | `minbias, qcd, dy, tt_hadronic, tt_leptonic, tt_semileptonic, wjets, ww, wz, ttw, ttz` | `scaling_study/README.md:22` |
| **Trained** (background, `granular_sm13`) | `minbias, qcd, dy, tt_hadronic, tt_leptonic, tt_semileptonic, wjets, ww_hadronic, ww_leptonic, ww_semileptonic, wz, ttw, ttz` | `analysis/scaling_axes_analysis.md:72-80` |
| **Evaluation-only, injected** (all three above) | `hh4b, hh, vbfh, ggh` | `scaling_study/README.md:22`; `analysis/scaling_axes_analysis.md:12-18` |

In the earlier `legacy_10class_hh4b_ref` / six-objective / 5-class families, **HH4B was instead a supervised training class**; HH/VBFH/ggH remained evaluation-only in all families where they appear at all. **Evidence:** `scaling_study/README.md:24`; `analysis/superclass_loss_objective_comparison.md`.

---

## Output 5 — Metric Inventory

**Source of truth:** `ngt_model/src/ngt_model/evaluation/external_higgs_anomaly.py` and `latent_diagnostics.py` / `latent_fisher.py` / `head_fisher.py`.

| Metric | Definition (verified in code) | File |
|---|---|---|
| `roc_auc_vs_sm_background` | `sklearn.metrics.roc_auc_score` on `y_true=[0]*n_bg+[1]*n_signal`, `y_score` = the chosen anomaly-score column, background = held-out `background_eval` split | `external_higgs_anomaly.py:594-596` |
| `background_p95_threshold` / `background_p99_threshold` | 95th / 99th percentile of the **background_eval-split** anomaly-score distribution — i.e. the score value below which 95% / 99% of held-out background falls | `external_higgs_anomaly.py:481-482,563-564` |
| `eff_at_95pct_bg` / `eff_at_99pct_bg` (and the by-method file's `eff_at_95_bg_rejection` / `eff_at_99_bg_rejection` — **same definition, different column name**, minor naming inconsistency between the two output files) | `mean(signal_scores > background_threshold)` — the fraction of a given signal's events that exceed the background threshold | `external_higgs_anomaly.py:631-632` |
| **Confirms:** "95% background rejection" = the operating threshold at the background's own 95th percentile, i.e. 95% of background is rejected (5% background acceptance) at that cut; `eff@95%BG` reports what fraction of *signal* survives the same cut. | | |
| `minbias_distance_score` | Mahalanobis distance (Ledoit-Wolf shrinkage covariance, PCA-whitened latent) to a mean/covariance fit on **MinBias-only** background-fit events | `external_higgs_anomaly.py:418-457,531,552` |
| `gmm_anomaly_score` | `-log p(z)` from a 2-component full-covariance `GaussianMixture` fit on the combined background-fit split | `external_higgs_anomaly.py:51-58,529` |
| `overall_anomaly_score` | `0.5·robust_z(gmm_anomaly_score; background_eval) + 0.5·robust_z(minbias_distance_score; background_eval)`, `robust_z=(x-median)/IQR` | `external_higgs_anomaly.py:460-469,574-576` |
| `effective_rank_entropy` | Entropy-based effective rank of the latent covariance eigen-spectrum, `exp(-Σp·log p)` | `latent_diagnostics.py:21-26,33-79` |
| `participation_ratio` | `(Σλ)² / Σλ²` | `latent_diagnostics.py:57` |
| `n_components_for_{50,80,90,95,99}pct` | # principal components needed for that cumulative explained variance | `latent_diagnostics.py:33-79` |
| Latent Fisher | empirical Fisher diagonal per latent dim, `F_j = mean((∂loss/∂z_j)²)` | `latent_fisher.py:8-13` |
| Head Fisher | same, over classifier-head parameters, with selectable scope (`classifier_head`/`latent_and_classifier`/`all_trainable`) | `head_fisher.py:9-14,44-49` |
| Classifier response | `classifier_top_predicted_class`, `classifier_top_predicted_fraction`, `fraction_predicted_<class>` per external signal | `external_higgs_anomaly.py` (classifier-response export); confirmed columns in `external_higgs_anomaly_metrics.csv` |
| Training metrics | per-epoch train/val loss & accuracy, plus per-term breakdown (`var_reg`, `cov_reg`, SupCon/VICReg terms) — logged to `training_history.csv` and to MLflow | `train_mha.py:863-889` |

---

## Output 6 — Existing Figure Inventory

Grouped by what each answers; all pre-existing (none generated for this audit).

| Figure(s) | Directory | Answers |
|---|---|---|
| `4class_best_val_loss_vs_scale*.png/pdf`, `4class_loss_vs_scale_summary.csv` | `analysis/figures/scaling_study/` | How does best validation loss change from 50k to 1M events/class? |
| `class_granularity_minbias_auc_summary.png/csv` | same | Does class diversity (`4class`/`10plus_sm`/`granular_sm13`) change mean MinBias-distance AUC? |
| `hh4b_efficiency_minbias_rejection_summary.csv`, `hh4b_granularity_updated_4class_noaug_eff95.png`, `hh4b_eff95_noaug_all_scored_runs_scatter.png`, `hh4b_efficiency_noaug_selected_scaling.png`, `hh4b_efficiency_granularity_and_input_size.png` | same | HH4B signal efficiency at fixed background rejection, across scale/granularity |
| `latent_dim_effective_rank_barplot.png`, `latent_dim_hh4b_f1_barplot.png`, `balanced_supcon_latent_dim_effective_rank_pr.png`, `latent_dim_sweep_presentation_summary.png` | `analysis/figures/latent_dim_sweep_test/` | Does nominal `latent_dim` (16/32/64/128) change effective rank / HH4B F1? (5-class best-config study) |
| `effective_rank_vs_latent_dim_granular_labels.png/csv` | `analysis/figures/granular_latent_dim_sweep/` | Does label granularity (splitting Top/WW) raise effective latent dimensionality? |
| `loss_objective_accuracy_vs_eval_quality.png`, `loss_objective_latent_head_fisher_all6.png`, `loss_objective_summary_metrics.png`, per-objective `*_latent_pca.png` / `*_training_curves.png` | `analysis/figures/superclass_loss_objective_comparison/` | Six-objective comparison: accuracy vs. eval-quality trade-off; latent/head-Fisher dimensionality per objective |
| `external_higgs_eval_quality_score.png`, `external_higgs_eval_quality_subscores_heatmap.png`, `external_higgs_gmm_gap_log.png`, `external_higgs_mahalanobis.png`, `external_higgs_participation_ratio.png` | `analysis/figures/external_higgs_signal_eval/` | 4-class vs. 10-class superclass comparison on external Higgs-like probes |
| `minbias_distance_distributions.png`, `minbias_distance_distributions_per_signal.png`, `minbias_vs_anomalies_latent_tsne.png`, `minbias_vs_anomalies_latent_pca.png`, `external_higgs_auc.png`, `signal_efficiency_fixed_background.png`, `external_higgs_classifier_response.png` (`classifier_response_matrix.png`), `latent_effective_dim_spectrum.png` | per-run `figures/external_higgs_anomaly/` and `figures/external_classifier_response/` under every scaling-study run directory | Per-run: MinBias-distance histogram, qualitative t-SNE/PCA separation, AUC bar, fixed-BG efficiency, classifier response, latent spectrum |
| Mermaid pipeline diagram | `README.md:89-100` | Architecture flow (matches `mha_model.py` directly, verified) |

PCA and t-SNE figures are treated throughout the repository's own documentation as **qualitative diagnostics only** (`scaling_study/README.md:86-91`): "PCA is the quantitative collapse diagnostic... t-SNE is the qualitative clustering diagnostic." This report follows that same distinction and does not treat visual cluster separation in t-SNE plots as a quantitative anomaly-detection result.

---

## Output 7 — Research-Question → Experiment → Metric Mapping

| # | Question | Experiment(s) | Metric(s) | Evidence |
|---|---|---|---|---|
| RQ1 | Does more training statistics improve anomaly sensitivity? | Axis 2, 50k→1M at matched (or best-available) capacity | mean MinBias-distance AUC, eff@95/99%BG, best val loss | `scaling_study/summaries/main_4class_scaling_minbias_summary.csv`; `analysis/figures/scaling_study/4class_loss_vs_scale_summary.csv` |
| RQ2 | Does more encoder capacity improve the representation / anomaly separation? | Axis 2 capacity sweep (depth 2→3→4, `ff_dim` 128→512, `latent_dim` 64→256) | same + effective rank, training stability (collapse/no-collapse) | `analysis/scaling_axes_analysis.md`; `analysis/1m_500k_scaling_analysis.md` |
| RQ3 | Does more SM class diversity/granularity improve anomaly detection? | `4class` vs `10plus_sm` vs `granular_sm13`, matched at 50k/100k | mean MinBias-distance AUC, eff@95/99%BG | `analysis/figures/scaling_study/class_granularity_minbias_auc_summary.csv` |
| RQ4 | Does probe complexity (linear/MLP/MLP2) affect latent organization and/or anomaly separation? | Probe ablation, 4class/50k | effective rank, participation ratio, PCA dims@95%, mean AUC/eff per method | `scaling_study/summaries/probe_ablation_summary.csv` |
| RQ5 | How well do Higgs-like processes separate from the learned SM representation? | Every scaling-study run's external evaluation | per-signal AUC, eff@95/99%BG, classifier response | per-run `external_higgs_anomaly_metrics*.csv` |
| RQ6 | Is MinBias-referenced distance a useful anomaly score, and how does it compare to GMM/combined scores? | Every run's 3-method comparison | AUC/eff per method (`gmm_nll`, `minbias_mahalanobis`, `overall_robust_mean`) | per-run `external_higgs_anomaly_metrics_by_method.csv` |
| RQ7 | Does the representation collapse toward the number of supervised classes, and is that architecture- or objective-driven? | Latent-dim sweep, label-granularity test, six-objective comparison | effective rank, participation ratio, PCA dims, HH4B F1 | `analysis/hh4b_collapse_diagnostics.md`, `analysis/best_config_finding_and_latent_sweep.md`, `analysis/superclass_loss_objective_comparison.md` |

---

# Output 8 — Scientific Report

## Abstract

We audit `ngt_model`, a supervised PyTorch pipeline that learns a fixed-size latent representation of simulated LHC Level-1-trigger-level events (`100` L1 PUPPI candidates × 5 kinematic/PID features) using a multi-head-attention encoder, and evaluates whether that representation supports anomaly detection against injected Higgs-like processes (HH4B, HH, VBFH, ggH). Anomaly scoring is computed in the encoder's latent space relative to a MinBias reference distribution, via three verified methods: Mahalanobis distance to a MinBias-only fit, a background-fit Gaussian-mixture density, and a robust-z-score combination of the two. Across a documented experimental history, the project first trained HH4B as a supervised label (5-class and 10-class "SM superclass" studies) to study latent-space collapse under different pooling/head/objective choices, then moved to a design in which HH4B, HH, VBFH, and ggH are never trained labels — only injected evaluation-only probes — while the background class set (`4class`, `10plus_sm`, `granular_sm13`) and training scale/capacity (50k to 1M events/class; encoder depth, width, and latent dimension) are varied independently. The two central, code- and data-verified findings are: (1) increasing Standard-Model class diversity/granularity does **not** improve mean MinBias-distance AUC against the four injected probes (`4class` 0.78 vs. `10plus_sm` 0.57–0.59 vs. `granular_sm13` 0.55–0.56 at matched 50k/100k); and (2) capacity scaling is **selective and stability-limited** rather than monotonic — deepening the encoder (3–4 attention blocks) collapses training at 150k–200k events/class, while widening only the feed-forward dimension (256→512) at fixed depth and latent size is stable and improves the metric, and a stable 1M-events/class recipe (low learning rate, no augmentation) reaches the highest mean AUC observed (0.84–0.85). We report these as scale-specific, probe-specific findings, not universal claims, and identify concrete gaps — no repeated seeds, no independent test-set holdout beyond the internal eval split, coarse-label confounds in the `4class` result, and no quantization/online-latency validation — that should be closed before any deployment-oriented conclusion is drawn.

## 1. Introduction

Level-1 (L1) hardware triggers at the LHC must decide, within severe latency budgets and using only coarse L1-level information, which collision events are worth keeping. A learned event representation that (a) organizes Standard Model (SM) processes coherently and (b) leaves room for events unlike anything trained on to look "far away" is potentially useful both for physics clustering and for flagging rare, unmodeled event topologies — including possible beyond-Standard-Model (BSM) signatures — without requiring those signatures to be trained on directly.

`ngt_model` studies this question concretely: train a supervised multi-head-attention encoder on SM background processes reconstructed from L1-only candidate information, then ask whether the resulting latent space separates a set of injected Higgs-like processes (treated as BSM-like anomaly probes, never trained on in the current main study) from a MinBias reference distribution. The project's own documented central research questions, carried over unchanged from `README.md` and cited here as the anchor for the rest of this report, are:

a. Which learning process yields the optimal representation of LHC collider data?
b. How does representation quality evolve with model and dataset size?
c. Can scaling laws quantify anomaly-detection capability with scale?

**Evidence:** these three questions are the project's own framing, verified in presentation material derived from the same analysis notes audited here (`presentations/Moreno-Estefania-OpenLab.pdf`, slide "Motivation & Research Questions"); the underlying evidence for how they are actually being answered is the run data audited in Outputs 1–7 above.

## 2. Dataset and Event Representation

See Output 4 for full detail; summarized here.

Each event is represented as a fixed-size `100 × 5` tensor of top-`pT`-selected L1 PUPPI candidates (`pt, eta, phi, dxy, pid`), zero-padded, drawn from the `collide2v_updated` parquet production (**Evidence:** `collide2v_data.py`). After normalization (log1p+min-max for `pt`, min-max for `eta`, wrapped-and-scaled for `phi`) and one-hot PID encoding (vocabulary built from training data only), the model-facing input has `4 + n_pid_classes` features per candidate (24 in the one fully-documented example, **Evidence:** `handoff_anomaly_var01_cov0001_lat128_minbias/README.md`). Gen-level and full-reconstruction information are explicitly excluded from the model input (**Evidence:** `README.md:129`).

Two distinct background class sets are in current use for the main scaling study — `4class` (minbias, qcd, dy, top) and `10plus_sm` (11 classes, adds `tt_*`/`wjets`/`ww`/`wz`/`ttw`/`ttz`), plus a `granular_sm13` cross-check that further splits `tt_*` and `ww_*` final states — none of which include HH4B, HH, VBFH, or ggH as trained labels (**Evidence:** `scaling_study/README.md`). This is a deliberate change from the project's earlier phase, in which HH4B was trained alongside 4 or 9 other SM classes (**Evidence:** family B–D in Output 2.1); the current framing treats all four Higgs-like processes symmetrically, as evaluation-only injected probes.

## 3. Representation Learning Framework

```
Event (100×5 raw) → preprocess (100×24) → Input projection (Linear+ReLU)
   → [MHA block (8 heads, full attention) + residual] × {2,3,4}
   → Pool (mean, verified default) → Post-pool dense (Linear+ReLU ×2)
   → Latent z (dim 64/128/192/256) → Probe (linear / MLP / MLP2) → class logits
                                    ↘ Anomaly scoring (§6), independent of the probe
```

**Evidence:** `mha_model.py` (Output 3). The encoder and the anomaly-scoring pipeline share only the latent vector `z` — the classifier probe's job is purely to produce a supervised training signal; anomaly scores are computed directly on `z`, not on the probe's output, in every score method verified (§6). The total training loss is a weighted sum of cross-entropy (or a label-smoothed/focal variant) plus zero or more of: latent variance regularization, latent covariance regularization, supervised contrastive (SupCon) loss, and a VICReg-style two-view loss — all four extra terms confirmed implemented in `train_mha.py` and gated by independent weight hyperparameters that default to 0 (**Evidence:** §Output 3 objective table; `train_mha.py:241-329,426-432`).

## 4. Experimental Design

### 4.1 Training-statistics scaling

- **Independent variable:** events/class ∈ {50k, 100k, 150k, 200k, 500k, 1M}.
- **Held fixed (per scaling step, per Output 2.3 table):** class set = `4class`; probe = `linear`; architecture mostly held at `medium` (128/256/2) except where capacity was also varied (see 4.2).
- **Metric:** mean MinBias-distance AUC, eff@95/99%BG, best validation loss.
- **Question:** does more background statistics, at otherwise-fixed capacity, improve the anomaly metric?

### 4.2 Model-capacity scaling

- **Independent variables:** `ff_dim` (128→512), attention depth (2→4 blocks), `latent_dim` (64→256), varied both individually and jointly.
- **Held fixed:** class set = `4class`; probe = `linear`.
- **Metric:** same as 4.1, plus training-stability outcome (converged vs. collapsed to chance accuracy) and effective latent rank.
- **Question:** does more capacity improve the representation, and is scaling capacity safe to do naively?

### 4.3 Standard Model class-diversity scaling

- **Independent variable:** class set ∈ {`4class`, `10plus_sm`, `granular_sm13`}.
- **Held fixed:** samples/class ∈ {50k, 100k} (matched); probe = `linear`.
- **Metric:** mean MinBias-distance AUC, eff@95/99%BG.
- **Question:** does a richer/more granular SM background improve separation of the same four injected probes?

### 4.4 Probe complexity

- **Independent variable:** classifier head ∈ {`linear`, `mlp`, `mlp2`}.
- **Held fixed:** class set = `4class`; samples/class = 50k; encoder architecture = `small`.
- **Metric:** eval accuracy/macro-F1, effective rank, participation ratio, PCA-dims@95%, and per-method mean AUC/eff.
- **Question:** does the probe's own capacity reorganize the latent space, and does that reorganization help or hurt anomaly separation?

### 4.5 Input granularity scaling

Not found as an independently-varied axis distinct from 4.3 in the current scaling study — the granularity axis in this repository *is* the class-diversity axis (splitting SM processes into more final states), not a change to the L1 input representation itself (candidate count, feature set). **Not verifiable as a separate experiment from the repository**; treated as identical to 4.3 in this report.

## 5. Latent-Space Analysis

PCA-based diagnostics (effective rank, participation ratio, PCs needed for 90/95/99% variance) and t-SNE are both computed per run, but the project's own documentation is explicit that only the PCA-based quantities are treated as the quantitative collapse diagnostic; t-SNE is qualitative (**Evidence:** `scaling_study/README.md:86-91`). This report follows the same convention.

Quantitative evidence of latent compression, and of its partial decoupling from anomaly separation:

- At `4class`/50k, effective rank rises monotonically with probe complexity (`linear` 9.49 → `mlp` 10.56 → `mlp2` 13.81) while mean MinBias-distance AUC *falls* (0.784 → 0.760 → 0.747) (**Evidence:** `scaling_study/summaries/probe_ablation_summary.csv`).
- At 200k, `ff512_blocks2` has a *lower* effective rank than `medium` (3.85 vs. 8.94) but a *higher* mean AUC (0.787 vs. 0.753) (**Evidence:** `scaling_study/summaries/main_4class_scaling_minbias_summary.csv`; `analysis/scaling_axes_analysis.md:267-291`).
- At 1M, `ff512_blocks2 + CE+var/cov` has by far the largest effective rank among the four stable 1M runs (6.33 vs. 1.77–3.66) and also the highest mean AUC (0.8445), but by a small margin over `medium CE-only` (0.8425) (**Evidence:** `analysis/1m_500k_scaling_analysis.md:198-213`).

**Interpretation (hypothesis, not established fact):** effective latent dimensionality is a genuine diagnostic of how much of the nominal `latent_dim` is actively used, and it is demonstrably *not* the same quantity as MinBias-vs-anomaly separation; the two move together in some configurations (1M) and inversely in others (50k probe ablation, 200k `ff512` vs. `medium`). No general rule for when they align is established by the current evidence.

## 6. Anomaly Detection

Reference distribution: **MinBias**, treated as the in-distribution/background reference throughout the current scaling study (**Evidence:** `analysis/scaling_axes_analysis.md:22-24`, `scaling_study/README.md`).

Three scoring methods, all computed on PCA-whitened latents fit only on a `background_fit` split, with thresholds computed only on a disjoint `background_eval` split (**Evidence:** `external_higgs_anomaly.py`, §Output 5):

1. **`minbias_mahalanobis`** — Mahalanobis distance (Ledoit-Wolf shrinkage covariance) to a mean/covariance fit on MinBias-only background-fit events. This is the score referred to informally throughout the project's own notes as "distance from MinBias reference."
2. **`gmm_nll`** — negative log-likelihood under a 2-component full-covariance Gaussian mixture fit on the combined background-fit split (`-log p_GMM(z)`).
3. **`overall_robust_mean`** — `0.5·robust_z(gmm_nll) + 0.5·robust_z(minbias_mahalanobis)`, both robust-z'd against the held-out background-eval split.

External signals HH4B, HH, VBFH, ggH are scored with all three methods against the same background_eval threshold, at `n=2000` events per signal in every run inspected. Median/mean/p90/p95 of each score, AUC vs. background, and efficiency at the 95th/99th background percentile are reported per signal, per method, per run (**Evidence:** `external_higgs_anomaly_metrics.csv` / `_by_method.csv` schema, verified directly against a live file for the 1M `medium_ce_only_lr3e5_noaug` run).

No method dominates uniformly. In the one run compared in detail here (1M `medium_ce_only_lr3e5_noaug`), `minbias_mahalanobis` gives the highest mean AUC (0.897 for HH4B) versus `gmm_nll` (0.879) and `overall_robust_mean` (0.887) for the same signal (**Evidence:** direct read of that run's `external_higgs_anomaly_metrics_by_method.csv`).

## 7. Results

### 7.1 Class-diversity axis (RQ3)

**Observation:** at both 50k and 100k events/class, `4class` gives substantially higher mean MinBias-distance AUC than either `10plus_sm` or `granular_sm13`.

**Quantitative evidence** (**Evidence:** `analysis/figures/scaling_study/class_granularity_minbias_auc_summary.csv`):

| class config | 50k mean AUC | 100k mean AUC | 50k eff@95%BG | 100k eff@95%BG |
|---|---:|---:|---:|---:|
| `4class` | 0.7836 | 0.7830 | 0.1716 | 0.1666 |
| `10plus_sm` | 0.5890 | 0.5744 | 0.0480 | 0.0377 |
| `granular_sm13` | 0.5606 | 0.5538 | 0.0436 | 0.0389 |

**Interpretation:** more class diversity/granularity does not improve — and in this comparison, clearly worsens — the mean MinBias-distance AUC against the four current injected probes.

**Limitation:** the classifier-response analysis for `4class` shows HH4B/HH mapped mostly to `top` (~0.91–0.94 top-fraction at 1M scale) (**Evidence:** `analysis/1m_500k_scaling_analysis.md:215-223`), i.e. injected Higgs-like events are pushed toward one coarse non-MinBias class. The higher `4class` AUC may therefore partly reflect coarse-label geometry rather than a more physically meaningful representation; this caveat is stated explicitly in the source analysis notes themselves (`analysis/scaling_axes_analysis.md:105-111`) and is preserved here rather than resolved, since resolving it would require an experiment (e.g. a matched coarse-vs-fine comparison controlling for label geometry) not present in the repository.

### 7.2 Probe-complexity axis (RQ4)

**Observation:** classification accuracy is essentially flat across probes (0.665–0.668), but mean anomaly AUC decreases monotonically as probe capacity increases.

**Quantitative evidence** (**Evidence:** `scaling_study/summaries/probe_ablation_summary.csv`):

| probe | eval accuracy | effective rank | mean AUC (`minbias_mahalanobis`) | eff@95%BG |
|---|---:|---:|---:|---:|
| linear | 0.6653 | 9.49 | 0.7836 | 0.1716 |
| mlp | 0.6683 | 10.56 | 0.7603 | 0.1893 |
| mlp2 | 0.6671 | 13.81 | 0.7471 | 0.1530 |

**Interpretation:** a more expressive probe reduces low-dimensional latent concentration (higher effective rank) without improving — and here, worsening — classification-independent anomaly separation. This is consistent with the hypothesis (stated in the project's own notes, `scaling_study/README.md:130`) that the linear-probe pressure shapes the latent directly, but the ablation only covers `4class`/50k (§Output 2.4); it is not established at other scales.

### 7.3 Capacity-scaling axis (RQ2)

**Observation:** capacity scaling is selective. Increasing `ff_dim` alone (depth and `latent_dim` fixed) is stable and helps; increasing depth to 3 blocks destabilizes at ≥150k–200k; increasing `latent_dim`+`ff_dim` together (`wide2blocks`) destabilizes at 150k.

**Quantitative evidence** (**Evidence:** `analysis/scaling_axes_analysis.md`; `scaling_study/summaries/main_4class_scaling_minbias_summary.csv`):

| run | depth | `ff_dim` | outcome | mean AUC |
|---|---:|---:|---|---:|
| 150k medium | 2 | 256 | stable | 0.768 |
| 150k medium_depth3 | 3 | 256 | stable, lower classifier quality | 0.796 |
| 150k wide2blocks | 2 | 384 (+`latent_dim`=192) | **collapsed** (final val_acc = 0.25) | n/a |
| 150k large | 3 | 384 (+`latent_dim`=192) | **collapsed** | n/a |
| 200k medium | 2 | 256 | stable | 0.753 |
| 200k ff512_blocks2 | 2 | 512 | stable, best 200k point | 0.787 |
| 200k medium_depth3 / ff512_depth3 | 3 | 256 / 512 | **both collapsed** | n/a |

**Interpretation:** the collapse pattern implicates the *combination* of extra depth and/or wider latent/FF dimensions with the current optimization recipe (default `lr=1e-3`, no scheduler), not sample size — the 150k `wide2blocks` run reaches non-trivial accuracy in early epochs before collapsing (**Evidence:** `analysis/scaling_axes_analysis.md:166-176`). This is a training-stability finding, not evidence that more capacity is intrinsically worse for the physics task.

### 7.4 Training-statistics scaling and the 1M stable recipe (RQ1)

**Observation:** naive scaling to 500k/1M with the previous 200k recipe and default LR is unstable or fails outright (memory limits, then optimization collapse). A lower learning rate (`3e-5`) with augmentation disabled stabilizes training at 1M for both `medium` and `ff512_blocks2`, reaching the highest mean AUC values observed anywhere in the audited evidence.

**Quantitative evidence** (**Evidence:** `analysis/1m_500k_scaling_analysis.md:198-213`; cross-checked directly against `scaling_4class_1m_medium_ce_only_lr3e5_noaug_seed42/figures/external_higgs_anomaly/external_higgs_anomaly_metrics_by_method.csv`):

| run | `ff_dim` | objective | mean AUC | eff@95%BG | eff@99%BG |
|---|---:|---|---:|---:|---:|
| 1M medium, CE-only | 256 | CE | 0.8425 | 0.3890 | 0.1797 |
| 1M medium, CE+var/cov | 256 | CE+var/cov | 0.8408 | 0.3872 | 0.2240 |
| 1M ff512, CE-only | 512 | CE | 0.8421 | 0.3590 | 0.1716 |
| 1M ff512, CE+var/cov | 512 | CE+var/cov | 0.8445 | 0.3705 | 0.1736 |

Per-signal, the 1M `medium CE-only` run reaches AUC 0.897 (HH4B), 0.889 (HH), 0.831 (VBFH), 0.753 (ggH) (**Evidence:** direct CSV read, cited in §6).

**Interpretation:** the largest available training set (1M/class), combined with a stability fix (low LR, no augmentation) rather than capacity alone, gives the best MinBias-vs-anomaly separation seen in this audit. Whether this is attributable to sample size, the LR/augmentation fix, or their interaction cannot be disentangled from the current evidence, because the unstable higher-LR/augmented 500k and 1M attempts never produced comparable valid metrics (**Evidence:** `analysis/1m_500k_scaling_analysis.md:64-74`).

## 8. Scaling Behaviour (synthesis across §7)

| Axis | Does scaling it improve MinBias-vs-anomaly separation? | Evidence |
|---|---|---|
| Training statistics (events/class, at fixed/best-available capacity) | **Not cleanly separable from stability fixes**; best result (1M, AUC≈0.84) required a different LR/augmentation recipe, not statistics alone | §7.4 |
| Encoder capacity | **Selective, not monotonic.** Depth beyond 2 blocks is unstable at ≥150k; `ff_dim` alone (256→512) helps when stable | §7.3 |
| SM class diversity/granularity | **No** — `4class` (mean AUC ≈0.78) outperforms `10plus_sm`/`granular_sm13` (≈0.55–0.59) at every matched point tested, with a known coarse-label caveat | §7.1 |
| Probe complexity | **No** — richer probes raise effective rank but *lower* mean AUC at the one scale tested (4class/50k) | §7.2 |

Negative and null results are reported here exactly as found: **more class diversity does not help**, **more probe capacity does not help**, and **naive capacity/statistics scaling is unsafe without a stability fix**. These are stated as findings, not omitted.

## 9. Discussion

**What the representation appears to learn (evidence):** a latent space that classifies four (or more) coarse SM processes reasonably well (60–67% accuracy across the class-diversity/probe configurations examined) while using only a fraction of its nominal dimensionality (effective rank typically 2–14 against `latent_dim` 64–256).

**Whether additional SM diversity improves robustness (evidence + hypothesis):** the AUC evidence says no (§7.1); the classifier-response evidence suggests a plausible mechanism (coarse labels concentrate injected signals into one non-MinBias region, inflating apparent distance) but does not prove it — that remains a **hypothesis**, explicitly flagged as such in the source notes and preserved as such here.

**Whether model scaling helps (evidence):** conditionally — only along the `ff_dim`-at-fixed-depth direction within the tested range; depth and joint width/latent scaling are training-stability failures under the current recipe, not established physics negative results (§7.3, §9 caveat also in `analysis/scaling_axes_analysis.md:390-397`: *"wide2blocks and earlier large/XL runs should not be interpreted as evidence that more capacity hurts physics performance. They show a training-stability failure."*).

**Whether additional statistics help (evidence):** the only large-N regime tested (1M) required a different optimization recipe, so statistics and stability-fix are confounded in the current evidence (§7.4).

**Probe effects (evidence):** probe complexity measurably changes latent geometry (effective rank) without a corresponding anomaly-separation benefit, at the one scale where it was tested (§7.2) — a genuine finding, not extrapolated beyond `4class`/50k.

**MinBias-reference behavior (evidence):** MinBias is used as the sole anomaly reference throughout the current scaling study; earlier work (the handoff-package run) instead scored MinBias itself as the "signal" against a broader SM background and found `roc_auc_vs_sm_background=0.314` for that inverted framing, i.e. MinBias did *not* look anomalous relative to other SM background — consistent with MinBias behaving as an in-distribution control (**Evidence:** `handoff_anomaly_var01_cov0001_lat128_minbias/README.md:111-123`).

**Possible latent collapse (evidence):** effective rank stays well below nominal `latent_dim` in essentially every configuration examined (§5) — this is a repeated, quantitative observation, not a one-off.

**Implications for trigger-oriented anomaly detection (hypothesis):** none of the current results include quantization, FPGA synthesis, or online-latency measurements for `ngt_model` itself. The separate `ctl2_model` project independently explores a quantized Linformer path toward that goal (§13, Future Work), but no result in `ngt_model`'s own evidence base speaks to deployability. Any deployment implication is therefore explicitly a **hypothesis**, not evidence.

## 10. Limitations

- **No repeated seeds** were found for the current scaling-study runs (each configuration in Output 2.3 appears to be `seed42` only); run-to-run stochastic variation is therefore not quantified for the scaling axis. *(Earlier phases of the project did run seed sweeps — `analysis/hh4b_collapse_diagnostics.md` — but not the current scaling study.)*
- **Evaluation, not held-out test set:** metrics are computed on the internal `eval`/`background_eval` split from the same simulated sample; there is no independent test-set holdout described.
- **Limited signal set:** only 4 injected processes (HH4B, HH, VBFH, ggH), all Higgs-like; no non-Higgs BSM benchmark is present.
- **Coarse-label confound** in the `4class`-outperforms-diversity result (§7.1), not disentangled by any controlled experiment in the repository.
- **Training-recipe confound** in the 1M result (§7.4): LR and augmentation both changed alongside sample size.
- **Class imbalance across configurations:** `granular_sm13`'s finest sub-classes have far fewer available parquet files than coarser classes (e.g., `ww_hadronic`: 326 files vs. `qcd`: 16,014 files, per earlier dataset-diagnostic notes cross-referenced from this session's prior work but not re-verified in this pass — flagged as **not re-verified in this audit**).
- **No hardware/quantization validation** for `ngt_model`: it is a full-precision PyTorch pipeline; `ctl2_model` is a separate, architecturally different attempt at a quantized path.
- **MLflow logging is real in code and configured on (verified: `train_mha.py` calls `mlflow.set_tracking_uri/set_experiment/start_run/log_params/log_metrics/log_artifact`; every `run_config.json` has `mlflow_enable: true`), but its remote success is not independently confirmable from this filesystem-only audit** — no local `mlruns/` directory exists anywhere in the repo, and the one available summary CSV that has an `mlflow_run_url` field (`ngt_mha_nodes_aligned_outputs/scale_tests/unified_experiment_summary_complete_only.csv`) has it empty for every row it contains. Treat "logged to MLflow" as a code-verified *intent*, not a confirmed remote record.
- **Probe ablation coverage is partial** (§Output 2.4): only `4class`/50k has all three probes; 8 planned `mlp2` runs across other scales/class-sets were never executed (`scaling_study/summaries/master_results_missing.txt`).
- **This is an offline representation/anomaly-detection study**, explicitly not an online HLT/L1 throughput or FPGA result (stated repeatedly in the project's own notes, e.g. `analysis/scaling_axes_analysis.md:397`).

## 11. Conclusions

Answering the research questions from §1/Output 7 directly, in the scope of the evidence actually available:

- **(RQ3) Under the class-diversity configurations evaluated (4class, 10plus_sm, granular_sm13, at 50k and 100k events/class), increasing Standard-Model class diversity did not improve mean MinBias-distance AUC against the four current injected Higgs-like probes** — it decreased it substantially (0.78 → 0.55–0.59).
- **(RQ4) Under the probe configurations evaluated (linear, MLP, two-layer MLP, at 4class/50k), increasing probe capacity did not improve mean MinBias-distance AUC**, despite increasing latent effective rank.
- **(RQ2) Under the capacity configurations evaluated, scaling model capacity did not improve outcomes monotonically**: widening the feed-forward dimension alone (fixed depth, fixed latent size) improved the anomaly metric when it remained stable; increasing depth or jointly increasing latent size and feed-forward width caused outright training collapse at 150k–200k events/class under the recipe used.
- **(RQ1) Under the sample-size configurations evaluated, the largest tested scale (1M events/class) gave the best MinBias-distance AUC observed (0.84–0.85), but only once the training recipe (learning rate, augmentation) was also changed** — a pure statistics-only scaling law is not established by the current evidence.
- **(RQ6) MinBias-referenced Mahalanobis distance is a usable anomaly score**, comparable to (and in the one detailed comparison, slightly ahead of) a GMM-density score and their robust-z combination; no method is shown to dominate uniformly across the (limited) comparisons available.
- **(RQ7) The learned latent space is measurably lower-dimensional than its nominal size in every configuration inspected**, and this compression is not a reliable predictor — in either direction — of anomaly-separation quality.

## 12. Future Work

Derived directly from the limitations in §10:

- Repeat the current `4class`/1M and `ff512_blocks2` recommended configurations across multiple seeds to establish variance before treating any single-seed number as a scaling law.
- Design a matched-granularity experiment that controls for coarse-label geometry (e.g., evaluate `4class` and `granular_sm13` classifier response to injected signals side by side, as already partly done in §7.1, but extended to isolate the "pushed into one class" effect quantitatively).
- Separate the sample-size and training-recipe variables at 1M (e.g., a 1M run at the previous higher LR *with* the memory fix, if numerically stabilizable, or a 200k run at `lr=3e-5` to isolate the LR effect from the sample-size effect).
- Complete the probe ablation (`mlp2`) at the scales/class-sets listed as missing in `scaling_study/summaries/master_results_missing.txt`.
- Extend the injected-probe set beyond Higgs-like processes to test whether the class-diversity and capacity conclusions generalize to a different anomaly topology.
- Pursue the already-scoped quantization path (`ctl2_model`, or a dedicated `ngt_model`-architecture HGQ2 port, noted as non-trivial because `StandardMHA` in `mha_model.py` is not a drop-in match for HGQ2's `QMultiHeadAttention`) before making any deployment claim.

---

## Output 9 — Proposed Figure Sequence

| # | Title | Source | Variables shown | Message | Exists already? |
|---|---|---|---|---|---|
| 1 | Model / analysis pipeline | `README.md` Mermaid diagram (cross-checked against `mha_model.py`) | architecture flow | What the encoder computes | **Exists** (as a diagram in README, not a rendered image file) |
| 2 | Best validation loss vs. scale | `analysis/figures/scaling_study/4class_best_val_loss_vs_scale*.png` | samples/class 50k→1M | Loss improves with scale | **Exists** |
| 3 | Class-diversity AUC summary | `analysis/figures/scaling_study/class_granularity_minbias_auc_summary.png` | `4class`/`10plus_sm`/`granular_sm13` × 50k/100k | Diversity does not help the current metric | **Exists** |
| 4 | HH4B efficiency vs. granularity and input size | `analysis/figures/scaling_study/hh4b_efficiency_granularity_and_input_size.png` | class config × scale | Combined view of RQ1+RQ3 | **Exists** |
| 5 | Probe ablation: accuracy vs. eval-quality/AUC trade-off | would combine `scaling_study/summaries/probe_ablation_summary.csv` (no dedicated PNG found) | linear/mlp/mlp2 | Probe capacity vs. anomaly separation | **Would need generating** — a plot of this specific table was not found as an existing file |
| 6 | Capacity-scaling stability map (150k/200k, stable vs. collapsed) | would combine `analysis/scaling_axes_analysis.md` tables (no dedicated PNG found) | depth × width × outcome | Where capacity scaling breaks | **Would need generating** |
| 7 | MinBias-distance histogram (signal vs. background) | e.g. `scaling_study/main_4class_scaling/4class/1m/medium_ce_only_lr3e5_noaug/.../minbias_distance_distributions.png` | per-run | Visual separation at the best 1M point | **Exists**, per-run |
| 8 | Latent effective rank vs. `latent_dim` (5-class best-config study) | `analysis/figures/latent_dim_sweep_test/latent_dim_effective_rank_barplot.png` | nominal latent_dim | Capacity ≠ effective usage | **Exists** |
| 9 | Six-objective comparison: accuracy vs. eval-quality | `analysis/figures/superclass_loss_objective_comparison/loss_objective_accuracy_vs_eval_quality.png` | 6 CE-based objectives | Objective design changes geometry more than accuracy | **Exists** |

Per this task's explicit instruction, **no missing figure (5, 6) was generated during this audit.**

---

## Output 10 — Missing Evidence / Experiments Required Before Publication

1. Seed-repeated runs for the current `4class` scaling study (only single-seed evidence exists for every row of `scaling_study/summaries/main_4class_scaling_minbias_summary.csv`).
2. A controlled experiment isolating the coarse-label-geometry hypothesis from the class-diversity result (§7.1, §12).
3. A controlled experiment isolating sample-size effects from the LR/augmentation-recipe change at 1M (§7.4, §12).
4. Completion of the probe ablation across scales/class-sets (§10, §Output 2.4).
5. An independent held-out test set, separate from the internal `eval`/`background_eval` split used both to fit thresholds and to report final metrics.
6. Any quantization, hardware-latency, or FPGA-synthesis validation for the `ngt_model` architecture specifically (not the separate `ctl2_model` Linformer path).
7. A non-Higgs-like anomaly benchmark, to test whether conclusions generalize beyond the four processes currently used.
8. An explanation for why `150k/large`, `200k/xl`, and `200k/ff512_depth3` — all real run directories with `run_config.json` — are excluded from `scaling_study/summaries/main_4class_scaling_minbias_summary.csv` without any documenting note found in the repository (§2.6). This should be resolved with the project team before treating that summary table as exhaustive.
9. Completion of `eval_quality_scores_n2000.json`-style latent-quality diagnostics for the current scaling study, which is missing from all but 2 of the 12 probe-ablation runs and absent from every `main_4class_scaling`/`priority2`/`granular_sm13` run (§2.6).

## Output 11 — Claims: Strongly Supported vs. Preliminary

**Strongly supported** (direct, code- and/or data-verified, multiple corroborating files):
- The encoder architecture and preprocessing pipeline as described in Output 3/4 (verified directly in source).
- The three anomaly-score definitions and the metric formulas in Output 5/§6 (verified directly in source, cross-checked against live CSV values).
- Class diversity (`4class` vs `10plus_sm` vs `granular_sm13`) does not improve mean MinBias-distance AUC at 50k/100k (§7.1) — consistent across two independent evidence paths (analysis note + curated summary CSV).
- Probe complexity raises effective rank but not mean AUC at `4class`/50k (§7.2) — single dedicated ablation, internally consistent.
- Depth-3 (and joint width+latent) capacity scaling collapses training at 150k–200k under the current recipe (§7.3) — directly observed via `run_config.json` + training metrics, described the same way in two analysis notes.
- The stable 1M recipe and its resulting AUCs (§7.4) — verified directly against a live per-run CSV, not only the summarizing note.

**Preliminary / single-configuration / hypothesis-flagged:**
- Whether the `4class` diversity advantage is a coarse-label artifact (explicit hypothesis in source notes, not isolated experimentally).
- Whether the 1M improvement is a sample-size effect, a recipe effect, or both (confounded, §7.4).
- Any generalization beyond the four Higgs-like probes.
- Any statement about deployability, latency, or quantization readiness of `ngt_model` specifically.
- The `granular_sm13` per-class sample-availability imbalance claim in §10 (flagged there as **not re-verified in this audit**).

## Output 12 — Repository Files Used as Evidence

```
README.md
analysis/README.md
analysis/best_config_finding_and_latent_sweep.md
analysis/superclass_loss_objective_comparison.md
analysis/external_higgs_signal_evaluation.md
analysis/scaling_axes_analysis.md
analysis/1m_500k_scaling_analysis.md
analysis/scaling_analysis.md
ngt_model/src/ngt_model/models/mha_model.py
ngt_model/src/ngt_model/datasets/collide2v_data.py
ngt_model/src/ngt_model/training/train_mha.py
ngt_model/src/ngt_model/evaluation/external_higgs_anomaly.py
ngt_model/src/ngt_model/evaluation/latent_diagnostics.py
ngt_model/src/ngt_model/evaluation/latent_fisher.py
ngt_model/src/ngt_model/evaluation/head_fisher.py
scaling_study/README.md
scaling_study/summaries/main_4class_scaling_minbias_summary.csv
scaling_study/summaries/probe_ablation_summary.csv
scaling_study/summaries/master_results_missing.txt
scaling_study/main_4class_scaling/4class/1m/medium_ce_only_lr3e5_noaug/runs/2026-08-16/scaling_4class_1m_medium_ce_only_lr3e5_noaug_seed42/run_config.json
scaling_study/main_4class_scaling/4class/1m/medium_ce_only_lr3e5_noaug/runs/2026-08-16/scaling_4class_1m_medium_ce_only_lr3e5_noaug_seed42/figures/external_higgs_anomaly/external_higgs_anomaly_metrics_by_method.csv
analysis/figures/scaling_study/4class_loss_vs_scale_summary.csv
analysis/figures/scaling_study/class_granularity_minbias_auc_summary.csv
analysis/figures/ (directory listing, for Output 6)
handoff_anomaly_var01_cov0001_lat128_minbias/README.md
ngt_mha_nodes_aligned_outputs/scale_tests/unified_experiment_summary_complete_only.before_2026_08_06_update.csv
scaling_study/main_4class_scaling/4class/500k/ff512_blocks2/runs/2026-08-16/scaling_4class_500k_ff512_blocks2_linear_seed42/failed_run_evidence/README.md
scaling_study/main_4class_scaling/4class/1m/ff512_blocks2_ce_only_noaug/runs/2026-08-15/scaling_4class_1m_ff512_blocks2_ce_only_noaug_seed42/failed_run_evidence/README.md
scaling_study/summaries/master_results_missing.txt
```

**Note on method:** this file list, and the report above, were produced across two passes: (1) direct file reads and targeted verification performed by the main analysis, plus (2) two independent read-only sub-audits — one of `ngt_model/src/` (source code, cited throughout §3/§Output 3/§Output 5/§6) and one of every run directory with a `run_config.json` (cited throughout Output 2, §7). Where the second pass surfaced a claim in the first draft that could not be independently confirmed (the "5-class best-config" study, §Output 2.5), it was corrected in place rather than left standing — see the inline correction note there.

*(This list covers files this report directly cites with a specific claim or number. Output 1's repository map and Output 2's inventory additionally reflect direct directory listings of the workspace, not individually enumerated here.)*

---

## Phase 12 — Final Audit (self-check, performed before delivering this report)

- Did I modify any existing file? **No** — this file and its containing directory (`scientific_report/`) are new; nothing under `ngt_model/`, `scaling_study/`, `analysis/`, `ctl2_model/`, or any other pre-existing path was written to.
- Did I assume experiments not present? **One instance caught and corrected**: a first-pass draft asserted a "5-class best-config" study; an independent second-pass file search found no such file and the claim was rewritten in place (§Output 2.5) rather than left uncorrected. Every remaining experiment family in Output 2 is tied to a specific directory or analysis note found by direct listing/reading; "Input granularity scaling" (§4.5) is explicitly marked as not a separate axis rather than invented.
- Did I invent metrics? **No** — every metric in Output 5 is quoted from source-code logic, not assumed from a column name.
- Did I invent physics conclusions? **No** — all conclusions in §11 are qualified to the exact configurations evaluated.
- Did I distinguish training classes from evaluation-only anomalies? **Yes** — explicitly, repeatedly, starting in Output 2.1 and carried through every results section.
- Did I distinguish qualitative PCA/t-SNE observations from quantitative anomaly metrics? **Yes** — stated as a repository-documented convention and followed throughout §5, §Output 6.
- Did I trace major conclusions back to files/results? **Yes** — see inline **Evidence:** annotations throughout.
- Did I report negative results rather than hiding them? **Yes** — §7.1, §7.2, §8 state the null/negative findings as headline results, not caveats.
- Did I clearly identify missing information? **Yes** — "Not verifiable from the repository" used in §3, §4.5; explicit "not re-verified in this audit" flags in §10/Output 11 for the one claim I could not directly re-check in this pass.
