# 2026-07-09 Dataset Diagnostics and DY-MinBias Runs [HF Collide-2V, ROOT samples.json, Updated Collide-2V]

## Scope

This analysis documents the checks performed to understand why the latent space and the `dxy` input distributions looked different across three data sources:

- HF Collide-2V parquet samples.
- ROOT samples from `ctl2_model/data/01_raw/samples.json`.
- Updated Collide-2V production parquet samples from:

```text
/eos/project/f/foundational-model-dataset/samples/production_final
```

The same top-pT L1 candidate selection logic was used in the diagnostic notebooks. The comparison focused on whether the selected L1 branches were equivalent, especially for `dxy`.

## Dataset Source Summary

| Source | Selected L1 branch family | Selected dxy-like branch | Main behavior |
|---|---|---|---|
| HF Collide-2V | `L1T_PFCand_*` | `L1T_PFCand_D0` | Almost all `dxy/D0` values are exactly zero, with rare large outliers. |
| ROOT `samples.json` | `L1PuppiCands_*` | `L1PuppiCands_dxy` | Positive, extended, discretized `dxy` distribution between 0 and about 1.17. |
| Updated Collide-2V `production_final` | `L1T_PUPPIPart_*` | `L1T_PUPPIPart_D0` | Similar to HF: mostly zero `D0`, with rare large outliers. |

## Run Source Map

The run folders from this day correspond to two different training/evaluation sources, while the updated Collide-2V dataset was used for diagnostic comparison only.

| Run folder | Dataset source identified from outputs | Evidence |
|---|---|---|
| `dy_minbias_wide_attn_aug_top_pt_minbias100000_top100000_qcd100000_dy100000_102125` | HF Collide-2V | DY was split into `dy_ee`, `dy_mumu`, and `dy_tautau`; the raw `dxy` plot is centered at zero with rare large outliers, matching the HF `L1T_PFCand_D0` behavior. |
| `dy_minbias_wide_attn_aug_top_pt_minbias80000_top80000_qcd80000_dy80000_111829` | ROOT `samples.json` | The raw `dxy` plot has the positive 0-1.2 structure expected from `L1PuppiCands_dxy`; only DY-MinBias diagnostic outputs were saved. |
| `dy_minbias_wide_attn_aug_top_pt_minbias80000_top80000_qcd80000_dy80000_115417` | ROOT `samples.json` | The raw `dxy` plot matches ROOT `L1PuppiCands_dxy`; this folder also contains diagnostic/training outputs but not the full final evaluation table. |
| `dy_minbias_wide_attn_aug_top_pt_minbias80000_top80000_qcd80000_dy80000_122835` | ROOT `samples.json` | The raw `dxy` plot matches ROOT `L1PuppiCands_dxy`; this is the complete ROOT 80k evaluation run. |

## HF Collide-2V vs ROOT Diagnostic

Diagnostic outputs:

```text
ngt_linformer_nodes_aligned_outputs/diagnostics/dxy_hf_vs_root/
```

Selected branches:

| Source | pt | eta | phi | dxy | pid |
|---|---|---|---|---|---|
| HF Collide-2V | `L1T_PFCand_PT` | `L1T_PFCand_Eta` | `L1T_PFCand_Phi` | `L1T_PFCand_D0` | `L1T_PFCand_PID` |
| ROOT | `L1PuppiCands_pt` | `L1PuppiCands_eta` | `L1PuppiCands_phi` | `L1PuppiCands_dxy` | `L1PuppiCands_pdgId` |

`dxy` summary:

| Source | Class | Median | Max | Fraction exactly zero |
|---|---|---:|---:|---:|
| HF | MinBias | 0.000 | 3.50 | 0.99988 |
| HF | Top | 0.000 | 17.67 | 0.94069 |
| HF | QCD | 0.000 | 23.06 | 0.99296 |
| HF | DY | 0.000 | 19.45 | 0.99943 |
| ROOT | MinBias | 0.203 | 1.17 | 0.12687 |
| ROOT | Top | 0.141 | 1.17 | 0.26816 |
| ROOT | QCD | 0.141 | 1.17 | 0.25187 |
| ROOT | DY | 0.188 | 1.17 | 0.16655 |

Interpretation:

- HF and ROOT are not using the same `dxy` branch definition.
- HF `L1T_PFCand_D0` is almost always zero for MinBias, DY, and QCD.
- ROOT `L1PuppiCands_dxy` is not mostly zero and has a physically structured positive distribution.
- Latent-space comparisons between HF-trained runs and ROOT-trained runs should not be interpreted as a pure model effect, because the input distributions are different.

## Updated Collide-2V vs ROOT Diagnostic

Diagnostic outputs:

```text
ngt_linformer_nodes_aligned_outputs/diagnostics/collide2v_vs_root_dxy/
```

The updated production parquet files contain both `L1T_*` and `FullReco_*` columns. The diagnostic loader was kept L1-only and selected the `L1T_PUPPIPart_*` family.

Selected branches:

| Source | pt | eta | phi | dxy | pid |
|---|---|---|---|---|---|
| Updated Collide-2V | `L1T_PUPPIPart_PT` | `L1T_PUPPIPart_Eta` | `L1T_PUPPIPart_Phi` | `L1T_PUPPIPart_D0` | `L1T_PUPPIPart_PID` |
| ROOT | `L1PuppiCands_pt` | `L1PuppiCands_eta` | `L1PuppiCands_phi` | `L1PuppiCands_dxy` | `L1PuppiCands_pdgId` |

Schema inspection:

| Source | Example class | L1 columns | FullReco columns | Selected L1 dxy |
|---|---|---:|---:|---|
| Updated Collide-2V | MinBias | 110 | 109-111 | `L1T_PUPPIPart_D0` |
| Updated Collide-2V | Top | 112 | 113 | `L1T_PUPPIPart_D0` |
| Updated Collide-2V | QCD | 108-110 | 109-113 | `L1T_PUPPIPart_D0` |
| Updated Collide-2V | DY | 108 | 107-111 | `L1T_PUPPIPart_D0` |

`dxy` summary:

| Source | Class | Median | Max | Fraction exactly zero |
|---|---|---:|---:|---:|
| Updated Collide-2V | MinBias | 0.000 | 2.37 | 0.99313 |
| Updated Collide-2V | Top | 0.000 | 19.71 | 0.93756 |
| Updated Collide-2V | QCD | 0.000 | 17.38 | 0.98791 |
| Updated Collide-2V | DY | 0.000 | 6.51 | 0.99296 |
| ROOT | MinBias | 0.203 | 1.17 | 0.12687 |
| ROOT | Top | 0.141 | 1.17 | 0.26816 |
| ROOT | QCD | 0.141 | 1.17 | 0.25187 |
| ROOT | DY | 0.188 | 1.17 | 0.16655 |

Interpretation:

- Updated Collide-2V provides L1 information, but the selected L1 branch family is `L1T_PUPPIPart_*`, not `L1PuppiCands_*`.
- The updated Collide-2V `D0` distribution behaves much more like HF `L1T_PFCand_D0` than ROOT `L1PuppiCands_dxy`.
- The ROOT samples remain distinct in `dxy` behavior.
- This explains why a ROOT latent PCA can look very different from HF or updated Collide-2V runs.

## HF DY Final-State Run

Run:

```text
ngt_linformer_nodes_aligned_outputs/runs/2026-07-09/
dy_minbias_wide_attn_aug_top_pt_minbias100000_top100000_qcd100000_dy100000_102125/
```

Dataset source:

```text
HF Collide-2V parquet samples with DY split into gen-level final-state labels
```

The target classes included:

- `minbias`
- `top`
- `qcd`
- `dy_ee`
- `dy_mumu`
- `dy_tautau`

Classification result:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| MinBias | 0.470 | 0.936 | 0.626 | 15,000 |
| Top | 0.911 | 0.958 | 0.934 | 15,000 |
| QCD | 0.817 | 0.740 | 0.777 | 15,000 |
| DY->ee | 0.825 | 0.043 | 0.081 | 6,855 |
| DY->mumu | 0.837 | 0.049 | 0.093 | 6,849 |
| DY->tautau | 0.000 | 0.000 | 0.000 | 1,296 |

Confusion behavior:

| True class | Main confusion |
|---|---|
| `DY->ee` | 6,176 / 6,855 classified as MinBias |
| `DY->mumu` | 6,082 / 6,849 classified as MinBias |
| `DY->tautau` | 1,063 / 1,296 classified as MinBias |

Interpretation:

- Splitting DY into final-state labels did not make the network recover DY subgroups.
- `DY->ee`, `DY->mumu`, and `DY->tautau` still collapse mostly into MinBias.
- The result supports the earlier latent-space conclusion: the DY-MinBias overlap is already present before the final classifier.

## ROOT 80k Run With Improved DY-MinBias Decision Boundary

Run:

```text
ngt_linformer_nodes_aligned_outputs/runs/2026-07-09/
dy_minbias_wide_attn_aug_top_pt_minbias80000_top80000_qcd80000_dy80000_122835/
```

Dataset/source note:

```text
ROOT samples from ctl2_model/data/01_raw/samples.json, using L1PuppiCands_* inputs.
```

Classification result:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| MinBias | 0.652 | 0.906 | 0.758 | 12,000 |
| Top | 0.760 | 0.773 | 0.767 | 12,000 |
| QCD | 0.724 | 0.598 | 0.655 | 12,000 |
| DY | 0.792 | 0.608 | 0.688 | 12,000 |

Overall:

```text
accuracy = 0.721
macro F1 = 0.717
```

Confusion matrix:

| True class | Pred MinBias | Pred Top | Pred QCD | Pred DY |
|---|---:|---:|---:|---:|
| MinBias | 10,867 | 110 | 396 | 627 |
| Top | 140 | 9,277 | 1,863 | 720 |
| QCD | 1,823 | 2,434 | 7,180 | 563 |
| DY | 3,849 | 383 | 478 | 7,290 |

DY-MinBias behavior:

```text
DY -> MinBias = 3,849 / 12,000 = 32.1%
DY -> DY      = 7,290 / 12,000 = 60.8%
```

Mahalanobis medians:

| Class | Median |
|---|---:|
| MinBias | -29.87 |
| DY | 55.53 |
| Top | 321.28 |
| QCD | 358.91 |

Interpretation:

- This run produced much better DY recall than the July 8 latent-space run.
- DY is no longer almost completely absorbed into MinBias.
- The result suggests that the ROOT L1 input definition and/or the updated run configuration changed the decision boundary in a useful direction.
- The Mahalanobis median for DY moved away from MinBias, unlike earlier HF runs where DY and MinBias medians were almost identical.

Two earlier ROOT 80k folders from the same day, `111829` and `115417`, saved DY-MinBias diagnostic outputs with the same binary summary:

| Binary class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| MinBias | 0.737 | 0.918 | 0.817 | 24,000 |
| DY | 0.891 | 0.672 | 0.766 | 24,000 |

Binary confusion matrix:

| True class | Pred MinBias | Pred DY |
|---|---:|---:|
| MinBias | 22,026 | 1,974 |
| DY | 7,870 | 16,130 |

## Overall Dataset Diagnostic Conclusion

The main finding from the day is that the three data sources are not interchangeable at the L1 branch level.

Key conclusions:

- HF Collide-2V and updated Collide-2V both use `D0`-like L1 branches that are mostly zero with rare outliers.
- ROOT uses `L1PuppiCands_dxy`, which has a very different positive distribution.
- The updated Collide-2V production files include both L1 and FullReco columns; the diagnostic comparison used only L1 columns.
- Differences in latent PCA between ROOT and HF/updated Collide-2V are expected because the selected input variables are not equivalent.
- DY-MinBias remains sensitive to dataset definition, branch semantics, and training configuration.
