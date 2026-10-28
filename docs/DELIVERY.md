# CERN delivery: NGT 100M and mixed-precision quantization

Prepared 2026-10-28 for `cms-l1-ad/openlab_ss26` with three separate projects and a documented selective INT8 CPU baseline.

## Start here

| Item | Location |
|---|---|
| Original Keras/HGQ/Kedro project | `ctl2_model/` |
| Current FP32 code and training/evaluation tools | `ngt_model/` |
| Selected 100M run, config, history and FP32 weights | `ngt_model/final_run/` |
| INT8 conversion, inference and comparison package | `quantized_model/` |
| Delivered mixed-precision weights | `quantized_model/models/` |
| Quantization manifests and measured evidence | `quantized_model/evidence/` |
| Scaling launch/resume scripts and configs | `ngt_model/scaling_study/large_100m_80m/` |
| Updated analysis notes | `docs/analysis/` |
| Existing technical report, historical context | `docs/reports/` |
| Editable current and historical diagrams | `docs/diagrams/README.md` |

The original research workspace is preserved. A separate `ctl2_model/` contains the original Kedro/HGQ Linformer project. Existing historical
notebooks/results remain available. Credentials, raw training caches and new
per-event tensors are not added to this delivery. The report in this repository
predates this quantization work; it is not a newly approved quantization report.

## Source selection

The source run is the CE + InfoNCE 80M-event configuration from 2026-08-24:

`scaling_4class_80m_dmodel2016_ff8064_100m_400ep_noaug_ce_infonce005_lr1e5_seed42`

It has 99,889,413 parameters, raw PID input with five features per candidate,
attention pooling, four attention blocks and four trained classes
(`minbias,qcd,dy,top`). This differs from the older 24-feature one-hot MHA runs.
The delivered epoch-4 checkpoint has the lowest recorded validation loss among
the five completed epochs. `400ep` is an epoch budget in the run name, not a
claim that 400 epochs completed. The full architecture and preprocessing are
frozen in both packages' source run configurations.

The FP32 checkpoint and the selected quantized bundle are stored with Git LFS.
After cloning, run `git lfs pull` before inference. Source datasets stay on CERN
storage. Raw 80M-event caches are not duplicated in GitLab.

## Quantization result and limits

| Measured item | Result |
|---|---:|
| Source checkpoint bytes | 399,573,270 |
| Selected mixed-precision bundle bytes | 302,201,316 |
| Reduction in artifact bytes | 24.37% |
| INT8 layers | `post_pool.0`, `post_pool.2` |
| Numerical pilot events | 512, 128 per trained class |
| FP32 / INT8 prediction agreement | 510 / 512 = 99.609375% |
| FP32 / INT8 pilot classification accuracy | 0.501953125 / 0.501953125 |
| Latent relative L2 difference | 0.0052094454 |
| Latent mean cosine similarity | 0.9999935033 |

The package quantizes the two post-pooling dense layers with per-channel dynamic
INT8. Attention, input projection, pooling-score network, latent projection and
classifier remain FP32. CPU throughput in this pilot did not improve
(17.44 events/s FP32 versus 15.15 events/s mixed precision); no speed or hardware
latency benefit is claimed.

The 22-layer dense INT8 experiment was rejected: it reduced artifact size to
100.8MB but only 27.34% of predictions matched FP32. Its measurements and the
layer sensitivity study are retained as evidence. That rejected binary is not
part of the delivered model.

This is a numerical prototype for CPU inference. It is **not** a QAT model,
integer-only attention implementation, FPGA/HLS design or L1-trigger deployment.
QKeras was not used because a Keras port would require an additional graph and
weight equivalence study. The existing PyTorch checkpoint is retained directly.

The pilot uses historical cached events, may overlap training, and informed
the policy choice. Its anomaly numbers use a direct Ledoit-Wolf MinBias latent
distance and are not interchangeable with historical PCA/GMM results. Only
128 events per external signal and 512 background events were used; the 99.5%
tail numbers are diagnostic and do not establish physics acceptance.

## Reproduction and validation

The [package README](../quantized_model/README.md) contains complete setup, conversion, prediction and comparison
commands. Python 3.11 and PyTorch 2.5.1 are used. The locally measured dependency
versions are recorded in `quantized_model/evidence/runtime.json`.

Local validation completed:

- Five functional tests: actual INT8 layer conversion and exact bundle reload,
  padded-event inference, frozen raw preprocessing, strict checkpoint loading,
  split recovery equivalence, and invalid layer exclusion.
- Conversion of the real 99,889,413-parameter checkpoint and reload smoke check.
- FP32-versus-quantized prediction, latent, logit, classification and anomaly
  comparisons on the same numerical pilot inputs.
- Source SHA256 and delivered artifact SHA256 recorded in the conversion manifest.
- Syntax parsing of all 14 Mermaid diagram definitions and embedded blocks,
  including current and historical diagrams; 16 shell scripts checked.
- Both PyTorch packages build as wheels and current README navigation resolves.

The GitLab CI job runs the small functional suite without downloading weights.
Its remote result must be read from GitLab after publication; local
success does not imply a successful remote pipeline.

Still required for physics/deployment acceptance: rerun on the original held-out
DDP splits (the extraction tool is included), use sufficiently large independent
background/signal samples, agree acceptance tolerances, and validate the intended
target hardware. The original large caches were not fully recovered locally
through CERNBox; use EOS for the held-out extraction.
