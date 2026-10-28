# NGT WP3: representation learning for LHC triggers

CERN Openlab Summer Student 2026 project. The delivery contains the current
PyTorch NGT workflow, the selected 99,889,413-parameter checkpoint, and a separate
mixed-precision CPU quantization package.

**[Read the delivery guide](DELIVERY.md)** for source selection, measured results,
reproduction commands and remaining physics/hardware validation.

| Project | Entry point |
|---|---|
| FP32 NGT model | [ngt_model](ctl2_embedding_model/ngt_model/README.md) |
| Selected 100M checkpoint | [final_run](ctl2_embedding_model/ngt_model/final_run/README.md) |
| Quantized model | [quantized_model](ctl2_embedding_model/quantized_model/README.md) |
| Large-run launch/resume tools | [scaling study](ctl2_embedding_model/scaling_study/large_100m_80m/README.md) |
| Updated analyses | [analysis index](ctl2_embedding_model/analysis/README.md) |
| Historical project overview | [August overview](docs/historical_overview_2026_08.md) |

## Get the weights

```bash
git lfs install
git clone https://gitlab.cern.ch/cms-l1-ad/openlab_ss26.git
cd openlab_ss26
git switch cern-final-2026-10-28
git lfs pull
```

The source checkpoint is epoch 4 of the CE + InfoNCE run; five epochs are present
in the history. The 400-epoch run-name suffix is a budget, not completed training.

The selected quantization converts two post-pooling dense layers to INT8 and
retains the attention graph in FP32. The delivered artifact is 24.4% smaller;
prediction agreement is 99.61% on a 512-event numerical pilot. The pilot informed
policy selection and may overlap training. It is not independent physics or FPGA
acceptance. Aggressive attention quantization failed and is documented in the
evidence instead of being presented as a deployable model.

Raw datasets remain on CERN storage; credentials are excluded. Historical
notebooks/results already tracked in the repository are preserved.
