# NGT model workspace

This directory contains the FP32 research workflow and the separate quantization
project for the actual 100M MHA checkpoint.

- [Delivery guide](../DELIVERY.md): model provenance, measured results and limits.
- [ngt_model](ngt_model/README.md): current training, data loading and evaluation code.
- [Selected FP32 run](ngt_model/final_run/README.md): epoch-4 checkpoint, configuration,
  five-epoch history and historical evaluation evidence.
- [quantized_model](quantized_model/README.md): conversion, loading, CPU inference,
  layer sensitivity, comparison and functional tests.
- [Large scaling study](scaling_study/large_100m_80m/README.md): original launch,
  resume, benchmark and analysis scripts.
- [Analysis index](analysis/README.md): scientific background and earlier studies.

Run commands from this directory after setting the package source paths:

```bash
export PYTHONPATH="$PWD/ngt_model/src:$PWD/quantized_model/src:${PYTHONPATH:-}"
```

The authoritative final-run input has 100 candidates and five normalized/raw-PID
features. The historical one-hot 24-feature runs remain available but are a
different model configuration. Original data caches are external; use `git lfs
pull` to obtain the delivered FP32 and mixed-precision weights.
