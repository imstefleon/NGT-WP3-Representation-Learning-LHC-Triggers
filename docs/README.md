# Documentation

| Material | Contents |
|---|---|
| [Delivery guide](DELIVERY.md) | Selected model, measured quantization results and reproduction scope |
| [Mermaid diagrams](diagrams/README.md) | Current project map, MHA architecture, quantization flow and CTL2 pipeline |
| [Analysis index](analysis/README.md) | Research notes and figures by topic |
| [Reports](reports/) | Technical report from the FP32 study |
| [Presentations](presentations/) | Existing project presentations |
| [Local checks](LOCAL_CHECKS.json) | Recorded functional, packaging, script and integrity checks |
| [Delivery inventory](DELIVERY_INVENTORY.json) | Curated file paths, sizes and SHA256 values |
| [File preservation](TRACKED_FILE_PRESERVATION.json) | Current location of every previously tracked delivery file |
| [Reorganization map](reorganization_map.json) | Folder moves used to separate the three projects |

The root projects are [ngt_model](../ngt_model/README.md),
[quantized_model](../quantized_model/README.md) and
[ctl2_model](../ctl2_model/README.md). Earlier notebooks and outputs are in
[archive/](../archive/README.md).

The report and historical descriptions predate the selective INT8 prototype.
Use `DELIVERY.md` and the quantization evidence for its measured status. Older
paths and experimental configurations are preserved as provenance in historical
documents. The [CTL2 CI reference](ci/ctl2_training_pipeline.yml) is separate from
the root PyTorch test pipeline.
