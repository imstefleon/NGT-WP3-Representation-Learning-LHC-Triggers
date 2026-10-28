# Selected 100M MHA checkpoint

Source run:
`scaling_4class_80m_dmodel2016_ff8064_100m_400ep_noaug_ce_infonce005_lr1e5_seed42`

The original run directory is:

```text
/eos/home-i03/e/emorenol/ctl2_embedding_model/outputs/scaling_study/large_100m_80m/one_24h_100m/runs/2026-08-24/scaling_4class_80m_dmodel2016_ff8064_100m_400ep_noaug_ce_infonce005_lr1e5_seed42
```

The delivered `models/checkpoint_epoch_004.pt` is selected from the five locally
recorded epochs by the lowest validation loss (1.268218801498413). The `400ep`
run name indicates a budget; the available history does not show 400 completed
epochs. The last recorded epoch is 5. These files do not imply a finished
400-epoch training campaign or a physics acceptance decision.

`run_config.json` freezes architecture, feature order, raw PID encoding,
normalization, label order, split sizes, DDP sharding and training settings.
The model has 99,889,413 parameters, attention pooling, 16 heads, four attention
blocks, width 2016, post-pooling width 8064, and a 128-dimensional latent vector.

The checkpoint contains `model_state_dict`, epoch number, validation metrics and
best validation loss. It contains no optimizer/RNG state and is suitable for
inference or a new fine-tuning run, not exact optimizer-state resume.

The preprocessing intentionally follows the existing implementation, including
its padding mask and raw PID behavior. `dxy` and PID remain in their original
scale. Normalization must not be recomputed during inference.

Historical anomaly-injection evidence and evolution figures are included under
`figures/`. Those results were produced by the FP32 workflow; the new quantized
comparison is under `../../quantized_model/evidence/` and uses a different
pilot/reference procedure. Do not mix the two sets of metrics.

Weights use Git LFS. Run `git lfs pull` after cloning. The source checkpoint
SHA256 is recorded in `../../quantized_model/evidence/conversion_manifest.json`.
