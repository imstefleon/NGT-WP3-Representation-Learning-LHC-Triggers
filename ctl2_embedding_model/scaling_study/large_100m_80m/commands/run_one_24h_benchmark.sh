#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."
export PYTHONPATH="${PWD}/ngt_model/src:${PYTHONPATH:-}"
export MPLBACKEND=Agg

torchrun --standalone --nproc_per_node=4 \
  scaling_study/large_100m_80m/commands/benchmark_100m_ddp.py \
  --batch-sizes "${BATCH_SIZES:-1,2,4,8,16,24,32,48,64}" \
  --warmup-steps "${WARMUP_STEPS:-5}" \
  --measure-steps "${MEASURE_STEPS:-20}" \
  --d-model 2016 \
  --ff-dim 8064 \
  --latent-dim 128 \
  --num-heads 16 \
  --num-attention-blocks 4 \
  --n-features 5 \
  --output scaling_study/large_100m_80m/benchmark_100m_ddp_results.csv
