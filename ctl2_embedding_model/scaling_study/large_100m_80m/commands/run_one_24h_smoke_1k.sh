#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."
export PYTHONPATH="${PWD}/ngt_model/src:${PYTHONPATH:-}"
export MPLBACKEND=Agg
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

torchrun --standalone --nproc_per_node=4 -m ngt_model.training.train_mha \
  --data-source collide2v_updated \
  --data-loading-mode rank_sharded_materialized \
  --ddp-timeout-minutes 2880 \
  --pid-encoding raw \
  --class-names minbias,qcd,dy,top \
  --n-events-per-class 1000 \
  --balanced-val-events-per-class 50 \
  --balanced-eval-events-per-class 50 \
  --n-constituents 100 \
  --latent-dim 128 \
  --d-model 2016 \
  --ff-dim 8064 \
  --num-heads 16 \
  --num-attention-blocks 4 \
  --pooling-mode attention \
  --classifier-head linear \
  --loss-mode ce \
  --latent-var-reg-weight 0.01 \
  --latent-cov-reg-weight 0.001 \
  --augmentation-copies 0 \
  --lr 3e-5 \
  --lr-schedule cosine \
  --lr-eta-min 0.0 \
  --weight-decay 1e-4 \
  --batch-size "${SMOKE_BATCH_SIZE:-1}" \
  --num-workers 0 \
  --epochs 1 \
  --checkpoint-every-epochs 1 \
  --early-stopping-patience 0 \
  --grad-clip-norm 1.0 \
  --fail-on-nonfinite-loss \
  --max-loss-value 1000000 \
  --max-regularizer-value 100000000 \
  --seed 42 \
  --output-root outputs/scaling_study/large_100m_80m/smoke_1k_realdata \
  --run-name smoke_1k_dmodel2016_ff8064_rank_sharded_seed42
