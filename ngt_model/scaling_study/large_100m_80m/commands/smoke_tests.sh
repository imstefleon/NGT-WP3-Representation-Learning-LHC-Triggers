#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../../.."
export PYTHONPATH="${PWD}/ngt_model/src:${PYTHONPATH:-}"
export MPLBACKEND=Agg

COMMON_ARGS=(
  --data-source collide2v_updated
  --class-names minbias,qcd,dy,top
  --n-constituents 100
  --latent-dim 256
  --ff-dim 2304
  --num-heads 16
  --num-attention-blocks 4
  --pooling-mode attention
  --classifier-head mlp
  --loss-mode ce
  --latent-var-reg-weight 0.01
  --latent-cov-reg-weight 0.001
  --lr 3e-5
  --weight-decay 1e-4
  --batch-size 1
  --num-workers 0
  --epochs 1
  --early-stopping-patience 1
  --grad-clip-norm 1.0
  --fail-on-nonfinite-loss
)

python -m ngt_model.training.train_mha \
  "${COMMON_ARGS[@]}" \
  --n-events-per-class 1000 \
  --augmentation-copies 0 \
  --output-root outputs/scaling_study/large_100m_80m/smoke_single_aug0 \
  --run-name smoke_single_aug0

python -m ngt_model.training.train_mha \
  "${COMMON_ARGS[@]}" \
  --n-events-per-class 1000 \
  --augmentation-copies 1 \
  --output-root outputs/scaling_study/large_100m_80m/smoke_single_aug1 \
  --run-name smoke_single_aug1

torchrun --standalone --nproc_per_node=4 -m ngt_model.training.train_mha \
  "${COMMON_ARGS[@]}" \
  --data-loading-mode rank_sharded_materialized \
  --n-events-per-class 1000 \
  --augmentation-copies 0 \
  --output-root outputs/scaling_study/large_100m_80m/smoke_ddp_rank_sharded \
  --run-name smoke_ddp_rank_sharded_aug0
