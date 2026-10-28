#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 BATCH_SIZE_PER_GPU" >&2
  echo "Recommended after the var/cov collapse: use 1024 first." >&2
  exit 2
fi

BATCH_SIZE="$1"

cd "$(dirname "$0")/../../../.."
export PYTHONPATH="${PWD}/ngt_model/src:${PYTHONPATH:-}"
export MPLBACKEND=Agg
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

torchrun --standalone --nproc_per_node=4 -m ngt_model.training.train_mha \
  --data-source collide2v_updated \
  --data-loading-mode rank_sharded_materialized \
  --ddp-timeout-minutes 2880 \
  --pid-encoding raw \
  --class-names minbias,qcd,dy,top \
  --n-events-per-class 20000000 \
  --balanced-val-events-per-class 1000 \
  --balanced-eval-events-per-class 1000 \
  --n-constituents 100 \
  --latent-dim 128 \
  --d-model 2016 \
  --ff-dim 8064 \
  --num-heads 16 \
  --num-attention-blocks 4 \
  --pooling-mode attention \
  --classifier-head linear \
  --loss-mode ce \
  --latent-var-reg-weight 0.0 \
  --latent-cov-reg-weight 0.0 \
  --augmentation-copies 0 \
  --lr 1e-5 \
  --lr-schedule cosine \
  --lr-eta-min 0.0 \
  --weight-decay 1e-4 \
  --batch-size "${BATCH_SIZE}" \
  --num-workers 0 \
  --epochs 400 \
  --checkpoint-every-epochs 1 \
  --early-stopping-patience 0 \
  --grad-clip-norm 0.5 \
  --fail-on-nonfinite-loss \
  --max-loss-value 1000000 \
  --max-regularizer-value 100000000 \
  --seed 42 \
  --output-root outputs/scaling_study/large_100m_80m/one_24h_100m \
  --run-name scaling_4class_80m_dmodel2016_ff8064_100m_400ep_noaug_ceonly_lr1e5_seed42 \
  --mlflow-enable \
  --mlflow-tracking-uri https://ml.cern.ch/models \
  --mlflow-experiment ngt_model_scaling_100m_80m \
  --mlflow-run-name scaling_4class_80m_dmodel2016_ff8064_100m_400ep_noaug_ceonly_lr1e5_seed42
