#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-A}"
SEED=42
RUN_NAME="scaling_4class_80m_100m_noaug_varcov_seed42"
if [[ "${RUN_ID}" == "B" ]]; then
  SEED=43
  RUN_NAME="scaling_4class_80m_100m_noaug_varcov_seed43"
fi

cd "$(dirname "$0")/../../../.."
export PYTHONPATH="${PWD}/ngt_model/src:${PYTHONPATH:-}"
export MPLBACKEND=Agg

torchrun --standalone --nproc_per_node=4 -m ngt_model.training.train_mha \
  --data-source collide2v_updated \
  --data-loading-mode rank_sharded_materialized \
  --class-names minbias,qcd,dy,top \
  --n-events-per-class 20000000 \
  --n-constituents 100 \
  --latent-dim 256 \
  --ff-dim 2304 \
  --num-heads 16 \
  --num-attention-blocks 4 \
  --pooling-mode attention \
  --classifier-head mlp \
  --loss-mode ce \
  --latent-var-reg-weight 0.01 \
  --latent-cov-reg-weight 0.001 \
  --augmentation-copies 0 \
  --lr 3e-5 \
  --weight-decay 1e-4 \
  --batch-size 4 \
  --num-workers 4 \
  --epochs 50 \
  --early-stopping-patience 10 \
  --grad-clip-norm 1.0 \
  --fail-on-nonfinite-loss \
  --seed "${SEED}" \
  --output-root outputs/scaling_study/large_100m_80m/full/run_${RUN_ID} \
  --run-name "${RUN_NAME}" \
  --mlflow-enable \
  --mlflow-tracking-uri https://ml.cern.ch/models \
  --mlflow-experiment ngt_model_scaling_100m_80m \
  --mlflow-run-name "${RUN_NAME}"
