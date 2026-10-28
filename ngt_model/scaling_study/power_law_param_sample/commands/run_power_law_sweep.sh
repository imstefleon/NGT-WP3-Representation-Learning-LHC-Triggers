#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../../.."
export PYTHONPATH="${PWD}/ngt_model/src:${PYTHONPATH:-}"
export MPLBACKEND=Agg
PY="${PY:-python}"

COMMON_ARGS=(
  --data-source collide2v_updated
  --class-names minbias,qcd,dy,top
  --n-constituents 100
  --latent-dim 128
  --num-heads 8
  --num-attention-blocks 2
  --pooling-mode attention
  --classifier-head linear
  --loss-mode ce
  --latent-var-reg-weight 0.01
  --latent-cov-reg-weight 0.001
  --augmentation-copies 0
  --lr 3e-5
  --weight-decay 1e-4
  --batch-size 128
  --num-workers 0
  --epochs 50
  --early-stopping-patience 10
  --grad-clip-norm 1.0
  --fail-on-nonfinite-loss
  --seed 42
  --mlflow-enable
  --mlflow-tracking-uri https://ml.cern.ch/models
  --mlflow-experiment ngt_model_param_sample_power_law
)

run_point() {
  local point="$1"
  local events_per_class="$2"
  local ff_dim="$3"
  local run_name="scaling_4class_${point}_ff${ff_dim}_varcov_seed42"
  local output_root="outputs/scaling_study/power_law_param_sample/${point}_ff${ff_dim}"

  "${PY}" -m ngt_model.training.train_mha \
    "${COMMON_ARGS[@]}" \
    --n-events-per-class "${events_per_class}" \
    --ff-dim "${ff_dim}" \
    --output-root "${output_root}" \
    --run-name "${run_name}" \
    --mlflow-run-name "${run_name}"

  local run_dir="${output_root}/runs/$(date +%F)/${run_name}"
  "${PY}" -m ngt_model.evaluation.latent_diagnostics "${run_dir}"
  "${PY}" -m ngt_model.evaluation.external_higgs_anomaly \
    --run-dir "${run_dir}" \
    --signals hh4b,hh,vbfh,ggh \
    --n-background-per-class 2000 \
    --n-signal-per-class 2000 \
    --device cpu \
    --batch-size 256 \
    --tsne-max-events-per-class 600 \
    --tsne-perplexity 40 \
    --tsne-n-iter 1000 \
    --raise-errors
}

run_point plaw_p1e5_n1e4 2500 88
run_point plaw_p5e5_n5e4 12500 208
run_point plaw_p2p5e6_n2p5e5 62500 480
run_point plaw_p1e7_n1e6 250000 968
