#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:-outputs/scaling_study/large_100m_80m/one_24h_100m/runs/2026-08-24/scaling_4class_80m_dmodel2016_ff8064_100m_400ep_noaug_ce_infonce005_lr1e5_seed42}"
EVENTS_PER_CLASS="${2:-2000}"
SIGNALS="${3:-hh4b,vbfh}"
DEVICE="${4:-cuda}"

cd "$(dirname "$0")/../../.."
export PYTHONPATH="${PWD}/ngt_model/src:${PYTHONPATH:-}"
export MPLBACKEND=Agg

for MODEL in "${RUN_DIR}"/models/checkpoint_epoch_*.pt; do
  EPOCH="$(basename "${MODEL}" | sed -E 's/.*epoch_([0-9]+).*/\1/')"
  OUT_SUBDIR="figures/anomaly_injection_epoch_${EPOCH}"
  echo "evaluating anomaly injection for epoch ${EPOCH}: ${MODEL}"
  python -m ngt_model.evaluation.external_higgs_anomaly \
    --run-dir "${RUN_DIR}" \
    --model "${MODEL}" \
    --signals "${SIGNALS}" \
    --n-background-per-class "${EVENTS_PER_CLASS}" \
    --n-signal-per-class "${EVENTS_PER_CLASS}" \
    --device "${DEVICE}" \
    --batch-size 128 \
    --output-subdir "${OUT_SUBDIR}" \
    --tsne-max-events-per-class 600 \
    --tsne-perplexity 40 \
    --tsne-n-iter 1000 \
    --raise-errors
  if [[ ! -f "${RUN_DIR}/${OUT_SUBDIR}/external_higgs_anomaly_metrics.csv" ]]; then
    echo "missing metrics for epoch ${EPOCH}: ${RUN_DIR}/${OUT_SUBDIR}/external_higgs_anomaly_metrics.csv" >&2
    exit 1
  fi
done

python scaling_study/large_100m_80m/commands/plot_anomaly_injection_evolution_100m.py \
  --run-dir "${RUN_DIR}"
