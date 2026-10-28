#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:-outputs/scaling_study/large_100m_80m/one_24h_100m/runs/2026-08-24/scaling_4class_80m_dmodel2016_ff8064_100m_400ep_noaug_ce_infonce005_lr1e5_seed42}"
EVENTS_PER_CLASS="${2:-2000}"
DEVICE="${3:-cuda}"

cd "$(dirname "$0")/../../.."
export PYTHONPATH="${PWD}/ngt_model/src:${PYTHONPATH:-}"
export MPLBACKEND=Agg

python scaling_study/large_100m_80m/commands/plot_latent_evolution_100m.py \
  --run-dir "${RUN_DIR}" \
  --cache-dir outputs/scaling_study/large_100m_80m/one_24h_100m/cache \
  --events-per-class "${EVENTS_PER_CLASS}" \
  --device "${DEVICE}" \
  --batch-size 256 \
  --shard 0
