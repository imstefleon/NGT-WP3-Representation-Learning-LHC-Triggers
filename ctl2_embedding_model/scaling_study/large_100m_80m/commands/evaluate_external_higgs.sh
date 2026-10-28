#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 RUN_DIR [EVENTS_PER_CLASS]" >&2
  exit 2
fi

RUN_DIR="$1"
EVENTS_PER_CLASS="${2:-20000}"

cd "$(dirname "$0")/../../.."
export PYTHONPATH="${PWD}/ngt_model/src:${PYTHONPATH:-}"
export MPLBACKEND=Agg

python -m ngt_model.evaluation.external_higgs_anomaly \
  --run-dir "${RUN_DIR}" \
  --signals hh4b,hh,vbfh,ggh \
  --n-background-per-class "${EVENTS_PER_CLASS}" \
  --n-signal-per-class "${EVENTS_PER_CLASS}" \
  --device cuda \
  --batch-size 128 \
  --tsne-max-events-per-class 2000 \
  --tsne-perplexity 40 \
  --tsne-n-iter 1000 \
  --raise-errors
