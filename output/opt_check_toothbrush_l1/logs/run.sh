#!/usr/bin/env bash
set -euo pipefail
cd /content/AmbiLAGE
export PYTHONUNBUFFERED=1
OUT_ROOT="output/opt_check_toothbrush_l1"
echo "=== [L1+VAE] Generate 09_toothbrush at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
.venv/bin/python -u -X utf8 tools/run_experiment.py generate \
  --group 09_toothbrush \
  --method ambilage \
  --output "${OUT_ROOT}" \
  --device cuda
echo "=== [L1+VAE] Evaluate Env 4 09_toothbrush at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
.venv/bin/python -u -X utf8 "${OUT_ROOT}/logs/eval_group_env4.py" \
  --output-root "${OUT_ROOT}" \
  --groups 09_toothbrush \
  --device cuda
echo "=== [L1+VAE] COMPLETE at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
