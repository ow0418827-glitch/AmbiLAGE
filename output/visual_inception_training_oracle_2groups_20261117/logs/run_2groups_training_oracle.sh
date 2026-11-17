#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/content/AmbiLAGE"
cd "$PROJECT_ROOT"

export PYTHONUTF8=1
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

OUT_ROOT="output/visual_inception_training_oracle_2groups_20261117"
LOG_DIR="${PROJECT_ROOT}/${OUT_ROOT}/logs"
STATUS_FILE="${LOG_DIR}/pipeline.status"
mkdir -p "${LOG_DIR}"

echo "RUNNING generation 07_shampoo,09_toothbrush at $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${STATUS_FILE}"
echo "=== Phase 1/2: Generate visual_inception (07_shampoo, 09_toothbrush) at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

.venv/bin/python -u -B -X utf8 tools/run_experiment.py generate \
  --group 07_shampoo,09_toothbrush \
  --method visual_inception \
  --output "${OUT_ROOT}" \
  --device cuda

for g in "07_shampoo" "09_toothbrush"; do
  if [[ ! -f "${PROJECT_ROOT}/${OUT_ROOT}/${g}/visual_inception/FROZEN_IMAGE.json" ]]; then
    echo "FAILED missing FROZEN_IMAGE.json for ${g} at $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${STATUS_FILE}"
    exit 1
  fi
done

echo "RUNNING evaluation training_oracle at $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${STATUS_FILE}"
echo "=== Phase 2/2: Evaluate training_oracle (50/30/20 text, 4 encoders, 2-group simultaneous pool replacement) at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

.venv/bin/python -u -B -X utf8 "${LOG_DIR}/eval_training_oracle.py" \
  --output-root "${OUT_ROOT}" \
  --device cuda

echo "COMPLETED at $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${STATUS_FILE}"
echo "=== ALL DONE at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
