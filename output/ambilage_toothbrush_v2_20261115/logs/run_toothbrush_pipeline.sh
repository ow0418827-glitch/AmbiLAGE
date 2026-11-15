#!/usr/bin/env bash
set -euo pipefail

cd /content/AmbiLAGE
export PYTHONUNBUFFERED=1

OUT_ROOT="output/ambilage_toothbrush_v2_20261115"
LOG_DIR="/content/AmbiLAGE/${OUT_ROOT}/logs"
DONE_FILE="${LOG_DIR}/pipeline.status"

cleanup_err() {
  rc=$?
  echo "[ERROR] Failed at $(date -u +%Y-%m-%dT%H:%M:%SZ) with exit code ${rc}"
  echo "FAILED rc=${rc} at $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${DONE_FILE}"
  exit "${rc}"
}
trap cleanup_err ERR

echo "RUNNING generation at $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${DONE_FILE}"
echo "=== [1/2] Generate 09_toothbrush/ambilage (updated OKLab/bg_color_shift AmbiLAGE) at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
.venv/bin/python -u -X utf8 tools/run_experiment.py generate \
  --group 09_toothbrush \
  --method ambilage \
  --output "${OUT_ROOT}" \
  --device cuda

echo "RUNNING evaluation at $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${DONE_FILE}"
echo "=== [2/2] Evaluate 09_toothbrush/ambilage under Env 4 (feature text + target image, none defense) at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
.venv/bin/python -u -X utf8 "${LOG_DIR}/eval_toothbrush_env4.py" \
  --output-root "${OUT_ROOT}" \
  --device cuda

echo "=== PIPELINE COMPLETE at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "COMPLETED at $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${DONE_FILE}"
