#!/usr/bin/env bash
set -euo pipefail

cd /content/AmbiLAGE
export PYTHONUNBUFFERED=1

OUT_ROOT="output/ambilage_toothbrush_20261115"
LOG_DIR="/content/AmbiLAGE/${OUT_ROOT}/logs"
DONE_FILE="${LOG_DIR}/lucid_reuse_env4.status"
LUCID_FROZEN="/content/AmbiLAGE/output/ambilage_2groups_20261114/09_toothbrush/lucid/FROZEN_IMAGE.json"

cleanup_err() {
  rc=$?
  echo "[ERROR] Failed at $(date -u +%Y-%m-%dT%H:%M:%SZ) with exit code ${rc}"
  echo "FAILED rc=${rc} at $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${DONE_FILE}"
  exit "${rc}"
}
trap cleanup_err ERR

echo "RUNNING lucid_reuse evaluation at $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${DONE_FILE}"
echo "=== Evaluate 09_toothbrush/lucid_reuse under Env 4 (feature text + target image, none defense) at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
.venv/bin/python -u -X utf8 "${LOG_DIR}/eval_toothbrush_env4.py" \
  --output-root "${OUT_ROOT}" \
  --lucid-frozen "${LUCID_FROZEN}" \
  --device cuda

echo "=== LUCID REUSE ENV4 EVALUATION COMPLETE at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "COMPLETED at $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${DONE_FILE}"
