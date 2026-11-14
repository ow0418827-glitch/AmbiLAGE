#!/usr/bin/env bash
set -euo pipefail

cd /content/AmbiLAGE
export PYTHONUNBUFFERED=1

LOG_DIR="/root/.gemini/antigravity/brain/e0388b53-0e9a-492d-97fa-c6472a1cb191/scratch"
DONE_FILE="${LOG_DIR}/gen_toothbrush_lucid.status"

cleanup_err() {
  rc=$?
  echo "[ERROR] Failed at $(date -u +%Y-%m-%dT%H:%M:%SZ) with exit code ${rc}"
  echo "FAILED rc=${rc} at $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${DONE_FILE}"
  exit "${rc}"
}
trap cleanup_err ERR

echo "RUNNING started at $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${DONE_FILE}"
echo "=== [1/3] Clean interrupted 09_toothbrush/ambilage and restart at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
rm -rf /content/AmbiLAGE/output/ambilage_choice1_2groups_20261114/09_toothbrush/ambilage
.venv/bin/python -u -X utf8 tools/run_experiment.py generate --group 09_toothbrush --method ambilage --output output/ambilage_choice1_2groups_20261114 --device cuda

echo "=== [2/3] Generate 07_shampoo/lucid at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
.venv/bin/python -u -X utf8 tools/run_experiment.py generate --group 07_shampoo --method lucid --output output/ambilage_choice1_2groups_20261114 --device cuda

echo "=== [3/3] Generate 09_toothbrush/lucid at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
.venv/bin/python -u -X utf8 tools/run_experiment.py generate --group 09_toothbrush --method lucid --output output/ambilage_choice1_2groups_20261114 --device cuda

echo "=== ALL GENERATION COMPLETE at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "COMPLETED at $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${DONE_FILE}"
