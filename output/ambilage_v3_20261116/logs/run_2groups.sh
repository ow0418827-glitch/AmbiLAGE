#!/usr/bin/env bash
set -euo pipefail
cd /content/AmbiLAGE
export PYTHONUNBUFFERED=1

OUT_ROOT="output/ambilage_v3_20261116"
LOG_DIR="/content/AmbiLAGE/${OUT_ROOT}/logs"
DONE_FILE="${LOG_DIR}/pipeline.status"
PROGRESS_FILE="${LOG_DIR}/group_progress.log"

TARGET_GROUPS=("09_toothbrush" "06_jeans")
FAILURES=0

: > "${PROGRESS_FILE}"

for g in "${TARGET_GROUPS[@]}"; do
  echo "RUNNING generation ${g} at $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${DONE_FILE}"
  echo "=== [${g}] Step 1/2: Generate (${g}/ambilage) at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  set +e
  .venv/bin/python -u -X utf8 tools/run_experiment.py generate \
    --group "${g}" \
    --method ambilage \
    --output "${OUT_ROOT}" \
    --device cuda
  gen_rc=$?
  set -e

  if [[ ${gen_rc} -ne 0 ]] || [[ ! -f "/content/AmbiLAGE/${OUT_ROOT}/${g}/ambilage/FROZEN_IMAGE.json" ]]; then
    echo "[ERROR] ${g} generation failed (rc=${gen_rc}) at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "${g} GENERATE_FAILED rc=${gen_rc} at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${PROGRESS_FILE}"
    FAILURES=$((FAILURES + 1))
    continue
  fi
  echo "${g} GENERATED at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${PROGRESS_FILE}"

  echo "RUNNING evaluation ${g} at $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${DONE_FILE}"
  echo "=== [${g}] Step 2/2: Evaluate Env 4 (${g}/ambilage, none defense) at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  set +e
  .venv/bin/python -u -X utf8 "${LOG_DIR}/eval_group_env4.py" \
    --output-root "${OUT_ROOT}" \
    --groups "${g}" \
    --device cuda
  eval_rc=$?
  set -e

  if [[ ${eval_rc} -ne 0 ]]; then
    echo "[ERROR] ${g} evaluation failed (rc=${eval_rc}) at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "${g} EVAL_FAILED rc=${eval_rc} at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${PROGRESS_FILE}"
    FAILURES=$((FAILURES + 1))
    continue
  fi
  echo "${g} EVALUATED_AND_VERIFIED at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${PROGRESS_FILE}"
done

if [[ ${FAILURES} -eq 0 ]]; then
  echo "=== 2 GROUPS (09_toothbrush, 06_jeans) COMPLETE at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  echo "COMPLETED at $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${DONE_FILE}"
  exit 0
else
  echo "=== FINISHED WITH ${FAILURES} GROUP FAILURE(S) at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  echo "COMPLETED_WITH_FAILURES count=${FAILURES} at $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${DONE_FILE}"
  exit 1
fi
