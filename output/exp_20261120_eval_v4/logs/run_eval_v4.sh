#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export PYTHONPATH="/content/AmbiLAGE"
REPO="/content/AmbiLAGE"
DRIVE_ROOT="/content/drive/MyDrive/AmbiLAGE_backup"
EXP_REL="output/exp_20261120_eval_v4"
EXP_DIR="${REPO}/${EXP_REL}"
LOG_DIR="${EXP_DIR}/logs"
STATUS_FILE="${LOG_DIR}/eval_v4.status"
PID_FILE="${LOG_DIR}/eval_v4.pid"
PY="${REPO}/.venv/bin/python"

cd "${REPO}"
mkdir -p "${LOG_DIR}" "${EXP_DIR}/eval" "${EXP_DIR}/memory_vectors" "${EXP_DIR}/defended_cache"
echo "$$" > "${PID_FILE}"

sync_drive() {
  mkdir -p "${DRIVE_ROOT}/${EXP_REL}" "${DRIVE_ROOT}/src" "${DRIVE_ROOT}/tools" "${DRIVE_ROOT}/docs" "${DRIVE_ROOT}/resources/visual_inception_evaluation"
  rsync -rt --inplace "${EXP_DIR}/" "${DRIVE_ROOT}/${EXP_REL}/" || true
  rsync -rt --inplace --exclude '__pycache__' "${REPO}/src/" "${DRIVE_ROOT}/src/" || true
  rsync -rt --inplace --exclude '__pycache__' "${REPO}/tools/" "${DRIVE_ROOT}/tools/" || true
  rsync -rt --inplace "${REPO}/docs/" "${DRIVE_ROOT}/docs/" || true
  rsync -rt --inplace "${REPO}/resources/visual_inception_evaluation/" "${DRIVE_ROOT}/resources/visual_inception_evaluation/" || true
}

SYNC_PID=""
cleanup() {
  local rc=$?
  if [[ -n "${SYNC_PID}" ]]; then
    kill "${SYNC_PID}" 2>/dev/null || true
    wait "${SYNC_PID}" 2>/dev/null || true
  fi
  if [[ ${rc} -ne 0 ]]; then
    printf '%s | FAILED(rc=%s)\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${rc}" > "${STATUS_FILE}"
  fi
  sync_drive
}
trap cleanup EXIT

(
  while true; do
    sleep 600
    sync_drive
  done
) </dev/null >/dev/null 2>&1 &
SYNC_PID=$!

update_status() {
  printf '%s | %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" > "${STATUS_FILE}"
}

now_sec() {
  "${PY}" -c "import time; print(f'{time.perf_counter():.6f}')"
}

elapsed_sec() {
  "${PY}" -c "print(f'{float($2) - float($1):.6f}')"
}

update_status "RUNNING:fast_evaluate_panel_v4"
echo "[START] $(date -u +%Y-%m-%dT%H:%M:%SZ) eval_v4 (AmbiLAGE + VI-0.8/0.2 across training_oracle_v4, overlap_30_v4, zero_overlap_v4)"
T0=$(now_sec)
"${PY}" -u -X utf8 tools/fast_evaluate_panel.py eval-suites \
  --spec "ambilage:ambilage:${REPO}/output/ambilage_single_clip_l14_9groups_20261118" \
  --spec "vi_08_02:visual_inception:${REPO}/output/exp_20261120_vi_08_02_9groups" \
  --suites training_oracle_v4 overlap_30_v4 zero_overlap_v4 \
  --output-dir "${EXP_DIR}/eval" \
  --key-mode image \
  --defense-mode none \
  --defended-cache-dir "${EXP_DIR}/defended_cache" \
  --memory-vectors-dir "${EXP_DIR}/memory_vectors" \
  --batch-size 16 \
  --text-batch-size 64 \
  --verify-mode in_process \
  --device cuda
T1=$(now_sec)
EVAL_V4_SEC=$(elapsed_sec "${T0}" "${T1}")
echo "[DONE] $(date -u +%Y-%m-%dT%H:%M:%SZ) eval_v4 elapsed=${EVAL_V4_SEC}s"

"${PY}" -c "
import json
json.dump({'eval_v4_sec': float('${EVAL_V4_SEC}')}, open('${LOG_DIR}/timings.json', 'w', encoding='utf-8'), indent=2)
"

update_status "SUMMARIZE"
"${PY}" -u -X utf8 "${LOG_DIR}/summarize_eval_v4.py"
update_status "COMPLETED"
echo "[ALL_DONE] $(date -u +%Y-%m-%dT%H:%M:%SZ)"
