#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export PYTHONPATH="/content/AmbiLAGE"
REPO="/content/AmbiLAGE"
DRIVE_ROOT="/content/drive/MyDrive/AmbiLAGE_backup"
OUT_REL="output/eval_9groups_20261119"
STAGE2_REL="${OUT_REL}/stage2_defenses"
STAGE2_DIR="${REPO}/${STAGE2_REL}"
DEFENDED_DIR="${STAGE2_DIR}/defended_cache"
VEC_BATCH_DIR="${STAGE2_DIR}/memory_vectors_eval_batch"
BENCH_JSON="${STAGE2_DIR}/logs/batch_micro_benchmark.json"
STATUS_FILE="${STAGE2_DIR}/logs/stage2_fast.status"
PY="${REPO}/.venv/bin/python"

cd "${REPO}"
mkdir -p "${STAGE2_DIR}/logs" "${VEC_BATCH_DIR}"

sync_results_drive() {
  mkdir -p "${DRIVE_ROOT}/${STAGE2_REL}" "${DRIVE_ROOT}/src" "${DRIVE_ROOT}/tools"
  rsync -rt --inplace --exclude 'defended_cache/' --exclude 'memory_vectors/' --exclude 'memory_vectors_eval_batch/' \
        "${STAGE2_DIR}/" "${DRIVE_ROOT}/${STAGE2_REL}/" || true
  rsync -rt --inplace --exclude '__pycache__' "${REPO}/src/" "${DRIVE_ROOT}/src/" || true
  rsync -rt --inplace --exclude '__pycache__' "${REPO}/tools/" "${DRIVE_ROOT}/tools/" || true
  rsync -rt --inplace "${REPO}/.gitignore" "${DRIVE_ROOT}/.gitignore" || true
}

sync_all_drive() {
  sync_results_drive
  mkdir -p "${DRIVE_ROOT}/${STAGE2_REL}/defended_cache" "${DRIVE_ROOT}/${STAGE2_REL}/memory_vectors_eval_batch"
  rsync -rt --ignore-existing "${DEFENDED_DIR}/" "${DRIVE_ROOT}/${STAGE2_REL}/defended_cache/" || true
  rsync -rt --ignore-existing "${VEC_BATCH_DIR}/" "${DRIVE_ROOT}/${STAGE2_REL}/memory_vectors_eval_batch/" || true
}

SYNC_PID=""
cleanup() {
  if [[ -n "${SYNC_PID}" ]]; then
    kill "${SYNC_PID}" 2>/dev/null || true
    wait "${SYNC_PID}" 2>/dev/null || true
  fi
  sync_all_drive
}
trap cleanup EXIT

(
  while true; do
    sleep 120
    sync_all_drive
  done
) </dev/null >/dev/null 2>&1 &
SYNC_PID=$!

update_status() {
  printf '%s | %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" > "${STATUS_FILE}"
}

if [[ ! -f "${BENCH_JSON}" ]]; then
  echo "[ERROR] Missing Gate 1 micro-benchmark file: ${BENCH_JSON}" >&2
  update_status "FAILED:MISSING_BENCHMARK"
  exit 1
fi

update_status "RUNNING_FAST_PANEL"
echo "[FAST STAGE 2 START] $(date -u +%Y-%m-%dT%H:%M:%SZ)"
"${PY}" -u -X utf8 -m src.experiments.fast_stage2_panel run \
  --stage2-dir "${STAGE2_DIR}" \
  --defense-mode panel_with_sd21 \
  --defended-cache-dir "${DEFENDED_DIR}" \
  --memory-vectors-dir "${VEC_BATCH_DIR}" \
  --batch-config "${BENCH_JSON}" \
  --num-workers 4 \
  --device cuda

update_status "SUMMARIZING"
"${PY}" -u -X utf8 "${STAGE2_DIR}/logs/summarize_stage2.py"
sync_all_drive
update_status "COMPLETED"
echo "[ALL STAGE 2 FAST COMPLETED] $(date -u +%Y-%m-%dT%H:%M:%SZ)"
