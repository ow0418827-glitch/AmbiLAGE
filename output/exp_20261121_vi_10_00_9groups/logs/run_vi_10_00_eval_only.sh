#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export PYTHONPATH="/content/AmbiLAGE"
REPO="/content/AmbiLAGE"
DRIVE_ROOT="/content/drive/MyDrive/AmbiLAGE_backup"
GEN_REL="output/exp_20261121_vi_10_00_9groups"
GEN_DIR="${REPO}/${GEN_REL}"
EVAL_REL="output/exp_20261121_eval_v4_cleaned"
EVAL_DIR="${REPO}/${EVAL_REL}"
LOG_DIR="${GEN_DIR}/logs"
STATUS_FILE="${LOG_DIR}/vi_10_00_eval.status"
PID_FILE="${LOG_DIR}/vi_10_00_eval.pid"
SYNC_LOG="${LOG_DIR}/drive_sync.log"
VEC_DIR="${EVAL_DIR}/memory_vectors_vi_10_00"
GEN_SEC="189.090387"
PY="${REPO}/.venv/bin/python"

cd "${REPO}"
mkdir -p "${LOG_DIR}" "${EVAL_DIR}/eval" "${VEC_DIR}" "${EVAL_DIR}/defended_cache"
echo "$$" > "${PID_FILE}"

sync_drive() {
  local ts
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  {
    echo "[SYNC_START] ${ts}"
    mkdir -p \
      "${DRIVE_ROOT}/${GEN_REL}" \
      "${DRIVE_ROOT}/${EVAL_REL}" \
      "${DRIVE_ROOT}/resources/visual_inception_evaluation"
    rsync -rt --inplace "${GEN_DIR}/" "${DRIVE_ROOT}/${GEN_REL}/"
    rsync -rt --inplace "${EVAL_DIR}/" "${DRIVE_ROOT}/${EVAL_REL}/"
    rsync -rt --inplace "${REPO}/resources/visual_inception_evaluation/" "${DRIVE_ROOT}/resources/visual_inception_evaluation/"
    echo "[SYNC_DONE] $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } >> "${SYNC_LOG}" 2>&1 || echo "[SYNC_WARN] $(date -u +%Y-%m-%dT%H:%M:%SZ) rsync failed" >> "${SYNC_LOG}"
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

update_status "EVAL:VI_10_00_CLEANED_V4"
echo "[START] $(date -u +%Y-%m-%dT%H:%M:%SZ) EVAL VI 1.0/0.0 across cleaned training_oracle_v4, overlap_30_v4, zero_overlap_v4 (clean vec_dir)"
T2=$(now_sec)
"${PY}" -u -X utf8 tools/fast_evaluate_panel.py eval-suites \
  --spec "vi_10_00:visual_inception:${GEN_DIR}" \
  --suites training_oracle_v4 overlap_30_v4 zero_overlap_v4 \
  --output-dir "${EVAL_DIR}/eval" \
  --key-mode image \
  --defense-mode none \
  --defended-cache-dir "${EVAL_DIR}/defended_cache" \
  --memory-vectors-dir "${VEC_DIR}" \
  --batch-size 16 \
  --text-batch-size 64 \
  --verify-mode in_process \
  --device cuda
T3=$(now_sec)
EVAL_SEC=$(elapsed_sec "${T2}" "${T3}")
echo "[DONE] $(date -u +%Y-%m-%dT%H:%M:%SZ) EVAL VI 1.0/0.0 elapsed=${EVAL_SEC}s"

"${PY}" -c "
import json
from pathlib import Path
eval_timings_path = Path('${EVAL_DIR}/logs/timings.json')
data = json.loads(eval_timings_path.read_text(encoding='utf-8')) if eval_timings_path.is_file() else {}
data['vi_10_00_gen_sec'] = float('${GEN_SEC}')
data['vi_10_00_eval_sec'] = float('${EVAL_SEC}')
eval_timings_path.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
Path('${LOG_DIR}/timings.json').write_text(json.dumps({
    'vi_10_00_gen_sec': float('${GEN_SEC}'),
    'vi_10_00_eval_sec': float('${EVAL_SEC}'),
}, indent=2) + '\n', encoding='utf-8')
"

update_status "SUMMARIZE"
"${PY}" -u -X utf8 "${EVAL_DIR}/logs/summarize_eval_v4_cleaned.py"
update_status "COMPLETED"
echo "[ALL_DONE] $(date -u +%Y-%m-%dT%H:%M:%SZ)"
