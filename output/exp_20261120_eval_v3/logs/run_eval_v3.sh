#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export PYTHONPATH="/content/AmbiLAGE"
REPO="/content/AmbiLAGE"
DRIVE_ROOT="/content/drive/MyDrive/AmbiLAGE_backup"
EXP_REL="output/exp_20261120_eval_v3"
VEC_DIR="${REPO}/output/eval_9groups_20261119/stage1_none/memory_vectors"
LOG_DIR="${REPO}/${EXP_REL}/logs"
STATUS_FILE="${LOG_DIR}/eval_v3.status"
PID_FILE="${LOG_DIR}/eval_v3.pid"
PY="${REPO}/.venv/bin/python"

cd "${REPO}"
mkdir -p "${LOG_DIR}" "${REPO}/${EXP_REL}/eval" "${VEC_DIR}"
echo "$$" > "${PID_FILE}"

sync_drive() {
  mkdir -p "${DRIVE_ROOT}/${EXP_REL}" "${DRIVE_ROOT}/src" "${DRIVE_ROOT}/tools" "${DRIVE_ROOT}/resources/visual_inception_evaluation"
  rsync -rt --inplace "${REPO}/${EXP_REL}/" "${DRIVE_ROOT}/${EXP_REL}/" || true
  rsync -rt --inplace --exclude '__pycache__' "${REPO}/src/" "${DRIVE_ROOT}/src/" || true
  rsync -rt --inplace --exclude '__pycache__' "${REPO}/tools/" "${DRIVE_ROOT}/tools/" || true
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

run_eval() {
  local tag="$1"
  local method="$2"
  local images_root="$3"
  local suite="$4"
  local target_dir="${REPO}/${EXP_REL}/eval/${tag}_${suite}"
  if [[ -f "${target_dir}/transfer_panel_summary.json" ]]; then
    echo "[SKIP] ${tag}_${suite} already evaluated"
    return 0
  fi
  update_status "EVAL:${tag}_${suite}"
  echo "[START] $(date -u +%Y-%m-%dT%H:%M:%SZ) EVAL ${tag} on ${suite}"
  "${PY}" -u -X utf8 tools/evaluate_visual_inception_category.py evaluate-transfer-panel \
    --images-root "${images_root}" \
    --output "${target_dir}" \
    --method "${method}" \
    --group all \
    --replacement-groups all \
    --query-suite "${suite}" \
    --key-mode image \
    --defense-mode none \
    --device cuda \
    --memory-vectors-dir "${VEC_DIR}"
  echo "[DONE] $(date -u +%Y-%m-%dT%H:%M:%SZ) EVAL ${tag} on ${suite}"
  sync_drive
}

for suite in training_oracle_v3 overlap_30_v3 zero_overlap_v3; do
  run_eval "ambilage" "ambilage" "${REPO}/output/ambilage_single_clip_l14_9groups_20261118" "${suite}"
  run_eval "vi_08_02" "visual_inception" "${REPO}/output/exp_20261120_vi_08_02_9groups" "${suite}"
done

update_status "SUMMARIZE"
"${PY}" -u -X utf8 "${LOG_DIR}/summarize_eval_v3.py"
update_status "COMPLETED"
echo "[ALL_DONE] $(date -u +%Y-%m-%dT%H:%M:%SZ)"
