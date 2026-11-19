#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export PYTHONPATH="/content/AmbiLAGE"
REPO="/content/AmbiLAGE"
DRIVE_ROOT="/content/drive/MyDrive/AmbiLAGE_backup"
OUT_REL="output/eval_9groups_20261119"
STAGE_REL="${OUT_REL}/stage1_none"
VEC_DIR="${REPO}/${STAGE_REL}/memory_vectors"
STATUS_FILE="${REPO}/${OUT_REL}/logs/stage1.status"
PY="${REPO}/.venv/bin/python"

cd "${REPO}"
mkdir -p "${REPO}/${STAGE_REL}" "${VEC_DIR}" "${REPO}/${OUT_REL}/logs"

sync_drive() {
  mkdir -p "${DRIVE_ROOT}/${OUT_REL}" "${DRIVE_ROOT}/src" "${DRIVE_ROOT}/tools" \
           "${DRIVE_ROOT}/resources/visual_inception_evaluation" "${DRIVE_ROOT}/output/deployment"
  rsync -rt --inplace "${REPO}/${OUT_REL}/" "${DRIVE_ROOT}/${OUT_REL}/" || true
  rsync -rt --inplace --exclude '__pycache__' "${REPO}/src/" "${DRIVE_ROOT}/src/" || true
  rsync -rt --inplace --exclude '__pycache__' "${REPO}/tools/" "${DRIVE_ROOT}/tools/" || true
  rsync -rt --inplace "${REPO}/resources/visual_inception_evaluation/category_query_overlap_30_v1/" \
        "${DRIVE_ROOT}/resources/visual_inception_evaluation/category_query_overlap_30_v1/" || true
  rsync -rt --inplace "${REPO}/output/deployment/" "${DRIVE_ROOT}/output/deployment/" || true
}

SYNC_PID=""
cleanup() {
  if [[ -n "${SYNC_PID}" ]]; then
    kill "${SYNC_PID}" 2>/dev/null || true
    wait "${SYNC_PID}" 2>/dev/null || true
  fi
  sync_drive
}
trap cleanup EXIT

(
  while true; do
    sleep 45
    sync_drive
  done
) </dev/null >/dev/null 2>&1 &
SYNC_PID=$!

update_status() {
  printf '%s | %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" > "${STATUS_FILE}"
}

run_panel() {
  local name="$1"
  local images_root="$2"
  local method="$3"
  local suite="$4"
  local target_dir="${REPO}/${STAGE_REL}/${name}"
  if [[ -f "${target_dir}/transfer_panel_summary.json" ]]; then
    echo "[SKIP] ${name} already has transfer_panel_summary.json"
    return 0
  fi
  update_status "RUNNING:${name}"
  echo "[START] $(date -u +%Y-%m-%dT%H:%M:%SZ) ${name}"
  "${PY}" -u -X utf8 tools/evaluate_visual_inception_category.py evaluate-transfer-panel \
    --images-root "${images_root}" \
    --output "${target_dir}" \
    --method "${method}" \
    --query-suite "${suite}" \
    --key-mode image \
    --defense-mode none \
    --device cuda \
    --memory-vectors-dir "${VEC_DIR}"
  echo "[DONE] $(date -u +%Y-%m-%dT%H:%M:%SZ) ${name}"
  sync_drive
}

update_status "STARTED"
run_panel "ambilage_zero_overlap" "${REPO}/output/ambilage_single_clip_l14_9groups_20261118" "ambilage" "zero_overlap"
run_panel "vi_zero_overlap" "${REPO}/output/vi_single_clip_l14_9groups_20261118" "visual_inception" "zero_overlap"
run_panel "ambilage_overlap_30" "${REPO}/output/ambilage_single_clip_l14_9groups_20261118" "ambilage" "overlap_30"
run_panel "vi_overlap_30" "${REPO}/output/vi_single_clip_l14_9groups_20261118" "visual_inception" "overlap_30"

update_status "SUMMARIZING"
"${PY}" -u -X utf8 "${REPO}/${OUT_REL}/logs/summarize_stage1.py"
sync_drive
update_status "COMPLETED"
echo "[ALL STAGE 1 COMPLETED] $(date -u +%Y-%m-%dT%H:%M:%SZ)"
