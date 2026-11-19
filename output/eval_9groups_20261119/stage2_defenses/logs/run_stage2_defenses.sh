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
VEC_DIR="${STAGE2_DIR}/memory_vectors"
STATUS_FILE="${STAGE2_DIR}/logs/stage2.status"
PREWARM_DONE="${STAGE2_DIR}/logs/prewarm.done"
PY="${REPO}/.venv/bin/python"

cd "${REPO}"
mkdir -p "${STAGE2_DIR}/logs" "${DEFENDED_DIR}" "${VEC_DIR}"

# Seed clean/none image vectors computed in Stage 1 into Stage 2's own cache without modifying Stage 1.
if [[ -d "${REPO}/${OUT_REL}/stage1_none/memory_vectors" ]]; then
  rsync -rt --ignore-existing "${REPO}/${OUT_REL}/stage1_none/memory_vectors/" "${VEC_DIR}/" || true
fi

sync_results_drive() {
  mkdir -p "${DRIVE_ROOT}/${STAGE2_REL}" "${DRIVE_ROOT}/${OUT_REL}/logs"
  rsync -rt --inplace --exclude 'defended_cache/' --exclude 'memory_vectors/' \
        "${STAGE2_DIR}/" "${DRIVE_ROOT}/${STAGE2_REL}/" || true
}

sync_all_drive() {
  sync_results_drive
  mkdir -p "${DRIVE_ROOT}/${STAGE2_REL}/defended_cache" "${DRIVE_ROOT}/${STAGE2_REL}/memory_vectors"
  rsync -rt --ignore-existing "${DEFENDED_DIR}/" "${DRIVE_ROOT}/${STAGE2_REL}/defended_cache/" || true
  rsync -rt --ignore-existing "${VEC_DIR}/" "${DRIVE_ROOT}/${STAGE2_REL}/memory_vectors/" || true
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

if [[ ! -f "${PREWARM_DONE}" ]]; then
  update_status "PREWARMING_DEFENDED_CACHE"
  echo "[PREWARM START] $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  "${PY}" -u -X utf8 "${STAGE2_DIR}/logs/prewarm_defended_cache.py"
  touch "${PREWARM_DONE}"
  echo "[PREWARM DONE] $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  sync_all_drive
else
  echo "[PREWARM SKIP] already done"
fi

run_panel() {
  local name="$1"
  local images_root="$2"
  local method="$3"
  local suite="$4"
  local target_dir="${STAGE2_DIR}/${name}"
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
    --defense-mode panel_with_sd21 \
    --device cuda \
    --defended-cache-dir "${DEFENDED_DIR}" \
    --memory-vectors-dir "${VEC_DIR}"
  echo "[DONE] $(date -u +%Y-%m-%dT%H:%M:%SZ) ${name}"
  sync_results_drive
}

run_panel "ambilage_training_oracle" "${REPO}/output/ambilage_single_clip_l14_9groups_20261118" "ambilage" "training_oracle"
run_panel "vi_training_oracle" "${REPO}/output/vi_single_clip_l14_9groups_20261118" "visual_inception" "training_oracle"
run_panel "ambilage_overlap_30" "${REPO}/output/ambilage_single_clip_l14_9groups_20261118" "ambilage" "overlap_30"
run_panel "vi_overlap_30" "${REPO}/output/vi_single_clip_l14_9groups_20261118" "visual_inception" "overlap_30"
run_panel "ambilage_zero_overlap" "${REPO}/output/ambilage_single_clip_l14_9groups_20261118" "ambilage" "zero_overlap"
run_panel "vi_zero_overlap" "${REPO}/output/vi_single_clip_l14_9groups_20261118" "visual_inception" "zero_overlap"

update_status "SUMMARIZING"
"${PY}" -u -X utf8 "${STAGE2_DIR}/logs/summarize_stage2.py"
sync_all_drive
update_status "COMPLETED"
echo "[ALL STAGE 2 COMPLETED] $(date -u +%Y-%m-%dT%H:%M:%SZ)"
