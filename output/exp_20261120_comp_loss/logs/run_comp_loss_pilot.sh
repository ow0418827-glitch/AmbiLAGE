#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export PYTHONPATH="/content/AmbiLAGE"
REPO="/content/AmbiLAGE"
DRIVE_ROOT="/content/drive/MyDrive/AmbiLAGE_backup"
EXP_REL="output/exp_20261120_comp_loss"
VEC_DIR="${REPO}/output/eval_9groups_20261119/stage1_none/memory_vectors"
STATUS_FILE="${REPO}/${EXP_REL}/logs/comp_loss.status"
PID_FILE="${REPO}/${EXP_REL}/logs/comp_loss.pid"
PY="${REPO}/.venv/bin/python"
TARGET_GROUPS="04_curtain,07_shampoo,09_toothbrush"

cd "${REPO}"
mkdir -p "${REPO}/${EXP_REL}/logs" "${REPO}/${EXP_REL}/eval" "${VEC_DIR}"
echo "$$" > "${PID_FILE}"

sync_drive() {
  mkdir -p "${DRIVE_ROOT}/${EXP_REL}" "${DRIVE_ROOT}/src" "${DRIVE_ROOT}/tools"
  rsync -rt --inplace "${REPO}/${EXP_REL}/" "${DRIVE_ROOT}/${EXP_REL}/" || true
  rsync -rt --inplace --exclude '__pycache__' "${REPO}/src/" "${DRIVE_ROOT}/src/" || true
  rsync -rt --inplace --exclude '__pycache__' "${REPO}/tools/" "${DRIVE_ROOT}/tools/" || true
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

run_gen() {
  local name="$1"
  local cfg="$2"
  local out_dir="${REPO}/${EXP_REL}/${name}"
  if [[ -f "${out_dir}/09_toothbrush/visual_inception/FROZEN_IMAGE.json" ]]; then
    echo "[SKIP] generation ${name} already complete"
    return 0
  fi
  update_status "GEN:${name}"
  echo "[START] $(date -u +%Y-%m-%dT%H:%M:%SZ) GEN ${name}"
  "${PY}" -u -X utf8 tools/run_experiment.py generate \
    --group "${TARGET_GROUPS}" \
    --method visual_inception \
    --visual-inception-config "${cfg}" \
    --training-bundles-root "${REPO}/data/public/visual_inception" \
    --output "${out_dir}" \
    --device cuda
  echo "[DONE] $(date -u +%Y-%m-%dT%H:%M:%SZ) GEN ${name}"
  sync_drive
}

run_eval() {
  local name="$1"
  local images_root="$2"
  local suite="$3"
  local target_dir="${REPO}/${EXP_REL}/eval/${name}_${suite}"
  if [[ -f "${target_dir}/transfer_panel_summary.json" ]]; then
    echo "[SKIP] eval ${name}_${suite} already complete"
    return 0
  fi
  update_status "EVAL:${name}_${suite}"
  echo "[START] $(date -u +%Y-%m-%dT%H:%M:%SZ) EVAL ${name}_${suite}"
  "${PY}" -u -X utf8 tools/evaluate_visual_inception_category.py evaluate-transfer-panel \
    --images-root "${images_root}" \
    --output "${target_dir}" \
    --method visual_inception \
    --group "${TARGET_GROUPS}" \
    --replacement-groups selected \
    --query-suite "${suite}" \
    --key-mode image \
    --defense-mode none \
    --device cuda \
    --memory-vectors-dir "${VEC_DIR}"
  echo "[DONE] $(date -u +%Y-%m-%dT%H:%M:%SZ) EVAL ${name}_${suite}"
  sync_drive
}

update_status "STARTED"
run_gen "vi_10_00" "${REPO}/${EXP_REL}/configs/vi_blend_10_00.json"
run_gen "vi_08_02" "${REPO}/${EXP_REL}/configs/vi_blend_08_02.json"

for suite in training_oracle zero_overlap; do
  run_eval "blend_03_07" "${REPO}/output/vi_single_clip_l14_9groups_20261118" "${suite}"
  run_eval "blend_08_02" "${REPO}/${EXP_REL}/vi_08_02" "${suite}"
  run_eval "blend_10_00" "${REPO}/${EXP_REL}/vi_10_00" "${suite}"
done

update_status "SUMMARIZING"
"${PY}" -u -X utf8 "${REPO}/${EXP_REL}/logs/summarize_comp_loss.py"
sync_drive
update_status "COMPLETED"
echo "[ALL STAGE 1.2 COMPLETED] $(date -u +%Y-%m-%dT%H:%M:%SZ)"
