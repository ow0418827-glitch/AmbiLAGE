#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export PYTHONPATH="/content/AmbiLAGE"
REPO="/content/AmbiLAGE"
DRIVE_ROOT="/content/drive/MyDrive/AmbiLAGE_backup"
EXP_REL="output/exp_20261120_grad_protect"
LOG_DIR="${REPO}/${EXP_REL}/logs"
STATUS_FILE="${LOG_DIR}/grad_protect.status"
PID_FILE="${LOG_DIR}/grad_protect.pid"
PY="${REPO}/.venv/bin/python"
if [[ ! -x "${PY}" ]]; then
  PY="$(command -v python3)"
fi
TARGET_GROUPS="02_chair,05_hair_extension,09_toothbrush"
ST1_VEC_DIR="${REPO}/${EXP_REL}/eval_stage1/_memory_vectors"
CFG="${REPO}/${EXP_REL}/configs/ambilage_grad_protect.json"

cd "${REPO}"
mkdir -p "${LOG_DIR}" "${REPO}/${EXP_REL}/gen" "${REPO}/${EXP_REL}/eval_stage1" "${REPO}/${EXP_REL}/eval_env4" "${ST1_VEC_DIR}"
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

for grp in 02_chair 05_hair_extension 09_toothbrush; do
  if [[ ! -f "${REPO}/${EXP_REL}/gen/${grp}/ambilage/FROZEN_IMAGE.json" ]]; then
    rm -f "${REPO}/${EXP_REL}/gen/${grp}/ambilage/PUBLIC_RUN.json"
    rm -f "${REPO}/${EXP_REL}/gen/${grp}/ambilage/generation/FAILED.json"
    update_status "GEN:gp_pilot3:${grp}"
    echo "[START] $(date -u +%Y-%m-%dT%H:%M:%SZ) GEN gp_pilot3 ${grp}"
    "${PY}" -u -X utf8 tools/run_experiment.py generate \
      --group "${grp}" \
      --method ambilage \
      --ambilage-config "${CFG}" \
      --training-bundles-root "${REPO}/data/public/visual_inception" \
      --output "${REPO}/${EXP_REL}/gen" \
      --device cuda
    echo "[DONE] $(date -u +%Y-%m-%dT%H:%M:%SZ) GEN gp_pilot3 ${grp}"
    sync_drive
  fi
done

rm -rf "${ST1_VEC_DIR}" "${REPO}/${EXP_REL}/eval_stage1"/gp_pilot3_*
update_status "ST1:fast_eval_v4"
echo "[START] $(date -u +%Y-%m-%dT%H:%M:%SZ) ST1 fast_eval_v4"
"${PY}" -u -X utf8 tools/fast_evaluate_panel.py eval-suites \
  --spec "baseline:ambilage:${REPO}/output/ambilage_single_clip_l14_9groups_20261118" \
  --spec "gp_pilot3:ambilage:${REPO}/${EXP_REL}/gen" \
  --suites training_oracle_v4 overlap_30_v4 zero_overlap_v4 \
  --output-dir "${REPO}/${EXP_REL}/eval_stage1" \
  --group "${TARGET_GROUPS}" \
  --replacement-groups selected \
  --key-mode image \
  --defense-mode none \
  --memory-vectors-dir "${ST1_VEC_DIR}" \
  --verify-mode in_process \
  --batch-size 16 \
  --text-batch-size 64 \
  --device cuda
echo "[DONE] $(date -u +%Y-%m-%dT%H:%M:%SZ) ST1 fast_eval_v4"
sync_drive

update_status "ENV4"
echo "[START] $(date -u +%Y-%m-%dT%H:%M:%SZ) ENV4 pilot3"
"${PY}" -u -X utf8 "${LOG_DIR}/eval_env4_pilot3.py" --device cuda
echo "[DONE] $(date -u +%Y-%m-%dT%H:%M:%SZ) ENV4 pilot3"

update_status "SUMMARIZE"
"${PY}" -u -X utf8 "${LOG_DIR}/summarize_grad_protect.py"
update_status "COMPLETED"
echo "[ALL_DONE] $(date -u +%Y-%m-%dT%H:%M:%SZ)"
