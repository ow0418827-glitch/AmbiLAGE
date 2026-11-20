#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export PYTHONPATH="/content/AmbiLAGE"
REPO="/content/AmbiLAGE"
DRIVE_ROOT="/content/drive/MyDrive/AmbiLAGE_backup"
EXP_REL="output/exp_20261120_vi_08_02_9groups"
LOG_DIR="${REPO}/${EXP_REL}/logs"
STATUS_FILE="${LOG_DIR}/vi_9groups.status"
PID_FILE="${LOG_DIR}/vi_9groups.pid"
PY="${REPO}/.venv/bin/python"

cd "${REPO}"
mkdir -p "${LOG_DIR}"
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

update_status "GEN:ALL_9_GROUPS"
echo "[START] $(date -u +%Y-%m-%dT%H:%M:%SZ) GEN VI 0.8/0.2 9 groups"

"${PY}" -u -X utf8 tools/run_experiment.py generate \
  --group all \
  --method visual_inception \
  --visual-inception-config "${REPO}/src/visual/config/visual_inception.example.json" \
  --training-bundles-root "${REPO}/data/public/visual_inception" \
  --output "${REPO}/${EXP_REL}" \
  --device cuda

echo "[DONE] $(date -u +%Y-%m-%dT%H:%M:%SZ) GEN VI 0.8/0.2 9 groups"
update_status "COMPLETED"
