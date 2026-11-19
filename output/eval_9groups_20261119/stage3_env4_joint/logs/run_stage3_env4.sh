#!/usr/bin/env bash
set -euo pipefail

PHASE="${1:-whitebox}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="/content/AmbiLAGE"
REPO="/content/AmbiLAGE"
DRIVE_ROOT="/content/drive/MyDrive/AmbiLAGE_backup"
OUT_REL="output/eval_9groups_20261119"
STAGE3_REL="${OUT_REL}/stage3_env4_joint"
STAGE3_DIR="${REPO}/${STAGE3_REL}"
STATUS_FILE="${STAGE3_DIR}/logs/stage3_${PHASE}.status"
PY="${REPO}/.venv/bin/python"

cd "${REPO}"
mkdir -p "${STAGE3_DIR}/logs" "${STAGE3_DIR}/memory_vectors"

sync_drive() {
  mkdir -p "${DRIVE_ROOT}/${STAGE3_REL}"
  rsync -rt --inplace --exclude 'memory_vectors/' "${STAGE3_DIR}/" "${DRIVE_ROOT}/${STAGE3_REL}/" || true
  mkdir -p "${DRIVE_ROOT}/${STAGE3_REL}/memory_vectors"
  rsync -rt --ignore-existing "${STAGE3_DIR}/memory_vectors/" "${DRIVE_ROOT}/${STAGE3_REL}/memory_vectors/" || true
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

update_status "RUNNING:${PHASE}"
echo "[STAGE 3 START] $(date -u +%Y-%m-%dT%H:%M:%SZ) phase=${PHASE}"
"${PY}" -u -X utf8 "${STAGE3_DIR}/logs/eval_env4_9groups.py" --phase "${PHASE}" --device cuda
sync_drive
update_status "COMPLETED:${PHASE}"
echo "[STAGE 3 DONE] $(date -u +%Y-%m-%dT%H:%M:%SZ) phase=${PHASE}"
