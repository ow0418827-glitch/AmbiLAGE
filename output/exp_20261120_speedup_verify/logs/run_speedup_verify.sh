#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export PYTHONPATH="/content/AmbiLAGE"
REPO="/content/AmbiLAGE"
DRIVE_ROOT="/content/drive/MyDrive/AmbiLAGE_backup"
EXP_REL="output/exp_20261120_speedup_verify"
EXP_DIR="${REPO}/${EXP_REL}"
LOG_DIR="${EXP_DIR}/logs"
STATUS_FILE="${LOG_DIR}/speedup_verify.status"
PID_FILE="${LOG_DIR}/speedup_verify.pid"
PY="${REPO}/.venv/bin/python"

cd "${REPO}"
mkdir -p "${LOG_DIR}"
echo "$$" > "${PID_FILE}"

sync_drive() {
  mkdir -p "${DRIVE_ROOT}/${EXP_REL}" "${DRIVE_ROOT}/output/exp_20261120_eval_v3" "${DRIVE_ROOT}/src" "${DRIVE_ROOT}/tools" "${DRIVE_ROOT}/resources/visual_inception_evaluation"
  rsync -rt --inplace "${EXP_DIR}/" "${DRIVE_ROOT}/${EXP_REL}/" || true
  rsync -rt --inplace "${REPO}/output/exp_20261120_eval_v3/" "${DRIVE_ROOT}/output/exp_20261120_eval_v3/" || true
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

now_sec() {
  "${PY}" -c "import time; print(f'{time.perf_counter():.6f}')"
}

elapsed_sec() {
  "${PY}" -c "print(f'{float($2) - float($1):.6f}')"
}

update_status "BENCH1:plan_a_vi_oracle_v3"
echo "[START] $(date -u +%Y-%m-%dT%H:%M:%SZ) Benchmark 1: Plan A vi_08_02_training_oracle_v3 (--verify-mode in_process)"
rm -rf "${EXP_DIR}/plan_a_vi_oracle_v3"
T0=$(now_sec)
"${PY}" -u -X utf8 tools/evaluate_visual_inception_category.py evaluate-transfer-panel \
  --images-root "${REPO}/output/exp_20261120_vi_08_02_9groups" \
  --output "${EXP_DIR}/plan_a_vi_oracle_v3" \
  --method visual_inception \
  --group all \
  --replacement-groups all \
  --query-suite training_oracle_v3 \
  --key-mode image \
  --defense-mode none \
  --device cuda \
  --verify-mode in_process \
  --memory-vectors-dir "${REPO}/output/eval_9groups_20261119/stage1_none/memory_vectors"
T1=$(now_sec)
PLAN_A_SEC=$(elapsed_sec "${T0}" "${T1}")
echo "[DONE] $(date -u +%Y-%m-%dT%H:%M:%SZ) Benchmark 1 elapsed=${PLAN_A_SEC}s"

update_status "BENCH2:plan_b_all6_inproc"
echo "[START] $(date -u +%Y-%m-%dT%H:%M:%SZ) Benchmark 2: Plan B all 6 panels (--verify-mode in_process, bs=16, tbs=64)"
rm -rf "${EXP_DIR}/plan_b_all6_inproc" "${EXP_DIR}/memory_vectors" "${EXP_DIR}/defended_cache"
T2=$(now_sec)
"${PY}" -u -X utf8 tools/fast_evaluate_panel.py eval-suites \
  --spec "ambilage:ambilage:${REPO}/output/ambilage_single_clip_l14_9groups_20261118" \
  --spec "vi_08_02:visual_inception:${REPO}/output/exp_20261120_vi_08_02_9groups" \
  --suites training_oracle_v3 overlap_30_v3 zero_overlap_v3 \
  --output-dir "${EXP_DIR}/plan_b_all6_inproc" \
  --key-mode image \
  --defense-mode none \
  --defended-cache-dir "${EXP_DIR}/defended_cache" \
  --memory-vectors-dir "${EXP_DIR}/memory_vectors" \
  --batch-size 16 \
  --text-batch-size 64 \
  --verify-mode in_process \
  --device cuda
T3=$(now_sec)
PLAN_B_SEC=$(elapsed_sec "${T2}" "${T3}")
echo "[DONE] $(date -u +%Y-%m-%dT%H:%M:%SZ) Benchmark 2 elapsed=${PLAN_B_SEC}s"

update_status "BENCH3:plan_b_subproc_smoke"
echo "[START] $(date -u +%Y-%m-%dT%H:%M:%SZ) Benchmark 3: Plan B subprocess smoke check (01_clip_l14)"
rm -rf "${EXP_DIR}/plan_b_subproc_smoke" "${EXP_DIR}/memory_vectors_subproc"
T4=$(now_sec)
"${PY}" -u -X utf8 tools/fast_evaluate_panel.py eval-suites \
  --spec "ambilage:ambilage:${REPO}/output/ambilage_single_clip_l14_9groups_20261118" \
  --spec "vi_08_02:visual_inception:${REPO}/output/exp_20261120_vi_08_02_9groups" \
  --suites training_oracle_v3 overlap_30_v3 zero_overlap_v3 \
  --output-dir "${EXP_DIR}/plan_b_subproc_smoke" \
  --key-mode image \
  --defense-mode none \
  --defended-cache-dir "${EXP_DIR}/defended_cache" \
  --memory-vectors-dir "${EXP_DIR}/memory_vectors_subproc" \
  --batch-size 16 \
  --text-batch-size 64 \
  --verify-mode subprocess \
  --encoder-slugs 01_clip_l14 \
  --device cuda
T5=$(now_sec)
SMOKE_SEC=$(elapsed_sec "${T4}" "${T5}")
echo "[DONE] $(date -u +%Y-%m-%dT%H:%M:%SZ) Benchmark 3 elapsed=${SMOKE_SEC}s"

"${PY}" -c "
import json
json.dump(
    {
        'plan_a_sec': float('${PLAN_A_SEC}'),
        'plan_b_all6_inproc_sec': float('${PLAN_B_SEC}'),
        'plan_b_subproc_smoke_sec': float('${SMOKE_SEC}'),
    },
    open('${LOG_DIR}/timings.json', 'w', encoding='utf-8'),
    indent=2,
)
"

update_status "COMPARE_VS_OFFICIAL_V3"
"${PY}" -u -X utf8 "${LOG_DIR}/compare_vs_official_v3.py"
update_status "COMPLETED"
echo "[ALL_DONE] $(date -u +%Y-%m-%dT%H:%M:%SZ)"
