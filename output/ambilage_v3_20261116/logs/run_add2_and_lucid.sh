#!/usr/bin/env bash
set -euo pipefail
cd /content/AmbiLAGE
export PYTHONUNBUFFERED=1

AMBI_OUT="output/ambilage_v3_20261116"
AMBI_LOGS="/content/AmbiLAGE/${AMBI_OUT}/logs"
AMBI_STATUS="${AMBI_LOGS}/pipeline.status"
AMBI_PROGRESS="${AMBI_LOGS}/group_progress.log"

LUCID_OUT="output/lucid_4groups_20261116"
LUCID_LOGS="/content/AmbiLAGE/${LUCID_OUT}/logs"
LUCID_STATUS="${LUCID_LOGS}/pipeline.status"
LUCID_PROGRESS="${LUCID_LOGS}/group_progress.log"

FAILURES=0

echo "REUSED 07_shampoo (from ambilage_choice1_2groups_20261114) at $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${LUCID_PROGRESS}"
echo "REUSED 09_toothbrush (from ambilage_toothbrush_20261115) at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${LUCID_PROGRESS}"
echo "QUEUED waiting for AmbiLAGE 07_shampoo & 08_speaker at $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${LUCID_STATUS}"

for g in "07_shampoo" "08_speaker"; do
  echo "RUNNING generation ${g} at $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${AMBI_STATUS}"
  echo "=== [AmbiLAGE ${g}] Step 1/2: Generate (${g}/ambilage) at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  set +e
  .venv/bin/python -u -X utf8 tools/run_experiment.py generate \
    --group "${g}" \
    --method ambilage \
    --output "${AMBI_OUT}" \
    --device cuda
  gen_rc=$?
  set -e

  if [[ ${gen_rc} -ne 0 ]] || [[ ! -f "/content/AmbiLAGE/${AMBI_OUT}/${g}/ambilage/FROZEN_IMAGE.json" ]]; then
    echo "[ERROR] AmbiLAGE ${g} generation failed (rc=${gen_rc}) at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "${g} GENERATE_FAILED rc=${gen_rc} at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${AMBI_PROGRESS}"
    FAILURES=$((FAILURES + 1))
    continue
  fi
  echo "${g} GENERATED at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${AMBI_PROGRESS}"

  echo "RUNNING evaluation ${g} at $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${AMBI_STATUS}"
  echo "=== [AmbiLAGE ${g}] Step 2/2: Evaluate Env 4 (${g}/ambilage, none defense) at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  set +e
  .venv/bin/python -u -X utf8 "${AMBI_LOGS}/eval_group_env4.py" \
    --output-root "${AMBI_OUT}" \
    --groups "${g}" \
    --device cuda
  eval_rc=$?
  set -e

  if [[ ${eval_rc} -ne 0 ]]; then
    echo "[ERROR] AmbiLAGE ${g} evaluation failed (rc=${eval_rc}) at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "${g} EVAL_FAILED rc=${eval_rc} at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${AMBI_PROGRESS}"
    FAILURES=$((FAILURES + 1))
    continue
  fi
  echo "${g} EVALUATED_AND_VERIFIED at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${AMBI_PROGRESS}"
done

if [[ ${FAILURES} -eq 0 ]]; then
  echo "COMPLETED (4 groups: 09_toothbrush, 06_jeans, 07_shampoo, 08_speaker) at $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${AMBI_STATUS}"
else
  echo "COMPLETED_WITH_FAILURES count=${FAILURES} at $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${AMBI_STATUS}"
fi

echo "=== [LUCID] Evaluate Env 4 (06_jeans, 08_speaker) at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
for g in "06_jeans" "08_speaker"; do
  echo "RUNNING evaluation ${g} at $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${LUCID_STATUS}"
  echo "=== [LUCID ${g}] Evaluate Env 4 (${g}/lucid, none defense) at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  set +e
  .venv/bin/python -u -X utf8 "${LUCID_LOGS}/eval_group_env4.py" \
    --output-root "${LUCID_OUT}" \
    --groups "${g}" \
    --method lucid \
    --device cuda
  lucid_rc=$?
  set -e

  if [[ ${lucid_rc} -ne 0 ]]; then
    echo "[ERROR] LUCID ${g} evaluation failed (rc=${lucid_rc}) at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "${g} EVAL_FAILED rc=${lucid_rc} at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${LUCID_PROGRESS}"
    FAILURES=$((FAILURES + 1))
  else
    echo "${g} EVALUATED_AND_VERIFIED at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${LUCID_PROGRESS}"
  fi
done

if [[ ${FAILURES} -eq 0 ]]; then
  .venv/bin/python -u -X utf8 "${LUCID_LOGS}/finalize_lucid_summary.py"
  echo "=== ALL DONE (AmbiLAGE 07_shampoo/08_speaker + LUCID 4-group Env 4) at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  echo "COMPLETED at $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${LUCID_STATUS}"
  exit 0
else
  echo "=== FINISHED WITH ${FAILURES} FAILURE(S) at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  echo "COMPLETED_WITH_FAILURES count=${FAILURES} at $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${LUCID_STATUS}"
  exit 1
fi
