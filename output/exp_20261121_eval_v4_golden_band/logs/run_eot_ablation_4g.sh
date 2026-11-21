#!/usr/bin/env bash
set -euo pipefail

cd /content/AmbiLAGE
export PYTHONUNBUFFERED=1
PY=/content/AmbiLAGE/.venv/bin/python
DRIVE=/content/drive/MyDrive/AmbiLAGE_backup
EXP_DIR=/content/AmbiLAGE/output/exp_20261121_eval_v4_golden_band
LOG_DIR="$EXP_DIR/logs"
EVAL_DIR="$EXP_DIR/eval_eot_ablation_4g"
MEM_VEC_DIR="$EXP_DIR/memory_vectors_eot_ablation_4g"
STATUS_FILE="$LOG_DIR/DONE_EOT_ABLATION_4G.status"

EOT_ON_OUT=/content/AmbiLAGE/output/exp_20261121_ambilage_golden_band_9groups
EOT_OFF_OUT=/content/AmbiLAGE/output/exp_20261121_ambilage_noeot_golden_band_9groups
VI_OUT=/content/AmbiLAGE/output/exp_20261121_vi_10_00_golden_band_9groups

mkdir -p "$LOG_DIR" "$EVAL_DIR"
rm -f "$STATUS_FILE"

sync_ablation_to_drive() {
  local tag="${1:-periodic}"
  if [ -d "$DRIVE" ]; then
    echo "[DRIVE_SYNC:$tag] $(date -Is) starting rsync to $DRIVE ..."
    mkdir -p "$DRIVE/output/exp_20261121_eval_v4_golden_band" "$DRIVE/output/cache/precomputed_eval_vectors"
    rsync -rt --inplace "$EXP_DIR/" "$DRIVE/output/exp_20261121_eval_v4_golden_band/" || true
    if [ -d /content/AmbiLAGE/output/cache/precomputed_eval_vectors ]; then
      rsync -rt --inplace /content/AmbiLAGE/output/cache/precomputed_eval_vectors/ "$DRIVE/output/cache/precomputed_eval_vectors/" || true
    fi
    echo "[DRIVE_SYNC:$tag] $(date -Is) completed."
  else
    echo "[DRIVE_SYNC:$tag] WARNING: $DRIVE not found!"
  fi
}

SYNC_PID=""
cleanup() {
  local rc=$?
  if [ -n "${SYNC_PID}" ]; then
    kill "${SYNC_PID}" 2>/dev/null || true
    wait "${SYNC_PID}" 2>/dev/null || true
  fi
  if [ "${rc}" -ne 0 ] && [ ! -f "${STATUS_FILE}" ]; then
    printf 'FAILED(rc=%s) %s\n' "${rc}" "$(date -Is)" > "${STATUS_FILE}"
  fi
  sync_ablation_to_drive "final_on_exit"
}
trap cleanup EXIT

(
  while true; do
    sleep 600
    sync_ablation_to_drive "10min_loop"
  done
) </dev/null >/dev/null 2>&1 &
SYNC_PID=$!

echo "[EOT ABLATION 4G START] $(date -Is) Evaluating EoT=ON vs EoT=OFF vs VI (01_camera,02_chair,06_jeans,07_shampoo)..."
"$PY" -u -X utf8 -B tools/fast_evaluate_panel.py eval-suites \
  --base golden_band_9groups \
  --group 01_camera,02_chair,06_jeans,07_shampoo \
  --replacement-groups selected \
  --spec "ambilage_eot_on_set2_4g:ambilage:$EOT_ON_OUT" \
  --spec "ambilage_eot_off_set2_4g:ambilage:$EOT_OFF_OUT" \
  --spec "vi_10_00_set2_4g:visual_inception:$VI_OUT" \
  --suites training_oracle_v4 overlap_30_v4 zero_overlap_v4 \
  --output-dir "$EVAL_DIR" \
  --memory-vectors-dir "$MEM_VEC_DIR" \
  --verify-mode in_process \
  --device cuda

echo "[VERIFY <=1e-7] $(date -Is) Checking all 27 independent verification files..."
"$PY" -c "
import json
from pathlib import Path
eval_dir = Path('$EVAL_DIR')
count = 0
for tag in ('ambilage_eot_on_set2_4g', 'ambilage_eot_off_set2_4g', 'vi_10_00_set2_4g'):
    for s in ('training_oracle_v4', 'overlap_30_v4', 'zero_overlap_v4'):
        for e in ('01_clip_l14', '02_openclip_bigG14', '03_siglip_so400m'):
            vp = eval_dir / f'{tag}_{s}' / e / 'category_independent_verification.json'
            assert vp.is_file(), f'Missing {vp}'
            v = json.loads(vp.read_text(encoding='utf-8'))
            err = float(v['max_score_error'])
            assert err <= 1e-7, f'Score error > 1e-7 in {vp}: {err}'
            count += 1
            print(f'[VERIFIED {count}/27] {tag}_{s}/{e}: max_score_error={err}')
assert count == 27
"

echo "[SUMMARIZE START] $(date -Is) Writing EOT_ABLATION_4G_SUMMARY.md & .json..."
"$PY" -u -X utf8 -B "$LOG_DIR/summarize_eot_ablation_4g.py"

printf 'SUCCESS %s\n' "$(date -Is)" > "${STATUS_FILE}"
echo "[EOT ABLATION 4G COMPLETED SUCCESSFULLY] $(date -Is)"
