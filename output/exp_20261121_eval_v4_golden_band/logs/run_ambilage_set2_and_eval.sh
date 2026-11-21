#!/usr/bin/env bash
set -euo pipefail

cd /content/AmbiLAGE
export PYTHONUNBUFFERED=1
PY=/content/AmbiLAGE/.venv/bin/python
DRIVE=/content/drive/MyDrive/AmbiLAGE_backup
EXP_DIR=/content/AmbiLAGE/output/exp_20261121_eval_v4_golden_band
AMB_OUT=/content/AmbiLAGE/output/exp_20261121_ambilage_golden_band_9groups
LOG_DIR="$EXP_DIR/logs"
STATUS_FILE="$LOG_DIR/DONE_AMBILAGE.status"
mkdir -p "$LOG_DIR" "$AMB_OUT"

sync_to_drive() {
  local tag="${1:-periodic}"
  if [ -d "$DRIVE" ]; then
    echo "[DRIVE_SYNC:$tag] $(date -Is) starting rsync to $DRIVE ..."
    mkdir -p "$DRIVE/output/exp_20261121_ambilage_golden_band_9groups" "$DRIVE/output/exp_20261121_eval_v4_golden_band"
    if [ -d "$AMB_OUT" ]; then
      rsync -rt --inplace "$AMB_OUT/" "$DRIVE/output/exp_20261121_ambilage_golden_band_9groups/" || true
    fi
    if [ -d "$EXP_DIR" ]; then
      rsync -rt --inplace "$EXP_DIR/" "$DRIVE/output/exp_20261121_eval_v4_golden_band/" || true
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
  if [ "${rc}" -ne 0 ]; then
    printf 'FAILED(rc=%s) %s\n' "${rc}" "$(date -Is)" > "${STATUS_FILE}"
  fi
  sync_to_drive "final_on_exit"
}
trap cleanup EXIT

(
  while true; do
    sleep 600
    sync_to_drive "10min_loop"
  done
) </dev/null >/dev/null 2>&1 &
SYNC_PID=$!

echo "[STAGE A START] $(date -Is) Generating 9 groups with AmbiLAGE (jw=0.15, tw=1.50) on Set 2 (golden_band_9groups) using 2 parallel shard workers..."
T0=$("$PY" -c "import time; print(f'{time.perf_counter():.6f}')")

"$PY" -u -X utf8 -B tools/run_experiment.py generate \
  --base golden_band_9groups \
  --group 01_camera,02_chair,03_cookies,04_curtain,05_hair_extension \
  --method ambilage \
  --ambilage-config src/visual/config/ambilage.clip_l14.example.json \
  --training-bundles-root data/public/visual_inception_golden_band \
  --output "$AMB_OUT" \
  --device cuda > "$LOG_DIR/ambilage_w1.log" 2>&1 &
PID1=$!

"$PY" -u -X utf8 -B tools/run_experiment.py generate \
  --base golden_band_9groups \
  --group 06_jeans,07_shampoo,08_speaker,09_toothbrush \
  --method ambilage \
  --ambilage-config src/visual/config/ambilage.clip_l14.example.json \
  --training-bundles-root data/public/visual_inception_golden_band \
  --output "$AMB_OUT" \
  --device cuda > "$LOG_DIR/ambilage_w2.log" 2>&1 &
PID2=$!

RC1=0
RC2=0
wait "$PID1" || RC1=$?
wait "$PID2" || RC2=$?
echo "[SHARDS DONE] $(date -Is) rc1=$RC1 rc2=$RC2"
if [ "$RC1" -ne 0 ] || [ "$RC2" -ne 0 ]; then
  echo "[ERROR] AmbiLAGE shard worker failed (rc1=$RC1, rc2=$RC2). Check ambilage_w1.log and ambilage_w2.log."
  exit 1
fi

echo "[STAGE A CHECK] $(date -Is) Running sequential completion & FROZEN_IMAGE verification pass (--group all)..."
"$PY" -u -X utf8 -B tools/run_experiment.py generate \
  --base golden_band_9groups \
  --group all \
  --method ambilage \
  --ambilage-config src/visual/config/ambilage.clip_l14.example.json \
  --training-bundles-root data/public/visual_inception_golden_band \
  --output "$AMB_OUT" \
  --device cuda

T1=$("$PY" -c "import time; print(f'{time.perf_counter():.6f}')")
echo "[STAGE A DONE] $(date -Is)"

"$PY" -c "
import json
from pathlib import Path
base_m = json.loads(Path('/content/AmbiLAGE/resources/evaluation_base_golden_band/manifest.json').read_text())
amb_out = Path('$AMB_OUT')
for g, gspec in base_m['groups'].items():
    fp = amb_out / g / 'ambilage/FROZEN_IMAGE.json'
    assert fp.is_file(), f'Missing {fp}'
    fdoc = json.loads(fp.read_text())
    cfg = fdoc['metadata']['config']
    assert abs(float(cfg['joint_weight']) - 0.15) < 1e-6, f'{g} joint_weight={cfg[\"joint_weight\"]}'
    assert abs(float(cfg['text_weight']) - 1.50) < 1e-6, f'{g} text_weight={cfg[\"text_weight\"]}'
    assert fdoc['source']['id'] == gspec['source_id'], f'{g} source id mismatch'
    assert fdoc['source']['image_sha256'] == gspec['source_sha256'], f'{g} source sha mismatch'
    print(f'[FROZEN VERIFIED] {g}: source={fdoc[\"source\"][\"id\"]} jw={cfg[\"joint_weight\"]} tw={cfg[\"text_weight\"]} image_sha256={fdoc[\"image_sha256\"][:12]}...')
"

sync_to_drive "after_stage_a"

echo "[STAGE B START] $(date -Is) Evaluating AmbiLAGE on Set 2 across 3 exact_overlap_v4 suites x 3 encoders..."
T2=$("$PY" -c "import time; print(f'{time.perf_counter():.6f}')")
"$PY" -u -X utf8 -B tools/fast_evaluate_panel.py eval-suites \
  --base golden_band_9groups \
  --spec "ambilage_set2:ambilage:$AMB_OUT" \
  --suites training_oracle_v4 overlap_30_v4 zero_overlap_v4 \
  --output-dir "$EXP_DIR/eval" \
  --memory-vectors-dir "$EXP_DIR/memory_vectors_ambilage_set2" \
  --verify-mode in_process \
  --device cuda
T3=$("$PY" -c "import time; print(f'{time.perf_counter():.6f}')")
echo "[STAGE B DONE] $(date -Is)"

"$PY" -c "
import json
from pathlib import Path
tp = Path('$LOG_DIR/timings.json')
timings = json.loads(tp.read_text(encoding='utf-8')) if tp.is_file() else {}
timings['ambilage_set2_generate_9groups_seconds'] = round(float('$T1') - float('$T0'), 2)
timings['ambilage_set2_eval_v4_3suites_3encoders_seconds'] = round(float('$T3') - float('$T2'), 2)
tp.write_text(json.dumps(timings, indent=2) + '\n', encoding='utf-8')
print('Updated timings:', timings)
"

echo "[STAGE C START] $(date -Is) Summarizing & verifying independent verification files..."
"$PY" -u -X utf8 -B "$LOG_DIR/summarize_eval_v4_golden_band.py"

"$PY" -c "
import json
from pathlib import Path
eval_dir = Path('$EXP_DIR/eval')
for s in ('training_oracle_v4', 'overlap_30_v4', 'zero_overlap_v4'):
    for e in ('01_clip_l14', '02_openclip_bigG14', '03_siglip_so400m'):
        vp = eval_dir / f'ambilage_set2_{s}' / e / 'category_independent_verification.json'
        assert vp.is_file(), f'Missing {vp}'
        v = json.loads(vp.read_text())
        assert v.get('status') == 'verified' or v.get('verified') is True or float(v.get('max_score_error', 1.0)) <= 1e-7, f'Verification failed: {vp}: {v}'
        assert float(v.get('max_score_error', 0.0)) <= 1e-7, f'Score error > 1e-7 in {vp}: {v}'
        print(f'[VERIFIED <=1e-7] ambilage_set2_{s} / {e}: max_score_error={v.get(\"max_score_error\")}')
"

printf 'SUCCESS %s\n' "$(date -Is)" > "${STATUS_FILE}"
echo "[ALL AMBILAGE SET2 STAGES COMPLETED SUCCESSFULLY] $(date -Is)"
