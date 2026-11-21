#!/usr/bin/env bash
set -euo pipefail

cd /content/AmbiLAGE
PY=/content/AmbiLAGE/.venv/bin/python
DRIVE=/content/drive/MyDrive/AmbiLAGE_backup
EXP_DIR=/content/AmbiLAGE/output/exp_20261121_eval_v4_golden_band
VI_OUT=/content/AmbiLAGE/output/exp_20261121_vi_10_00_golden_band_9groups
LOG_DIR="$EXP_DIR/logs"
mkdir -p "$LOG_DIR"

sync_to_drive() {
  local tag="${1:-periodic}"
  if [ -d "$DRIVE" ]; then
    echo "[DRIVE_SYNC:$tag] $(date -Is) starting rsync to $DRIVE ..."
    mkdir -p "$DRIVE/resources" "$DRIVE/data/public" "$DRIVE/output/public_pair_selection" "$DRIVE/output"
    rsync -a /content/AmbiLAGE/src/ "$DRIVE/src/" || true
    rsync -a /content/AmbiLAGE/tools/ "$DRIVE/tools/" || true
    rsync -a /content/AmbiLAGE/resources/evaluation_base_golden_band/ "$DRIVE/resources/evaluation_base_golden_band/" || true
    rsync -a /content/AmbiLAGE/data/public/visual_inception_golden_band/ "$DRIVE/data/public/visual_inception_golden_band/" || true
    rsync -a \
      --exclude='data/images' \
      --exclude='data/image_records' \
      --exclude='data/selection_features' \
      --exclude='data/prior_rank_features' \
      --exclude='data/calibration_features' \
      --exclude='expansion_rounds' \
      /content/AmbiLAGE/output/public_pair_selection/groups9_golden_band_set2/ \
      "$DRIVE/output/public_pair_selection/groups9_golden_band_set2/" || true
    if [ -d "$VI_OUT" ]; then
      rsync -a "$VI_OUT/" "$DRIVE/output/exp_20261121_vi_10_00_golden_band_9groups/" || true
    fi
    if [ -d "$EXP_DIR" ]; then
      rsync -a "$EXP_DIR/" "$DRIVE/output/exp_20261121_eval_v4_golden_band/" || true
    fi
    echo "[DRIVE_SYNC:$tag] $(date -Is) completed."
  else
    echo "[DRIVE_SYNC:$tag] WARNING: $DRIVE not found!"
  fi
}

# Initial sync of newly built Set 2 evaluation base and selection artifacts
sync_to_drive "initial"

# Background 10-minute Google Drive sync loop
(
  while true; do
    sleep 600
    sync_to_drive "10min_loop"
  done
) &
SYNC_PID=$!
trap 'kill $SYNC_PID 2>/dev/null || true; sync_to_drive "final_on_exit"' EXIT

echo "[STAGE A START] $(date -Is) Generating 9 groups with Visual Inception (default 1.0/0.0) on Set 2 (golden_band_9groups)..."
T0=$(date +%s.%N)
"$PY" -X utf8 -B tools/run_experiment.py generate \
  --base golden_band_9groups \
  --group all \
  --method visual_inception \
  --visual-inception-config src/visual/config/visual_inception.example.json \
  --training-bundles-root data/public/visual_inception_golden_band \
  --output "$VI_OUT" \
  --device cuda
T1=$(date +%s.%N)
echo "[STAGE A DONE] $(date -Is)"

"$PY" -c "
import json
from pathlib import Path
base_m = json.loads(Path('/content/AmbiLAGE/resources/evaluation_base_golden_band/manifest.json').read_text())
vi_out = Path('$VI_OUT')
for g, gspec in base_m['groups'].items():
    fp = vi_out / g / 'visual_inception/FROZEN_IMAGE.json'
    assert fp.is_file(), f'Missing {fp}'
    fdoc = json.loads(fp.read_text())
    mech = fdoc['metadata']['training_objective']['mechanism_parameters']
    assert float(mech['bank_loss_blend']) == 1.0, f'{g} bank_loss_blend={mech[\"bank_loss_blend\"]}'
    assert float(mech['hard_margin_loss_blend']) == 0.0, f'{g} hard_margin_loss_blend={mech[\"hard_margin_loss_blend\"]}'
    assert fdoc['source']['id'] == gspec['source_id'], f'{g} source id mismatch'
    assert fdoc['source']['image_sha256'] == gspec['source_sha256'], f'{g} source sha mismatch'
    print(f'[FROZEN VERIFIED] {g}: source={fdoc[\"source\"][\"id\"]} weights=(1.0, 0.0) image_sha256={fdoc[\"image_sha256\"][:12]}...')
"

sync_to_drive "after_stage_a"

echo "[STAGE B START] $(date -Is) Evaluating 3 exact_overlap_v4 suites x 3 encoders on Set 2 (golden_band_9groups)..."
T2=$(date +%s.%N)
"$PY" -X utf8 -B tools/fast_evaluate_panel.py eval-suites \
  --base golden_band_9groups \
  --spec "vi_10_00_set2:visual_inception:$VI_OUT" \
  --suites training_oracle_v4 overlap_30_v4 zero_overlap_v4 \
  --output-dir "$EXP_DIR/eval" \
  --memory-vectors-dir "$EXP_DIR/memory_vectors_vi_set2" \
  --verify-mode in_process \
  --device cuda
T3=$(date +%s.%N)
echo "[STAGE B DONE] $(date -Is)"

"$PY" -c "
import json
from pathlib import Path
t0, t1, t2, t3 = float('$T0'), float('$T1'), float('$T2'), float('$T3')
timings = {
    'vi_10_00_set2_generate_9groups_seconds': round(t1 - t0, 2),
    'eval_v4_set2_3suites_3encoders_seconds': round(t3 - t2, 2),
    'total_pipeline_seconds': round(t3 - t0, 2),
}
Path('$LOG_DIR/timings.json').write_text(json.dumps(timings, indent=2) + '\n', encoding='utf-8')
print('Timings:', timings)
"

echo "[STAGE C START] $(date -Is) Summarizing & verifying independent verification files..."
"$PY" -X utf8 -B "$LOG_DIR/summarize_eval_v4_golden_band.py"

"$PY" -c "
import json
from pathlib import Path
eval_dir = Path('$EXP_DIR/eval')
for s in ('training_oracle_v4', 'overlap_30_v4', 'zero_overlap_v4'):
    for e in ('01_clip_l14', '02_openclip_bigG14', '03_siglip_so400m'):
        vp = eval_dir / f'vi_10_00_set2_{s}' / e / 'category_independent_verification.json'
        assert vp.is_file(), f'Missing {vp}'
        v = json.loads(vp.read_text())
        assert v.get('status') == 'verified' or v.get('verified') is True or float(v.get('max_score_error', 1.0)) <= 1e-7, f'Verification failed: {vp}: {v}'
        assert float(v.get('max_score_error', 0.0)) <= 1e-7, f'Score error > 1e-7 in {vp}: {v}'
        print(f'[VERIFIED <=1e-7] {s} / {e}: max_score_error={v.get(\"max_score_error\")}')
"

echo "SUCCESS $(date -Is)" > "$LOG_DIR/DONE.status"
echo "[ALL STAGES COMPLETED SUCCESSFULLY] $(date -Is)"
