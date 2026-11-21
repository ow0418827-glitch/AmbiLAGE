#!/usr/bin/env bash
set -euo pipefail

cd /content/AmbiLAGE
export PYTHONUNBUFFERED=1
PY=/content/AmbiLAGE/.venv/bin/python
DRIVE=/content/drive/MyDrive/AmbiLAGE_backup
EXP_DIR=/content/AmbiLAGE/output/exp_20261121_eval_v4_golden_band
MS_OUT=/content/AmbiLAGE/output/exp_20261121_ambilage_ms_v3_geom_photo_golden_band_9groups
LOG_DIR="$EXP_DIR/logs"
EVAL_DIR="$EXP_DIR/eval_eot_ablation_4g"
MEM_VEC_DIR="$EXP_DIR/memory_vectors_eot_ablation_4g"
CFG_PATH=/content/AmbiLAGE/src/visual/config/ambilage.clip_l14.ms_v3_geom_photo.json
STATUS_FILE="$LOG_DIR/DONE_MS_V3_GEOM_PHOTO_4G.status"

mkdir -p "$LOG_DIR" "$MS_OUT" "$EVAL_DIR"
rm -f "$STATUS_FILE"

sync_to_drive() {
  local tag="${1:-periodic}"
  if [ -d "$DRIVE" ]; then
    echo "[DRIVE_SYNC:$tag] $(date -Is) starting rsync to $DRIVE ..."
    mkdir -p \
      "$DRIVE/src/visual/_ambilage" \
      "$DRIVE/src/visual/config" \
      "$DRIVE/src/tests" \
      "$DRIVE/output/cache/precomputed_eval_vectors" \
      "$DRIVE/output/exp_20261121_ambilage_ms_v3_geom_photo_golden_band_9groups" \
      "$DRIVE/output/exp_20261121_eval_v4_golden_band"
    rsync -rt --inplace /content/AmbiLAGE/src/visual/_ambilage/ "$DRIVE/src/visual/_ambilage/" || true
    rsync -rt --inplace /content/AmbiLAGE/src/visual/config/ "$DRIVE/src/visual/config/" || true
    rsync -rt --inplace /content/AmbiLAGE/src/visual/training_objectives.py "$DRIVE/src/visual/" || true
    rsync -rt --inplace /content/AmbiLAGE/src/tests/test_single_model_and_category_abstraction.py "$DRIVE/src/tests/" || true
    if [ -d /content/AmbiLAGE/output/cache/precomputed_eval_vectors ]; then
      rsync -rt --inplace /content/AmbiLAGE/output/cache/precomputed_eval_vectors/ "$DRIVE/output/cache/precomputed_eval_vectors/" || true
    fi
    if [ -d "$MS_OUT" ]; then
      rsync -rt --inplace "$MS_OUT/" "$DRIVE/output/exp_20261121_ambilage_ms_v3_geom_photo_golden_band_9groups/" || true
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
  if [ "${rc}" -ne 0 ] && [ ! -f "${STATUS_FILE}" ]; then
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

echo "[STAGE A START] $(date -Is) Generating 4 Set 2 groups (01_camera,02_chair,06_jeans,07_shampoo) with ms_v3_geom_photo..."
"$PY" -u -X utf8 -B tools/run_experiment.py generate \
  --base golden_band_9groups \
  --group 01_camera,02_chair \
  --method ambilage \
  --ambilage-config "$CFG_PATH" \
  --training-bundles-root data/public/visual_inception_golden_band \
  --output "$MS_OUT" \
  --device cuda > "$LOG_DIR/ms_v3_geom_photo_w1.log" 2>&1 &
WPID1=$!

"$PY" -u -X utf8 -B tools/run_experiment.py generate \
  --base golden_band_9groups \
  --group 06_jeans,07_shampoo \
  --method ambilage \
  --ambilage-config "$CFG_PATH" \
  --training-bundles-root data/public/visual_inception_golden_band \
  --output "$MS_OUT" \
  --device cuda > "$LOG_DIR/ms_v3_geom_photo_w2.log" 2>&1 &
WPID2=$!

WRC1=0
WRC2=0
wait "$WPID1" || WRC1=$?
wait "$WPID2" || WRC2=$?
echo "[STAGE A SHARDS DONE] $(date -Is) w1=$WRC1 w2=$WRC2"
if [ "$WRC1" -ne 0 ] || [ "$WRC2" -ne 0 ]; then
  echo "[ERROR] Generation worker failed (w1=$WRC1, w2=$WRC2)"
  exit 1
fi

echo "[STAGE A VERIFY] $(date -Is) Verifying all 4 ms_v3_geom_photo FROZEN_IMAGE.json..."
"$PY" -u -X utf8 -B -c "
import json
from pathlib import Path
from src.paths import resolve_evaluation_base
from src.experiments.visual_inception_category_evaluation import load_frozen_suite

groups = ['01_camera', '02_chair', '06_jeans', '07_shampoo']
base = resolve_evaluation_base('golden_band_9groups')
out = Path('$MS_OUT')
suite = load_frozen_suite(out, method='ambilage', base=base, selected_groups=groups)
assert len(suite) == 4
for item in suite:
    g = item['group']
    meta = item['metadata']
    cfg = meta['config']
    tobj = meta['training_objective']
    gate = meta['final_naturalness']
    assert cfg['eot_enabled'] is False, f'{g} cfg eot_enabled={cfg[\"eot_enabled\"]}'
    assert cfg['diffusion_eot_enabled'] is False, f'{g} cfg diff_eot={cfg[\"diffusion_eot_enabled\"]}'
    assert cfg['multiscale_eot_enabled'] is True, f'{g} cfg ms_eot={cfg[\"multiscale_eot_enabled\"]}'
    assert cfg['multiscale_eot_profile'] == 'geom_photo_v3', f'{g} cfg profile={cfg[\"multiscale_eot_profile\"]}'
    assert abs(float(cfg['multiscale_identity_weight_end']) - 0.35) < 1e-9, f'{g} cfg id_end={cfg[\"multiscale_identity_weight_end\"]}'
    assert tobj['eot_enabled'] is False, f'{g} obj eot_enabled={tobj[\"eot_enabled\"]}'
    assert tobj['diffusion_eot_enabled'] is False, f'{g} obj diff_eot={tobj[\"diffusion_eot_enabled\"]}'
    assert float(tobj['diffusion_weight']) == 0.0, f'{g} obj diff_w={tobj[\"diffusion_weight\"]}'
    assert tobj['multiscale_eot_enabled'] is True, f'{g} obj ms_eot={tobj[\"multiscale_eot_enabled\"]}'
    assert tobj['multiscale_eot_profile'] == 'geom_photo_v3', f'{g} obj profile={tobj[\"multiscale_eot_profile\"]}'
    assert tobj['eot_paths'] == ['identity', 'multiscale_grid_breaking_geom_photo_v3_light_photometric'], f'{g} eot_paths={tobj[\"eot_paths\"]}'
    assert gate.get('passed') is True, f'{g} naturalness passed={gate.get(\"passed\")}'
    print(f'[VERIFIED MS-V3 FROZEN] {g}: eot_paths={tobj[\"eot_paths\"]} ssim={gate[\"metrics\"][\"source_ssim\"]:.4f} tv={gate[\"metrics\"][\"fg_chroma_tv_levels\"]:.4f}')
"

sync_to_drive "after_ms_v3_gen"

echo "[STAGE B START] $(date -Is) Evaluating ms_v3_geom_photo on Set 2 4 groups (--replacement-groups selected)..."
"$PY" -u -X utf8 -B tools/fast_evaluate_panel.py eval-suites \
  --base golden_band_9groups \
  --group 01_camera,02_chair,06_jeans,07_shampoo \
  --replacement-groups selected \
  --spec "ambilage_ms_v3_geom_photo_set2_4g:ambilage:$MS_OUT" \
  --suites training_oracle_v4 overlap_30_v4 zero_overlap_v4 \
  --output-dir "$EVAL_DIR" \
  --memory-vectors-dir "$MEM_VEC_DIR" \
  --verify-mode in_process \
  --device cuda

echo "[STAGE C VERIFY] $(date -Is) Checking all 9 independent verification files <= 1e-7..."
"$PY" -c "
import json
from pathlib import Path
eval_dir = Path('$EVAL_DIR')
count = 0
for s in ('training_oracle_v4', 'overlap_30_v4', 'zero_overlap_v4'):
    for e in ('01_clip_l14', '02_openclip_bigG14', '03_siglip_so400m'):
        vp = eval_dir / f'ambilage_ms_v3_geom_photo_set2_4g_{s}' / e / 'category_independent_verification.json'
        assert vp.is_file(), f'Missing {vp}'
        v = json.loads(vp.read_text(encoding='utf-8'))
        err = float(v['max_score_error'])
        assert err <= 1e-7, f'Score error > 1e-7 in {vp}: {err}'
        count += 1
        print(f'[VERIFIED {count}/9] ambilage_ms_v3_geom_photo_set2_4g_{s}/{e}: max_score_error={err}')
"

printf 'SUCCESS %s\n' "$(date -Is)" > "${STATUS_FILE}"
echo "[MS_V3_GEOM_PHOTO 4G TEST COMPLETED SUCCESSFULLY] $(date -Is)"
