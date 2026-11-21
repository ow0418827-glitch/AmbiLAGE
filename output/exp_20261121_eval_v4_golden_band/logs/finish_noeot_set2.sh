#!/usr/bin/env bash
set -euo pipefail

cd /content/AmbiLAGE
export PYTHONUNBUFFERED=1
PY=/content/AmbiLAGE/.venv/bin/python
DRIVE=/content/drive/MyDrive/AmbiLAGE_backup
EXP_DIR=/content/AmbiLAGE/output/exp_20261121_eval_v4_golden_band
AMB_NOEOT_OUT=/content/AmbiLAGE/output/exp_20261121_ambilage_noeot_golden_band_9groups
LOG_DIR="$EXP_DIR/logs"
STATUS_FILE="$LOG_DIR/DONE_AMBILAGE_NOEOT.status"
CFG_PATH=/content/AmbiLAGE/src/visual/config/ambilage.clip_l14.example.json

mkdir -p "$LOG_DIR" "$AMB_NOEOT_OUT"
rm -f "$STATUS_FILE"

sync_noeot_to_drive() {
  local tag="${1:-periodic}"
  if [ -d "$DRIVE" ]; then
    echo "[DRIVE_SYNC:$tag] $(date -Is) starting rsync to $DRIVE ..."
    mkdir -p \
      "$DRIVE/src/visual/_ambilage" \
      "$DRIVE/src/visual/config" \
      "$DRIVE/output/cache/precomputed_eval_vectors" \
      "$DRIVE/output/exp_20261121_ambilage_noeot_golden_band_9groups" \
      "$DRIVE/output/exp_20261121_eval_v4_golden_band"
    rsync -rt --inplace /content/AmbiLAGE/src/visual/_ambilage/ "$DRIVE/src/visual/_ambilage/" || true
    rsync -rt --inplace /content/AmbiLAGE/src/visual/config/ "$DRIVE/src/visual/config/" || true
    rsync -rt --inplace /content/AmbiLAGE/src/visual/precomputed_eval_store.py "$DRIVE/src/visual/" || true
    if [ -d /content/AmbiLAGE/output/cache/precomputed_eval_vectors ]; then
      rsync -rt --inplace /content/AmbiLAGE/output/cache/precomputed_eval_vectors/ "$DRIVE/output/cache/precomputed_eval_vectors/" || true
    fi
    if [ -d "$AMB_NOEOT_OUT" ]; then
      rsync -rt --inplace "$AMB_NOEOT_OUT/" "$DRIVE/output/exp_20261121_ambilage_noeot_golden_band_9groups/" || true
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
  sync_noeot_to_drive "final_on_exit"
}
trap cleanup EXIT

(
  while true; do
    sleep 600
    sync_noeot_to_drive "10min_loop"
  done
) </dev/null >/dev/null 2>&1 &
SYNC_PID=$!

echo "[STAGE A VERIFY] $(date -Is) Validating all 9 Set 2 EoT=OFF groups via load_frozen_suite & manifest checks..."
"$PY" -u -X utf8 -B -c "
import json
from pathlib import Path
from src.paths import resolve_evaluation_base
from src.visual.audit import digest, read_json
from src.experiments.visual_inception_category_evaluation import load_frozen_suite

base = resolve_evaluation_base('golden_band_9groups')
out = Path('$AMB_NOEOT_OUT')
suite = load_frozen_suite(out, method='ambilage', base=base)
assert len(suite) == 9, f'Expected 9 frozen items, got {len(suite)}'

base_m = json.loads(Path('/content/AmbiLAGE/resources/evaluation_base_golden_band/manifest.json').read_text(encoding='utf-8'))
for g, gspec in base_m['groups'].items():
    groot = out / g / 'ambilage'
    fp = groot / 'FROZEN_IMAGE.json'
    pr = groot / 'PUBLIC_RUN.json'
    assert fp.is_file(), f'Missing {fp}'
    assert pr.is_file(), f'Missing {pr}'
    fdoc = read_json(fp)
    pdoc = read_json(pr)
    assert fdoc['public_binding_sha256'] == digest(pdoc), f'{g} public_binding_sha256 != digest(PUBLIC_RUN.json)'
    meta = fdoc['metadata']
    cfg = meta['config']
    tobj = meta['training_objective']
    gate = meta['final_naturalness']
    assert cfg['eot_enabled'] is False, f'{g} config eot_enabled={cfg[\"eot_enabled\"]}'
    assert tobj['eot_enabled'] is False, f'{g} obj eot_enabled={tobj[\"eot_enabled\"]}'
    assert tobj['diffusion_eot_enabled'] is False, f'{g} diff_eot={tobj[\"diffusion_eot_enabled\"]}'
    assert float(tobj['diffusion_weight']) == 0.0, f'{g} diff_w={tobj[\"diffusion_weight\"]}'
    assert abs(float(cfg['joint_weight']) - 0.15) < 1e-6, f'{g} jw={cfg[\"joint_weight\"]}'
    assert abs(float(cfg['text_weight']) - 1.50) < 1e-6, f'{g} tw={cfg[\"text_weight\"]}'
    assert gate.get('passed') is True, f'{g} final_naturalness.passed={gate.get(\"passed\")}'
    assert fdoc['source']['id'] == gspec['source_id'], f'{g} source id mismatch'
    assert fdoc['source']['image_sha256'] == gspec['source_sha256'], f'{g} source sha mismatch'
    assert Path(fdoc['image']).is_file(), f'{g} frozen image missing'
    print(f'[NOEOT FROZEN VERIFIED] {g}: source={fdoc[\"source\"][\"id\"]} tv={gate[\"metrics\"][\"fg_chroma_tv_levels\"]:.4f} ssim={gate[\"metrics\"][\"source_ssim\"]:.4f} sha256={fdoc[\"image_sha256\"][:12]}...')
"

sync_noeot_to_drive "after_noeot_gen"

echo "[STAGE B START] $(date -Is) Evaluating AmbiLAGE (EoT=OFF) on Set 2 across 3 exact_overlap_v4 suites x 3 encoders..."
T2=$("$PY" -c "import time; print(f'{time.perf_counter():.6f}')")

"$PY" -u -X utf8 -B tools/fast_evaluate_panel.py eval-suites \
  --base golden_band_9groups \
  --spec "ambilage_noeot_set2:ambilage:$AMB_NOEOT_OUT" \
  --suites training_oracle_v4 overlap_30_v4 zero_overlap_v4 \
  --output-dir "$EXP_DIR/eval" \
  --memory-vectors-dir "$EXP_DIR/memory_vectors_ambilage_noeot_set2" \
  --verify-mode in_process \
  --device cuda

T3=$("$PY" -c "import time; print(f'{time.perf_counter():.6f}')")
echo "[STAGE B DONE] $(date -Is)"

"$PY" -c "
import json
from pathlib import Path
tp = Path('$LOG_DIR/timings.json')
timings = json.loads(tp.read_text(encoding='utf-8')) if tp.is_file() else {}
meta05 = json.loads(Path('$AMB_NOEOT_OUT/05_hair_extension/ambilage/generation/metadata.json').read_text(encoding='utf-8'))
timings['ambilage_noeot_set2_05_hair_extension_regen_seconds'] = round(float(meta05['elapsed_seconds']), 2)
timings['ambilage_noeot_set2_eval_v4_3suites_3encoders_seconds'] = round(float('$T3') - float('$T2'), 2)
tp.write_text(json.dumps(timings, indent=2) + '\n', encoding='utf-8')
print('Updated timings:', timings)
"

echo "[STAGE C START] $(date -Is) Summarizing & verifying <=1e-7 independent verification..."
"$PY" -u -X utf8 -B "$LOG_DIR/summarize_eval_v4_golden_band.py"

"$PY" -c "
import json
from pathlib import Path
eval_dir = Path('$EXP_DIR/eval')
for s in ('training_oracle_v4', 'overlap_30_v4', 'zero_overlap_v4'):
    for e in ('01_clip_l14', '02_openclip_bigG14', '03_siglip_so400m'):
        vp = eval_dir / f'ambilage_noeot_set2_{s}' / e / 'category_independent_verification.json'
        assert vp.is_file(), f'Missing {vp}'
        v = json.loads(vp.read_text(encoding='utf-8'))
        assert float(v['max_score_error']) <= 1e-7, f'Score error > 1e-7 in {vp}: {v}'
        print(f'[VERIFIED <=1e-7] ambilage_noeot_set2_{s} / {e}: max_score_error={v.get(\"max_score_error\")}')
"

printf 'SUCCESS %s\n' "$(date -Is)" > "${STATUS_FILE}"
echo "[ALL AMBILAGE NOEOT SET2 STAGES COMPLETED SUCCESSFULLY] $(date -Is)"
