#!/usr/bin/env bash
set -euo pipefail

cd /content/AmbiLAGE
export PYTHONUNBUFFERED=1
PY=/content/AmbiLAGE/.venv/bin/python
DRIVE=/content/drive/MyDrive/AmbiLAGE_backup
EXP_DIR=/content/AmbiLAGE/output/exp_20261121_eval_v4_golden_band
AMB_EOT_OUT=/content/AmbiLAGE/output/exp_20261121_ambilage_golden_band_9groups
AMB_NOEOT_OUT=/content/AmbiLAGE/output/exp_20261121_ambilage_noeot_golden_band_9groups
LOG_DIR="$EXP_DIR/logs"
STATUS_FILE="$LOG_DIR/DONE_AMBILAGE_NOEOT.status"
CFG_PATH=/content/AmbiLAGE/src/visual/config/ambilage.clip_l14.example.json

mkdir -p "$LOG_DIR" "$AMB_NOEOT_OUT"

OLD_PARENT_PID="${OLD_PARENT_PID:-181178}"
OLD_W1_PID="${OLD_W1_PID:-181185}"
OLD_W2_PID="${OLD_W2_PID:-181186}"

echo "[PHASE 1 START] $(date -Is) Waiting for currently running EoT=ON groups (02_chair in $OLD_W1_PID, 07_shampoo in $OLD_W2_PID)..."

stop_pid() {
  local pid="$1"
  local name="$2"
  if kill -0 "$pid" 2>/dev/null; then
    echo "[STOP] $(date -Is) Stopping $name (PID=$pid)..."
    kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 10); do
      if ! kill -0 "$pid" 2>/dev/null; then
        echo "[STOP] $(date -Is) $name (PID=$pid) exited cleanly."
        return 0
      fi
      sleep 1
    done
    kill -KILL "$pid" 2>/dev/null || true
    echo "[STOP] $(date -Is) $name (PID=$pid) killed."
  else
    echo "[STOP] $(date -Is) $name (PID=$pid) already exited."
  fi
}

W1_STOPPED=0
W2_STOPPED=0
while true; do
  if [ "$W1_STOPPED" -eq 0 ]; then
    if [ -s "$AMB_EOT_OUT/02_chair/ambilage/FROZEN_IMAGE.json" ]; then
      echo "[PHASE 1] $(date -Is) 02_chair frozen! Stopping W1 before it runs 03_cookies..."
      sleep 2
      stop_pid "$OLD_W1_PID" "EoT-ON W1"
      W1_STOPPED=1
    elif ! kill -0 "$OLD_W1_PID" 2>/dev/null; then
      echo "[PHASE 1] $(date -Is) EoT-ON W1 exited on its own."
      W1_STOPPED=1
    fi
  fi

  if [ "$W2_STOPPED" -eq 0 ]; then
    if [ -s "$AMB_EOT_OUT/07_shampoo/ambilage/FROZEN_IMAGE.json" ]; then
      echo "[PHASE 1] $(date -Is) 07_shampoo frozen! Stopping W2 before it runs 08_speaker..."
      sleep 2
      stop_pid "$OLD_W2_PID" "EoT-ON W2"
      W2_STOPPED=1
    elif ! kill -0 "$OLD_W2_PID" 2>/dev/null; then
      echo "[PHASE 1] $(date -Is) EoT-ON W2 exited on its own."
      W2_STOPPED=1
    fi
  fi

  if [ "$W1_STOPPED" -eq 1 ] && [ "$W2_STOPPED" -eq 1 ]; then
    break
  fi
  sleep 2
done

# Stop old parent runner (181178) so it doesn't interfere with rsync or trigger failure handlers later
stop_pid "$OLD_PARENT_PID" "Old EoT-ON parent runner"
sleep 2

# Clean any barely-started incomplete group directories without FROZEN_IMAGE.json
for g in 03_cookies 04_curtain 05_hair_extension 08_speaker 09_toothbrush; do
  if [ -d "$AMB_EOT_OUT/$g" ] && [ ! -f "$AMB_EOT_OUT/$g/ambilage/FROZEN_IMAGE.json" ]; then
    echo "[CLEANUP] Removing incomplete paused leftover dir: $AMB_EOT_OUT/$g"
    rm -rf "${AMB_EOT_OUT:?}/$g"
  fi
done

echo "[PHASE 2 START] $(date -Is) Verifying & archiving 4 completed EoT=ON groups to Google Drive..."
"$PY" -c "
import json
from pathlib import Path
out = Path('$AMB_EOT_OUT')
completed = []
for g in ('01_camera', '02_chair', '06_jeans', '07_shampoo'):
    fp = out / g / 'ambilage/FROZEN_IMAGE.json'
    assert fp.is_file() and fp.stat().st_size > 0, f'Missing frozen {g}'
    d = json.loads(fp.read_text(encoding='utf-8'))
    assert d['metadata']['config']['eot_enabled'] is True
    assert abs(float(d['metadata']['config']['joint_weight']) - 0.15) < 1e-6
    assert abs(float(d['metadata']['config']['text_weight']) - 1.50) < 1e-6
    assert Path(d['image']).is_file()
    completed.append(g)
    print(f'[EOT=ON SAVED] {g}: source={d[\"source\"][\"id\"]} sha256={d[\"image_sha256\"][:12]}...')
(out / 'PAUSED_AT_4_GROUPS.status').write_text(
    json.dumps({'status': 'paused_by_user_after_running_groups', 'completed_groups': completed}, ensure_ascii=False, indent=2) + '\n',
    encoding='utf-8',
)
"

if [ -d "$DRIVE" ]; then
  mkdir -p "$DRIVE/output/exp_20261121_ambilage_golden_band_9groups"
  rsync -rt --inplace "$AMB_EOT_OUT/" "$DRIVE/output/exp_20261121_ambilage_golden_band_9groups/"
  echo "[PHASE 2] $(date -Is) Saved 4 completed EoT=ON groups to Drive."
fi

echo "[PHASE 3 START] $(date -Is) Setting eot_enabled=false in $CFG_PATH..."
"$PY" -c "
import json
from pathlib import Path
from src.visual._ambilage.config import AmbiLAGEConfig

p = Path('$CFG_PATH')
raw = json.loads(p.read_text(encoding='utf-8'))
raw['eot_enabled'] = False
cfg = AmbiLAGEConfig.from_dict(raw)
assert cfg.eot_enabled is False, f'Expected eot_enabled=False, got {cfg.eot_enabled}'
assert cfg.effective_diffusion_eot_enabled() is False, 'Expected diffusion eot disabled'
assert abs(cfg.joint_weight - 0.15) < 1e-6 and abs(cfg.text_weight - 1.50) < 1e-6
p.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print('[CONFIG UPDATED & VALIDATED] eot_enabled=False, jw=0.15, tw=1.50')
"

if [ -d "$DRIVE" ]; then
  mkdir -p "$DRIVE/src/visual/config"
  rsync -rt --inplace "$CFG_PATH" "$DRIVE/src/visual/config/"
fi

# Phase 4: background 10-minute Drive sync for EoT=OFF run
sync_noeot_to_drive() {
  local tag="${1:-periodic}"
  if [ -d "$DRIVE" ]; then
    echo "[DRIVE_SYNC:$tag] $(date -Is) starting rsync to $DRIVE ..."
    mkdir -p "$DRIVE/output/exp_20261121_ambilage_noeot_golden_band_9groups" "$DRIVE/output/exp_20261121_eval_v4_golden_band" "$DRIVE/src/visual/config"
    rsync -rt --inplace "$CFG_PATH" "$DRIVE/src/visual/config/" || true
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

echo "[PHASE 4 STAGE A START] $(date -Is) Generating 9 groups with AmbiLAGE (jw=0.15, tw=1.50, EoT=OFF) on Set 2..."
T0=$("$PY" -c "import time; print(f'{time.perf_counter():.6f}')")

"$PY" -u -X utf8 -B tools/run_experiment.py generate \
  --base golden_band_9groups \
  --group 01_camera,02_chair,03_cookies,04_curtain,05_hair_extension \
  --method ambilage \
  --ambilage-config "$CFG_PATH" \
  --training-bundles-root data/public/visual_inception_golden_band \
  --output "$AMB_NOEOT_OUT" \
  --device cuda > "$LOG_DIR/ambilage_noeot_w1.log" 2>&1 &
NPID1=$!

"$PY" -u -X utf8 -B tools/run_experiment.py generate \
  --base golden_band_9groups \
  --group 06_jeans,07_shampoo,08_speaker,09_toothbrush \
  --method ambilage \
  --ambilage-config "$CFG_PATH" \
  --training-bundles-root data/public/visual_inception_golden_band \
  --output "$AMB_NOEOT_OUT" \
  --device cuda > "$LOG_DIR/ambilage_noeot_w2.log" 2>&1 &
NPID2=$!

NRC1=0
NRC2=0
wait "$NPID1" || NRC1=$?
wait "$NPID2" || NRC2=$?
echo "[NOEOT SHARDS DONE] $(date -Is) rc1=$NRC1 rc2=$NRC2"
if [ "$NRC1" -ne 0 ] || [ "$NRC2" -ne 0 ]; then
  echo "[ERROR] AmbiLAGE EoT=OFF shard worker failed (rc1=$NRC1, rc2=$NRC2)."
  exit 1
fi

echo "[PHASE 4 STAGE A CHECK] $(date -Is) Sequential verification pass (--group all)..."
"$PY" -u -X utf8 -B tools/run_experiment.py generate \
  --base golden_band_9groups \
  --group all \
  --method ambilage \
  --ambilage-config "$CFG_PATH" \
  --training-bundles-root data/public/visual_inception_golden_band \
  --output "$AMB_NOEOT_OUT" \
  --device cuda

T1=$("$PY" -c "import time; print(f'{time.perf_counter():.6f}')")
echo "[PHASE 4 STAGE A DONE] $(date -Is)"

"$PY" -c "
import json
from pathlib import Path
base_m = json.loads(Path('/content/AmbiLAGE/resources/evaluation_base_golden_band/manifest.json').read_text())
out = Path('$AMB_NOEOT_OUT')
for g, gspec in base_m['groups'].items():
    fp = out / g / 'ambilage/FROZEN_IMAGE.json'
    assert fp.is_file(), f'Missing {fp}'
    fdoc = json.loads(fp.read_text())
    cfg = fdoc['metadata']['config']
    tobj = fdoc['metadata']['training_objective']
    assert cfg['eot_enabled'] is False, f'{g} config eot_enabled={cfg[\"eot_enabled\"]}'
    assert tobj['eot_enabled'] is False, f'{g} obj eot_enabled={tobj[\"eot_enabled\"]}'
    assert tobj['diffusion_eot_enabled'] is False, f'{g} diff_eot={tobj[\"diffusion_eot_enabled\"]}'
    assert float(tobj['diffusion_weight']) == 0.0, f'{g} diff_w={tobj[\"diffusion_weight\"]}'
    assert abs(float(cfg['joint_weight']) - 0.15) < 1e-6, f'{g} jw={cfg[\"joint_weight\"]}'
    assert abs(float(cfg['text_weight']) - 1.50) < 1e-6, f'{g} tw={cfg[\"text_weight\"]}'
    assert fdoc['source']['id'] == gspec['source_id'], f'{g} source id mismatch'
    assert fdoc['source']['image_sha256'] == gspec['source_sha256'], f'{g} source sha mismatch'
    print(f'[NOEOT FROZEN VERIFIED] {g}: source={fdoc[\"source\"][\"id\"]} eot=False diff_w=0.0 jw={cfg[\"joint_weight\"]} tw={cfg[\"text_weight\"]} sha256={fdoc[\"image_sha256\"][:12]}...')
"

sync_noeot_to_drive "after_noeot_gen"

echo "[PHASE 4 STAGE B START] $(date -Is) Evaluating AmbiLAGE (EoT=OFF) on Set 2 across 3 exact_overlap_v4 suites x 3 encoders..."
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
echo "[PHASE 4 STAGE B DONE] $(date -Is)"

"$PY" -c "
import json
from pathlib import Path
tp = Path('$LOG_DIR/timings.json')
timings = json.loads(tp.read_text(encoding='utf-8')) if tp.is_file() else {}
timings['ambilage_noeot_set2_generate_9groups_seconds'] = round(float('$T1') - float('$T0'), 2)
timings['ambilage_noeot_set2_eval_v4_3suites_3encoders_seconds'] = round(float('$T3') - float('$T2'), 2)
tp.write_text(json.dumps(timings, indent=2) + '\n', encoding='utf-8')
print('Updated timings:', timings)
"

echo "[PHASE 4 STAGE C START] $(date -Is) Summarizing & verifying <=1e-7 independent verification..."
"$PY" -u -X utf8 -B "$LOG_DIR/summarize_eval_v4_golden_band.py"

"$PY" -c "
import json
from pathlib import Path
eval_dir = Path('$EXP_DIR/eval')
for s in ('training_oracle_v4', 'overlap_30_v4', 'zero_overlap_v4'):
    for e in ('01_clip_l14', '02_openclip_bigG14', '03_siglip_so400m'):
        vp = eval_dir / f'ambilage_noeot_set2_{s}' / e / 'category_independent_verification.json'
        assert vp.is_file(), f'Missing {vp}'
        v = json.loads(vp.read_text())
        assert v.get('status') == 'verified' or v.get('verified') is True or float(v.get('max_score_error', 1.0)) <= 1e-7, f'Verification failed: {vp}: {v}'
        assert float(v.get('max_score_error', 0.0)) <= 1e-7, f'Score error > 1e-7 in {vp}: {v}'
        print(f'[VERIFIED <=1e-7] ambilage_noeot_set2_{s} / {e}: max_score_error={v.get(\"max_score_error\")}')
"

printf 'SUCCESS %s\n' "$(date -Is)" > "${STATUS_FILE}"
echo "[ALL AMBILAGE NOEOT SET2 STAGES COMPLETED SUCCESSFULLY] $(date -Is)"
