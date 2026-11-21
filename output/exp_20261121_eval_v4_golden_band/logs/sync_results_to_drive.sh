#!/usr/bin/env bash
set -euo pipefail

SRC_ROOT="/content/AmbiLAGE"
DRIVE="/content/drive/MyDrive/AmbiLAGE_backup"
LOG_DIR="$SRC_ROOT/output/exp_20261121_eval_v4_golden_band/logs"
LOG_FILE="$LOG_DIR/drive_sync_cron.log"
LOCK_FILE="/tmp/ambilage_drive_sync.lock"

mkdir -p "$LOG_DIR"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "[SYNC_SKIP] $(date -Is) another sync process already holds $LOCK_FILE" | tee -a "$LOG_FILE"
  exit 0
fi

if [ ! -d "$DRIVE" ]; then
  echo "[SYNC_ERROR] $(date -Is) Drive backup root not found: $DRIVE" | tee -a "$LOG_FILE"
  exit 1
fi

{
  echo "[SYNC_START] $(date -Is) rsync -> $DRIVE"
  mkdir -p \
    "$DRIVE/tools" \
    "$DRIVE/src/experiments" \
    "$DRIVE/src/visual/_ambilage" \
    "$DRIVE/src/visual/config" \
    "$DRIVE/src/tests" \
    "$DRIVE/output/cache/precomputed_eval_vectors" \
    "$DRIVE/output/exp_20261121_eval_v4_golden_band" \
    "$DRIVE/output/exp_20261121_ambilage_multiscale_only_golden_band_9groups" \
    "$DRIVE/output/exp_20261121_ambilage_ms_v2_geom_golden_band_9groups" \
    "$DRIVE/output/exp_20261121_ambilage_ms_v2_geom_set2_9groups_results" \
    "$DRIVE/output/exp_20261121_ambilage_ms_v3_geom_photo_golden_band_9groups" \
    "$DRIVE/output/exp_20261121_ambilage_noeot_golden_band_9groups" \
    "$DRIVE/output/exp_20261121_vi_paper_minimal_margin_golden_band_9groups"

  rsync -rt --inplace "$SRC_ROOT/tools/generate.py" "$DRIVE/tools/"
  rsync -rt --inplace "$SRC_ROOT/src/experiments/base_workflow.py" "$DRIVE/src/experiments/"
  rsync -rt --inplace "$SRC_ROOT/src/visual/_ambilage/" "$DRIVE/src/visual/_ambilage/"
  rsync -rt --inplace "$SRC_ROOT/src/visual/config/" "$DRIVE/src/visual/config/"
  rsync -rt --inplace "$SRC_ROOT/src/visual/training_objectives.py" "$DRIVE/src/visual/"
  rsync -rt --inplace "$SRC_ROOT/src/visual/precomputed_eval_store.py" "$DRIVE/src/visual/"
  rsync -rt --inplace "$SRC_ROOT/src/tests/test_single_model_and_category_abstraction.py" "$DRIVE/src/tests/"

  if [ -d "$SRC_ROOT/output/cache/precomputed_eval_vectors" ]; then
    rsync -rt --inplace "$SRC_ROOT/output/cache/precomputed_eval_vectors/" "$DRIVE/output/cache/precomputed_eval_vectors/"
  fi

  for d in \
    exp_20261121_eval_v4_golden_band \
    exp_20261121_ambilage_multiscale_only_golden_band_9groups \
    exp_20261121_ambilage_ms_v2_geom_golden_band_9groups \
    exp_20261121_ambilage_ms_v2_geom_set2_9groups_results \
    exp_20261121_ambilage_ms_v3_geom_photo_golden_band_9groups \
    exp_20261121_ambilage_noeot_golden_band_9groups \
    exp_20261121_vi_paper_minimal_margin_golden_band_9groups; do
    if [ -d "$SRC_ROOT/output/$d" ]; then
      rsync -rt --inplace "$SRC_ROOT/output/$d/" "$DRIVE/output/$d/"
    fi
  done

  echo "[SYNC_DONE] $(date -Is) completed."
} 2>&1 | tee -a "$LOG_FILE"
