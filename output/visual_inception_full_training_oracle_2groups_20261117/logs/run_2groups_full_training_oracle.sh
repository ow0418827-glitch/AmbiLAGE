#!/usr/bin/env bash
set -euo pipefail

cd /content/AmbiLAGE
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

OUT="output/visual_inception_full_training_oracle_2groups_20261117"
LOG_DIR="$OUT/logs"
STATUS_FILE="$LOG_DIR/pipeline.status"
mkdir -p "$LOG_DIR"

write_status() {
  printf '%s at %s\n' "$1" "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" > "$STATUS_FILE"
}

on_err() {
  local code=$?
  write_status "FAILED (exit $code)"
  exit "$code"
}
trap on_err ERR

echo "=== Phase 1/2: Generate FULL visual_inception (07_shampoo, 09_toothbrush; 100 queries, batch=16, 90-neg retrieval) at $(date -u +"%Y-%m-%dT%H:%M:%SZ") ==="
write_status "RUNNING generate full_visual_inception"

for g in 07_shampoo 09_toothbrush; do
  if [ -f "$OUT/$g/visual_inception/FROZEN_IMAGE.json" ]; then
    echo "=== SKIP already frozen $g ==="
  else
    .venv/bin/python -u -B -X utf8 tools/run_experiment.py generate \
      --group "$g" \
      --method visual_inception \
      --training-bundles-root data/public/visual_inception \
      --output "$OUT" \
      --device cuda
  fi
done

for g in 07_shampoo 09_toothbrush; do
  if [ ! -f "$OUT/$g/visual_inception/FROZEN_IMAGE.json" ]; then
    echo "ERROR: missing $OUT/$g/visual_inception/FROZEN_IMAGE.json" >&2
    exit 1
  fi
done

echo "=== Phase 2/2: Evaluate training_oracle (official 50/30/20 text, 4 encoders, 2-group simultaneous pool replacement) at $(date -u +"%Y-%m-%dT%H:%M:%SZ") ==="
write_status "RUNNING evaluation training_oracle"

.venv/bin/python -u -B -X utf8 "$LOG_DIR/eval_full_training_oracle.py" \
  --output-root "$OUT" \
  --device cuda

write_status "COMPLETED training_oracle full 2groups"
echo "=== ALL DONE at $(date -u +"%Y-%m-%dT%H:%M:%SZ") ==="
