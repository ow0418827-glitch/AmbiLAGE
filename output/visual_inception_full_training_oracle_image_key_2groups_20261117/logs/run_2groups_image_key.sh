#!/usr/bin/env bash
set -euo pipefail

cd /content/AmbiLAGE
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

OUT="output/visual_inception_full_training_oracle_image_key_2groups_20261117"
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

echo "=== Evaluate frozen full visual_inception with DEFAULT key_mode=image (07_shampoo, 09_toothbrush) at $(date -u +"%Y-%m-%dT%H:%M:%SZ") ==="
write_status "RUNNING evaluation image_key training_oracle"

.venv/bin/python -u -B -X utf8 "$LOG_DIR/eval_image_key_training_oracle.py" \
  --output-root "$OUT" \
  --device cuda

write_status "COMPLETED image_key training_oracle full 2groups"
echo "=== ALL DONE at $(date -u +"%Y-%m-%dT%H:%M:%SZ") ==="
