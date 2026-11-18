#!/usr/bin/env bash
set -euo pipefail

cd /content/AmbiLAGE
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

OUT="output/visual_inception_enhanced_v3_2groups_20261118"
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

echo "=== Phase 1/4: Generate Visual Inception enhanced_v3 (07_shampoo, 09_toothbrush) at $(date -u +"%Y-%m-%dT%H:%M:%SZ") ==="
write_status "RUNNING phase 1/4 generate enhanced_v3"

if [ ! -f "$OUT/07_shampoo/visual_inception/FROZEN_IMAGE.json" ] || [ ! -f "$OUT/09_toothbrush/visual_inception/FROZEN_IMAGE.json" ]; then
  .venv/bin/python -u -B -X utf8 tools/run_experiment.py generate \
    --group 07_shampoo,09_toothbrush \
    --method visual_inception \
    --training-bundles-root data/public/visual_inception \
    --visual-inception-config src/visual/config/visual_inception.example.json \
    --output "$OUT" \
    --device cuda
else
  echo "=== SKIP generate (both FROZEN_IMAGE.json already exist) ==="
fi

.venv/bin/python -u -B -X utf8 -c '
import json
from pathlib import Path
from src.visual._visual_inception.evidence import ENHANCED_OBJECTIVE_KIND, validate_ready

out = Path("output/visual_inception_enhanced_v3_2groups_20261118")
for g in ("07_shampoo", "09_toothbrush"):
    frozen = json.loads((out / g / "visual_inception" / "FROZEN_IMAGE.json").read_text(encoding="utf-8"))
    assert frozen["status"] == "frozen", g
    assert frozen["method"] == "visual_inception", g
    validate_ready(out / g / "visual_inception" / "generation", require_production=True)
    meta = frozen["metadata"]
    obj = meta["training_objective"]
    assert obj["kind"] == ENHANCED_OBJECTIVE_KIND, obj["kind"]
    assert obj["mechanism_profile"] == "enhanced_v3", obj.get("mechanism_profile")
    assert meta["decoded_image_audit"]["within_budget"] is True
    assert meta["decoded_image_audit"]["max_absolute_pixel_difference_levels"] <= 8
    print(
        "VERIFIED frozen",
        g,
        "kind=" + obj["kind"],
        "max_linf=" + str(meta["decoded_image_audit"]["max_absolute_pixel_difference_levels"]),
        "secs=" + str(round(meta["optimization_seconds"], 2)),
    )
'

echo "=== Phase 2/4: Official category eval + verify on held-out eval_g14 (key_mode=image, training_oracle) at $(date -u +"%Y-%m-%dT%H:%M:%SZ") ==="
write_status "RUNNING phase 2/4 eval g14_image_key"

if [ ! -f "$OUT/eval_training_oracle_image_key/g14_image_key/category_independent_verification.json" ]; then
  .venv/bin/python -u -B -X utf8 tools/evaluate_visual_inception_category.py evaluate \
    --images-root "$OUT" \
    --output "$OUT/eval_training_oracle_image_key/g14_image_key" \
    --method visual_inception \
    --group 07_shampoo,09_toothbrush \
    --replacement-groups selected \
    --query-suite training_oracle \
    --key-mode image \
    --defense-mode none \
    --device cuda

  .venv/bin/python -u -B -X utf8 tools/evaluate_visual_inception_category.py verify \
    --output "$OUT/eval_training_oracle_image_key/g14_image_key" \
    --device cuda
else
  echo "=== SKIP phase 2/4 g14_image_key (already verified) ==="
fi

echo "=== Phase 3/4: Official category eval + verify on held-out ViT-B-32 (key_mode=image, training_oracle) at $(date -u +"%Y-%m-%dT%H:%M:%SZ") ==="
write_status "RUNNING phase 3/4 eval b32_image_key"

.venv/bin/python -u -B -X utf8 tools/evaluate_visual_inception_category.py evaluate \
  --images-root "$OUT" \
  --output "$OUT/eval_training_oracle_image_key/b32_image_key" \
  --method visual_inception \
  --group 07_shampoo,09_toothbrush \
  --replacement-groups selected \
  --query-suite training_oracle \
  --key-mode image \
  --defense-mode none \
  --encoder-config "$LOG_DIR/eval_b32_heldout.json" \
  --device cuda

.venv/bin/python -u -B -X utf8 tools/evaluate_visual_inception_category.py verify \
  --output "$OUT/eval_training_oracle_image_key/b32_image_key" \
  --device cuda

echo "=== Phase 4/4: 4-Encoder Full Eval (eval_g14 + 3 white-box surrogates) at $(date -u +"%Y-%m-%dT%H:%M:%SZ") ==="
write_status "RUNNING phase 4/4 eval 4 encoders"

.venv/bin/python -u -B -X utf8 "$LOG_DIR/eval_enhanced_v3_image_key_training_oracle.py" \
  --output-root "$OUT" \
  --device cuda

write_status "COMPLETED enhanced_v3 2groups key_mode=image training_oracle"
echo "=== ALL DONE at $(date -u +"%Y-%m-%dT%H:%M:%SZ") ==="
