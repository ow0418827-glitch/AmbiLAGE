#!/usr/bin/env bash
set -euo pipefail
cd /content/AmbiLAGE
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

OUT_BASE="output/ambilage_v3_20261116_training_oracle_image_key_eval"

echo "=== [1/4] AmbiLAGE v3 training_oracle key_mode=image (ViT-g-14) at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
.venv/bin/python -u -X utf8 tools/evaluate_visual_inception_category.py evaluate \
  --images-root output/ambilage_v3_20261116 \
  --output "${OUT_BASE}/g14_image_key" \
  --method ambilage \
  --group 07_shampoo,09_toothbrush \
  --replacement-groups selected \
  --query-suite training_oracle \
  --key-mode image \
  --defense-mode none \
  --device cuda

echo "=== [2/4] Verify AmbiLAGE v3 (ViT-g-14) at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
.venv/bin/python -u -X utf8 tools/evaluate_visual_inception_category.py verify \
  --output "${OUT_BASE}/g14_image_key" \
  --device cuda

echo "=== [3/4] AmbiLAGE v3 training_oracle key_mode=image (ViT-B-32 matched) at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
.venv/bin/python -u -X utf8 tools/evaluate_visual_inception_category.py evaluate \
  --images-root output/ambilage_v3_20261116 \
  --output "${OUT_BASE}/b32_image_key" \
  --method ambilage \
  --group 07_shampoo,09_toothbrush \
  --replacement-groups selected \
  --query-suite training_oracle \
  --key-mode image \
  --defense-mode none \
  --encoder-config output/ambilage_v3_20261116/logs/eval_b32_matched.json \
  --device cuda

echo "=== [4/4] Verify AmbiLAGE v3 (ViT-B-32 matched) at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
.venv/bin/python -u -X utf8 tools/evaluate_visual_inception_category.py verify \
  --output "${OUT_BASE}/b32_image_key" \
  --device cuda

echo "=== ALL V3 IMAGE-KEY EVALUATIONS COMPLETED at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
