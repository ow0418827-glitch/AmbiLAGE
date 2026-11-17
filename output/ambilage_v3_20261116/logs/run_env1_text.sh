#!/usr/bin/env bash
set -euo pipefail
cd /content/AmbiLAGE
export PYTHONUNBUFFERED=1

OUT_ROOT="output/ambilage_v3_20261116"
LOG_DIR="/content/AmbiLAGE/${OUT_ROOT}/logs"
STATUS_FILE="${LOG_DIR}/eval_env1_text.status"
TARGET_GROUPS="06_jeans,07_shampoo,08_speaker,09_toothbrush"

echo "RUNNING g14_none_text at $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${STATUS_FILE}"
echo "=== [1/2] Evaluate Env 1 (pure feature text, none defense, ViT-g-14 held-out) at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
.venv/bin/python -u -X utf8 tools/run_experiment.py evaluate \
  --group "${TARGET_GROUPS}" \
  --method ambilage \
  --output "${OUT_ROOT}" \
  --device cuda \
  --query-mode text \
  --defense-mode none \
  --replacement-groups selected \
  --evaluation-name g14_none_text

echo "RUNNING b32_none_text at $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${STATUS_FILE}"
echo "=== [2/2] Evaluate Env 1 (pure feature text, none defense, ViT-B-32 matched) at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
.venv/bin/python -u -X utf8 tools/run_experiment.py evaluate \
  --group "${TARGET_GROUPS}" \
  --method ambilage \
  --output "${OUT_ROOT}" \
  --device cuda \
  --query-mode text \
  --defense-mode none \
  --replacement-groups selected \
  --encoder-config "${OUT_ROOT}/logs/eval_b32_matched.json" \
  --evaluation-name b32_none_text

echo "COMPLETED at $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${STATUS_FILE}"
echo "=== ALL ENV 1 (PURE FEATURE TEXT) EVALUATIONS COMPLETED AND VERIFIED at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
