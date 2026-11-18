#!/usr/bin/env bash
set -euo pipefail

cd /content/AmbiLAGE
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

OUT_ROOT="output/ambilage_v4_2groups_20261118"
LOG_DIR="/content/AmbiLAGE/${OUT_ROOT}/logs"
STATUS_FILE="${LOG_DIR}/pipeline.status"
DRIVE_DEST="/content/drive/MyDrive/AmbiLAGE_backup/${OUT_ROOT}"

mkdir -p "${LOG_DIR}" "${DRIVE_DEST}"

sync_to_drive() {
  echo "[SYNC] Incremental rsync to ${DRIVE_DEST} at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  mkdir -p "${DRIVE_DEST}"
  rsync -a \
    --exclude='memory_vector_cache/' \
    --exclude='memory_vectors/' \
    "/content/AmbiLAGE/${OUT_ROOT}/" \
    "${DRIVE_DEST}/" || echo "[WARN] rsync failed at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}

on_err() {
  local code=$?
  echo "FAILED (exit ${code}) at $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${STATUS_FILE}"
  sync_to_drive
  exit "${code}"
}
trap on_err ERR

# 1. Generate 07_shampoo (AmbiLAGE v4)
echo "RUNNING generate 07_shampoo at $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${STATUS_FILE}"
echo "=== [1/6] Generate AmbiLAGE v4: 07_shampoo at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
.venv/bin/python -u -X utf8 tools/run_experiment.py generate \
  --group 07_shampoo \
  --method ambilage \
  --output "${OUT_ROOT}" \
  --device cuda
sync_to_drive

# 2. Generate 09_toothbrush (AmbiLAGE v4)
echo "RUNNING generate 09_toothbrush at $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${STATUS_FILE}"
echo "=== [2/6] Generate AmbiLAGE v4: 09_toothbrush at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
.venv/bin/python -u -X utf8 tools/run_experiment.py generate \
  --group 09_toothbrush \
  --method ambilage \
  --output "${OUT_ROOT}" \
  --device cuda
sync_to_drive

# 3. Eval 1a: Visual Inception protocol (key_mode=image, training_oracle) on ViT-g-14 (held-out)
echo "RUNNING eval1 g14_image_key at $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${STATUS_FILE}"
echo "=== [3/6] Evaluate AmbiLAGE v4: training_oracle + key_mode=image (ViT-g-14) at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
.venv/bin/python -u -X utf8 tools/evaluate_visual_inception_category.py evaluate \
  --images-root "${OUT_ROOT}" \
  --output "${OUT_ROOT}/eval_training_oracle_image_key/g14_image_key" \
  --method ambilage \
  --group 07_shampoo,09_toothbrush \
  --replacement-groups selected \
  --query-suite training_oracle \
  --key-mode image \
  --defense-mode none \
  --device cuda
.venv/bin/python -u -X utf8 tools/evaluate_visual_inception_category.py verify \
  --output "${OUT_ROOT}/eval_training_oracle_image_key/g14_image_key" \
  --device cuda
sync_to_drive

# 4. Eval 1b: Visual Inception protocol (key_mode=image, training_oracle) on ViT-B-32 (matched surrogate)
echo "RUNNING eval1 b32_image_key at $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${STATUS_FILE}"
echo "=== [4/6] Evaluate AmbiLAGE v4: training_oracle + key_mode=image (ViT-B-32 matched) at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
.venv/bin/python -u -X utf8 tools/evaluate_visual_inception_category.py evaluate \
  --images-root "${OUT_ROOT}" \
  --output "${OUT_ROOT}/eval_training_oracle_image_key/b32_image_key" \
  --method ambilage \
  --group 07_shampoo,09_toothbrush \
  --replacement-groups selected \
  --query-suite training_oracle \
  --key-mode image \
  --defense-mode none \
  --encoder-config "${LOG_DIR}/eval_b32_matched.json" \
  --device cuda
.venv/bin/python -u -X utf8 tools/evaluate_visual_inception_category.py verify \
  --output "${OUT_ROOT}/eval_training_oracle_image_key/b32_image_key" \
  --device cuda
sync_to_drive

# 5. Eval 2: Original Env 4 (none_target_joint, key_mode=joint, ViT-g-14 & ViT-B-32)
echo "RUNNING eval2 env4 at $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${STATUS_FILE}"
echo "=== [5/6] Evaluate AmbiLAGE v4: Env 4 (none_target_joint) on ViT-g-14 & ViT-B-32 at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
.venv/bin/python -u -X utf8 "${LOG_DIR}/eval_group_env4.py" \
  --output-root "${OUT_ROOT}" \
  --groups 07_shampoo 09_toothbrush \
  --device cuda
sync_to_drive

echo "COMPLETED at $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${STATUS_FILE}"
echo "=== [6/6] ALL AmbiLAGE v4 GENERATION & DUAL-ENV EVALUATIONS COMPLETED AND SYNCED at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
