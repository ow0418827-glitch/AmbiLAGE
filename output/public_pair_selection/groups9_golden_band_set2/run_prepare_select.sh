#!/usr/bin/env bash
set -euo pipefail
cd /content/AmbiLAGE
echo "[START prepare] $(date -Is)"
/content/AmbiLAGE/.venv/bin/python -X utf8 -B tools/select_public_pairs.py prepare \
  --config src/experiments/config/selection_groups9_golden_band_set2.json \
  --products resources/webshop/catalog/catalog.sqlite \
  --index output/datasets/public/public_products_v2.sqlite \
  --root output/public_pair_selection/groups9_golden_band_set2 \
  --device cuda
echo "[END prepare / START select] $(date -Is)"
/content/AmbiLAGE/.venv/bin/python -X utf8 -B tools/select_public_pairs.py select \
  --config src/experiments/config/selection_groups9_golden_band_set2.json \
  --products resources/webshop/catalog/catalog.sqlite \
  --index output/datasets/public/public_products_v2.sqlite \
  --root output/public_pair_selection/groups9_golden_band_set2 \
  --device cuda
echo "[DONE select] $(date -Is)"
