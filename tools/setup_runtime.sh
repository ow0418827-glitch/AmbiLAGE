#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONUTF8=1
RUNTIME_PYTHON="${AMBILAGE_PYTHON:-python3.11}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"
"$RUNTIME_PYTHON" -m venv .venv
.venv/bin/python -m pip install --upgrade pip --index-url "$PIP_INDEX_URL"
.venv/bin/python -m pip install 'torch>=2.6,<3' 'torchvision>=0.21,<1' --index-url "$TORCH_INDEX_URL"
.venv/bin/python -m pip install -r requirements/requirements.txt --index-url "$PIP_INDEX_URL"
.venv/bin/python -m pip install --no-deps --no-build-isolation -e .
.venv/bin/python -X utf8 tools/preflight.py --code-only --hardware --output output/deployment/preflight.json
