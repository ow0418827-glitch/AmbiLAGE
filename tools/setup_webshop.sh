#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONUTF8=1
RUNTIME_PYTHON="${AMBILAGE_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
SPACY_MODEL_WHEEL="${AMBILAGE_SPACY_MODEL_WHEEL:-https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.8.0/en_core_web_lg-3.8.0-py3-none-any.whl}"
if [[ ! -x "$RUNTIME_PYTHON" ]]; then
  echo "Create the project environment with tools/setup_runtime.sh, or set AMBILAGE_PYTHON." >&2
  exit 1
fi
"$RUNTIME_PYTHON" -B -X utf8 tools/check_webshop_runtime.py --java-only --require
"$RUNTIME_PYTHON" -m pip install -r requirements/requirements-webshop.txt --index-url "$PIP_INDEX_URL"
"$RUNTIME_PYTHON" -m pip install --no-deps "$SPACY_MODEL_WHEEL"
mkdir -p output/deployment
"$RUNTIME_PYTHON" -m pip freeze > output/deployment/requirements-webshop-resolved.txt
"$RUNTIME_PYTHON" -B -X utf8 tools/check_webshop_runtime.py --require --output output/deployment/webshop_runtime.json
