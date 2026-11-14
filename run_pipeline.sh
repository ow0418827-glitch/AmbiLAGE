#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$PROJECT_ROOT"
export PYTHONUTF8=1
RUNTIME_PYTHON="${AMBILAGE_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
if [[ ! -x "$RUNTIME_PYTHON" ]]; then RUNTIME_PYTHON=python3; fi
exec "$RUNTIME_PYTHON" -X utf8 -B -m src.experiments.pipeline "$@"
