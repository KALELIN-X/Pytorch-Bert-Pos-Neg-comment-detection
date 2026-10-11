#!/usr/bin/env bash
# Explicit demo launcher; installation/model download are described in README.
# Never import private datasets, train, or start/delete database volumes implicitly.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ -x .venv/bin/python ]]; then
  python_cmd=.venv/bin/python
elif [[ -x .venv/Scripts/python.exe ]]; then
  python_cmd=.venv/Scripts/python.exe
else
  python_cmd=python
fi
exec "$python_cmd" -m uvicorn src.api:app --host 127.0.0.1 --port 8000
