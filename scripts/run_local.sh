#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8001}"
VENV_DIR="${VENV_DIR:-$HOME/venvs/omnivoice-api}"

if [ -x "$VENV_DIR/bin/uvicorn" ]; then
  exec "$VENV_DIR/bin/uvicorn" app:app --host "$HOST" --port "$PORT"
fi

if command -v uvicorn >/dev/null 2>&1; then
  exec uvicorn app:app --host "$HOST" --port "$PORT"
fi

echo "uvicorn not found. Run ./scripts/install_thunder.sh first, or activate the venv." >&2
exit 1
