#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${ENV_FILE:-.env}"
if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  . "$ENV_FILE"
  set +a
fi

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-6006}"
VENV_DIR="${VENV_DIR:-$HOME/venvs/omnivoice-tts}"
APP_MODULE="${APP_MODULE:-omnivoice_api.tts.server:app}"

if [ -x "$VENV_DIR/bin/uvicorn" ]; then
  exec "$VENV_DIR/bin/uvicorn" "$APP_MODULE" --host "$HOST" --port "$PORT"
fi

if command -v uvicorn >/dev/null 2>&1; then
  exec uvicorn "$APP_MODULE" --host "$HOST" --port "$PORT"
fi

echo "uvicorn not found. Run ./scripts/install_tts.sh first, or activate the TTS venv." >&2
exit 1
