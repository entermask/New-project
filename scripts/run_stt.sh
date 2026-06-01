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
VENV_DIR="${VENV_DIR:-$HOME/venvs/qwen3-asr-api}"
APP_MODULE="${APP_MODULE:-omnivoice_api.stt.server:app}"

if [ -x "$VENV_DIR/bin/python" ]; then
  PYTHON_VERSION=$("$VENV_DIR/bin/python" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "3.12")
  SITE_PACKAGES="$VENV_DIR/lib/python$PYTHON_VERSION/site-packages"
  EXTRA_PATHS=""
  for lib_dir in \
    "$SITE_PACKAGES/nvidia/cu13/lib" \
    "$SITE_PACKAGES/nvidia/cuda_runtime/lib" \
    "$SITE_PACKAGES/nvidia/cudnn/lib" \
    "$SITE_PACKAGES/nvidia/cublas/lib" \
    "$SITE_PACKAGES/nvidia/nccl/lib"; do
    if [ -d "$lib_dir" ]; then
      if [ -n "$EXTRA_PATHS" ]; then
        EXTRA_PATHS="$EXTRA_PATHS:$lib_dir"
      else
        EXTRA_PATHS="$lib_dir"
      fi
    fi
  done
  if [ -n "$EXTRA_PATHS" ]; then
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$EXTRA_PATHS"
  fi
fi

if [ -x "$VENV_DIR/bin/uvicorn" ]; then
  exec "$VENV_DIR/bin/uvicorn" "$APP_MODULE" --host "$HOST" --port "$PORT"
fi

if command -v uvicorn >/dev/null 2>&1; then
  exec uvicorn "$APP_MODULE" --host "$HOST" --port "$PORT"
fi

echo "uvicorn not found. Run ./scripts/install_stt.sh first, or activate the STT venv." >&2
exit 1
