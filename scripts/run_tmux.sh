#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="${SESSION_NAME:-omnivoice}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_APP_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
APP_DIR="${APP_DIR:-$DEFAULT_APP_DIR}"
VENV_DIR="${VENV_DIR:-$HOME/venvs/omnivoice-api}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
RESTART="${RESTART:-1}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed. Run: sudo apt-get update && sudo apt-get install -y tmux" >&2
  exit 1
fi

if [ ! -f "$APP_DIR/.env" ]; then
  echo "Missing $APP_DIR/.env" >&2
  exit 1
fi

if [ ! -x "$VENV_DIR/bin/uvicorn" ]; then
  echo "Missing $VENV_DIR/bin/uvicorn. Run ./scripts/install_thunder.sh first." >&2
  exit 1
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  if [ "$RESTART" = "1" ]; then
    tmux kill-session -t "$SESSION_NAME"
  else
    echo "tmux session '$SESSION_NAME' already exists. Attach with: tmux attach -t $SESSION_NAME"
    exit 0
  fi
fi

# Kill any process currently occupying the port
if command -v lsof >/dev/null 2>&1; then
  PIDS=$(lsof -t -i :"$PORT" || true)
  if [ -n "$PIDS" ]; then
    echo "Port $PORT is in use. Killing process(es): $PIDS"
    echo "$PIDS" | xargs kill -9 || true
    sleep 1
  fi
fi

# Detect site-packages path inside VENV_DIR to auto-configure Nvidia CUDA library paths
PYTHON_VERSION=$( "$VENV_DIR/bin/python" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "3.10" )
SITE_PACKAGES="$VENV_DIR/lib/python$PYTHON_VERSION/site-packages"

EXTRA_PATHS=""
if [ -d "$SITE_PACKAGES/nvidia/cublas/lib" ]; then
  EXTRA_PATHS="$SITE_PACKAGES/nvidia/cublas/lib"
fi
if [ -d "$SITE_PACKAGES/nvidia/cudnn/lib" ]; then
  if [ -n "$EXTRA_PATHS" ]; then
    EXTRA_PATHS="$EXTRA_PATHS:$SITE_PACKAGES/nvidia/cudnn/lib"
  else
    EXTRA_PATHS="$SITE_PACKAGES/nvidia/cudnn/lib"
  fi
fi

tmux new-session -d -s "$SESSION_NAME" \
  "cd '$APP_DIR' && \
   source '$VENV_DIR/bin/activate' && \
   set -a && source .env && set +a && \
   ${EXTRA_PATHS:+export LD_LIBRARY_PATH=\"\$LD_LIBRARY_PATH:$EXTRA_PATHS\" &&} \
   exec '$VENV_DIR/bin/uvicorn' app:app --host '$HOST' --port '$PORT'"

echo "Started OmniVoice API in tmux session '$SESSION_NAME' on $HOST:$PORT"
echo "Attach logs: tmux attach -t $SESSION_NAME"
echo "Detach: Ctrl+B then D"
