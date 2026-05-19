#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="${SESSION_NAME:-omnivoice}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_APP_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
APP_DIR="${APP_DIR:-$DEFAULT_APP_DIR}"
VENV_DIR="${VENV_DIR:-$HOME/venvs/omnivoice-api}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8001}"
RESTART="${RESTART:-1}"
WORKERS="${WORKERS:-5}"
USE_MPS="${USE_MPS:-1}"

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

# Enable NVIDIA MPS if requested and nvidia-smi is available
if [ "$USE_MPS" = "1" ] && command -v nvidia-smi >/dev/null 2>&1; then
  echo "Setting up NVIDIA MPS (Multi-Process Service)..."
  
  # 1. Try to set EXCLUSIVE_PROCESS mode for GPU 0
  if nvidia-smi -i 0 -c EXCLUSIVE_PROCESS >/dev/null 2>&1; then
    echo "GPU 0 set to EXCLUSIVE_PROCESS mode."
  elif sudo nvidia-smi -i 0 -c EXCLUSIVE_PROCESS >/dev/null 2>&1; then
    echo "GPU 0 set to EXCLUSIVE_PROCESS mode using sudo."
  else
    echo "Warning: Could not set GPU 0 to EXCLUSIVE_PROCESS mode. Skipping mode change..."
  fi

  # 2. Start the MPS control daemon
  if nvidia-cuda-mps-control -d >/dev/null 2>&1; then
    echo "NVIDIA MPS control daemon started."
  else
    if pgrep nvidia-cuda-mps >/dev/null 2>&1; then
      echo "NVIDIA MPS control daemon is already running."
    else
      echo "Warning: Failed to start NVIDIA MPS control daemon."
    fi
  fi
fi

tmux new-session -d -s "$SESSION_NAME" \
  "cd '$APP_DIR' && \
   source '$VENV_DIR/bin/activate' && \
   set -a && source .env && set +a && \
   ${EXTRA_PATHS:+export LD_LIBRARY_PATH=\"\$LD_LIBRARY_PATH:$EXTRA_PATHS\" &&} \
   exec '$VENV_DIR/bin/uvicorn' app:app --host '$HOST' --port '$PORT' --workers '$WORKERS'"

echo "Started OmniVoice API in tmux session '$SESSION_NAME' on $HOST:$PORT"
echo "Attaching to session '$SESSION_NAME' now... (Detach with Ctrl+B then D)"

exec tmux attach-session -t "$SESSION_NAME"
