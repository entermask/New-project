#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="${SESSION_NAME:-omnivoice}"
APP_DIR="${APP_DIR:-$HOME/New-project}"
VENV_DIR="${VENV_DIR:-$HOME/venvs/omnivoice-api}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8001}"
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

tmux new-session -d -s "$SESSION_NAME" \
  "cd '$APP_DIR' && \
   source '$VENV_DIR/bin/activate' && \
   set -a && source .env && set +a && \
   exec '$VENV_DIR/bin/uvicorn' app:app --host '$HOST' --port '$PORT'"

echo "Started OmniVoice API in tmux session '$SESSION_NAME' on $HOST:$PORT"
echo "Attach logs: tmux attach -t $SESSION_NAME"
echo "Detach: Ctrl+B then D"
