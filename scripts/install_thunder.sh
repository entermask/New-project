#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${VENV_DIR:-$HOME/venvs/omnivoice-api}"

python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install -U pip wheel
python -m pip install -r requirements.txt

if command -v sudo >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y ffmpeg
else
  apt-get update
  apt-get install -y ffmpeg
fi

echo "Installed OmniVoice API dependencies in $VENV_DIR"
