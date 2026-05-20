#!/usr/bin/env bash
# This script sets up a systemd service that automatically launches the
# OmniVoice API inside a tmux session on system boot or reboot.
set -euo pipefail

SERVICE_NAME="omnivoice-tmux"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

# Ensure the script is run as root
if [ "$EUID" -ne 0 ]; then
  echo "Please run this script as root (sudo)." >&2
  exit 1
fi

# Ensure tmux is installed
if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed! Installing tmux..."
  if command -v apt-get >/dev/null 2>&1; then
    apt-get update && apt-get install -y tmux
  else
    echo "Could not find apt-get. Please install tmux manually." >&2
    exit 1
  fi
fi

TMUX_PATH="$(command -v tmux)"

echo "Creating systemd service file at ${SERVICE_FILE}..."
cat <<EOF > "${SERVICE_FILE}"
[Unit]
Description=OmniVoice Tmux Service
After=network.target

[Service]
Type=forking
User=root
WorkingDirectory=${APP_DIR}
ExecStart=${APP_DIR}/scripts/run_tmux.sh
ExecStop=${TMUX_PATH} kill-session -t omnivoice
Restart=on-failure
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

echo "Reloading systemd daemon..."
systemctl daemon-reload

echo "Enabling ${SERVICE_NAME}.service to run on system boot..."
systemctl enable "${SERVICE_NAME}.service"

echo "Starting ${SERVICE_NAME}.service..."
systemctl restart "${SERVICE_NAME}.service"

echo "--------------------------------------------------------"
echo "OmniVoice Tmux Service successfully configured and started!"
echo "Check the service status: systemctl status ${SERVICE_NAME}"
echo "Attach to the running tmux session: tmux attach -t omnivoice"
echo "--------------------------------------------------------"
