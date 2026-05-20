#!/usr/bin/env bash
# Script to automate setting up the systemd service for running OmniVoice in tmux on boot

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
SERVICE_NAME="omnivoice-tmux"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

if [ "$EUID" -ne 0 ]; then
  echo "Error: This script must be run as root (or with sudo)." >&2
  exit 1
fi

echo "Creating Systemd service file at ${SERVICE_FILE}..."

cat <<EOF > "${SERVICE_FILE}"
[Unit]
Description=OmniVoice Tmux Service
After=network.target

[Service]
Type=forking
User=root
WorkingDirectory=${APP_DIR}
ExecStart=${APP_DIR}/scripts/run_tmux.sh
ExecStop=/usr/bin/tmux kill-session -t omnivoice
Restart=on-failure
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

echo "Reloading systemd daemon..."
systemctl daemon-reload

echo "Enabling ${SERVICE_NAME}.service to run on boot..."
systemctl enable "${SERVICE_NAME}.service"

echo "Starting ${SERVICE_NAME}.service now..."
systemctl start "${SERVICE_NAME}.service"

echo "Systemd service setup complete!"
echo "You can check the status with: systemctl status ${SERVICE_NAME}.service"
echo "You can attach to the tmux session with: tmux a"
