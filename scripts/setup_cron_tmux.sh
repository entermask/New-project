#!/usr/bin/env bash
# This script sets up a crontab entry to automatically run the run_tmux.sh
# script on system boot/reboot using @reboot.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

# Ensure cron is installed
if ! command -v crontab >/dev/null 2>&1; then
  echo "cron is not installed! Installing cron..."
  if command -v apt-get >/dev/null 2>&1; then
    apt-get update && apt-get install -y cron
  else
    echo "Could not find apt-get. Please install cron manually." >&2
    exit 1
  fi
fi

# Ensure cron daemon is started
echo "Starting cron daemon..."
if command -v service >/dev/null 2>&1; then
  service cron start || true
elif command -v systemctl >/dev/null 2>&1; then
  systemctl start cron || true
else
  /usr/sbin/cron || true
fi

CRON_ENTRY="@reboot /bin/bash ${APP_DIR}/scripts/run_tmux.sh"

echo "Configuring crontab..."
# Fetch existing crontab (ignoring any existing run_tmux.sh entries to avoid duplicates)
# and append the new @reboot entry.
(crontab -l 2>/dev/null | grep -v "run_tmux.sh" || true; echo "$CRON_ENTRY") | crontab -

echo "--------------------------------------------------------"
echo "OmniVoice Cron @reboot successfully configured!"
echo "Current crontab entries:"
crontab -l
echo "--------------------------------------------------------"
echo "Note: The script will now automatically run in tmux on system reboot."
echo "To launch the tmux session right now without rebooting, run:"
echo "  bash ${APP_DIR}/scripts/run_tmux.sh"
echo "--------------------------------------------------------"
