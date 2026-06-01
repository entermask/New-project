#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${VENV_DIR:-$HOME/venvs/qwen3-asr-api}"
APP_PORT="${APP_PORT:-6006}"
QWEN_ASR_INSTALL_MODE="${QWEN_ASR_INSTALL_MODE:-stable}"
INSTALL_NVIDIA_DRIVER="${INSTALL_NVIDIA_DRIVER:-auto}"
NVIDIA_DRIVER_VERSION="${NVIDIA_DRIVER_VERSION:-}"
ALLOW_DEADSNAKES_PPA="${ALLOW_DEADSNAKES_PPA:-1}"
CONFIGURE_UFW="${CONFIGURE_UFW:-1}"
ENABLE_UFW="${ENABLE_UFW:-0}"
SSH_PORT="${SSH_PORT:-22}"

if command -v sudo >/dev/null 2>&1; then
  SUDO=(sudo)
else
  SUDO=()
fi

apt_update() { "${SUDO[@]}" apt-get update; }
apt_install() { "${SUDO[@]}" apt-get install -y "$@"; }

python_is_312_plus() {
  "$1" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)' >/dev/null 2>&1
}

find_python() {
  local candidate
  for candidate in "${PYTHON_BIN:-}" python3.13 python3.12 python3; do
    if [ -n "$candidate" ] && command -v "$candidate" >/dev/null 2>&1 && python_is_312_plus "$candidate"; then
      command -v "$candidate"
      return 0
    fi
  done
  return 1
}

install_python_312() {
  apt_update
  if apt_install python3.12 python3.12-venv python3.12-dev; then
    return 0
  fi
  if [ "$ALLOW_DEADSNAKES_PPA" = "1" ]; then
    apt_install software-properties-common ca-certificates
    "${SUDO[@]}" add-apt-repository -y ppa:deadsnakes/ppa
    apt_update
    apt_install python3.12 python3.12-venv python3.12-dev
    return 0
  fi
  echo "Could not install Python 3.12. Use Ubuntu 24.04+ or set PYTHON_BIN." >&2
  exit 1
}

install_system_packages() {
  apt_update
  apt_install ffmpeg tmux curl ca-certificates lsb-release build-essential gcc g++ ufw
}

configure_firewall() {
  if [ "$CONFIGURE_UFW" = "0" ] || [ "$CONFIGURE_UFW" = "false" ]; then
    return 0
  fi
  "${SUDO[@]}" ufw allow "$SSH_PORT/tcp"
  "${SUDO[@]}" ufw allow "$APP_PORT/tcp"
  if [ "$ENABLE_UFW" = "1" ] || [ "$ENABLE_UFW" = "true" ]; then
    "${SUDO[@]}" ufw --force enable
  fi
  "${SUDO[@]}" ufw status verbose || true
}

install_nvidia_driver_if_needed() {
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
    return 0
  fi
  if [ "$INSTALL_NVIDIA_DRIVER" = "0" ] || [ "$INSTALL_NVIDIA_DRIVER" = "false" ]; then
    return 0
  fi
  apt_update
  apt_install ubuntu-drivers-common pciutils
  if [ -n "$NVIDIA_DRIVER_VERSION" ]; then
    apt_install "nvidia-driver-$NVIDIA_DRIVER_VERSION"
  elif ubuntu-drivers install --help 2>/dev/null | grep -q -- "--gpgpu"; then
    "${SUDO[@]}" ubuntu-drivers install --gpgpu
  else
    "${SUDO[@]}" ubuntu-drivers autoinstall
  fi
  echo "NVIDIA driver install finished. Reboot may be required."
}

install_system_packages
configure_firewall
install_nvidia_driver_if_needed

PYTHON="$(find_python || true)"
if [ -z "$PYTHON" ]; then
  install_python_312
  PYTHON="$(find_python || true)"
fi
[ -n "$PYTHON" ] || { echo "Python >=3.12 unavailable." >&2; exit 1; }

if command -v uv >/dev/null 2>&1; then
  UV_BIN="$(command -v uv)"
else
  curl -LsSf https://astral.sh/uv/install.sh | sh
  UV_BIN="$HOME/.local/bin/uv"
fi

"$UV_BIN" venv "$VENV_DIR" --python "$PYTHON" --clear
source "$VENV_DIR/bin/activate"

if [ "$QWEN_ASR_INSTALL_MODE" = "nightly" ]; then
  "$UV_BIN" pip install -U vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly
  "$UV_BIN" pip install -U qwen-asr -r requirements-common.txt
elif [ "$QWEN_ASR_INSTALL_MODE" = "stable" ]; then
  "$UV_BIN" pip install -r requirements-stt.txt
else
  echo "QWEN_ASR_INSTALL_MODE must be nightly or stable." >&2
  exit 1
fi

if [ ! -f .env ]; then
  cp .env.stt.example .env
fi

echo "Installed Qwen3-ASR API dependencies in $VENV_DIR"
exec ./scripts/run_stt.sh
