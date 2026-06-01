#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${VENV_DIR:-$HOME/venvs/omnivoice-tts}"
APP_PORT="${APP_PORT:-6006}"
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

"$PYTHON" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install -U pip wheel

IS_5090=0
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true)
  if echo "$GPU_NAME" | grep -qiE "5090|blackwell"; then
    IS_5090=1
  fi
fi

if [ "$IS_5090" = "1" ]; then
  python -m pip install -U --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
else
  python -m pip install torch==2.5.1+cu121 torchaudio==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
fi

python -m pip install -r requirements-tts.txt
python -m pip install git+https://github.com/entermask/omnivoice-triton.git --no-deps

if [ ! -f .env ]; then
  cp .env.tts.example .env
fi

echo "Installed OmniVoice TTS API dependencies in $VENV_DIR"
if [ "${INSTALL_ONLY:-0}" = "1" ] || [ "${INSTALL_ONLY:-0}" = "true" ]; then
  exit 0
fi

exec ./scripts/run_tts.sh
