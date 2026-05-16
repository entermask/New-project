#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${VENV_DIR:-$HOME/venvs/omnivoice-api}"
INSTALL_NVIDIA_DRIVER="${INSTALL_NVIDIA_DRIVER:-auto}"
NVIDIA_DRIVER_VERSION="${NVIDIA_DRIVER_VERSION:-}"
ALLOW_DEADSNAKES_PPA="${ALLOW_DEADSNAKES_PPA:-1}"
CONFIGURE_UFW="${CONFIGURE_UFW:-1}"
ENABLE_UFW="${ENABLE_UFW:-0}"
APP_PORT="${APP_PORT:-8001}"
SSH_PORT="${SSH_PORT:-22}"
INSTALL_TORCH_CUDA="${INSTALL_TORCH_CUDA:-auto}"
TORCH_CUDA_INDEX_URL="${TORCH_CUDA_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

if command -v sudo >/dev/null 2>&1; then
  SUDO=(sudo)
else
  SUDO=()
fi

apt_update() {
  "${SUDO[@]}" apt-get update
}

apt_install() {
  "${SUDO[@]}" apt-get install -y "$@"
}

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
  echo "Python >=3.12 not found; installing Python 3.12..."
  apt_update
  if apt_install python3.12 python3.12-venv python3.12-dev; then
    return 0
  fi

  if [ "$ALLOW_DEADSNAKES_PPA" = "1" ]; then
    echo "python3.12 packages were not available; trying deadsnakes PPA..."
    apt_install software-properties-common ca-certificates
    "${SUDO[@]}" add-apt-repository -y ppa:deadsnakes/ppa
    apt_update
    apt_install python3.12 python3.12-venv python3.12-dev
    return 0
  fi

  echo "Could not install Python 3.12. Use an Ubuntu 24.04+ image or set PYTHON_BIN=/path/to/python3.12." >&2
  exit 1
}

install_system_packages() {
  apt_update
  apt_install ffmpeg tmux curl ca-certificates lsb-release build-essential gcc g++ ufw
}

configure_firewall() {
  if [ "$CONFIGURE_UFW" = "0" ] || [ "$CONFIGURE_UFW" = "false" ]; then
    echo "Skipping UFW configuration because CONFIGURE_UFW=$CONFIGURE_UFW"
    return 0
  fi

  echo "Configuring UFW rules for SSH port $SSH_PORT and app port $APP_PORT..."
  "${SUDO[@]}" ufw allow "$SSH_PORT/tcp"
  "${SUDO[@]}" ufw allow "$APP_PORT/tcp"

  if [ "$ENABLE_UFW" = "1" ] || [ "$ENABLE_UFW" = "true" ]; then
    "${SUDO[@]}" ufw --force enable
  else
    echo "UFW rules added but firewall not enabled. To enable now, run: sudo ufw enable"
  fi

  "${SUDO[@]}" ufw status verbose || true
}

install_nvidia_driver_if_needed() {
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    echo "NVIDIA driver already works:"
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
    return 0
  fi

  if [ "$INSTALL_NVIDIA_DRIVER" = "0" ] || [ "$INSTALL_NVIDIA_DRIVER" = "false" ]; then
    echo "NVIDIA driver is not working, but INSTALL_NVIDIA_DRIVER=$INSTALL_NVIDIA_DRIVER; skipping."
    return 0
  fi

  echo "NVIDIA driver not detected; installing Ubuntu NVIDIA driver packages..."
  apt_update
  apt_install ubuntu-drivers-common pciutils

  if [ -n "$NVIDIA_DRIVER_VERSION" ]; then
    apt_install "nvidia-driver-$NVIDIA_DRIVER_VERSION"
  elif ubuntu-drivers install --help 2>/dev/null | grep -q -- "--gpgpu"; then
    "${SUDO[@]}" ubuntu-drivers install --gpgpu
  else
    "${SUDO[@]}" ubuntu-drivers autoinstall
  fi

  echo "NVIDIA driver install finished. A reboot or instance restart may be required before nvidia-smi works."
}

detect_gpu_name() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1 || true
  fi
}

should_install_torch_cuda() {
  case "$INSTALL_TORCH_CUDA" in
    1|true|yes)
      return 0
      ;;
    0|false|no)
      return 1
      ;;
    auto)
      local gpu_name
      gpu_name="$(detect_gpu_name)"
      [[ "$gpu_name" == *"L40S"* ]]
      return
      ;;
    *)
      echo "INSTALL_TORCH_CUDA must be auto, 1, or 0." >&2
      exit 1
      ;;
  esac
}

install_torch_cuda_if_needed() {
  if should_install_torch_cuda; then
    echo "Installing PyTorch CUDA wheels from $TORCH_CUDA_INDEX_URL..."
    python -m pip install torch torchvision torchaudio --index-url "$TORCH_CUDA_INDEX_URL"
  else
    echo "Skipping explicit PyTorch CUDA wheel install. Set INSTALL_TORCH_CUDA=1 to force it."
  fi
}

install_system_packages
configure_firewall
install_nvidia_driver_if_needed

PYTHON="$(find_python || true)"
if [ -z "$PYTHON" ]; then
  install_python_312
  PYTHON="$(find_python || true)"
fi

if [ -z "$PYTHON" ]; then
  echo "Python >=3.12 is still unavailable after install." >&2
  exit 1
fi

echo "Using Python: $("$PYTHON" --version) at $PYTHON"
"$PYTHON" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install -U pip wheel
install_torch_cuda_if_needed
python -m pip install -r requirements.txt

cp .env.example .env

echo "Installed OmniVoice API dependencies in $VENV_DIR"
echo "Installed system packages: ffmpeg tmux curl ca-certificates lsb-release build-essential gcc g++ ufw"
echo "Run service with: ./scripts/run_tmux.sh"
