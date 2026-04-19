#!/bin/bash
# Setup script for the fine-tuning sidecar.
# Installs torch (with correct CUDA version) then unsloth + FastAPI.
#
# Usage:
#   cd server/sidecar
#   bash setup.sh
#
# Requires: Python 3.10+, CUDA GPU, nvidia-smi

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/venv"

echo "=== Tanrenai Sidecar Setup ==="

# Detect CUDA version
if command -v nvidia-smi &>/dev/null; then
    CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    echo "NVIDIA driver: ${CUDA_VERSION}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "ERROR: nvidia-smi not found. A CUDA GPU is required for training."
    exit 1
fi

# Detect CUDA toolkit version for torch index
CUDA_TAG="cu121"  # safe default
if command -v nvcc &>/dev/null; then
    NVCC_VERSION=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
    echo "CUDA toolkit: ${NVCC_VERSION}"
    case "${NVCC_VERSION}" in
        11.8*) CUDA_TAG="cu118" ;;
        12.1*) CUDA_TAG="cu121" ;;
        12.4*|12.5*|12.6*) CUDA_TAG="cu124" ;;
        *) echo "Note: CUDA ${NVCC_VERSION} detected, using cu121 index as fallback" ;;
    esac
else
    echo "Note: nvcc not found, assuming CUDA 12.1 (cu121)"
fi
echo "Using PyTorch index: ${CUDA_TAG}"
echo ""

# Create venv
if [ ! -d "${VENV_DIR}" ]; then
    echo "--- Creating virtual environment ---"
    python3 -m venv "${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"

echo "--- Installing PyTorch (this may take a while) ---"
pip install -q --upgrade pip
pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

echo "--- Installing unsloth ---"
pip install unsloth

echo "--- Installing FastAPI + uvicorn ---"
pip install fastapi uvicorn

echo ""
echo "=== Setup complete ==="
echo "To activate: source ${VENV_DIR}/bin/activate"
echo "To start:    cd ${SCRIPT_DIR} && uvicorn main:app --host 127.0.0.1 --port 18082"
