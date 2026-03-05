#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ENV_NAME="lpb"

activate_conda_env() {
  local conda_base
  conda_base="$(conda info --base)"
  # shellcheck disable=SC1090
  source "$conda_base/etc/profile.d/conda.sh"

  if ! conda env list | awk "{print \$1}" | grep -qx "$ENV_NAME"; then
    conda env create -f conda_environment.yaml
  fi
  conda activate "$ENV_NAME"
}

activate_venv_env() {
  local pybin="python3"
  if command -v python3.9 >/dev/null 2>&1; then
    pybin="python3.9"
  fi

  if [[ ! -d .venv ]]; then
    "$pybin" -m venv .venv
  fi

  # shellcheck disable=SC1091
  source .venv/bin/activate

  # gym==0.21.0 metadata breaks with newer pip; keep pip below 24.1
  python -m pip install --upgrade "pip<24.1" "setuptools<70" wheel

  if command -v nvidia-smi >/dev/null 2>&1; then
    python -m pip install \
      torch==1.12.1+cu116 \
      torchvision==0.13.1+cu116 \
      torchaudio==0.12.1 \
      --extra-index-url https://download.pytorch.org/whl/cu116
  else
    python -m pip install \
      torch==1.12.1 \
      torchvision==0.13.1 \
      torchaudio==0.12.1
  fi

  python -m pip install \
    diffusers==0.11.1 \
    huggingface_hub==0.12.1 \
    transformers==4.24.0 \
    numpy==1.23.3 \
    numba==0.56.4 \
    matplotlib==3.6.1 \
    pymunk==6.2.1 \
    imageio==2.22.0 \
    imageio-ffmpeg==0.4.7 \
    hydra-core==1.2.0 \
    einops==0.4.1 \
    tqdm==4.64.1 \
    dill==0.3.5.1 \
    zarr==2.12.0 \
    numcodecs==0.10.2 \
    h5py==3.7.0 \
    wandb==0.13.3 \
    scipy==1.9.1 \
    scikit-image==0.19.3 \
    scikit-video==1.1.11 \
    shapely==1.8.4 \
    termcolor==2.0.1 \
    psutil==5.9.2 \
    click==8.0.4 \
    boto3==1.24.96 \
    accelerate==0.13.2 \
    datasets==2.6.1 \
    gym==0.21.0 \
    pygame==2.1.2 \
    robomimic==0.3.0 \
    opencv-python==4.6.0.66 \
    "Cython<3" \
    mujoco-py==2.1.2.14
}

if command -v conda >/dev/null 2>&1; then
  activate_conda_env
else
  activate_venv_env
fi

python - <<"PY"
import torch
print(f"Torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in current environment.")
print(f"CUDA device count: {torch.cuda.device_count()}")
PY

echo "[setup_env] Environment ready."
