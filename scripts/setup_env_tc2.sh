#!/bin/bash
set -euo pipefail

module load miniconda3/25.5.1
source /apps/miniconda3/etc/profile.d/conda.sh

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >/dev/null 2>&1 || true

ENV_NAME=lpb_tc2_cf
if conda env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
  echo "[setup] Conda env $ENV_NAME already exists"
else
  echo "[setup] Creating conda env $ENV_NAME"
  conda create -y -n "$ENV_NAME" python=3.9
fi

conda activate "$ENV_NAME"
READY_FILE="$CONDA_PREFIX/.lpb_cf_ready"

pip_retry() {
  local n=0
  local max=5
  until [ $n -ge $max ]; do
    if python -m pip --default-timeout=120 --retries 10 "$@"; then
      return 0
    fi
    n=$((n+1))
    echo "[setup] pip retry $n/$max failed for: $*"
    sleep 10
  done
  return 1
}

if [ ! -f "$READY_FILE" ]; then
  python -m pip install --upgrade "pip==23.2.1" "setuptools==65.7.0" "wheel==0.38.4"
  pip_retry install --extra-index-url https://download.pytorch.org/whl/cu121 torch==2.1.2 torchvision==0.16.2
  pip_retry install \
    numpy==1.23.3 scipy==1.9.1 matplotlib==3.6.1 \
    hydra-core==1.2.0 dill==0.3.5.1 wandb==0.13.3 \
    gym==0.21.0 pygame==2.1.2 pymunk==6.2.1 shapely==1.8.4 \
    opencv-python==4.6.0.66 scikit-image==0.19.3 imageio==2.22.0 imageio-ffmpeg==0.4.7 \
    einops==0.4.1 diffusers==0.11.1 huggingface_hub==0.11.1 transformers==4.28.1 \
    numcodecs==0.10.2 zarr==2.12.0 pyyaml==6.0.2
  pip_retry install --no-deps robomimic==0.3.0
fi

python - <<'PY'
import torch
mods = [
    "hydra", "wandb", "pymunk", "gym", "cv2", "skimage", "dill", "einops",
    "diffusers", "transformers", "numcodecs", "zarr", "pygame", "shapely", "robomimic"
]
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
for m in mods:
    __import__(m)
print("[setup] Import checks passed")
PY

touch "$READY_FILE"
echo "[setup] Marked ready at $READY_FILE"
