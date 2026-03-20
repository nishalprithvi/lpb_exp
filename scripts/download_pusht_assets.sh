#!/bin/bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

mkdir -p data/pretrained/pusht data/pusht

CKPT_PATH="data/pretrained/pusht/pusht_image_dp.ckpt"
validate_ckpt() {
  python - "$CKPT_PATH" <<'PY'
import sys
import torch
path = sys.argv[1]
obj = torch.load(path, map_location='cpu')
if not isinstance(obj, dict):
    raise RuntimeError('checkpoint root is not dict')
print('[assets] checkpoint validated keys=', len(obj.keys()))
PY
}

download_ckpt() {
  local url="$1"
  echo "[assets] trying checkpoint URL: $url"
  rm -f "$CKPT_PATH"
  wget -O "$CKPT_PATH" "$url"
  validate_ckpt
  echo "$url" > data/pretrained/pusht/checkpoint_source_url.txt
}

NEED_CKPT=1
if [ -f "$CKPT_PATH" ] && [ -s "$CKPT_PATH" ]; then
  if validate_ckpt >/dev/null 2>&1; then
    NEED_CKPT=0
    echo "[assets] Valid checkpoint already present: $CKPT_PATH"
  else
    echo "[assets] Existing checkpoint invalid; re-downloading"
    rm -f "$CKPT_PATH"
  fi
fi

if [ "$NEED_CKPT" -eq 1 ]; then
  for u in \
    "https://diffusion-policy.cs.columbia.edu/data/experiments/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt" \
    "https://diffusion-policy.cs.columbia.edu/data/experiments/image/pusht/diffusion_policy_cnn/train_0/checkpoints/epoch=0500-test_mean_score=0.884.ckpt"
  do
    if wget -q --spider "$u"; then
      if download_ckpt "$u"; then
        break
      fi
    fi
  done
fi

if [ ! -f "$CKPT_PATH" ] || [ ! -s "$CKPT_PATH" ]; then
  echo "[assets] ERROR: no valid PushT checkpoint downloaded"
  exit 2
fi

DATA_ZIP="data/pusht/pusht.zip"
if [ ! -f "$DATA_ZIP" ]; then
  wget -O "$DATA_ZIP" "https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip"
else
  echo "[assets] Dataset zip already present: $DATA_ZIP"
fi

if [ ! -d "data/pusht/pusht/pusht_cchi_v7_replay.zarr" ] && [ ! -d "data/pusht/replay_buffer.zarr" ]; then
  unzip -o "$DATA_ZIP" -d data/pusht/
else
  echo "[assets] Dataset already extracted"
fi

echo "[assets] done"
