#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=06:00:00
#SBATCH --mem=30G
#SBATCH --job-name=lpb_pusht_libero_phase2
#SBATCH --output=./job_logs/output_libero_%j.out
#SBATCH --error=./job_logs/error_libero_%j.err

set -euo pipefail

REPO_ROOT="/home/msai/prithvi005/latent_research/lpb_exp"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
mkdir -p job_logs outputs_phase2_pusht outputs_phase2_libero outputs_phase2_combined
TS="$(date +%Y%m%d_%H%M%S)"
MAIN_LOG="job_logs/phase2_pusht_libero_${TS}.log"
exec > >(tee -a "$MAIN_LOG") 2>&1

STATE_FILE="outputs_phase2_combined/state.json"
PUSHT_TMP_DIR="outputs_phase2_pusht_next"
LIBERO_TMP_DIR="outputs_phase2_libero_next"
PUSHT_FINAL_DIR="outputs_phase2_pusht"
LIBERO_FINAL_DIR="outputs_phase2_libero"

PUSHT_DATASET_PATH="data/pusht/pusht_cchi_v7_replay.zarr"
LIBERO_DATASET_GLOB="data/libero10/data/expert_demonstration/libero_10/*.hdf5"

PUSHT_PRETRAINED_URL="${PUSHT_PRETRAINED_URL:-https://diffusion-policy.cs.columbia.edu/data/experiments/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt}"
PUSHT_PRETRAINED_CKPT="${PUSHT_PRETRAINED_CKPT:-outputs_phase2_pusht/train_run/checkpoints/latest.ckpt}"
PUSHT_FALLBACK_CKPT="${PUSHT_FALLBACK_CKPT:-outputs_phase2_pusht/train_run/checkpoints/latest.ckpt}"
PUSHT_MIN_CKPT_BYTES="${PUSHT_MIN_CKPT_BYTES:-3000000000}"

LIBERO_PRETRAINED_CKPT="${LIBERO_PRETRAINED_CKPT:-models/pretrained/libero10/base_policy/checkpoints/160.ckpt}"
LIBERO_TASK_DATASET="${LIBERO_TASK_DATASET:-data/libero10/data/expert_demonstration/libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_demo.hdf5}"

EVAL_NUM_SEEDS="${EVAL_NUM_SEEDS:-50}"
EVAL_EPISODES_PER_SEED="${EVAL_EPISODES_PER_SEED:-1}"

JOB_START_EPOCH="$(date +%s)"
MAX_RUNTIME=$((6 * 3600))
SAFE_BUFFER=$((15 * 60))

remaining_seconds() {
  local now elapsed rem
  now="$(date +%s)"
  elapsed=$((now - JOB_START_EPOCH))
  rem=$((MAX_RUNTIME - elapsed - SAFE_BUFFER))
  if [[ "$rem" -lt 0 ]]; then
    rem=0
  fi
  echo "$rem"
}

require_time_or_exit() {
  local phase="$1"
  if [[ "$(remaining_seconds)" -le 0 ]]; then
    echo "No safe time left before ${phase}; exiting cleanly for resume."
    exit 0
  fi
}

ckpt_size_ok() {
  local p="$1"
  [[ -f "$p" ]] || return 1
  local s
  s="$(stat -c%s "$p")"
  [[ "$s" -ge "$PUSHT_MIN_CKPT_BYTES" ]]
}

init_state() {
  if [[ ! -f "$STATE_FILE" ]]; then
    cat > "$STATE_FILE" <<JSON
{
  "env_ready": false,
  "pusht_ready": false,
  "libero_ready": false,
  "push_eval_done": false,
  "push_analysis_done": false,
  "libero_eval_done": false,
  "libero_analysis_done": false,
  "finalized": false,
  "pusht_ckpt": null,
  "libero_ckpt": null
}
JSON
  fi
}

update_state() {
  local key="$1"
  local value="$2"
  python - "$STATE_FILE" "$key" "$value" <<PY
import json
import sys
from pathlib import Path
state_path = Path(sys.argv[1])
key = sys.argv[2]
value = sys.argv[3]
state = json.loads(state_path.read_text())
parts = key.split('.')
obj = state
for p in parts[:-1]:
    obj = obj[p]
if value == 'true':
    obj[parts[-1]] = True
elif value == 'false':
    obj[parts[-1]] = False
elif value == 'null':
    obj[parts[-1]] = None
else:
    obj[parts[-1]] = value
state_path.write_text(json.dumps(state, indent=2))
PY
}

state_get() {
  local key="$1"
  python - "$STATE_FILE" "$key" <<PY
import json
import sys
from pathlib import Path
obj = json.loads(Path(sys.argv[1]).read_text())
for p in sys.argv[2].split('.'):
    obj = obj[p]
if isinstance(obj, bool):
    print('true' if obj else 'false')
elif obj is None:
    print('null')
else:
    print(obj)
PY
}

ensure_env() {
  if command -v conda >/dev/null 2>&1; then
    conda_base="$(conda info --base)"
    # shellcheck disable=SC1090
    source "$conda_base/etc/profile.d/conda.sh"
    conda activate lpb || source scripts/setup_env.sh
  elif [[ -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
  else
    source scripts/setup_env.sh
  fi
}

setup_mujoco_env() {
  export MUJOCO_PY_MUJOCO_PATH="${MUJOCO_PY_MUJOCO_PATH:-/home/msai/prithvi005/.mujoco/mujoco210}"
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:${MUJOCO_PY_MUJOCO_PATH}/bin:/usr/lib/nvidia:/usr/lib64/nvidia:/home/msai/prithvi005/.local/glew/lib64"
  export LIBRARY_PATH="${LIBRARY_PATH:-}:/home/msai/prithvi005/.local/glew/lib64"
  export CPATH="${CPATH:-}:/home/msai/prithvi005/.local/glew/include"
  export MUJOCO_GL="${MUJOCO_GL:-egl}"
  export CFLAGS="${CFLAGS:-} -Wno-error=incompatible-pointer-types -Wno-error=implicit-function-declaration -Wno-error=int-conversion"
  export PATH="/home/msai/prithvi005/.local/bin:${PATH:-}"
}

ensure_datasets() {
  [[ -d "$PUSHT_DATASET_PATH" ]] || { echo "Missing PushT dataset: $PUSHT_DATASET_PATH"; return 1; }
  ls $LIBERO_DATASET_GLOB >/dev/null 2>&1 || { echo "Missing LIBERO10 dataset files under data/libero10/data/expert_demonstration/libero_10"; return 1; }
  [[ -f "$LIBERO_TASK_DATASET" ]] || { echo "LIBERO task dataset file not found: $LIBERO_TASK_DATASET"; return 1; }
}

ensure_pretrained_ckpts() {
  mkdir -p "$(dirname "$PUSHT_PRETRAINED_CKPT")"

  if ! ckpt_size_ok "$PUSHT_PRETRAINED_CKPT"; then
    if ckpt_size_ok "$PUSHT_FALLBACK_CKPT"; then
      echo "PushT pretrained download missing/incomplete; using local fallback checkpoint: $PUSHT_FALLBACK_CKPT"
      PUSHT_PRETRAINED_CKPT="$PUSHT_FALLBACK_CKPT"
    else
      echo "Downloading PushT pretrained checkpoint from: $PUSHT_PRETRAINED_URL"
      wget -c -O "$PUSHT_PRETRAINED_CKPT" "$PUSHT_PRETRAINED_URL"
      ckpt_size_ok "$PUSHT_PRETRAINED_CKPT" || { echo "PushT checkpoint is incomplete after download."; return 1; }
    fi
  fi

  [[ -f "$LIBERO_PRETRAINED_CKPT" ]] || { echo "Missing LIBERO pretrained checkpoint: $LIBERO_PRETRAINED_CKPT"; return 1; }

  update_state pusht_ckpt "$PUSHT_PRETRAINED_CKPT"
  update_state libero_ckpt "$LIBERO_PRETRAINED_CKPT"
}

validate_ckpt_compat() {
  local ckpt="$1"
  local mode="$2"
  python - "$ckpt" "$mode" <<PY
import dill
import sys
import torch

ckpt = sys.argv[1]
mode = sys.argv[2]
payload = torch.load(ckpt, pickle_module=dill, map_location="cpu")
cfg = payload.get("cfg", None)
target = ""
if cfg is not None:
    target = str(getattr(cfg, "_target_", ""))
if mode == "pusht_image" and "lowdim" in target.lower():
    raise SystemExit(f"Incompatible PushT checkpoint for image evaluator: target={target}")
if mode == "libero" and "lowdim" in target.lower():
    raise SystemExit(f"Incompatible LIBERO checkpoint (expected image workspace): target={target}")
print(f"[ckpt-ok] {ckpt} target={target}")
PY
}

preflight_libero_runtime() {
  setup_mujoco_env
  python - <<PY
import mujoco_py
print("[preflight] mujoco_py import ok")
PY
}

run_pusht_phase() {
  mkdir -p "$PUSHT_TMP_DIR/videos" "$PUSHT_TMP_DIR/analysis"
  if [[ "$(state_get push_eval_done)" != "true" ]]; then
    require_time_or_exit "PushT evaluation"
    seed_args=()
    for ((s=0; s<EVAL_NUM_SEEDS; s++)); do
      seed_args+=("$s")
    done

    python scripts/run_phase2_eval.py \
      --checkpoint "$PUSHT_PRETRAINED_CKPT" \
      --output-dir "$PUSHT_TMP_DIR" \
      --device cuda:0 \
      --seeds "${seed_args[@]}" \
      --episodes-per-seed "$EVAL_EPISODES_PER_SEED" \
      --max-steps 400 \
      --num-inference-steps 20

    update_state push_eval_done true
  else
    echo "PushT evaluation already complete (state)."
  fi

  if [[ "$(state_get push_analysis_done)" != "true" ]]; then
    require_time_or_exit "PushT analysis"
    python scripts/eval_and_plot.py \
      --metrics-csv "$PUSHT_TMP_DIR/videos/per_episode_metrics.csv" \
      --output-dir "$PUSHT_TMP_DIR/analysis"
    update_state push_analysis_done true
  else
    echo "PushT analysis already complete (state)."
  fi
}

run_libero_phase() {
  setup_mujoco_env
  preflight_libero_runtime
  mkdir -p "$LIBERO_TMP_DIR/videos" "$LIBERO_TMP_DIR/analysis"
  if [[ "$(state_get libero_eval_done)" != "true" ]]; then
    require_time_or_exit "LIBERO evaluation"
    seed_args=()
    for ((s=0; s<EVAL_NUM_SEEDS; s++)); do
      seed_args+=("$s")
    done

    python scripts/run_libero_phase2_eval.py \
      --checkpoint "$LIBERO_PRETRAINED_CKPT" \
      --dataset-path "$LIBERO_TASK_DATASET" \
      --output-dir "$LIBERO_TMP_DIR" \
      --device cuda:0 \
      --seeds "${seed_args[@]}" \
      --episodes-per-seed "$EVAL_EPISODES_PER_SEED" \
      --max-steps 400 \
      --num-inference-steps 20

    update_state libero_eval_done true
  else
    echo "LIBERO evaluation already complete (state)."
  fi

  if [[ "$(state_get libero_analysis_done)" != "true" ]]; then
    require_time_or_exit "LIBERO analysis"
    python scripts/libero_eval_and_plot.py \
      --metrics-csv "$LIBERO_TMP_DIR/videos/per_episode_metrics.csv" \
      --output-dir "$LIBERO_TMP_DIR/analysis" \
      --title-prefix "LIBERO Phase-2"
    update_state libero_analysis_done true
  else
    echo "LIBERO analysis already complete (state)."
  fi
}

finalize_outputs() {
  if [[ "$(state_get finalized)" == "true" ]]; then
    echo "Outputs already finalized."
    return 0
  fi

  [[ -f "$PUSHT_TMP_DIR/videos/per_episode_metrics.csv" ]] || { echo "Missing PushT metrics in tmp output"; return 1; }
  [[ -f "$LIBERO_TMP_DIR/videos/per_episode_metrics.csv" ]] || { echo "Missing LIBERO metrics in tmp output"; return 1; }

  mkdir -p "$PUSHT_FINAL_DIR" "$LIBERO_FINAL_DIR"

  rm -rf "$PUSHT_FINAL_DIR/videos" "$PUSHT_FINAL_DIR/analysis"
  cp -a "$PUSHT_TMP_DIR/videos" "$PUSHT_FINAL_DIR/videos"
  cp -a "$PUSHT_TMP_DIR/analysis" "$PUSHT_FINAL_DIR/analysis"

  rm -rf "$LIBERO_FINAL_DIR/videos" "$LIBERO_FINAL_DIR/analysis"
  cp -a "$LIBERO_TMP_DIR/videos" "$LIBERO_FINAL_DIR/videos"
  cp -a "$LIBERO_TMP_DIR/analysis" "$LIBERO_FINAL_DIR/analysis"

  update_state finalized true
}

init_state

if [[ "$(state_get env_ready)" != "true" ]]; then
  require_time_or_exit "environment setup"
  ensure_env
  update_state env_ready true
else
  ensure_env
fi

if [[ "$(state_get pusht_ready)" != "true" || "$(state_get libero_ready)" != "true" ]]; then
  require_time_or_exit "dataset/checkpoint preflight"
  ensure_datasets
  ensure_pretrained_ckpts
  validate_ckpt_compat "$PUSHT_PRETRAINED_CKPT" pusht_image
  validate_ckpt_compat "$LIBERO_PRETRAINED_CKPT" libero
  update_state pusht_ready true
  update_state libero_ready true
else
  ensure_pretrained_ckpts
  validate_ckpt_compat "$PUSHT_PRETRAINED_CKPT" pusht_image
  validate_ckpt_compat "$LIBERO_PRETRAINED_CKPT" libero
fi

run_libero_phase
run_pusht_phase
finalize_outputs

echo "Pretrained PushT + LIBERO phase-2 pipeline completed successfully."
