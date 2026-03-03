#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=06:00:00
#SBATCH --mem=30G
#SBATCH --job-name=lpb_pusht_phase2
#SBATCH --output=./job_logs/output_%j.out
#SBATCH --error=./job_logs/error_%j.err

set -euo pipefail

REPO_ROOT="/home/msai/prithvi005/latent_research/lpb_exp"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
mkdir -p job_logs outputs_phase2_pusht data/pusht outputs_phase2_pusht/videos outputs_phase2_pusht/analysis
TS="$(date +%Y%m%d_%H%M%S)"
MAIN_LOG="job_logs/phase2_${TS}.log"
exec > >(tee -a "$MAIN_LOG") 2>&1

STATE_FILE="outputs_phase2_pusht/state.json"
TRAIN_OUT="outputs_phase2_pusht/train_run"
CKPT_DIR="$TRAIN_OUT/checkpoints"
LATEST_CKPT="$CKPT_DIR/latest.ckpt"
DATASET_PATH="data/pusht/pusht_cchi_v7_replay.zarr"

TRAIN_NUM_EPOCHS="${TRAIN_NUM_EPOCHS:-1000}"
MIN_EVAL_EPOCH="${MIN_EVAL_EPOCH:-100}"
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

init_state() {
  if [[ ! -f "$STATE_FILE" ]]; then
    cat > "$STATE_FILE" <<JSON
{
  "env_ready": false,
  "dataset_ready": false,
  "trained_checkpoint": null,
  "evaluations_done": {
    "baseline": false,
    "obstacle_no_guidance": false,
    "obstacle_guidance": false
  },
  "analysis_done": false
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
parts = key.split(".")
obj = state
for p in parts[:-1]:
    obj = obj[p]

if value == "true":
    obj[parts[-1]] = True
elif value == "false":
    obj[parts[-1]] = False
elif value == "null":
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

state_path = Path(sys.argv[1])
key = sys.argv[2]
obj = json.loads(state_path.read_text())
for part in key.split("."):
    obj = obj[part]
if isinstance(obj, bool):
    print("true" if obj else "false")
elif obj is None:
    print("null")
else:
    print(obj)
PY
}

prune_checkpoints() {
  mkdir -p "$CKPT_DIR"
  find "$CKPT_DIR" -maxdepth 1 -type f -name "*.ckpt" ! -name "latest.ckpt" -print -delete || true
}

get_ckpt_epoch() {
  local ckpt="$1"
  if [[ ! -f "$ckpt" ]]; then
    echo "-1"
    return 0
  fi

  python - "$ckpt" <<PY
import pickle
import sys
import torch

ckpt = sys.argv[1]
try:
    obj = torch.load(ckpt, map_location="cpu")
    ep = obj.get("pickles", {}).get("epoch", None)
    if ep is None:
        print(-1)
    else:
        print(int(pickle.loads(ep)))
except Exception:
    print(-1)
PY
}

reset_eval_outputs() {
  rm -f outputs_phase2_pusht/videos/*.mp4 || true
  rm -f outputs_phase2_pusht/videos/per_episode_metrics.csv || true
  rm -f outputs_phase2_pusht/videos/per_episode_metrics.json || true
  rm -f outputs_phase2_pusht/analysis/summary.csv || true
  rm -f outputs_phase2_pusht/analysis/summary.json || true
  rm -f outputs_phase2_pusht/analysis/success_rate.png || true
  rm -f outputs_phase2_pusht/analysis/collision_rate.png || true
}

require_time_or_exit() {
  local phase="$1"
  if [[ "$(remaining_seconds)" -le 0 ]]; then
    echo "No safe time left before ${phase}; exiting cleanly for resume."
    exit 0
  fi
}

ensure_dataset() {
  if [[ -d "$DATASET_PATH" ]]; then
    return 0
  fi

  echo "PushT dataset missing. Attempting download..."
  local tmp_zip
  tmp_zip="/tmp/pusht_dataset_${SLURM_JOB_ID:-manual}.zip"

  local -a urls=()
  if [[ -n "${PUSHT_DATASET_URL:-}" ]]; then
    urls+=("$PUSHT_DATASET_URL")
  fi
  urls+=(
    "https://diffusion-policy.cs.columbia.edu/data/training/pusht_cchi_v7_replay.zarr.zip"
    "https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip"
  )

  local ok=0
  for url in "${urls[@]}"; do
    echo "Trying dataset URL: $url"
    rm -f "$tmp_zip"
    if wget -q --show-progress -O "$tmp_zip" "$url"; then
      mkdir -p data/pusht
      unzip -o "$tmp_zip" -d data/pusht >/dev/null
      rm -f "$tmp_zip"
      ok=1
      break
    fi
  done

  if [[ "$ok" -ne 1 ]]; then
    echo "Failed to download PushT dataset. Set PUSHT_DATASET_URL to a valid link and resubmit."
    return 1
  fi

  if [[ ! -d "$DATASET_PATH" ]]; then
    candidate="$(find data/pusht -maxdepth 4 -type d -name pusht_cchi_v7_replay.zarr | head -n 1 || true)"
    if [[ -n "${candidate:-}" ]]; then
      mkdir -p "$(dirname "$DATASET_PATH")"
      cp -a "$candidate" "$DATASET_PATH"
    fi
  fi

  [[ -d "$DATASET_PATH" ]]
}

init_state

require_time_or_exit "environment setup"
if [[ "$(state_get env_ready)" != "true" ]]; then
  source scripts/setup_env.sh
  update_state env_ready true
else
  echo "Environment already marked ready in state.json"
  if command -v conda >/dev/null 2>&1; then
    conda_base="$(conda info --base)"
    # shellcheck disable=SC1090
    source "$conda_base/etc/profile.d/conda.sh"
    conda activate lpb
  elif [[ -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
  else
    source scripts/setup_env.sh
    update_state env_ready true
  fi
fi

require_time_or_exit "dataset setup"
if [[ "$(state_get dataset_ready)" != "true" ]]; then
  ensure_dataset
  update_state dataset_ready true
else
  echo "Dataset already marked ready in state.json"
fi

if [[ ! -d "$DATASET_PATH" ]]; then
  echo "Dataset directory missing at $DATASET_PATH despite state; resetting dataset flag."
  update_state dataset_ready false
  exit 1
fi

prune_checkpoints

did_train=0
ckpt_epoch="$(get_ckpt_epoch "$LATEST_CKPT")"
if [[ -f "$LATEST_CKPT" ]]; then
  echo "Found existing checkpoint: $LATEST_CKPT (epoch=${ckpt_epoch})"
else
  echo "No checkpoint found; training required."
fi

if [[ ! -f "$LATEST_CKPT" || "$ckpt_epoch" -lt "$MIN_EVAL_EPOCH" ]]; then
  require_time_or_exit "training"
  train_time="$(remaining_seconds)"
  echo "Starting/continuing training for up to ${train_time}s (target num_epochs=${TRAIN_NUM_EPOCHS}, min_eval_epoch=${MIN_EVAL_EPOCH})"

  set +e
  timeout "${train_time}" python train.py --config-dir=. --config-name=image_pusht_diffusion_policy_cnn.yaml \
    training.seed=0 \
    training.device=cuda:0 \
    training.resume=true \
    training.num_epochs="${TRAIN_NUM_EPOCHS}" \
    training.checkpoint_every=5 \
    task.dataset.zarr_path="$DATASET_PATH" \
    task.env_runner.n_train=1 \
    task.env_runner.n_train_vis=0 \
    task.env_runner.n_test=1 \
    task.env_runner.n_test_vis=0 \
    logging.mode=offline \
    hydra.run.dir="$TRAIN_OUT"
  train_rc=$?
  set -e

  if [[ "$train_rc" -ne 0 && "$train_rc" -ne 124 ]]; then
    echo "Training failed with rc=$train_rc"
    exit "$train_rc"
  fi
  did_train=1
fi

prune_checkpoints

if [[ ! -f "$LATEST_CKPT" ]]; then
  echo "Checkpoint not ready yet; exiting cleanly for resume."
  exit 0
fi

ckpt_epoch="$(get_ckpt_epoch "$LATEST_CKPT")"
echo "Checkpoint after training check: epoch=${ckpt_epoch}"
update_state trained_checkpoint "$LATEST_CKPT"

if [[ "$did_train" -eq 1 ]]; then
  echo "Training executed in this run; resetting evaluation/analysis state and stale artifacts."
  update_state evaluations_done.baseline false
  update_state evaluations_done.obstacle_no_guidance false
  update_state evaluations_done.obstacle_guidance false
  update_state analysis_done false
  reset_eval_outputs
fi

if [[ "$ckpt_epoch" -lt "$MIN_EVAL_EPOCH" ]]; then
  echo "Checkpoint epoch ${ckpt_epoch} is below MIN_EVAL_EPOCH=${MIN_EVAL_EPOCH}; exiting for resume before evaluation."
  exit 0
fi

need_eval=0
[[ "$(state_get evaluations_done.baseline)" != "true" ]] && need_eval=1
[[ "$(state_get evaluations_done.obstacle_no_guidance)" != "true" ]] && need_eval=1
[[ "$(state_get evaluations_done.obstacle_guidance)" != "true" ]] && need_eval=1

if [[ "$need_eval" -eq 1 ]]; then
  require_time_or_exit "evaluation"
  seed_args=()
  for ((s=0; s<EVAL_NUM_SEEDS; s++)); do
    seed_args+=("$s")
  done

  python scripts/run_phase2_eval.py \
    --checkpoint "$LATEST_CKPT" \
    --output-dir outputs_phase2_pusht \
    --device cuda:0 \
    --seeds "${seed_args[@]}" \
    --episodes-per-seed "$EVAL_EPISODES_PER_SEED" \
    --max-steps 400 \
    --num-inference-steps 20

  update_state evaluations_done.baseline true
  update_state evaluations_done.obstacle_no_guidance true
  update_state evaluations_done.obstacle_guidance true
else
  echo "Evaluation already marked complete in state.json"
fi

if [[ "$(state_get analysis_done)" != "true" ]]; then
  require_time_or_exit "analysis"
  python scripts/eval_and_plot.py \
    --metrics-csv outputs_phase2_pusht/videos/per_episode_metrics.csv \
    --output-dir outputs_phase2_pusht/analysis
  update_state analysis_done true
else
  echo "Analysis already marked complete in state.json"
fi

echo "Phase-2 pipeline completed successfully."
