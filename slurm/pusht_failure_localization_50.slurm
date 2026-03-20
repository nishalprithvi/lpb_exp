#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=6:00:00
#SBATCH --mem=30G
#SBATCH --job-name=pusht_cf_50
#SBATCH --output=./job_logs/pusht_cf_50_%j.out
#SBATCH --error=./job_logs/pusht_cf_50_%j.err

set -euo pipefail

PROJECT_ROOT=$(pwd)
LOG_DIR="$PROJECT_ROOT/job_logs"
mkdir -p "$LOG_DIR"

RUN_TS=$(date "+%Y%m%d_%H%M%S")
PIPE_LOG="$LOG_DIR/pipeline_pusht_cf_50_${RUN_TS}.log"

log() {
  echo "[$(date "+%Y-%m-%d %H:%M:%S")] $1" | tee -a "$PIPE_LOG"
}

RUN_NAME="pusht_run2_50_tc2_${RUN_TS}"
OUTPUT_ROOT="$PROJECT_ROOT/outputs/failure_localization_${RUN_NAME}"
mkdir -p "$OUTPUT_ROOT"

log "Start PushT counterfactual failure-localization run"
log "Project: $PROJECT_ROOT"
log "Output root: $OUTPUT_ROOT"
log "Slurm job: ${SLURM_JOB_ID:-local}"

module load miniconda3/25.5.1
source /apps/miniconda3/etc/profile.d/conda.sh

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >/dev/null 2>&1 || true

log "Running environment setup/verification"
bash scripts/setup_env_tc2.sh 2>&1 | tee -a "$PIPE_LOG"
conda activate lpb_tc2_cf

log "Python: $(which python)"
python - <<"PY" 2>&1 | tee -a "$PIPE_LOG"
import torch
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
PY

log "Downloading/validating PushT assets"
bash scripts/download_pusht_assets.sh 2>&1 | tee -a "$PIPE_LOG"

CKPT="$PROJECT_ROOT/data/pretrained/pusht/pusht_image_dp.ckpt"
if [ ! -f "$CKPT" ]; then
  log "ERROR: checkpoint missing at $CKPT"
  exit 2
fi

log "Running full pipeline (50 episodes, guidance disabled)"
python -m experiments.failure_localization.main run_pipeline \
  --task pusht \
  --checkpoint "$CKPT" \
  --output-root "$OUTPUT_ROOT" \
  --guidance-mode threshold_disabled \
  --episodes 50 \
  --test-start-seed 200000 \
  --max-steps 300 \
  --chunk-size 20 \
  --device cuda \
  --rho 0.3 \
  --coarse-n 3 \
  --refine-n 5 \
  --verify-n 10 \
  2>&1 | tee -a "$PIPE_LOG"

log "Pipeline completed successfully"
log "Final output root: $OUTPUT_ROOT"
printf "%s\n" "$OUTPUT_ROOT" > "$LOG_DIR/latest_pusht_cf_output_root.txt"
