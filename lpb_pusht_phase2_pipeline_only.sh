#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=06:00:00
#SBATCH --mem=30G
#SBATCH --job-name=lpb_pusht_phase2_only
#SBATCH --output=./job_logs/output_pusht_only_%j.out
#SBATCH --error=./job_logs/error_pusht_only_%j.err

set -euo pipefail

REPO_ROOT="/home/msai/prithvi005/latent_research/lpb_exp"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
mkdir -p job_logs outputs_phase2_pusht

TS="$(date +%Y%m%d_%H%M%S)"
MAIN_LOG="job_logs/pusht_phase2_only_${TS}.log"
exec > >(tee -a "$MAIN_LOG") 2>&1

CHECKPOINT_LOCAL="data/pretrained/pusht/pusht_dp_epoch0550.ckpt"
PUSHT_DEVICE="${PUSHT_DEVICE:-cpu}"
SEED_OFFSET="${SEED_OFFSET:-10000}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-0.5}"

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

sanity_baseline() {
  echo "Running low-dim sanity baseline check (seeds 0..4 with seed_offset=${SEED_OFFSET})."
  rm -rf outputs_phase2_pusht/sanity
  python scripts/run_phase2_eval_lowdim.py \
    --checkpoint "$CHECKPOINT_LOCAL" \
    --output-dir outputs_phase2_pusht/sanity \
    --device "$PUSHT_DEVICE" \
    --seeds 0 1 2 3 4 \
    --seed-offset "$SEED_OFFSET" \
    --scenario-mode baseline_only \
    --max-steps 300 \
    --num-inference-steps 100

  python - <<'PY'
import csv
from pathlib import Path
p = Path('outputs_phase2_pusht/sanity/videos/per_episode_metrics.csv')
rows = list(csv.DictReader(p.open()))
success = sum(int(float(r['success'])) for r in rows)
rate = success / len(rows)
print(f'[sanity] success={success}/{len(rows)} rate={rate:.3f}')
if rate <= 0.60:
    raise SystemExit(f'Sanity baseline gate failed: {rate:.3f} <= 0.60')
PY
}

run_final_eval() {
  echo "Running final low-dim PushT eval with guidance scale=${GUIDANCE_SCALE}"
  python scripts/run_phase2_eval_lowdim.py \
    --checkpoint "$CHECKPOINT_LOCAL" \
    --output-dir outputs_phase2_pusht \
    --device "$PUSHT_DEVICE" \
    --seeds 0 1 2 \
    --seed-offset "$SEED_OFFSET" \
    --scenario-mode full \
    --obstacle-guidance-scale "$GUIDANCE_SCALE" \
    --obstacle-margin 8.0 \
    --guidance-start-timestep 20 \
    --max-steps 300 \
    --num-inference-steps 100

  python scripts/eval_and_plot.py \
    --metrics-csv outputs_phase2_pusht/videos/per_episode_metrics.csv \
    --output-dir outputs_phase2_pusht/analysis
}

print_summary() {
  echo "===== FINAL SUMMARY ====="
  echo "Checkpoint local path: $CHECKPOINT_LOCAL"
  echo "--- videos ---"
  ls -lh outputs_phase2_pusht/videos | sed -n "1,200p"
  echo "--- analysis ---"
  ls -lh outputs_phase2_pusht/analysis | sed -n "1,200p"
  echo "--- success/collision summary ---"
  python - <<'PY'
import csv
from pathlib import Path
p=Path('outputs_phase2_pusht/analysis/summary.csv')
rows=list(csv.DictReader(p.open()))
for r in rows:
    print(f"{r['scenario']}: success={float(r['success_rate']):.3f}, collision={float(r['collision_rate']):.3f}, n={r['num_episodes']}")
PY
}

ensure_env

[[ -f "$CHECKPOINT_LOCAL" ]] || { echo "Missing low-dim checkpoint: $CHECKPOINT_LOCAL"; exit 1; }

echo "Cleaning old PushT outputs only."
rm -rf outputs_phase2_pusht/*
mkdir -p outputs_phase2_pusht/videos outputs_phase2_pusht/analysis

sanity_baseline
run_final_eval
print_summary

echo "PushT-only phase-2 pipeline (low-dim pretrained) completed successfully."
