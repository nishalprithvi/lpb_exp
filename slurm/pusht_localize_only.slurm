#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=3:00:00
#SBATCH --mem=24G
#SBATCH --job-name=pusht_localize
#SBATCH --output=./job_logs/pusht_localize_%j.out
#SBATCH --error=./job_logs/pusht_localize_%j.err

set -euo pipefail

PROJECT_ROOT=$(pwd)
if [ $# -lt 1 ]; then
  echo "Usage: sbatch slurm/pusht_localize_only.slurm <OUTPUT_ROOT> [RHO] [COARSE_N] [REFINE_N] [VERIFY_N]"
  exit 1
fi

bash scripts/localize_only_tc2.sh "$@"
