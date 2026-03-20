#!/bin/bash
set -euo pipefail

# Re-run only counterfactual localization (no rollout regeneration)
# Usage:
#   bash scripts/localize_only_tc2.sh <OUTPUT_ROOT> [RHO] [COARSE_N] [REFINE_N] [VERIFY_N]

OUTPUT_ROOT=${1:?"Need OUTPUT_ROOT, e.g. /home/.../outputs/failure_localization_pusht_run2_50_tc2_YYYYMMDD_HHMMSS"}
RHO=${2:-0.3}
COARSE_N=${3:-3}
REFINE_N=${4:-5}
VERIFY_N=${5:-10}

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CKPT="$PROJECT_ROOT/data/pretrained/pusht/pusht_image_dp.ckpt"

module load miniconda3/25.5.1
source /apps/miniconda3/etc/profile.d/conda.sh
conda activate lpb_tc2_cf

python -m experiments.failure_localization.main localize_failure_chunks \
  --task pusht \
  --checkpoint "$CKPT" \
  --output-root "$OUTPUT_ROOT" \
  --guidance-mode threshold_disabled \
  --device cuda \
  --rho "$RHO" \
  --coarse-n "$COARSE_N" \
  --refine-n "$REFINE_N" \
  --verify-n "$VERIFY_N"

python -m experiments.failure_localization.main summarize_results \
  --task pusht \
  --checkpoint "$CKPT" \
  --output-root "$OUTPUT_ROOT" \
  --guidance-mode threshold_disabled \
  --device cuda
