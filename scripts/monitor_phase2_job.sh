#!/bin/bash
set -euo pipefail

REPO_ROOT="/home/msai/prithvi005/latent_research/lpb_exp"
cd "$REPO_ROOT"

JOB_ID="${1:?usage: monitor_phase2_job.sh <job_id>}"
POLL_SEC="${POLL_SEC:-60}"
LOG="job_logs/auto_monitor_phase2.log"

echo "[$(date -Is)] monitor start job_id=${JOB_ID}" >> "$LOG"

while true; do
  state="$(sacct -j "$JOB_ID" --format=JobIDRaw,State -n | awk 'NR==1{print $2}')"
  echo "[$(date -Is)] job=${JOB_ID} state=${state}" >> "$LOG"

  case "$state" in
    COMPLETED)
      echo "[$(date -Is)] completed job=${JOB_ID}" >> "$LOG"
      exit 0
      ;;
    FAILED|TIMEOUT|CANCELLED)
      echo "[$(date -Is)] resubmitting after state=${state} for job=${JOB_ID}" >> "$LOG"
      JOB_ID="$(sbatch --parsable lpb_pusht_phase2_pipeline.sh)"
      echo "[$(date -Is)] new_job=${JOB_ID}" >> "$LOG"
      ;;
    RUNNING|PENDING|CONFIGURING|COMPLETING)
      ;;
    "")
      ;;
    *)
      echo "[$(date -Is)] unexpected_state=${state} job=${JOB_ID}" >> "$LOG"
      ;;
  esac

  sleep "$POLL_SEC"
done
