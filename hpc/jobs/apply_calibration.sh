#!/bin/bash
#SBATCH --job-name=apply_calibration
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=hpc/logs/slurm/%x-%A_%a.out
#SBATCH --error=hpc/logs/slurm/%x-%A_%a.err

set -euo pipefail

# Apply the timmh (Haucke et al. 2022) per-camera depth calibration to a dap3 job's tracked
# detections (apps/camera_trap/calibration/apply.py). CPU-only -- no --nv needed. Isambard AI
# compute nodes are offline, so HF_HUB_OFFLINE=1 and no network access is assumed. Submit as an
# array (one task per shard):
#
#   JOB_DIR=hpc/runs/main_job CALIB_DIR=hpc/runs/ref_job/calib \
#     OUT_DIR=hpc/runs/main_job/calib_applied sbatch --array=0-3 hpc/jobs/apply_calibration.sh
#
# --array=0-(NUM_SHARDS-1) must match NUM_SHARDS below. After all array tasks finish, merge the
# shards into one calibrated_objects.csv + apply_summary.json:
#
#   JOB_DIR=hpc/runs/main_job CALIB_DIR=hpc/runs/ref_job/calib \
#     OUT_DIR=hpc/runs/main_job/calib_applied MERGE=1 sbatch hpc/jobs/apply_calibration.sh

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" && "${MERGE:-0}" != "1" ]]; then
  echo "SLURM_ARRAY_TASK_ID is not set. Submit with sbatch --array=0-(NUM_SHARDS-1), or MERGE=1 for the merge step." >&2
  exit 1
fi

REPO_ROOT="${REPO_ROOT:-$HOME/Unmarked-Anything}"
JOB_DIR="${JOB_DIR:?Set JOB_DIR to the dap3 output dir to calibrate}"
CALIB_DIR="${CALIB_DIR:?Set CALIB_DIR to the build_reference.py output dir (calib/)}"
OUT_DIR="${OUT_DIR:-$JOB_DIR/calib_applied}"
ALIGN_METHOD="${ALIGN_METHOD:-ransac}"
ALIGN_BLUR="${ALIGN_BLUR:-0}"
MIN_ALIGN_PIXELS="${MIN_ALIGN_PIXELS:-500}"
FORCE_POOLED="${FORCE_POOLED:-0}"
EXTRINSIC_RECALIBRATION="${EXTRINSIC_RECALIBRATION:-0}"
LIGHTGLUE_WEIGHTS="${LIGHTGLUE_WEIGHTS:-}"
VIDEO_DIR="${VIDEO_DIR:-}"
NUM_SHARDS="${NUM_SHARDS:-1}"
OVERWRITE="${OVERWRITE:-0}"
MERGE="${MERGE:-0}"

# apptainer container running the dap3-sam3 env (CPU only here; no --nv). CONDA_ENV is an
# alternative for a local (non-apptainer) install.
CONTAINER="${CONTAINER:-}"
CONDA_ENV="${CONDA_ENV:-}"

mkdir -p "$REPO_ROOT/hpc/logs/slurm" "$OUT_DIR"

export HF_HUB_OFFLINE=1
export TMPDIR=/tmp

cd "$REPO_ROOT"

if [[ "$MERGE" == "1" ]]; then
  CMD=(python "$REPO_ROOT/apps/camera_trap/calibration/apply.py" --job-dir "$JOB_DIR" --calib-dir "$CALIB_DIR" --out-dir "$OUT_DIR" --merge)
else
  SHARD_INDEX="$SLURM_ARRAY_TASK_ID"
  CMD=(
    python "$REPO_ROOT/apps/camera_trap/calibration/apply.py"
    --job-dir "$JOB_DIR"
    --calib-dir "$CALIB_DIR"
    --out-dir "$OUT_DIR"
    --align-method "$ALIGN_METHOD"
    --min-align-pixels "$MIN_ALIGN_PIXELS"
    --shard-index "$SHARD_INDEX"
    --num-shards "$NUM_SHARDS"
  )
  if [[ "$ALIGN_BLUR" == "1" ]]; then
    CMD+=(--align-blur)
  fi
  if [[ "$FORCE_POOLED" == "1" ]]; then
    CMD+=(--force-pooled)
  fi
  if [[ "$EXTRINSIC_RECALIBRATION" == "1" ]]; then
    CMD+=(--extrinsic-recalibration)
    if [[ -n "$LIGHTGLUE_WEIGHTS" ]]; then
      CMD+=(--lightglue-weights "$LIGHTGLUE_WEIGHTS")
    fi
  fi
  if [[ -n "$VIDEO_DIR" ]]; then
    CMD+=(--video-dir "$VIDEO_DIR")
  fi
  if [[ "$OVERWRITE" == "1" ]]; then
    CMD+=(--overwrite)
  fi
fi

echo "Host: $(hostname)"
echo "Start time: $(date)"
echo "JOB_DIR: $JOB_DIR"
echo "CALIB_DIR: $CALIB_DIR"
echo "OUT_DIR: $OUT_DIR"
echo "Command: ${CMD[*]}"

# Only $HOME is mounted into containers by default here; bind the shared filesystems too, so video,
# output and cache paths under /scratch, /projects or /lus resolve inside the container.
if [[ -z "${APPTAINER_BIND:-}" ]]; then
  APPTAINER_BIND=""
  for _p in /lus /scratch /projects /local; do [[ -d "$_p" ]] && APPTAINER_BIND+="${APPTAINER_BIND:+,}$_p"; done
  export APPTAINER_BIND
fi

if [[ -n "$CONTAINER" ]]; then
  apptainer exec \
    --env HF_HUB_OFFLINE=1 --env TMPDIR=/tmp \
    "$CONTAINER" "${CMD[@]}"
elif [[ -n "$CONDA_ENV" ]]; then
  source "$HOME/miniforge3/bin/activate"
  conda activate "$CONDA_ENV"
  "${CMD[@]}"
else
  echo "Set CONTAINER (the dap3-sam3 .sif) or CONDA_ENV before submitting." >&2
  exit 1
fi

echo "End time: $(date)"
