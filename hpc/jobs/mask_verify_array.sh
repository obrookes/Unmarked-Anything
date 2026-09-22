#!/bin/bash
#SBATCH --job-name=mask_verify_array
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=hpc/logs/slurm/%x-%A_%a.out
#SBATCH --error=hpc/logs/slurm/%x-%A_%a.err

set -euo pipefail

# Mask QC ("agent check") on a dap3 job's outputs, using a local VLM inside an offline vLLM
# container (Isambard AI compute nodes have no internet access, so the model weights must
# already be cached to $HF_HOME / $VLLM_CACHE_ROOT ahead of time -- that's on the user, not this
# script). Submit as an array (one task per shard):
#
#   JOB_DIR=hpc/runs/some_job OUT_DIR=hpc/runs/some_job/qc sbatch --array=0-3 hpc/jobs/mask_verify_array.sh
#
# --array=0-(NUM_SHARDS-1) must match NUM_SHARDS below (default 1: a single task processes every
# video). Sharding is by video index (mask_verify.py's --shard-index/--num-shards), which is a
# cheap, order-stable split -- no separate manifest needed.

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "SLURM_ARRAY_TASK_ID is not set. Submit with sbatch --array=0-(NUM_SHARDS-1)." >&2
  exit 1
fi

REPO_ROOT="${REPO_ROOT:-$HOME/Unmarked-Anything}"
JOB_DIR="${JOB_DIR:?Set JOB_DIR to the dap3 output dir to QC}"
OUT_DIR="${OUT_DIR:-$JOB_DIR/qc}"
VIDEO_DIR="${VIDEO_DIR:-}"
MODEL="${MODEL:-qwen}"
FRAMES_PER_TRACK="${FRAMES_PER_TRACK:-5}"
REJECT_FRAC_MAX="${REJECT_FRAC_MAX:-0.5}"
MIN_CHECKED="${MIN_CHECKED:-1}"
TP="${TP:-1}"
NUM_SHARDS="${NUM_SHARDS:-1}"
GALLERY="${GALLERY:-0}"
OVERWRITE="${OVERWRITE:-0}"

# vLLM container / env. Parameterised so a different container path or conda env can be swapped
# in without editing the script; either CONTAINER (a .sif run via apptainer/singularity) or
# CONDA_ENV (a local vLLM env) may be used -- set whichever matches how vLLM is installed here.
CONTAINER="${CONTAINER:-}"
CONDA_ENV="${CONDA_ENV:-}"
SCRATCH="${SCRATCH:-/scratch/$USER}"
HF_HOME="${HF_HOME:-$SCRATCH/hf-cache}"

mkdir -p "$REPO_ROOT/hpc/logs/slurm" "$OUT_DIR"

SHARD_INDEX="$SLURM_ARRAY_TASK_ID"

export HF_HUB_OFFLINE=1
export HF_HOME
export TMPDIR=/tmp

CMD=(
  python "$REPO_ROOT/apps/camera_trap/qc/mask_verify.py"
  --job-dir "$JOB_DIR"
  --out-dir "$OUT_DIR"
  --model "$MODEL"
  --frames-per-track "$FRAMES_PER_TRACK"
  --reject-frac-max "$REJECT_FRAC_MAX"
  --min-checked "$MIN_CHECKED"
  --tp "$TP"
  --shard-index "$SHARD_INDEX"
  --num-shards "$NUM_SHARDS"
)
if [[ -n "$VIDEO_DIR" ]]; then
  CMD+=(--video-dir "$VIDEO_DIR")
fi
if [[ "$GALLERY" != "0" ]]; then
  CMD+=(--gallery "$GALLERY")
fi
if [[ "$OVERWRITE" == "1" ]]; then
  CMD+=(--overwrite)
fi

echo "Host: $(hostname)"
echo "Start time: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID} | SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID} (shard $SHARD_INDEX/$NUM_SHARDS)"
echo "JOB_DIR: $JOB_DIR"
echo "OUT_DIR: $OUT_DIR"
echo "Command: ${CMD[*]}"

cd "$REPO_ROOT"

if [[ -n "$CONTAINER" ]]; then
  apptainer exec --nv \
    --env HF_HUB_OFFLINE=1 --env "HF_HOME=$HF_HOME" --env TMPDIR=/tmp \
    "$CONTAINER" "${CMD[@]}"
elif [[ -n "$CONDA_ENV" ]]; then
  source "$HOME/miniforge3/bin/activate"
  conda activate "$CONDA_ENV"
  "${CMD[@]}"
else
  echo "Set CONTAINER (a vLLM .sif) or CONDA_ENV (a local vLLM env) before submitting." >&2
  exit 1
fi

echo "End time: $(date)"
