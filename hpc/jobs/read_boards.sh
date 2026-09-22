#!/bin/bash
#SBATCH --job-name=read_boards
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=hpc/logs/slurm/%x-%A_%a.out
#SBATCH --error=hpc/logs/slurm/%x-%A_%a.err

set -euo pipefail

# Read PSS P3 distance-calibration board numbers from dap3-processed reference videos, using a
# local VLM inside an offline vLLM container (Isambard AI compute nodes have no internet access,
# so the model weights must already be cached to $HF_HOME / $VLLM_CACHE_ROOT ahead of time --
# that's on the user, not this script). Optionally also runs fit.py to produce calibration.json.
#
#   JOB_DIR=hpc/runs/ref_job OUT_DIR=hpc/runs/ref_job/calib REFERENCE_ROOT=/path/to/wcf-pps-p3 \
#     sbatch hpc/jobs/read_boards.sh

REPO_ROOT="${REPO_ROOT:-$HOME/Unmarked-Anything}"
JOB_DIR="${JOB_DIR:?Set JOB_DIR to the dap3 output dir for the reference videos}"
OUT_DIR="${OUT_DIR:-$JOB_DIR/calib}"
REFERENCE_MAP="${REFERENCE_MAP:-$REPO_ROOT/configs/pss_p3/reference_video_map.csv}"
REFERENCE_ROOT="${REFERENCE_ROOT:?Set REFERENCE_ROOT to the dir that reference_dir paths in REFERENCE_MAP are relative to}"
VIDEO_DIR="${VIDEO_DIR:-}"
VIDEO_CAM_CSV="${VIDEO_CAM_CSV:-}"
MODEL="${MODEL:-qwen}"
MAX_PER_TRACK="${MAX_PER_TRACK:-20}"
MIN_SAM_CONF="${MIN_SAM_CONF:-0.0}"
CONTACT_SHEET="${CONTACT_SHEET:-20}"
TP="${TP:-1}"
OVERWRITE="${OVERWRITE:-0}"
FIT="${FIT:-0}"
CAMERAS_CSV="${CAMERAS_CSV:-$REPO_ROOT/configs/pss_p3/camera_metadata.csv}"

# vLLM container / env. Parameterised so a different container path or conda env can be swapped
# in without editing the script; either CONTAINER (a .sif run via apptainer/singularity) or
# CONDA_ENV (a local vLLM env) may be used -- set whichever matches how vLLM is installed here.
CONTAINER="${CONTAINER:-}"
CONDA_ENV="${CONDA_ENV:-}"
SCRATCH="${SCRATCH:-/scratch/$USER}"
HF_HOME="${HF_HOME:-$SCRATCH/hf-cache}"

mkdir -p "$REPO_ROOT/hpc/logs/slurm" "$OUT_DIR"

export HF_HUB_OFFLINE=1
export HF_HOME
export TMPDIR=/tmp

CMD=(
  python "$REPO_ROOT/apps/camera_trap/calibration/read_boards.py"
  --job-dir "$JOB_DIR"
  --out-dir "$OUT_DIR"
  --reference-map "$REFERENCE_MAP"
  --reference-root "$REFERENCE_ROOT"
  --model "$MODEL"
  --max-per-track "$MAX_PER_TRACK"
  --min-sam-conf "$MIN_SAM_CONF"
  --contact-sheet "$CONTACT_SHEET"
  --tp "$TP"
)
if [[ -n "$VIDEO_DIR" ]]; then
  CMD+=(--video-dir "$VIDEO_DIR")
fi
if [[ -n "$VIDEO_CAM_CSV" ]]; then
  CMD+=(--video-cam-csv "$VIDEO_CAM_CSV")
fi
if [[ "$OVERWRITE" == "1" ]]; then
  CMD+=(--overwrite)
fi

echo "Host: $(hostname)"
echo "Start time: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-}"
echo "JOB_DIR: $JOB_DIR"
echo "OUT_DIR: $OUT_DIR"
echo "Command: ${CMD[*]}"

cd "$REPO_ROOT"

run_cmd() {
  if [[ -n "$CONTAINER" ]]; then
    apptainer exec --nv \
      --env HF_HUB_OFFLINE=1 --env "HF_HOME=$HF_HOME" --env TMPDIR=/tmp \
      "$CONTAINER" "$@"
  elif [[ -n "$CONDA_ENV" ]]; then
    source "$HOME/miniforge3/bin/activate"
    conda activate "$CONDA_ENV"
    "$@"
  else
    echo "Set CONTAINER (a vLLM .sif) or CONDA_ENV (a local vLLM env) before submitting." >&2
    exit 1
  fi
}

run_cmd "${CMD[@]}"

if [[ "$FIT" == "1" ]]; then
  echo "Fitting calibration from $OUT_DIR/calib_points.csv"
  run_cmd python "$REPO_ROOT/apps/camera_trap/calibration/fit.py" \
    --points "$OUT_DIR/calib_points.csv" \
    --out-dir "$OUT_DIR" \
    --cameras "$CAMERAS_CSV"
fi

echo "End time: $(date)"
