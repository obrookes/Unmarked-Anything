#!/bin/bash
#SBATCH --job-name=build_reference
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=hpc/logs/slurm/%x-%A_%a.out
#SBATCH --error=hpc/logs/slurm/%x-%A_%a.err

set -euo pipefail

# Build per-camera distance-calibration references (Haucke et al. 2022 method: SAM3 person mask
# + DA3 depth, disparity-space piecewise-linear fit) from a calibration_frames.csv of
# person-holding-a-distance-sign frames. Isambard AI compute nodes are offline, so both the SAM3
# and DA3 model weights must already be cached (HF_HOME) before submitting -- that's on the
# user, not this script.
#
#   CALIBRATION_FRAMES=configs/pss_p3/calibration_frames.csv \
#     REFERENCE_ROOT=/path/to/wcf-pps-p3 \
#     OUT_DIR=hpc/runs/reference_calib \
#     SAM3_MODEL_PATH=/path/to/sam3.pt \
#     sbatch hpc/jobs/build_reference.sh

REPO_ROOT="${REPO_ROOT:-$HOME/Unmarked-Anything}"
CALIBRATION_FRAMES="${CALIBRATION_FRAMES:?Set CALIBRATION_FRAMES to the calibration_frames.csv path}"
REFERENCE_ROOT="${REFERENCE_ROOT:?Set REFERENCE_ROOT to the dir that relative video_path entries are relative to}"
OUT_DIR="${OUT_DIR:?Set OUT_DIR for calibration outputs}"
SAM3_MODEL_PATH="${SAM3_MODEL_PATH:?Set SAM3_MODEL_PATH to the SAM3 checkpoint (.pt)}"
SAM3_PROMPT="${SAM3_PROMPT:-person}"
SAM3_DET_THRESHOLD="${SAM3_DET_THRESHOLD:-0.5}"
DA3_MODEL_ID="${DA3_MODEL_ID:-depth-anything/DA3NESTED-GIANT-LARGE}"
DA3_BATCH_SIZE="${DA3_BATCH_SIZE:-8}"
DEVICE="${DEVICE:-auto}"
CAMERAS_CSV="${CAMERAS_CSV:-$REPO_ROOT/configs/pss_p3/camera_metadata.csv}"
MIN_DEPTH="${MIN_DEPTH:-1.0}"
MAX_DEPTH="${MAX_DEPTH:-25.0}"
ONLY_CAMS="${ONLY_CAMS:-}"
OVERWRITE="${OVERWRITE:-0}"

# Offline apptainer container with sam3 + DA3 (weights pre-fetched to HF_HOME).
CONTAINER="${CONTAINER:-$REPO_ROOT/envs/containers/dap3-sam3.sif}"
SCRATCH="${SCRATCH:-/scratch/$USER}"
HF_HOME="${HF_HOME:-$SCRATCH/hf-cache}"

mkdir -p "$REPO_ROOT/hpc/logs/slurm" "$OUT_DIR"

export HF_HUB_OFFLINE=1
export HF_HOME
export TMPDIR=/tmp

if [[ "$CALIBRATION_FRAMES" != /* ]]; then
  CALIBRATION_FRAMES="$REPO_ROOT/$CALIBRATION_FRAMES"
fi
if [[ "$SAM3_MODEL_PATH" != /* ]]; then
  SAM3_MODEL_PATH="$REPO_ROOT/$SAM3_MODEL_PATH"
fi
if [[ "$CAMERAS_CSV" != /* ]]; then
  CAMERAS_CSV="$REPO_ROOT/$CAMERAS_CSV"
fi

CMD=(
  python3 "$REPO_ROOT/apps/camera_trap/calibration/build_reference.py"
  --calibration-frames "$CALIBRATION_FRAMES"
  --reference-root "$REFERENCE_ROOT"
  --out-dir "$OUT_DIR"
  --sam3-model-path "$SAM3_MODEL_PATH"
  --sam3-prompt "$SAM3_PROMPT"
  --sam3-det-threshold "$SAM3_DET_THRESHOLD"
  --da3-model-id "$DA3_MODEL_ID"
  --da3-batch-size "$DA3_BATCH_SIZE"
  --device "$DEVICE"
  --cameras "$CAMERAS_CSV"
  --min-depth "$MIN_DEPTH"
  --max-depth "$MAX_DEPTH"
)
if [[ -n "$ONLY_CAMS" ]]; then
  CMD+=(--only-cams "$ONLY_CAMS")
fi
if [[ "$OVERWRITE" == "1" ]]; then
  CMD+=(--overwrite)
fi

echo "Host: $(hostname)"
echo "Start time: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-}"
echo "CALIBRATION_FRAMES: $CALIBRATION_FRAMES"
echo "OUT_DIR: $OUT_DIR"
echo "Command: ${CMD[*]}"

cd "$REPO_ROOT"

# Only $HOME is mounted into containers by default here; bind the shared filesystems too, so video,
# output and cache paths under /scratch, /projects or /lus resolve inside the container.
if [[ -z "${APPTAINER_BIND:-}" ]]; then
  APPTAINER_BIND=""
  for _p in /lus /scratch /projects /local; do [[ -d "$_p" ]] && APPTAINER_BIND+="${APPTAINER_BIND:+,}$_p"; done
  export APPTAINER_BIND
fi

apptainer exec --nv \
  --env "PYTHONPATH=$REPO_ROOT:$REPO_ROOT/src" \
  --env HF_HUB_OFFLINE=1 --env "HF_HOME=$HF_HOME" --env TMPDIR=/tmp \
  --bind "$REPO_ROOT" --bind "$REFERENCE_ROOT" \
  "$CONTAINER" "${CMD[@]}"

echo "End time: $(date)"
