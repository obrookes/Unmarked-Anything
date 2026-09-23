#!/bin/bash
#SBATCH --job-name=find_sign_frames
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=hpc/logs/slurm/%x-%A_%a.out
#SBATCH --error=hpc/logs/slurm/%x-%A_%a.err

set -euo pipefail

# Find PSS P3 distance-calibration sign frames in reference videos, using a local VLM inside an
# offline vLLM container (Isambard AI compute nodes have no internet access, so the model weights
# must already be cached to $HF_HOME / $VLLM_CACHE_ROOT ahead of time -- that's on the user, not
# this script). Submit as an array (one task per shard):
#
#   OUT_DIR=hpc/runs/ref_job/sign_frames REFERENCE_ROOT=/path/to/wcf-pps-p3 \
#     sbatch --array=0-3 hpc/jobs/find_sign_frames.sh
#
# --array=0-(NUM_SHARDS-1) must match NUM_SHARDS below (default 1: a single task processes every
# video). Sharding is by video index (find_sign_frames.py's --shard-index/--num-shards), a cheap,
# order-stable split. Each task rebuilds calibration_frames.csv/events_summary.csv/contact_sheet.html
# from every scan*.jsonl present so far, so a build reflecting *all* shards' output needs a final
# separate call once every array task has finished -- set BUILD_ONLY=1 (and no --array) for that.

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" && "${BUILD_ONLY:-0}" != "1" ]]; then
  echo "SLURM_ARRAY_TASK_ID is not set. Submit with sbatch --array=0-(NUM_SHARDS-1), or set BUILD_ONLY=1 for a single build-only task." >&2
  exit 1
fi

REPO_ROOT="${REPO_ROOT:-$HOME/Unmarked-Anything}"
OUT_DIR="${OUT_DIR:?Set OUT_DIR for scan.jsonl / calibration_frames.csv}"
REFERENCE_MAP="${REFERENCE_MAP:-$REPO_ROOT/configs/pss_p3/reference_video_map.csv}"
REFERENCE_ROOT="${REFERENCE_ROOT:?Set REFERENCE_ROOT to the dir that reference_dir paths in REFERENCE_MAP are relative to}"
MODEL="${MODEL:-qwen}"
SCAN_FPS="${SCAN_FPS:-2}"
MAX_SIDE="${MAX_SIDE:-1536}"
TILE_FALLBACK="${TILE_FALLBACK:-1}"
MIN_AGREE="${MIN_AGREE:-2}"
MAX_GAP_S="${MAX_GAP_S:-1.0}"
CAMERAS="${CAMERAS:-}"
OVERRIDE_CSV="${OVERRIDE_CSV:-}"
TP="${TP:-1}"
NUM_SHARDS="${NUM_SHARDS:-1}"
OVERWRITE="${OVERWRITE:-0}"
BUILD_ONLY="${BUILD_ONLY:-0}"

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
  python "$REPO_ROOT/apps/camera_trap/calibration/find_sign_frames.py"
  --reference-map "$REFERENCE_MAP"
  --reference-root "$REFERENCE_ROOT"
  --out-dir "$OUT_DIR"
  --model "$MODEL"
  --scan-fps "$SCAN_FPS"
  --max-side "$MAX_SIDE"
  --min-agree "$MIN_AGREE"
  --max-gap-s "$MAX_GAP_S"
  --tp "$TP"
)
if [[ "$TILE_FALLBACK" == "1" ]]; then
  CMD+=(--tile-fallback)
fi
if [[ -n "$CAMERAS" ]]; then
  CMD+=(--cameras "$CAMERAS")
fi
if [[ -n "$OVERRIDE_CSV" ]]; then
  CMD+=(--override-csv "$OVERRIDE_CSV")
fi
if [[ "$OVERWRITE" == "1" ]]; then
  CMD+=(--overwrite)
fi

if [[ "$BUILD_ONLY" == "1" ]]; then
  CMD+=(--build-only)
else
  SHARD_INDEX="$SLURM_ARRAY_TASK_ID"
  CMD+=(--shard-index "$SHARD_INDEX" --num-shards "$NUM_SHARDS")
fi

echo "Host: $(hostname)"
echo "Start time: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-} | SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID:-} (BUILD_ONLY=$BUILD_ONLY)"
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
# vLLM runtime env for the GLM-5.3-Flash FP8 container (mirrors vision-llm-ann-verifier's run_vllm.sh):
# DeepGEMM's FP8 JIT needs nvcc (absent in the container) and JIT caches would otherwise land in ~/.cache.
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$SCRATCH/vllm-cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$SCRATCH/triton-cache}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-$SCRATCH/flashinfer-cache}"

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
