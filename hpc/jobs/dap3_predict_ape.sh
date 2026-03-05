#!/bin/bash
#SBATCH --job-name=dap3_predict_ape
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=hpc/logs/slurm/%x-%j.out
#SBATCH --error=hpc/logs/slurm/%x-%j.err

set -euo pipefail

echo "Host: $(hostname)"
echo "Start time: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"

REPO_ROOT="${REPO_ROOT:-$HOME/Unmarked-Anything}"
INPUT_VIDEO_DIR="${INPUT_VIDEO_DIR:-$REPO_ROOT/assets/videos}"
SAM3_MODEL_PATH="${SAM3_MODEL_PATH:-$REPO_ROOT/weights/sam3/safari_checkpoint_hf.pt}"
SAM3_TEXT_PROMPTS="${SAM3_TEXT_PROMPTS:-ape}"
DA3_MODEL_ID="${DA3_MODEL_ID:-depth-anything/DA3NESTED-GIANT-LARGE}"
DA3_MODE="${DA3_MODE:-batch}"
DA3_STREAM_CONFIG="${DA3_STREAM_CONFIG:-$REPO_ROOT/da3_streaming/configs/base_config.yaml}"
TARGET_FPS="${TARGET_FPS:-1.0}"
SAM3_MODE="${SAM3_MODE:-track}"
DEVICE="${DEVICE:-auto}"
USE_HALF="${USE_HALF:-0}"
OVERWRITE="${OVERWRITE:-0}"
MAX_VIDEOS="${MAX_VIDEOS:-}"
DA3_BATCH_SIZE="${DA3_BATCH_SIZE:-}"
SAM3_TRACK_ISOLATION="${SAM3_TRACK_ISOLATION:-recreate}"
SAM3_TRACK_TAIL_POLICY="${SAM3_TRACK_TAIL_POLICY:-warn_and_finalize}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/hpc/runs}"
USE_SCRATCH="${USE_SCRATCH:-1}"

mkdir -p "$REPO_ROOT/hpc/logs/slurm" "$OUTPUT_ROOT"

if [[ -z "$DA3_BATCH_SIZE" ]]; then
  echo "DA3_BATCH_SIZE is required and must be a positive integer." >&2
  exit 1
fi
if ! [[ "$DA3_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid DA3_BATCH_SIZE '$DA3_BATCH_SIZE'. Must be a positive integer." >&2
  exit 1
fi
case "$DA3_MODE" in
  batch|stream) ;;
  *)
    echo "Invalid DA3_MODE '$DA3_MODE'. Must be batch or stream." >&2
    exit 1
    ;;
esac
if [[ "$DA3_MODE" == "stream" && ! -f "$DA3_STREAM_CONFIG" ]]; then
  echo "DA3 stream config not found: $DA3_STREAM_CONFIG" >&2
  exit 1
fi
case "$SAM3_TRACK_ISOLATION" in
  recreate|reset|both) ;;
  *)
    echo "Invalid SAM3_TRACK_ISOLATION '$SAM3_TRACK_ISOLATION'. Must be recreate, reset, or both." >&2
    exit 1
    ;;
esac
case "$SAM3_TRACK_TAIL_POLICY" in
  warn_and_finalize|fail_fast) ;;
  *)
    echo "Invalid SAM3_TRACK_TAIL_POLICY '$SAM3_TRACK_TAIL_POLICY'. Must be warn_and_finalize or fail_fast." >&2
    exit 1
    ;;
esac

cd "$REPO_ROOT"

module purge
module load cuda/12.6

source "$HOME/miniforge3/bin/activate"
conda activate dap-3_py3-11

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

RUN_TAG="${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
FINAL_OUTPUT_DIR="$OUTPUT_ROOT/$RUN_TAG"
WORK_OUTPUT_DIR="$FINAL_OUTPUT_DIR"

if [[ "$USE_SCRATCH" == "1" && -n "${SLURM_TMPDIR:-}" ]]; then
  WORK_OUTPUT_DIR="$SLURM_TMPDIR/$RUN_TAG"
fi
mkdir -p "$WORK_OUTPUT_DIR"

IFS=',' read -r -a PROMPTS <<< "$SAM3_TEXT_PROMPTS"

CMD=(
  python "$REPO_ROOT/dap3_cli.py"
  --input-video-dir "$INPUT_VIDEO_DIR"
  --output-dir "$WORK_OUTPUT_DIR"
  --sam3-model-path "$SAM3_MODEL_PATH"
  --sam3-text-prompts "${PROMPTS[@]}"
  --da3-model-id "$DA3_MODEL_ID"
  --da3-mode "$DA3_MODE"
  --target-fps "$TARGET_FPS"
  --sam3-mode "$SAM3_MODE"
  --device "$DEVICE"
)
if [[ "$DA3_MODE" == "stream" ]]; then
  CMD+=(--da3-stream-config "$DA3_STREAM_CONFIG")
fi

if [[ "$USE_HALF" == "1" ]]; then
  CMD+=(--half)
fi

if [[ "$OVERWRITE" == "1" ]]; then
  CMD+=(--overwrite)
fi

if [[ -n "$MAX_VIDEOS" ]]; then
  CMD+=(--max-videos "$MAX_VIDEOS")
fi
CMD+=(--da3-batch-size "$DA3_BATCH_SIZE")
CMD+=(--sam3-track-isolation "$SAM3_TRACK_ISOLATION")
CMD+=(--sam3-track-tail-policy "$SAM3_TRACK_TAIL_POLICY")

echo "Command: ${CMD[*]}"
"${CMD[@]}"

if [[ "$WORK_OUTPUT_DIR" != "$FINAL_OUTPUT_DIR" ]]; then
  mkdir -p "$FINAL_OUTPUT_DIR"
  rsync -a "$WORK_OUTPUT_DIR/" "$FINAL_OUTPUT_DIR/"
fi

echo "Output written to: $FINAL_OUTPUT_DIR"
echo "End time: $(date)"
