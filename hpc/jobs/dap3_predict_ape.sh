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
INPUT_VIDEO_DIR="${INPUT_VIDEO_DIR:-$REPO_ROOT/demo/input}"
SAM3_MODEL_PATH="${SAM3_MODEL_PATH:-$REPO_ROOT/weights/sam3/safari_checkpoint_hf.pt}"
SAM3_TEXT_PROMPTS="${SAM3_TEXT_PROMPTS:-ape}"
TARGET_FPS="${TARGET_FPS:-1.0}"
SAM3_MODE="${SAM3_MODE:-track}"
DEVICE="${DEVICE:-auto}"
USE_HALF="${USE_HALF:-0}"
OVERWRITE="${OVERWRITE:-0}"
MAX_VIDEOS="${MAX_VIDEOS:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/hpc/runs}"
USE_SCRATCH="${USE_SCRATCH:-1}"

mkdir -p "$REPO_ROOT/hpc/logs/slurm" "$OUTPUT_ROOT"

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
  --target-fps "$TARGET_FPS"
  --sam3-mode "$SAM3_MODE"
  --device "$DEVICE"
)

if [[ "$USE_HALF" == "1" ]]; then
  CMD+=(--half)
fi

if [[ "$OVERWRITE" == "1" ]]; then
  CMD+=(--overwrite)
fi

if [[ -n "$MAX_VIDEOS" ]]; then
  CMD+=(--max-videos "$MAX_VIDEOS")
fi

echo "Command: ${CMD[*]}"
"${CMD[@]}"

if [[ "$WORK_OUTPUT_DIR" != "$FINAL_OUTPUT_DIR" ]]; then
  mkdir -p "$FINAL_OUTPUT_DIR"
  rsync -a "$WORK_OUTPUT_DIR/" "$FINAL_OUTPUT_DIR/"
fi

echo "Output written to: $FINAL_OUTPUT_DIR"
echo "End time: $(date)"
