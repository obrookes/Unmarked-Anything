#!/bin/bash
#SBATCH --job-name=dap3_predict_array
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=hpc/logs/slurm/%x-%A_%a.out
#SBATCH --error=hpc/logs/slurm/%x-%A_%a.err

set -euo pipefail

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "SLURM_ARRAY_TASK_ID is not set. Submit with sbatch --array=1-N." >&2
  exit 1
fi

REPO_ROOT="${REPO_ROOT:-$HOME/Unmarked-Anything}"
MANIFEST="${MANIFEST:-$REPO_ROOT/hpc/configs/job_manifest.tsv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/hpc/runs}"
USE_SCRATCH="${USE_SCRATCH:-1}"

mkdir -p "$REPO_ROOT/hpc/logs/slurm" "$OUTPUT_ROOT"

if [[ ! -f "$MANIFEST" ]]; then
  echo "Manifest not found: $MANIFEST" >&2
  exit 1
fi

mapfile -t JOB_LINES < <(grep -v '^[[:space:]]*#' "$MANIFEST" | sed '/^[[:space:]]*$/d')
JOB_COUNT="${#JOB_LINES[@]}"

if (( JOB_COUNT == 0 )); then
  echo "No runnable jobs found in manifest: $MANIFEST" >&2
  exit 1
fi

TASK_INDEX=$((SLURM_ARRAY_TASK_ID - 1))
if (( TASK_INDEX < 0 || TASK_INDEX >= JOB_COUNT )); then
  echo "Array index ${SLURM_ARRAY_TASK_ID} out of range (1..${JOB_COUNT})." >&2
  exit 1
fi

LINE="${JOB_LINES[$TASK_INDEX]}"
IFS=$'\t' read -r JOB_TAG INPUT_VIDEO_DIR SAM3_MODEL_PATH SAM3_TEXT_PROMPTS TARGET_FPS SAM3_MODE DEVICE USE_HALF OVERWRITE MAX_VIDEOS <<< "$LINE"

JOB_TAG="${JOB_TAG:-job${SLURM_ARRAY_TASK_ID}}"
TARGET_FPS="${TARGET_FPS:-1.0}"
SAM3_MODE="${SAM3_MODE:-track}"
DEVICE="${DEVICE:-auto}"
USE_HALF="${USE_HALF:-0}"
OVERWRITE="${OVERWRITE:-0}"

if [[ "$INPUT_VIDEO_DIR" != /* ]]; then
  INPUT_VIDEO_DIR="$REPO_ROOT/$INPUT_VIDEO_DIR"
fi
if [[ "$SAM3_MODEL_PATH" != /* ]]; then
  SAM3_MODEL_PATH="$REPO_ROOT/$SAM3_MODEL_PATH"
fi

if [[ -z "$SAM3_TEXT_PROMPTS" ]]; then
  echo "Empty SAM3_TEXT_PROMPTS in manifest line: $LINE" >&2
  exit 1
fi

cd "$REPO_ROOT"

module purge
module load cuda/12.6

source "$HOME/miniforge3/bin/activate"
conda activate dap-3_py3-11

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

RUN_TAG="${SLURM_JOB_NAME}-${JOB_TAG}-${SLURM_ARRAY_TASK_ID}-${SLURM_JOB_ID}"
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
if [[ -n "${MAX_VIDEOS:-}" ]]; then
  CMD+=(--max-videos "$MAX_VIDEOS")
fi

echo "Host: $(hostname)"
echo "Start time: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID} | SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "Manifest: $MANIFEST"
echo "Selected line: $LINE"
echo "Command: ${CMD[*]}"

"${CMD[@]}"

if [[ "$WORK_OUTPUT_DIR" != "$FINAL_OUTPUT_DIR" ]]; then
  mkdir -p "$FINAL_OUTPUT_DIR"
  rsync -a "$WORK_OUTPUT_DIR/" "$FINAL_OUTPUT_DIR/"
fi

echo "Output written to: $FINAL_OUTPUT_DIR"
echo "End time: $(date)"
