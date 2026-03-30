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
IFS=$'\t' read -r -a F <<< "$LINE"

# Manifest format columns:
# 1  input_video_dir
# 2  sam3_model_path
# 3  sam3_text_prompts_csv
# 4  da3_model_id
# 5  da3_mode (optional: batch|stream|all_frames, default batch)
# 6  da3_stream_config (optional path, used in stream mode)
# 7  target_fps
# 8  sam3_mode
# 9  device
# 10 conf
# 11 use_half
# 12 overwrite
# 13 max_videos
# 14 da3_batch_size
# 15 sam3_track_isolation (optional: recreate|reset|both)
# 16 sam3_track_tail_policy (optional: warn_and_finalize|fail_fast)
# 17 write_npz (optional: 0|1, default 0 — set to 1 to write *_arrays.npz)
INPUT_VIDEO_DIR="${F[0]:-}"
SAM3_MODEL_PATH="${F[1]:-}"
SAM3_TEXT_PROMPTS="${F[2]:-}"
DA3_MODEL_ID="${F[3]:-depth-anything/DA3NESTED-GIANT-LARGE}"
DA3_MODE="${F[4]:-batch}"
DA3_STREAM_CONFIG="${F[5]:-da3_streaming/configs/base_config.yaml}"
TARGET_FPS="${F[6]:-1.0}"
SAM3_MODE="${F[7]:-track}"
DEVICE="${F[8]:-auto}"
CONF="${F[9]:-0.25}"
USE_HALF="${F[10]:-0}"
OVERWRITE="${F[11]:-0}"
MAX_VIDEOS="${F[12]:-}"
DA3_BATCH_SIZE="${F[13]:-}"
SAM3_TRACK_ISOLATION="${F[14]:-recreate}"
SAM3_TRACK_TAIL_POLICY="${F[15]:-warn_and_finalize}"
WRITE_NPZ="${WRITE_NPZ:-${F[16]:-0}}"

if [[ -z "$INPUT_VIDEO_DIR" || -z "$SAM3_MODEL_PATH" || -z "$SAM3_TEXT_PROMPTS" || -z "$DA3_BATCH_SIZE" ]]; then
  echo "Invalid manifest line (missing required fields): $LINE" >&2
  exit 1
fi
if ! [[ "$DA3_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid da3_batch_size '$DA3_BATCH_SIZE' in manifest line: $LINE" >&2
  echo "da3_batch_size must be a positive integer." >&2
  exit 1
fi
case "$DA3_MODE" in
  batch|stream|all_frames) ;;
  *)
    echo "Invalid da3_mode '$DA3_MODE' in manifest line: $LINE" >&2
    echo "da3_mode must be one of: batch, stream, all_frames." >&2
    exit 1
    ;;
esac
case "$SAM3_TRACK_ISOLATION" in
  recreate|reset|both) ;;
  *)
    echo "Invalid sam3_track_isolation '$SAM3_TRACK_ISOLATION' in manifest line: $LINE" >&2
    echo "sam3_track_isolation must be one of: recreate, reset, both." >&2
    exit 1
    ;;
esac
case "$SAM3_TRACK_TAIL_POLICY" in
  warn_and_finalize|fail_fast) ;;
  *)
    echo "Invalid sam3_track_tail_policy '$SAM3_TRACK_TAIL_POLICY' in manifest line: $LINE" >&2
    echo "sam3_track_tail_policy must be one of: warn_and_finalize, fail_fast." >&2
    exit 1
    ;;
esac

if [[ "$INPUT_VIDEO_DIR" != /* ]]; then
  INPUT_VIDEO_DIR="$REPO_ROOT/$INPUT_VIDEO_DIR"
fi
if [[ "$SAM3_MODEL_PATH" != /* ]]; then
  SAM3_MODEL_PATH="$REPO_ROOT/$SAM3_MODEL_PATH"
fi
if [[ "$DA3_STREAM_CONFIG" != /* ]]; then
  DA3_STREAM_CONFIG="$REPO_ROOT/$DA3_STREAM_CONFIG"
fi
if [[ "$DA3_MODE" == "stream" && ! -f "$DA3_STREAM_CONFIG" ]]; then
  echo "DA3 stream config not found: $DA3_STREAM_CONFIG" >&2
  exit 1
fi

slugify() {
  local s="$1"
  s="${s,,}"                          # lowercase
  s="${s// /-}"                       # spaces to dash
  s="${s//,/-}"                       # commas to dash
  s="$(echo "$s" | sed -E 's/[^a-z0-9._-]+/-/g; s/^-+//; s/-+$//; s/-+/-/g')"
  printf '%s' "$s"
}

# Build job tag from config. Use only checkpoint file name, not full path.
MODEL_NAME="$(basename "$SAM3_MODEL_PATH")"
MODEL_NAME="${MODEL_NAME%.*}"
MODEL_TAG="$(slugify "$MODEL_NAME")"
DA3_NAME="${DA3_MODEL_ID##*/}"        # keep only final model token
DA3_NAME="${DA3_NAME%.*}"
DA3_TAG="$(slugify "$DA3_NAME")"
PROMPT_TAG="$(slugify "$SAM3_TEXT_PROMPTS")"
FPS_TAG="$(slugify "${TARGET_FPS//./p}")"
MODE_TAG="$(slugify "$SAM3_MODE")"
DA3_EXEC_TAG="$(slugify "$DA3_MODE")"
[[ -z "$MODEL_TAG" ]] && MODEL_TAG="sam3"
[[ -z "$DA3_TAG" ]] && DA3_TAG="da3"
[[ -z "$PROMPT_TAG" ]] && PROMPT_TAG="noprompt"
[[ -z "$FPS_TAG" ]] && FPS_TAG="1p0"
[[ -z "$MODE_TAG" ]] && MODE_TAG="track"
[[ -z "$DA3_EXEC_TAG" ]] && DA3_EXEC_TAG="batch"
JOB_TAG="${MODEL_TAG}-${DA3_TAG}-${DA3_EXEC_TAG}-${PROMPT_TAG}-fps${FPS_TAG}-${MODE_TAG}"
JOB_TAG="${JOB_TAG:0:80}"

cd "$REPO_ROOT"

module purge
module load cuda/12.6

source "$HOME/miniforge3/bin/activate"
conda activate dap3-stream

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
  --da3-model-id "$DA3_MODEL_ID"
  --da3-mode "$DA3_MODE"
  --target-fps "$TARGET_FPS"
  --sam3-mode "$SAM3_MODE"
  --conf "$CONF"
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
if [[ -n "${MAX_VIDEOS:-}" ]]; then
  CMD+=(--max-videos "$MAX_VIDEOS")
fi
CMD+=(--da3-batch-size "$DA3_BATCH_SIZE")
CMD+=(--sam3-track-isolation "$SAM3_TRACK_ISOLATION")
CMD+=(--sam3-track-tail-policy "$SAM3_TRACK_TAIL_POLICY")
if [[ "$WRITE_NPZ" == "1" ]]; then
  CMD+=(--npz)
fi

echo "Host: $(hostname)"
echo "Start time: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID} | SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "Manifest: $MANIFEST"
echo "Selected line: $LINE"
echo "Auto job tag: $JOB_TAG"
echo "Command: ${CMD[*]}"

"${CMD[@]}"

if [[ "$WORK_OUTPUT_DIR" != "$FINAL_OUTPUT_DIR" ]]; then
  mkdir -p "$FINAL_OUTPUT_DIR"
  rsync -a "$WORK_OUTPUT_DIR/" "$FINAL_OUTPUT_DIR/"
fi

echo "Output written to: $FINAL_OUTPUT_DIR"
echo "End time: $(date)"
