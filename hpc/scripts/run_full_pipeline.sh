#!/bin/bash
# Submit the full PSS P3 pipeline as dependency-chained SLURM jobs:
#
#   A.  find_sign_frames array (VLM agent scans reference videos for sign frames)
#   A'. find_sign_frames build-only merge          (depends on A)      -> calibration_frames.csv
#   B.  build_reference (SAM3 person mask + DA3 depth, per-camera fit) (depends on A') -> calib/
#   C.  dap3 main array on chimp videos, --npz                        (parallel with A/A'/B)
#   D.  mask_verify array                          (depends on C)
#   E1. apply_calibration array                    (depends on B, C)
#   E1'.apply_calibration merge                     (depends on E1)     -> calibrated_objects.csv
#   E2. export_job_distances_csv                   (depends on D, E1') -> distances.csv
#   F.  build_ctds_inputs + ctds_abundance.R        (depends on E2)     -> abundance/
#
# Every stage's underlying CLI resumes/skips existing outputs (--overwrite=0 by default), so
# re-running this script after a partial failure is safe. Individual stages can also be skipped
# outright with SKIP_<STAGE>=1 env vars; skipping a stage whose output another stage needs
# requires pointing the pipeline at that existing output (see the *_CSV / *_DIR override vars
# below).
#
# Usage:
#   PIPELINE_ENV=hpc/configs/my_pipeline.env hpc/scripts/run_full_pipeline.sh
#   DRY_RUN=1 PIPELINE_ENV=hpc/configs/my_pipeline.env hpc/scripts/run_full_pipeline.sh
#
# See hpc/configs/pipeline.env.example for the variables this script reads.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT_DEFAULT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PIPELINE_ENV="${PIPELINE_ENV:-$REPO_ROOT_DEFAULT/hpc/configs/pipeline.env.example}"
if [[ ! -f "$PIPELINE_ENV" ]]; then
  echo "Pipeline env file not found: $PIPELINE_ENV" >&2
  echo "Copy hpc/configs/pipeline.env.example, edit it, and set PIPELINE_ENV." >&2
  exit 1
fi
# shellcheck disable=SC1090
source "$PIPELINE_ENV"

REPO_ROOT="${REPO_ROOT:-$REPO_ROOT_DEFAULT}"
OUT_ROOT="${OUT_ROOT:?Set OUT_ROOT in the pipeline env file}"
VIDEO_DIR="${VIDEO_DIR:?Set VIDEO_DIR in the pipeline env file}"
REFERENCE_ROOT="${REFERENCE_ROOT:?Set REFERENCE_ROOT in the pipeline env file}"

REFERENCE_MAP="${REFERENCE_MAP:-$REPO_ROOT/configs/pss_p3/reference_video_map.csv}"
SCAN_FPS="${SCAN_FPS:-2}"

SAM3_CKPT="${SAM3_CKPT:?Set SAM3_CKPT in the pipeline env file}"
SAM3_BACKEND="${SAM3_BACKEND:-official}"
SAM3_DET_THRESHOLD="${SAM3_DET_THRESHOLD:-0.5}"
CALIB_SAM3_PROMPT="${CALIB_SAM3_PROMPT:-person}"
DA3_MODEL_ID="${DA3_MODEL_ID:-depth-anything/DA3NESTED-GIANT-LARGE}"
DA3_MODE="${DA3_MODE:-batch}"
DA3_BATCH="${DA3_BATCH:-48}"
TARGET_FPS="${TARGET_FPS:-6.0}"
DEPTH_INTERVAL_S="${DEPTH_INTERVAL_S:-2}"
MAIN_PROMPTS="${MAIN_PROMPTS:-ape}"
MIN_DEPTH="${MIN_DEPTH:-1}"
MAX_DEPTH="${MAX_DEPTH:-25}"
EXTRINSIC_RECALIBRATION="${EXTRINSIC_RECALIBRATION:-0}"
LIGHTGLUE_WEIGHTS="${LIGHTGLUE_WEIGHTS:-}"

DAP3_SAM3_CONTAINER="${DAP3_SAM3_CONTAINER:-}"
DAP3_CONDA_ENV="${DAP3_CONDA_ENV:-dap3-stream}"
VLM_CONTAINER="${VLM_CONTAINER:-}"
VLM_CONDA_ENV="${VLM_CONDA_ENV:-}"
VLM_MODEL="${VLM_MODEL:-qwen}"
VLM_GPUS="${VLM_GPUS:-1}"  # GPUs per VLM task; also the vLLM tensor-parallel size (glm needs 4)
R_CONDA_ENV="${R_CONDA_ENV:-}"
R_MODULE="${R_MODULE:-}"
HF_HOME="${HF_HOME:-}"
SLURM_PARTITION="${SLURM_PARTITION:-}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-}"
FIND_SIGN_NUM_SHARDS="${FIND_SIGN_NUM_SHARDS:-1}"
APPLY_NUM_SHARDS="${APPLY_NUM_SHARDS:-1}"
MASK_VERIFY_NUM_SHARDS="${MASK_VERIFY_NUM_SHARDS:-1}"

DRY_RUN="${DRY_RUN:-0}"
SKIP_SIGNS="${SKIP_SIGNS:-0}"
SKIP_REFCAL="${SKIP_REFCAL:-0}"
SKIP_MAIN="${SKIP_MAIN:-0}"
SKIP_QC="${SKIP_QC:-0}"
SKIP_APPLY="${SKIP_APPLY:-0}"
SKIP_EXPORT="${SKIP_EXPORT:-0}"
SKIP_ABUNDANCE="${SKIP_ABUNDANCE:-0}"

# Overrides for resuming from an existing run when the stage that produced it is skipped.
CALIBRATION_FRAMES_CSV="${CALIBRATION_FRAMES_CSV:-$OUT_ROOT/sign_frames/calibration_frames.csv}"
CALIB_DIR="${CALIB_DIR:-$OUT_ROOT/calibration/calib}"
MAIN_JOB_DIR="${MAIN_JOB_DIR:-$OUT_ROOT/main}"
TRACK_QC_CSV="${TRACK_QC_CSV:-$MAIN_JOB_DIR/qc/track_qc.csv}"
CALIBRATED_OBJECTS_CSV="${CALIBRATED_OBJECTS_CSV:-$MAIN_JOB_DIR/calib_applied/calibrated_objects.csv}"
DISTANCES_CSV="${DISTANCES_CSV:-$OUT_ROOT/distances.csv}"

MANIFEST_DIR="$OUT_ROOT/manifests"
mkdir -p "$MANIFEST_DIR"

# Manifest field order (see hpc/README.md "Array Jobs" for the full column list):
#   input_video_dir sam3_model_path sam3_text_prompts_csv da3_model_id da3_mode
#   da3_stream_config target_fps sam3_mode device conf use_half overwrite max_videos
#   da3_batch_size sam3_track_isolation sam3_track_tail_policy write_npz sam3_backend
#   sam3_det_threshold depth_interval_seconds
write_manifest() {
  local path="$1" input_dir="$2" prompt="$3"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$input_dir" "$SAM3_CKPT" "$prompt" "$DA3_MODEL_ID" "$DA3_MODE" \
    "da3_streaming/configs/base_config.yaml" "$TARGET_FPS" "track" "auto" "0.25" "1" "0" "" \
    "$DA3_BATCH" "recreate" "warn_and_finalize" "1" "$SAM3_BACKEND" "$SAM3_DET_THRESHOLD" \
    "$DEPTH_INTERVAL_S" \
    > "$path"
}

# The RUN_TAG dap3_predict_array.sh computes for a single-row (array=1-1) manifest, so this
# script can locate that job's output dir once sbatch hands back a job id. Mirrors the slugify +
# tag logic in hpc/jobs/dap3_predict_array.sh -- keep in sync if that logic changes.
slugify() {
  local s="$1"
  s="${s,,}"
  s="${s// /-}"
  s="${s//,/-}"
  s="$(echo "$s" | sed -E 's/[^a-z0-9._-]+/-/g; s/^-+//; s/-+$//; s/-+/-/g')"
  printf '%s' "$s"
}

run_tag_for() {
  local job_name="$1" model_path="$2" prompt="$3"
  local model_name model_tag da3_tag prompt_tag fps_tag mode_tag da3_exec_tag
  model_name="$(basename "$model_path")"; model_name="${model_name%.*}"
  model_tag="$(slugify "$model_name")"
  da3_tag="$(slugify "${DA3_MODEL_ID##*/}")"
  prompt_tag="$(slugify "$prompt")"
  fps_tag="$(slugify "${TARGET_FPS//./p}")"
  mode_tag="track"
  da3_exec_tag="$(slugify "$DA3_MODE")"
  [[ -z "$model_tag" ]] && model_tag="sam3"
  [[ -z "$da3_tag" ]] && da3_tag="da3"
  [[ -z "$prompt_tag" ]] && prompt_tag="noprompt"
  local job_tag="${model_tag}-${da3_tag}-${da3_exec_tag}-${prompt_tag}-fps${fps_tag}-${mode_tag}"
  job_tag="${job_tag:0:80}"
  printf '%s-%s-1-%s' "$job_name" "$job_tag" "$4"
}

sbatch_extra_args=()
[[ -n "$SLURM_PARTITION" ]] && sbatch_extra_args+=(--partition="$SLURM_PARTITION")
[[ -n "$SLURM_ACCOUNT" ]] && sbatch_extra_args+=(--account="$SLURM_ACCOUNT")

submit_job_id() {
  # Prints the sbatch command to stderr (so it's visible but not captured). In DRY_RUN mode,
  # nothing is submitted; a placeholder job id is returned instead so downstream stages still
  # print their --dependency wiring against something. Otherwise submits it, echoes sbatch's own
  # stdout to stderr, and returns the parsed job id on stdout.
  echo "sbatch command: $*" >&2
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRYRUN_JOBID"
    return 0
  fi
  local out
  out="$("$@")"
  echo "$out" >&2
  awk '/Submitted batch job/ {print $4}' <<<"$out" | tail -n1
}

echo "== PSS P3 pipeline =="
echo "PIPELINE_ENV: $PIPELINE_ENV"
echo "REPO_ROOT:    $REPO_ROOT"
echo "OUT_ROOT:     $OUT_ROOT"
echo "DRY_RUN:      $DRY_RUN"
echo ""

export REPO_ROOT

## -- Stage A: find_sign_frames array (VLM scan of reference videos) ----------
JOB_ID_A=""
if [[ "$SKIP_SIGNS" == "1" ]]; then
  echo "[A] SKIP_SIGNS=1 -- skipping find_sign_frames scan."
else
  echo "[A] Submitting find_sign_frames array..."
  export_vars="ALL,REPO_ROOT=$REPO_ROOT,OUT_DIR=$OUT_ROOT/sign_frames,REFERENCE_MAP=$REFERENCE_MAP,REFERENCE_ROOT=$REFERENCE_ROOT,MODEL=$VLM_MODEL,TP=$VLM_GPUS,SCAN_FPS=$SCAN_FPS,NUM_SHARDS=$FIND_SIGN_NUM_SHARDS"
  [[ -n "$VLM_CONTAINER" ]] && export_vars="$export_vars,CONTAINER=$VLM_CONTAINER"
  [[ -n "$VLM_CONDA_ENV" ]] && export_vars="$export_vars,CONDA_ENV=$VLM_CONDA_ENV"
  [[ -n "$HF_HOME" ]] && export_vars="$export_vars,HF_HOME=$HF_HOME"
  JOB_ID_A="$(submit_job_id sbatch "${sbatch_extra_args[@]}" --job-name=find_sign_frames --gres="gpu:$VLM_GPUS" \
    --array="0-$((FIND_SIGN_NUM_SHARDS - 1))" --export="$export_vars" \
    "$REPO_ROOT/hpc/jobs/find_sign_frames.sh")"
  echo "  job id: ${JOB_ID_A:-<dry-run>}"
fi
echo ""

## -- Stage A': find_sign_frames build-only merge ------------------------------
JOB_ID_A2=""
if [[ "$SKIP_SIGNS" == "1" ]]; then
  echo "[A'] SKIP_SIGNS=1 -- skipping find_sign_frames merge. Using CALIBRATION_FRAMES_CSV=$CALIBRATION_FRAMES_CSV"
else
  echo "[A'] Submitting find_sign_frames build-only merge (dependency on A)..."
  dep_args=()
  [[ -n "$JOB_ID_A" ]] && dep_args+=(--dependency="afterok:$JOB_ID_A")
  export_vars="ALL,REPO_ROOT=$REPO_ROOT,OUT_DIR=$OUT_ROOT/sign_frames,REFERENCE_MAP=$REFERENCE_MAP,REFERENCE_ROOT=$REFERENCE_ROOT,MODEL=$VLM_MODEL,TP=$VLM_GPUS,SCAN_FPS=$SCAN_FPS,BUILD_ONLY=1"
  [[ -n "$VLM_CONTAINER" ]] && export_vars="$export_vars,CONTAINER=$VLM_CONTAINER"
  [[ -n "$VLM_CONDA_ENV" ]] && export_vars="$export_vars,CONDA_ENV=$VLM_CONDA_ENV"
  [[ -n "$HF_HOME" ]] && export_vars="$export_vars,HF_HOME=$HF_HOME"
  JOB_ID_A2="$(submit_job_id sbatch "${sbatch_extra_args[@]}" "${dep_args[@]}" --job-name=find_sign_frames_build \
    --export="$export_vars" "$REPO_ROOT/hpc/jobs/find_sign_frames.sh")"
  echo "  job id: ${JOB_ID_A2:-<dry-run>}"
  CALIBRATION_FRAMES_CSV="$OUT_ROOT/sign_frames/calibration_frames.csv"
fi
echo ""

## -- Stage B: build_reference (SAM3 person mask + DA3 depth calibration) -----
JOB_ID_B=""
if [[ "$SKIP_REFCAL" == "1" ]]; then
  echo "[B] SKIP_REFCAL=1 -- skipping build_reference. Using CALIB_DIR=$CALIB_DIR"
else
  echo "[B] Submitting build_reference (dependency on A')..."
  dep_args=()
  [[ -n "$JOB_ID_A2" ]] && dep_args+=(--dependency="afterok:$JOB_ID_A2")
  export_vars="ALL,REPO_ROOT=$REPO_ROOT,CALIBRATION_FRAMES=$CALIBRATION_FRAMES_CSV,REFERENCE_ROOT=$REFERENCE_ROOT,OUT_DIR=$OUT_ROOT/calibration,SAM3_MODEL_PATH=$SAM3_CKPT,SAM3_PROMPT=$CALIB_SAM3_PROMPT,SAM3_DET_THRESHOLD=$SAM3_DET_THRESHOLD,DA3_MODEL_ID=$DA3_MODEL_ID,MIN_DEPTH=$MIN_DEPTH,MAX_DEPTH=$MAX_DEPTH"
  [[ -n "$DAP3_SAM3_CONTAINER" ]] && export_vars="$export_vars,CONTAINER=$DAP3_SAM3_CONTAINER"
  [[ -n "$HF_HOME" ]] && export_vars="$export_vars,HF_HOME=$HF_HOME"
  JOB_ID_B="$(submit_job_id sbatch "${sbatch_extra_args[@]}" "${dep_args[@]}" --job-name=build_reference \
    --export="$export_vars" "$REPO_ROOT/hpc/jobs/build_reference.sh")"
  echo "  job id: ${JOB_ID_B:-<dry-run>}"
  CALIB_DIR="$OUT_ROOT/calibration/calib"
fi
echo ""

## -- Stage C: dap3 main array on chimp videos --------------------------------
JOB_ID_C=""
if [[ "$SKIP_MAIN" == "1" ]]; then
  echo "[C] SKIP_MAIN=1 -- skipping main dap3 run. Using MAIN_JOB_DIR=$MAIN_JOB_DIR"
else
  echo "[C] Submitting dap3 main array on chimp videos..."
  MAIN_MANIFEST="$MANIFEST_DIR/main.tsv"
  write_manifest "$MAIN_MANIFEST" "$VIDEO_DIR" "$MAIN_PROMPTS"
  export_vars="ALL,REPO_ROOT=$REPO_ROOT,MANIFEST=$MAIN_MANIFEST,OUTPUT_ROOT=$OUT_ROOT/main,WRITE_NPZ=1,SAM3_BACKEND=$SAM3_BACKEND,SAM3_DET_THRESHOLD=$SAM3_DET_THRESHOLD,DEPTH_INTERVAL_SECONDS=$DEPTH_INTERVAL_S"
  [[ -n "$DAP3_SAM3_CONTAINER" ]] && export_vars="$export_vars,CONTAINER=$DAP3_SAM3_CONTAINER"
  [[ -n "$DAP3_CONDA_ENV" ]] && export_vars="$export_vars,CONDA_ENV=$DAP3_CONDA_ENV"
  [[ -n "$HF_HOME" ]] && export_vars="$export_vars,HF_HOME=$HF_HOME"
  JOB_ID_C="$(submit_job_id sbatch "${sbatch_extra_args[@]}" --job-name=dap3_main --array=1-1 \
    --export="$export_vars" "$REPO_ROOT/hpc/jobs/dap3_predict_array.sh")"
  if [[ -n "$JOB_ID_C" ]]; then
    MAIN_JOB_DIR="$OUT_ROOT/main/$(run_tag_for dap3_main "$SAM3_CKPT" "$MAIN_PROMPTS" "$JOB_ID_C")"
  fi
  echo "  job id: ${JOB_ID_C:-<dry-run>}"
  echo "  main job dir: ${MAIN_JOB_DIR:-<unknown until job runs>}"
fi
echo ""

## -- Stage D: mask QC array ---------------------------------------------------
JOB_ID_D=""
if [[ "$SKIP_QC" == "1" ]]; then
  echo "[D] SKIP_QC=1 -- skipping mask QC."
else
  echo "[D] Submitting mask QC array (dependency on C)..."
  dep_args=()
  [[ -n "$JOB_ID_C" ]] && dep_args+=(--dependency="afterok:$JOB_ID_C")
  QC_OUT_DIR="$MAIN_JOB_DIR/qc"
  export_vars="ALL,REPO_ROOT=$REPO_ROOT,JOB_DIR=$MAIN_JOB_DIR,OUT_DIR=$QC_OUT_DIR,MODEL=$VLM_MODEL,TP=$VLM_GPUS,NUM_SHARDS=$MASK_VERIFY_NUM_SHARDS"
  [[ -n "$VLM_CONTAINER" ]] && export_vars="$export_vars,CONTAINER=$VLM_CONTAINER"
  [[ -n "$VLM_CONDA_ENV" ]] && export_vars="$export_vars,CONDA_ENV=$VLM_CONDA_ENV"
  [[ -n "$HF_HOME" ]] && export_vars="$export_vars,HF_HOME=$HF_HOME"
  JOB_ID_D="$(submit_job_id sbatch "${sbatch_extra_args[@]}" "${dep_args[@]}" --job-name=mask_verify_array --gres="gpu:$VLM_GPUS" \
    --array="0-$((MASK_VERIFY_NUM_SHARDS - 1))" --export="$export_vars" \
    "$REPO_ROOT/hpc/jobs/mask_verify_array.sh")"
  echo "  job id: ${JOB_ID_D:-<dry-run>}"
  TRACK_QC_CSV="$QC_OUT_DIR/track_qc.csv"
fi
echo ""

## -- Stage E1: apply_calibration array ----------------------------------------
JOB_ID_E1=""
if [[ "$SKIP_APPLY" == "1" ]]; then
  echo "[E1] SKIP_APPLY=1 -- skipping apply_calibration. Using CALIBRATED_OBJECTS_CSV=$CALIBRATED_OBJECTS_CSV"
else
  echo "[E1] Submitting apply_calibration array (dependency on B, C)..."
  dep_parts=()
  [[ -n "$JOB_ID_B" ]] && dep_parts+=("afterok:$JOB_ID_B")
  [[ -n "$JOB_ID_C" ]] && dep_parts+=("afterok:$JOB_ID_C")
  dep_args=()
  if [[ ${#dep_parts[@]} -gt 0 ]]; then
    dep_str="$(IFS=,; echo "${dep_parts[*]}")"
    dep_args+=(--dependency="$dep_str")
  fi
  APPLY_OUT_DIR="$MAIN_JOB_DIR/calib_applied"
  export_vars="ALL,REPO_ROOT=$REPO_ROOT,JOB_DIR=$MAIN_JOB_DIR,CALIB_DIR=$CALIB_DIR,OUT_DIR=$APPLY_OUT_DIR,NUM_SHARDS=$APPLY_NUM_SHARDS"
  if [[ "$EXTRINSIC_RECALIBRATION" == "1" ]]; then
    export_vars="$export_vars,EXTRINSIC_RECALIBRATION=1"
    [[ -n "$LIGHTGLUE_WEIGHTS" ]] && export_vars="$export_vars,LIGHTGLUE_WEIGHTS=$LIGHTGLUE_WEIGHTS"
  fi
  [[ -n "$DAP3_SAM3_CONTAINER" ]] && export_vars="$export_vars,CONTAINER=$DAP3_SAM3_CONTAINER"
  [[ -n "$DAP3_CONDA_ENV" ]] && export_vars="$export_vars,CONDA_ENV=$DAP3_CONDA_ENV"
  JOB_ID_E1="$(submit_job_id sbatch "${sbatch_extra_args[@]}" "${dep_args[@]}" --job-name=apply_calibration \
    --array="0-$((APPLY_NUM_SHARDS - 1))" --export="$export_vars" \
    "$REPO_ROOT/hpc/jobs/apply_calibration.sh")"
  echo "  job id: ${JOB_ID_E1:-<dry-run>}"
fi
echo ""

## -- Stage E1': apply_calibration merge ---------------------------------------
JOB_ID_E1_MERGE=""
if [[ "$SKIP_APPLY" == "1" ]]; then
  echo "[E1'] SKIP_APPLY=1 -- skipping apply_calibration merge."
else
  echo "[E1'] Submitting apply_calibration merge (dependency on E1)..."
  dep_args=()
  [[ -n "$JOB_ID_E1" ]] && dep_args+=(--dependency="afterok:$JOB_ID_E1")
  APPLY_OUT_DIR="$MAIN_JOB_DIR/calib_applied"
  export_vars="ALL,REPO_ROOT=$REPO_ROOT,JOB_DIR=$MAIN_JOB_DIR,CALIB_DIR=$CALIB_DIR,OUT_DIR=$APPLY_OUT_DIR,MERGE=1"
  [[ -n "$DAP3_SAM3_CONTAINER" ]] && export_vars="$export_vars,CONTAINER=$DAP3_SAM3_CONTAINER"
  [[ -n "$DAP3_CONDA_ENV" ]] && export_vars="$export_vars,CONDA_ENV=$DAP3_CONDA_ENV"
  JOB_ID_E1_MERGE="$(submit_job_id sbatch "${sbatch_extra_args[@]}" "${dep_args[@]}" --job-name=apply_calibration_merge \
    --export="$export_vars" "$REPO_ROOT/hpc/jobs/apply_calibration.sh")"
  echo "  job id: ${JOB_ID_E1_MERGE:-<dry-run>}"
  CALIBRATED_OBJECTS_CSV="$APPLY_OUT_DIR/calibrated_objects.csv"
fi
echo ""

## -- Stage E2: export distances CSV -------------------------------------------
JOB_ID_E2=""
if [[ "$SKIP_EXPORT" == "1" ]]; then
  echo "[E2] SKIP_EXPORT=1 -- skipping export. Using DISTANCES_CSV=$DISTANCES_CSV"
else
  echo "[E2] Submitting export (dependency on D, E1')..."
  dep_parts=()
  [[ -n "$JOB_ID_D" ]] && dep_parts+=("afterok:$JOB_ID_D")
  [[ -n "$JOB_ID_E1_MERGE" ]] && dep_parts+=("afterok:$JOB_ID_E1_MERGE")
  dep_args=()
  if [[ ${#dep_parts[@]} -gt 0 ]]; then
    dep_str="$(IFS=,; echo "${dep_parts[*]}")"
    dep_args+=(--dependency="$dep_str")
  fi
  export_vars="ALL,REPO_ROOT=$REPO_ROOT,JOB_DIR=$MAIN_JOB_DIR,OUTPUT_CSV=$DISTANCES_CSV,CALIBRATED_OBJECTS=$CALIBRATED_OBJECTS_CSV,TRACK_QC=$TRACK_QC_CSV,INTERVAL_SECONDS=$DEPTH_INTERVAL_S"
  [[ -n "$DAP3_CONDA_ENV" ]] && export_vars="$export_vars,CONDA_ENV=$DAP3_CONDA_ENV"
  JOB_ID_E2="$(submit_job_id sbatch "${sbatch_extra_args[@]}" "${dep_args[@]}" --job-name=export_distances \
    --export="$export_vars" "$REPO_ROOT/hpc/jobs/export_distances.sh")"
  echo "  job id: ${JOB_ID_E2:-<dry-run>}"
fi
echo ""

## -- Stage F: build_ctds_inputs + ctds_abundance.R -----------------------------
JOB_ID_F=""
if [[ "$SKIP_ABUNDANCE" == "1" ]]; then
  echo "[F] SKIP_ABUNDANCE=1 -- skipping abundance estimation."
else
  echo "[F] Submitting abundance estimation (dependency on E2)..."
  dep_args=()
  [[ -n "$JOB_ID_E2" ]] && dep_args+=(--dependency="afterok:$JOB_ID_E2")
  export_vars="ALL,REPO_ROOT=$REPO_ROOT,DISTANCES=$DISTANCES_CSV,OUT_DIR=$OUT_ROOT/abundance"
  [[ -n "$DAP3_CONDA_ENV" ]] && export_vars="$export_vars,CONDA_ENV=$DAP3_CONDA_ENV"
  [[ -n "$R_CONDA_ENV" ]] && export_vars="$export_vars,R_CONDA_ENV=$R_CONDA_ENV"
  [[ -n "$R_MODULE" ]] && export_vars="$export_vars,R_MODULE=$R_MODULE"
  JOB_ID_F="$(submit_job_id sbatch "${sbatch_extra_args[@]}" "${dep_args[@]}" --job-name=ctds_abundance \
    --export="$export_vars" "$REPO_ROOT/hpc/jobs/ctds_abundance.sh")"
  echo "  job id: ${JOB_ID_F:-<dry-run>}"
fi
echo ""

echo "== Submitted job ids =="
echo "A  (find_sign_frames):        ${JOB_ID_A:-skipped}"
echo "A' (find_sign_frames build):  ${JOB_ID_A2:-skipped}"
echo "B  (build_reference):         ${JOB_ID_B:-skipped}"
echo "C  (main dap3):                ${JOB_ID_C:-skipped}"
echo "D  (mask QC):                  ${JOB_ID_D:-skipped}"
echo "E1 (apply_calibration):        ${JOB_ID_E1:-skipped}"
echo "E1'(apply_calibration merge):  ${JOB_ID_E1_MERGE:-skipped}"
echo "E2 (export):                   ${JOB_ID_E2:-skipped}"
echo "F  (abundance):                ${JOB_ID_F:-skipped}"
echo ""
echo "Outputs:"
echo "  calibration_frames: $CALIBRATION_FRAMES_CSV"
echo "  calib:              $CALIB_DIR"
echo "  calibrated_objects: $CALIBRATED_OBJECTS_CSV"
echo "  distances:          $DISTANCES_CSV"
echo "  abundance:           $OUT_ROOT/abundance"
