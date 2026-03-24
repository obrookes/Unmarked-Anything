#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
JOB_SCRIPT="${1:-$REPO_ROOT/hpc/jobs/dap3_predict_array.sh}"
MANIFEST="${2:-$REPO_ROOT/hpc/configs/job_manifest.tsv}"
EXPORT_JOB_SCRIPT="${EXPORT_JOB_SCRIPT:-$REPO_ROOT/hpc/jobs/dap3_export_overlays_array.sh}"

EXPORT_DEPENDENCY="${EXPORT_DEPENDENCY:-afterok}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/hpc/runs}"
EXPORT_STYLE="${EXPORT_STYLE:-rgb}"
DEPTH_SOURCE="${DEPTH_SOURCE:-new}"
EXPORT_SUBDIR="${EXPORT_SUBDIR:-overlay_videos}"
VIDEO_DIR_OVERRIDE="${VIDEO_DIR_OVERRIDE:-}"
EXPORT_FPS="${EXPORT_FPS:-}"
EXPORT_FOURCC="${EXPORT_FOURCC:-mp4v}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -f "$JOB_SCRIPT" ]]; then
  echo "Array job script not found: $JOB_SCRIPT" >&2
  exit 1
fi
if [[ ! -f "$MANIFEST" ]]; then
  echo "Manifest not found: $MANIFEST" >&2
  exit 1
fi
if [[ ! -f "$EXPORT_JOB_SCRIPT" ]]; then
  echo "Export job script not found: $EXPORT_JOB_SCRIPT" >&2
  exit 1
fi

case "$EXPORT_DEPENDENCY" in
  afterok|afterany) ;;
  *)
    echo "Invalid EXPORT_DEPENDENCY '$EXPORT_DEPENDENCY'. Use afterok or afterany." >&2
    exit 1
    ;;
esac

case "$EXPORT_STYLE" in
  rgb|analysis-depth) ;;
  *)
    echo "Invalid EXPORT_STYLE '$EXPORT_STYLE'. Use rgb or analysis-depth." >&2
    exit 1
    ;;
esac

case "$DEPTH_SOURCE" in
  new|old) ;;
  *)
    echo "Invalid DEPTH_SOURCE '$DEPTH_SOURCE'. Use new or old." >&2
    exit 1
    ;;
esac

if [[ ${#EXPORT_FOURCC} -ne 4 ]]; then
  echo "Invalid EXPORT_FOURCC '$EXPORT_FOURCC'. It must be 4 characters." >&2
  exit 1
fi

if [[ -n "$EXPORT_FPS" ]]; then
  if ! [[ "$EXPORT_FPS" =~ ^[0-9]*\.?[0-9]+$ ]]; then
    echo "Invalid EXPORT_FPS '$EXPORT_FPS'. Must be numeric." >&2
    exit 1
  fi
fi

count="$(awk '!/^[[:space:]]*#/ && /[^[:space:]]/ {c++} END{print c+0}' "$MANIFEST")"
if [[ "$count" == "0" ]]; then
  echo "No runnable lines found in manifest: $MANIFEST" >&2
  exit 1
fi

predict_cmd=(
  sbatch
  --array="1-${count}"
  --export=ALL,MANIFEST="$MANIFEST"
  "$JOB_SCRIPT"
)

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[DRY_RUN] Predict submit command:"
  echo "  ${predict_cmd[*]}"
  echo "[DRY_RUN] Export submit command will be generated after predict job id is known."
  exit 0
fi

echo "Submitting predict array job (${count} task(s))..."
echo "  job script: $JOB_SCRIPT"
echo "  manifest:   $MANIFEST"
predict_submit_output="$("${predict_cmd[@]}")"
echo "$predict_submit_output"

predict_job_id="$(awk '/Submitted batch job/ {print $4}' <<<"$predict_submit_output" | tail -n1)"
if [[ -z "$predict_job_id" ]]; then
  echo "Failed to parse array job id from sbatch output." >&2
  exit 1
fi

export_vars=(
  "ALL"
  "REPO_ROOT=$REPO_ROOT"
  "OUTPUT_ROOT=$OUTPUT_ROOT"
  "PREDICT_ARRAY_JOB_ID=$predict_job_id"
  "EXPORT_STYLE=$EXPORT_STYLE"
  "DEPTH_SOURCE=$DEPTH_SOURCE"
  "EXPORT_SUBDIR=$EXPORT_SUBDIR"
  "VIDEO_DIR_OVERRIDE=$VIDEO_DIR_OVERRIDE"
  "EXPORT_FPS=$EXPORT_FPS"
  "EXPORT_FOURCC=$EXPORT_FOURCC"
)
export_var_string="$(IFS=,; echo "${export_vars[*]}")"

export_cmd=(
  sbatch
  --dependency="${EXPORT_DEPENDENCY}:${predict_job_id}"
  --export="$export_var_string"
  "$EXPORT_JOB_SCRIPT"
)

echo "Submitting dependent export job..."
echo "  dependency: ${EXPORT_DEPENDENCY}:${predict_job_id}"
echo "  export job: $EXPORT_JOB_SCRIPT"
export_submit_output="$("${export_cmd[@]}")"
echo "$export_submit_output"

export_job_id="$(awk '/Submitted batch job/ {print $4}' <<<"$export_submit_output" | tail -n1)"
if [[ -z "$export_job_id" ]]; then
  echo "Warning: unable to parse export job id from sbatch output." >&2
fi

echo "Submitted workflow:"
echo "  predict array job id: $predict_job_id"
if [[ -n "$export_job_id" ]]; then
  echo "  export job id:        $export_job_id"
fi
echo "  output root:          $OUTPUT_ROOT"
echo "  run-dir match rule:   *-${predict_job_id}"
