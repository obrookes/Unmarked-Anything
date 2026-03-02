#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
JOB_SCRIPT="${1:-$REPO_ROOT/hpc/jobs/dap3_predict_array.sh}"
MANIFEST="${2:-$REPO_ROOT/hpc/configs/job_manifest.tsv}"

if [[ ! -f "$JOB_SCRIPT" ]]; then
  echo "Array job script not found: $JOB_SCRIPT" >&2
  exit 1
fi
if [[ ! -f "$MANIFEST" ]]; then
  echo "Manifest not found: $MANIFEST" >&2
  exit 1
fi

count=$(grep -v '^[[:space:]]*#' "$MANIFEST" | sed '/^[[:space:]]*$/d' | wc -l)
count=$(echo "$count" | tr -d '[:space:]')

if [[ "$count" == "0" ]]; then
  echo "No runnable lines found in manifest: $MANIFEST" >&2
  exit 1
fi

echo "Submitting array job with $count task(s) using manifest: $MANIFEST"
sbatch --array="1-${count}" --export=ALL,MANIFEST="$MANIFEST" "$JOB_SCRIPT"
