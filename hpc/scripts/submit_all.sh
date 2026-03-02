#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
JOBS_DIR="$REPO_ROOT/hpc/jobs"

if [[ ! -d "$JOBS_DIR" ]]; then
  echo "Jobs directory not found: $JOBS_DIR" >&2
  exit 1
fi

submitted=0
while IFS= read -r -d '' job; do
  name="$(basename "$job")"
  if [[ "$name" == "dap3_predict_array.sh" ]]; then
    continue
  fi
  echo "Submitting $name"
  sbatch "$job"
  submitted=$((submitted + 1))
done < <(find "$JOBS_DIR" -maxdepth 1 -type f -name '*.sh' -print0 | sort -z)

if (( submitted == 0 )); then
  echo "No non-array job scripts found in $JOBS_DIR"
else
  echo "Submitted $submitted job script(s)."
fi
