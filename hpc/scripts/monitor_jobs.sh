#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="$REPO_ROOT/hpc/logs/slurm"
USER_NAME="${USER}"
TAIL_LINES="${TAIL_LINES:-40}"

echo "== squeue ($USER_NAME) =="
squeue -u "$USER_NAME" || true

echo
echo "== recent logs ($LOG_DIR) =="
if [[ -d "$LOG_DIR" ]]; then
  mapfile -t logs < <(find "$LOG_DIR" -maxdepth 1 -type f \( -name '*.out' -o -name '*.err' \) -printf '%T@ %p\n' | sort -nr | head -n 3 | awk '{print $2}')
  if (( ${#logs[@]} == 0 )); then
    echo "No log files yet."
  else
    for f in "${logs[@]}"; do
      echo "--- $f (last ${TAIL_LINES} lines) ---"
      tail -n "$TAIL_LINES" "$f" || true
      echo
    done
  fi
else
  echo "Log directory does not exist yet: $LOG_DIR"
fi
