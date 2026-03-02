#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FROM_DIR="${1:-${SLURM_TMPDIR:-}}"
TO_DIR="${2:-$REPO_ROOT/hpc/runs/manual-collect-$(date +%Y%m%d-%H%M%S)}"

if [[ -z "$FROM_DIR" ]]; then
  echo "Source directory is empty. Pass it explicitly: collect_results.sh <from_dir> [to_dir]" >&2
  exit 1
fi
if [[ ! -d "$FROM_DIR" ]]; then
  echo "Source directory not found: $FROM_DIR" >&2
  exit 1
fi

mkdir -p "$TO_DIR"
rsync -a "$FROM_DIR/" "$TO_DIR/"
echo "Collected results from: $FROM_DIR"
echo "Collected results into: $TO_DIR"
