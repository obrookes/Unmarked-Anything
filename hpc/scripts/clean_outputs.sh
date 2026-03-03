#!/bin/bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  hpc/scripts/clean_outputs.sh [--logs] [--runs] [--all] [--dry-run] [--yes]

Description:
  Cleans HPC runtime artifacts under this repo:
  - --logs: remove *.out and *.err files from hpc/logs/slurm
  - --runs: remove all contents inside hpc/runs

Options:
  --logs      Clean SLURM .out/.err logs only.
  --runs      Clean run outputs only.
  --all       Clean both logs and runs (equivalent to --logs --runs).
  --dry-run   Show what would be deleted, but do not delete anything.
  --yes       Skip interactive confirmation.
  -h, --help  Show this help.

Examples:
  hpc/scripts/clean_outputs.sh --logs
  hpc/scripts/clean_outputs.sh --runs --yes
  hpc/scripts/clean_outputs.sh --all --dry-run
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="$REPO_ROOT/hpc/logs/slurm"
RUNS_DIR="$REPO_ROOT/hpc/runs"

CLEAN_LOGS=0
CLEAN_RUNS=0
DRY_RUN=0
AUTO_YES=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --logs)
      CLEAN_LOGS=1
      ;;
    --runs)
      CLEAN_RUNS=1
      ;;
    --all)
      CLEAN_LOGS=1
      CLEAN_RUNS=1
      ;;
    --dry-run)
      DRY_RUN=1
      ;;
    --yes)
      AUTO_YES=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
  shift
done

if (( CLEAN_LOGS == 0 && CLEAN_RUNS == 0 )); then
  echo "No cleanup target selected. Use --logs, --runs, or --all." >&2
  usage
  exit 1
fi

log_files=()
run_entries=()

if (( CLEAN_LOGS == 1 )) && [[ -d "$LOG_DIR" ]]; then
  mapfile -t log_files < <(find "$LOG_DIR" -maxdepth 1 -type f \( -name '*.out' -o -name '*.err' \) | sort)
fi
if (( CLEAN_RUNS == 1 )) && [[ -d "$RUNS_DIR" ]]; then
  mapfile -t run_entries < <(find "$RUNS_DIR" -mindepth 1 -maxdepth 1 | sort)
fi

echo "Cleanup plan:"
if (( CLEAN_LOGS == 1 )); then
  echo "- logs: ${#log_files[@]} file(s) in $LOG_DIR"
fi
if (( CLEAN_RUNS == 1 )); then
  echo "- runs: ${#run_entries[@]} entr$( (( ${#run_entries[@]} == 1 )) && echo 'y' || echo 'ies') in $RUNS_DIR"
fi

if (( DRY_RUN == 1 )); then
  echo
  echo "Dry run enabled; no files were deleted."
  exit 0
fi

if (( AUTO_YES == 0 )); then
  read -r -p "Proceed with deletion? [y/N] " reply
  if [[ ! "$reply" =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
  fi
fi

if (( CLEAN_LOGS == 1 )) && (( ${#log_files[@]} > 0 )); then
  rm -f -- "${log_files[@]}"
fi

if (( CLEAN_RUNS == 1 )) && (( ${#run_entries[@]} > 0 )); then
  rm -rf -- "${run_entries[@]}"
fi

echo "Cleanup complete."
