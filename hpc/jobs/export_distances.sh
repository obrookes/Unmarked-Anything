#!/bin/bash
#SBATCH --job-name=export_distances
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=hpc/logs/slurm/%x-%j.out
#SBATCH --error=hpc/logs/slurm/%x-%j.err

set -euo pipefail

# Export per-detection distances CSV from a dap3 job's outputs (JSON only -- no NPZ needed),
# applying depth calibration and mask-QC track drops. CPU-only.
#
#   JOB_DIR=hpc/runs/main_job CALIBRATED_OBJECTS=hpc/runs/main_job/calib_applied/calibrated_objects.csv \
#     TRACK_QC=hpc/runs/main_job/qc/track_qc.csv OUTPUT_CSV=hpc/runs/main_job/distances.csv \
#     sbatch hpc/jobs/export_distances.sh

REPO_ROOT="${REPO_ROOT:-$HOME/Unmarked-Anything}"
JOB_DIR="${JOB_DIR:?Set JOB_DIR to the dap3 output dir to export}"
OUTPUT_CSV="${OUTPUT_CSV:?Set OUTPUT_CSV to the destination CSV path}"
CALIBRATED_OBJECTS="${CALIBRATED_OBJECTS:-}"
TRACK_QC="${TRACK_QC:-}"
VIDEO_DIR="${VIDEO_DIR:-}"
VIDEO_TIMES="${VIDEO_TIMES:-}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-2.0}"
WINDOW_SECONDS="${WINDOW_SECONDS:-1.0}"
CONDA_ENV="${CONDA_ENV:-dap3-stream}"

mkdir -p "$REPO_ROOT/hpc/logs/slurm" "$(dirname "$OUTPUT_CSV")"

cd "$REPO_ROOT"

source "$HOME/miniforge3/bin/activate"
conda activate "$CONDA_ENV"

CMD=(
  python "$REPO_ROOT/apps/camera_trap/scripts/export_job_distances_csv.py"
  --job-dir "$JOB_DIR"
  --output-csv "$OUTPUT_CSV"
  --interval-seconds "$INTERVAL_SECONDS"
  --window-seconds "$WINDOW_SECONDS"
)
if [[ -n "$CALIBRATED_OBJECTS" ]]; then
  CMD+=(--calibrated-objects "$CALIBRATED_OBJECTS")
fi
if [[ -n "$TRACK_QC" ]]; then
  CMD+=(--track-qc "$TRACK_QC")
fi
if [[ -n "$VIDEO_DIR" ]]; then
  CMD+=(--video-dir "$VIDEO_DIR")
fi
if [[ -n "$VIDEO_TIMES" ]]; then
  CMD+=(--video-times "$VIDEO_TIMES")
fi

echo "Host: $(hostname)"
echo "Start time: $(date)"
echo "JOB_DIR: $JOB_DIR"
echo "OUTPUT_CSV: $OUTPUT_CSV"
echo "Command: ${CMD[*]}"

"${CMD[@]}"

echo "End time: $(date)"
