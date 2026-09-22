#!/bin/bash
#SBATCH --job-name=ctds_abundance
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=hpc/logs/slurm/%x-%j.out
#SBATCH --error=hpc/logs/slurm/%x-%j.err

set -euo pipefail

# Build CTDS (Distance-package) inputs from an exported distances CSV, then fit the
# density/abundance model with ctds_abundance.R. CPU-only.
#
#   DISTANCES=hpc/runs/main_job/distances.csv OUT_DIR=hpc/runs/main_job/abundance \
#     sbatch hpc/jobs/ctds_abundance.sh

REPO_ROOT="${REPO_ROOT:-$HOME/Unmarked-Anything}"
DISTANCES="${DISTANCES:?Set DISTANCES to the exported distances CSV}"
OUT_DIR="${OUT_DIR:?Set OUT_DIR for CTDS inputs and abundance outputs}"
METADATA="${METADATA:-$REPO_ROOT/configs/pss_p3/camera_metadata.csv}"
CTDS_CONFIG="${CTDS_CONFIG:-$REPO_ROOT/configs/pss_p3/ctds_config.yaml}"
MODEL="${MODEL:-auto}"
CONDA_ENV="${CONDA_ENV:-dap3-stream}"

# R invocation. Parameterised so a conda env with the Distance/dplyr/activity/yaml packages, or
# an environment module providing Rscript, can be swapped in without editing this script (same
# convention as CONTAINER/CONDA_ENV in the other job scripts).
R_CONDA_ENV="${R_CONDA_ENV:-}"
R_MODULE="${R_MODULE:-}"
RSCRIPT_BIN="${RSCRIPT_BIN:-Rscript}"

mkdir -p "$REPO_ROOT/hpc/logs/slurm" "$OUT_DIR"

cd "$REPO_ROOT"

source "$HOME/miniforge3/bin/activate"
conda activate "$CONDA_ENV"

echo "Host: $(hostname)"
echo "Start time: $(date)"
echo "DISTANCES: $DISTANCES"
echo "OUT_DIR: $OUT_DIR"

python "$REPO_ROOT/apps/camera_trap/abundance/build_ctds_inputs.py" \
  --distances "$DISTANCES" \
  --metadata "$METADATA" \
  --config "$CTDS_CONFIG" \
  --out-dir "$OUT_DIR"

conda deactivate

if [[ -n "$R_MODULE" ]]; then
  module load "$R_MODULE"
elif [[ -n "$R_CONDA_ENV" ]]; then
  source "$HOME/miniforge3/bin/activate"
  conda activate "$R_CONDA_ENV"
fi

RCMD=(
  "$RSCRIPT_BIN" "$REPO_ROOT/apps/camera_trap/abundance/ctds_abundance.R"
  --flatfile "$OUT_DIR/ctds_flatfile.csv"
  --activity "$OUT_DIR/activity_times.csv"
  --config "$CTDS_CONFIG"
  --out-dir "$OUT_DIR"
  --model "$MODEL"
)
echo "Command: ${RCMD[*]}"
"${RCMD[@]}"

echo "End time: $(date)"
