#!/bin/bash
#SBATCH --job-name=dap3_export_overlays
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=hpc/logs/slurm/%x-%j.out
#SBATCH --error=hpc/logs/slurm/%x-%j.err

set -euo pipefail

if [[ -z "${PREDICT_ARRAY_JOB_ID:-}" ]]; then
  echo "PREDICT_ARRAY_JOB_ID is required." >&2
  exit 1
fi

REPO_ROOT="${REPO_ROOT:-$HOME/Unmarked-Anything}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/hpc/runs}"
EXPORT_STYLE="${EXPORT_STYLE:-rgb}"
DEPTH_SOURCE="${DEPTH_SOURCE:-new}"
EXPORT_SUBDIR="${EXPORT_SUBDIR:-overlay_videos}"
VIDEO_DIR_OVERRIDE="${VIDEO_DIR_OVERRIDE:-}"
EXPORT_FPS="${EXPORT_FPS:-}"
EXPORT_FOURCC="${EXPORT_FOURCC:-mp4v}"

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

mkdir -p "$REPO_ROOT/hpc/logs/slurm" "$OUTPUT_ROOT"
cd "$REPO_ROOT"

module purge
module load cuda/12.6

source "$HOME/miniforge3/bin/activate"
conda activate dap3-stream

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

mapfile -t RUN_DIRS < <(find "$OUTPUT_ROOT" -mindepth 1 -maxdepth 1 -type d -name "*-${PREDICT_ARRAY_JOB_ID}" | sort)
if (( ${#RUN_DIRS[@]} == 0 )); then
  echo "No run directories matched '*-${PREDICT_ARRAY_JOB_ID}' under $OUTPUT_ROOT" >&2
  exit 1
fi

SUMMARY_TSV="$(mktemp)"
trap 'rm -f "$SUMMARY_TSV"' EXIT

echo "Host: $(hostname)"
echo "Start time: $(date)"
echo "Predict array job id: ${PREDICT_ARRAY_JOB_ID}"
echo "Matched run directories: ${#RUN_DIRS[@]}"

success_count=0
fail_count=0

for run_dir in "${RUN_DIRS[@]}"; do
  manifest_path="$run_dir/run_manifest.json"
  if [[ ! -f "$manifest_path" ]]; then
    msg="run_manifest.json not found"
    echo "Skipping $run_dir: $msg" >&2
    printf "%s\t%s\t%s\t%s\n" "$run_dir" "failed" "$msg" "" >> "$SUMMARY_TSV"
    fail_count=$((fail_count + 1))
    continue
  fi

  resolved_video_dir="$(python - "$manifest_path" "$VIDEO_DIR_OVERRIDE" <<'PY'
import json
import pathlib
import sys

manifest_path = pathlib.Path(sys.argv[1])
override = sys.argv[2].strip()
if override:
    print(override)
    raise SystemExit(0)

payload = json.loads(manifest_path.read_text(encoding="utf-8"))
input_video_dir = payload.get("input_video_dir")
if isinstance(input_video_dir, str) and input_video_dir.strip():
    print(input_video_dir.strip())
    raise SystemExit(0)

videos = payload.get("videos") or []
for row in videos:
    video_path = row.get("video_path")
    if isinstance(video_path, str) and video_path.strip():
        print(str(pathlib.Path(video_path).parent))
        raise SystemExit(0)

raise SystemExit(2)
PY
)"
  if [[ -z "$resolved_video_dir" ]]; then
    msg="failed to resolve video directory"
    echo "Skipping $run_dir: $msg" >&2
    printf "%s\t%s\t%s\t%s\n" "$run_dir" "failed" "$msg" "" >> "$SUMMARY_TSV"
    fail_count=$((fail_count + 1))
    continue
  fi
  if [[ "$resolved_video_dir" != /* ]]; then
    resolved_video_dir="$REPO_ROOT/$resolved_video_dir"
  fi

  write_dir="$run_dir/$EXPORT_SUBDIR"
  cmd=(
    python "$REPO_ROOT/apps/camera_trap/cli/visualize_test_output.py"
    --output-root "$run_dir"
    --video-dir "$resolved_video_dir"
    --no-gui
    --export-all
    --write-video-dir "$write_dir"
    --export-style "$EXPORT_STYLE"
    --depth-source "$DEPTH_SOURCE"
    --export-fourcc "$EXPORT_FOURCC"
  )
  if [[ -n "$EXPORT_FPS" ]]; then
    cmd+=(--export-fps "$EXPORT_FPS")
  fi

  echo
  echo "=== Exporting run: $run_dir ==="
  echo "Video dir: $resolved_video_dir"
  echo "Command: ${cmd[*]}"

  if "${cmd[@]}"; then
    printf "%s\t%s\t%s\t%s\n" "$run_dir" "success" "" "$write_dir" >> "$SUMMARY_TSV"
    success_count=$((success_count + 1))
  else
    msg="visualize_test_output.py failed"
    printf "%s\t%s\t%s\t%s\n" "$run_dir" "failed" "$msg" "$write_dir" >> "$SUMMARY_TSV"
    fail_count=$((fail_count + 1))
  fi
done

summary_path="$OUTPUT_ROOT/export-summary-${PREDICT_ARRAY_JOB_ID}.json"
python - "$SUMMARY_TSV" "$summary_path" "$PREDICT_ARRAY_JOB_ID" "$OUTPUT_ROOT" "$EXPORT_STYLE" "$DEPTH_SOURCE" "$EXPORT_SUBDIR" <<'PY'
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

tsv_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
predict_job_id = sys.argv[3]
output_root = sys.argv[4]
export_style = sys.argv[5]
depth_source = sys.argv[6]
export_subdir = sys.argv[7]

runs = []
for line in tsv_path.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    run_dir, status, message, write_dir = line.split("\t")
    runs.append(
        {
            "run_dir": run_dir,
            "status": status,
            "message": message or None,
            "write_video_dir": write_dir or None,
        }
    )

payload = {
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "predict_array_job_id": predict_job_id,
    "output_root": output_root,
    "export_style": export_style,
    "depth_source": depth_source,
    "export_subdir": export_subdir,
    "total_runs": len(runs),
    "successful_runs": sum(1 for r in runs if r["status"] == "success"),
    "failed_runs": sum(1 for r in runs if r["status"] != "success"),
    "runs": runs,
}

summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(f"Wrote export summary: {summary_path}")
PY

echo "End time: $(date)"
echo "Export runs: success=${success_count} failed=${fail_count} total=$((success_count + fail_count))"
if (( fail_count > 0 )); then
  exit 1
fi
