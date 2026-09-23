# Unmarked Anything

## Segment and Depth Anything 3 for Camera Trap Distance Sampling

This repository provides a workflow for camera trap–based distance sampling that integrates segmentation and monocular depth estimation to localise, track, and estimate animal-to-camera distances for a specified target species. The framework combines the wildlife-adapted [SAM-3 (SA-FARI)](https://arxiv.org/abs/2511.15622) with [Depth Anything 3](https://depth-anything-3.github.io/assets/da3_tech_report_2025.pdf) to generate frame-level detections, segmentation masks, object tracks, and depth-derived distance signals from camera trap video.

## Core Technologies

- **SAM-3 (SA-FARI)** for wildlife-specific segmentation/tracking: [https://arxiv.org/abs/2511.15622](https://arxiv.org/abs/2511.15622)
- **Depth Anything 3** for monocular depth estimation: [https://depth-anything-3.github.io/assets/da3_tech_report_2025.pdf](https://depth-anything-3.github.io/assets/da3_tech_report_2025.pdf)

## Installation

### Conda (recommended)

```bash
conda create -n dap-3_py3-11 "python>=3.7,<3.11"
conda activate dap-3_py3-11
pip install torch>=2 torchvision --index-url https://download.pytorch.org/whl/cu126
pip install xformers
git clone https://github.com/obrookes/Unmarked-Anything
cd Unmarked-Anything
pip install -e .
pip install gsplat
pip install ultralytics
```

For AArch/ARM HPC environments where `pycolmap` is unavailable, keep `pycolmap` disabled in [`pyproject.toml`](./pyproject.toml) (currently commented out).
For the same environment, `pycolmap` import is also commented in [`src/depth_anything_3/utils/export/colmap.py`](./src/depth_anything_3/utils/export/colmap.py), so COLMAP export paths that depend on `pycolmap` are disabled.

### Singularity Container

Container assets:
- Definition: `envs/containers/depth-anything-3.def`
- Local image: `envs/containers/dap3-ultralytics.sif` (git-ignored)

Build image (if needed):

```bash
singularity build envs/containers/dap3-ultralytics.sif envs/containers/depth-anything-3.def
```

Run CLI in container:

```bash
singularity exec --nv envs/containers/dap3-ultralytics.sif \
  python apps/camera_trap/cli/dap3_cli.py \
  --input-video-dir assets/videos \
  --output-dir outputs/demo \
  --sam3-model-path weights/sam3/safari_checkpoint_hf.pt \
  --sam3-text-prompts animal
```

Use `--nv` when you need GPU support in the container.

## Model Files

Store large local model files here (all git-ignored):
- `weights/sam3/` for SAM3 checkpoints
- `weights/dap3/` for DA3 checkpoints/artifacts

Current SAM3 checkpoint example:
- `weights/sam3/safari_checkpoint_hf.pt`

## Quick Start

Canonical CLI:

```bash
python apps/camera_trap/cli/dap3_cli.py \
  --input-video-dir assets/videos \
  --output-dir outputs/demo \
  --sam3-model-path weights/sam3/safari_checkpoint_hf.pt \
  --sam3-text-prompts animal \
  --da3-mode batch \
  --da3-batch-size 4
```

Default mask persistence now writes COCO RLE only. Use `--mask-storage-format both` when you need
paired raw+RLE outputs for transitional validation or storage comparisons.

Default `--sam3-backend official` runs the facebookresearch/sam3 package (SA-FARI checkpoints,
`--sam3-mode track` only). Pass `--sam3-backend ultralytics` to use the ultralytics SAM3 port
instead (needed for `--sam3-mode frame`, or for comparison runs).

Legacy wrapper also works: `python dap3_cli.py ...`

Streaming mode example (DA3-Streaming on all sampled frames):

```bash
python apps/camera_trap/cli/dap3_cli.py \
  --input-video-dir assets/videos \
  --output-dir outputs/demo_stream \
  --sam3-model-path weights/sam3/safari_checkpoint_hf.pt \
  --sam3-text-prompts animal \
  --da3-mode stream \
  --da3-stream-config da3_streaming/configs/base_config.yaml \
  --da3-batch-size 4
```

All-frames mode example (standard DA3 on all sampled frames; persist depth only for SAM-positive frames):

```bash
python apps/camera_trap/cli/dap3_cli.py \
  --input-video-dir assets/videos \
  --output-dir outputs/demo_all_frames \
  --sam3-model-path weights/sam3/safari_checkpoint_hf.pt \
  --sam3-text-prompts animal \
  --da3-mode all_frames \
  --da3-batch-size 4
```

## Sample Data

- Input video: `assets/videos/demo.MP4`
- Suggested output directory: `outputs/demo/`
- Visualization CLI: `python apps/camera_trap/cli/visualize_test_output.py ...`
- Visualization notebook: [`notebooks/camera_trap/visualize_test_output.ipynb`](./notebooks/camera_trap/visualize_test_output.ipynb)
- Pass `--video-dir` for stem-based auto-resolution and selection from `run_manifest.json`.
- Use `--video-path` only for direct single-video mode.
- Use `--depth-source new|old` to choose which depth maps are rendered (`old` expects merged `*_old` keys in NPZ).

Preprocessing helpers:
- `hpc/scripts/crop_videos.sh` crops a fixed bottom percentage from all videos in a directory (requires `ffmpeg`).
- `hpc/scripts/reduce_video_fps.sh` creates reduced-FPS duplicates for all videos in a directory.

Basic interactive example:

```bash
python apps/camera_trap/cli/visualize_test_output.py \
  --output-root outputs/demo \
  --video-stem demo \
  --video-dir assets/videos \
  --view both \
  --page-size 2
```

Overlay export example:

```bash
python apps/camera_trap/cli/visualize_test_output.py \
  --output-root outputs/demo \
  --video-stem demo \
  --video-dir assets/videos \
  --no-gui \
  --write-video outputs/demo/demo_overlay.mp4 \
  --export-style rgb
```

Analysis-depth overlay export:

```bash
python apps/camera_trap/cli/visualize_test_output.py \
  --output-root outputs/demo \
  --video-stem demo \
  --video-dir assets/videos \
  --no-gui \
  --write-video outputs/demo/demo_analysis_overlay.mp4 \
  --export-style analysis-depth
```

Batch export-all example:

```bash
python apps/camera_trap/cli/visualize_test_output.py \
  --output-root outputs/demo \
  --video-dir assets/videos \
  --no-gui \
  --export-all \
  --write-video-dir outputs/demo/overlay_videos \
  --export-style analysis-depth
```

Batch naming:
- `--export-style rgb` writes `<stem>_overlay.mp4`
- `--export-style analysis-depth` writes `<stem>_analysis_overlay.mp4`

Depth-analysis helpers are documented separately in [`apps/camera_trap/scripts/README.md`](./apps/camera_trap/scripts/README.md).

## HPC Execution

- For SLURM/HPC job structure, batch scripts, manifests, and helper utilities, see: [`hpc/README.md`](./hpc/README.md)

## PSS P3 end-to-end pipeline

The PSS P3 chimpanzee camera-trap distance-sampling analysis estimates each detection's distance
from the camera using the method of Haucke, Kühl, Hoyer & Steinhage (2022), "Overcoming the
distance estimation bottleneck in estimating animal abundance with camera traps," *Ecological
Informatics*, 68, 101536 — a monocular-depth-based disparity calibration, ported into this repo
from the [`timmh/distance-estimation`](https://github.com/timmh/distance-estimation) reference
implementation. (This replaces an earlier linear per-camera fit driven by VLM-read distance-board
numbers; that approach — `read_boards.py` → `calibration/fit.py` → `calibration.json` — has been
removed.)

The pipeline chains the following stages, all runnable on Isambard AI as SLURM jobs (see
`hpc/scripts/run_full_pipeline.sh`, the source of truth for stage names/dependencies/env vars):

1. **A/A'. Find sign frames** — `apps/camera_trap/calibration/find_sign_frames.py`
   (`hpc/jobs/find_sign_frames.sh`, a sharded array job) — a VLM agent scans the raw reference
   videos to find frames where a person holds a distance sign, producing `calibration_frames.csv`
   (plus `events_summary.csv` and `contact_sheet.html`, a QC artefact worth eyeballing). Stage A'
   is a build-only merge of the shards into the final CSV.
2. **B. Build reference** — `apps/camera_trap/calibration/build_reference.py`
   (`hpc/jobs/build_reference.sh`, GPU, runs in the `dap3-sam3` apptainer container) — runs a SAM3
   person-mask segmenter and DA3 depth on the calibration frames, producing per-camera
   `calib/<transect_cam>.npz`, a pooled `calib/_pooled.npz`, and `calibration_summary.csv` (check
   its `loo_mae_m` column — leave-one-out mean absolute error in metres — as the key calibration
   QC number), plus `instances/` overlay images and `calibration_plots/` (more QC artefacts).
3. **C. Main dap3 run** — `dap3_cli.py` on the actual chimp videos with `--npz` and
   `--depth-interval-seconds` (default 2s — DA3 depth is now computed on a time grid rather than
   every frame, for speed; NPZ depth is stored float16 at native resolution). Runs in parallel
   with stages 1–2.
4. **D. Mask QC** — `apps/camera_trap/qc/mask_verify.py` (VLM), sharded, on the main run's output
   → `track_qc.csv`. Depends on C.
5. **E1/E1'. Apply calibration** — `apps/camera_trap/calibration/apply.py`
   (`hpc/jobs/apply_calibration.sh`, sharded array job + merge) — depends on B and C. Aligns each
   frame's disparity onto the camera's reference anchor via RANSAC, then applies the per-camera
   piecewise-linear calibration curve, producing `calibrated_objects.csv` (per-detection distance,
   `calib_method`, alignment inlier fraction, etc.) and `apply_summary.json` (a QC artefact —
   counts per `calib_method`/camera and failures by reason). Optionally
   `--extrinsic-recalibration --lightglue-weights ... --video-dir ...` for cameras that moved
   between the reference and main runs.
6. **E2. Export** — `apps/camera_trap/scripts/export_job_distances_csv.py --calibrated-objects
   calibrated_objects.csv --track-qc track_qc.csv` → a per-detection `distances.csv`. Depends on D
   and E1'.
7. **F. Abundance** — `apps/camera_trap/abundance/build_ctds_inputs.py` (builds `Distance`-package
   flatfile + activity inputs) → `apps/camera_trap/abundance/ctds_abundance.R` (fits the
   detection function + activity model, produces density/abundance estimates).

### Running it

```bash
cp hpc/configs/pipeline.env.example hpc/configs/my_pipeline.env
# edit my_pipeline.env: REPO_ROOT, VIDEO_DIR, REF_VIDEO_DIR, REFERENCE_ROOT, OUT_ROOT, SAM3_CKPT,
# container/env paths, VLM model, Slurm partition/account, etc.

# see the sbatch commands without submitting anything:
DRY_RUN=1 PIPELINE_ENV=hpc/configs/my_pipeline.env hpc/scripts/run_full_pipeline.sh

# submit the full chain (dependency-chained sbatch jobs, prints job ids):
PIPELINE_ENV=hpc/configs/my_pipeline.env hpc/scripts/run_full_pipeline.sh
```

Each stage's underlying CLI resumes/skips existing outputs by default, so re-running the script
after a partial failure is safe. Individual stages can be skipped with `SKIP_SIGNS=1`,
`SKIP_REFCAL=1`, `SKIP_MAIN=1`, `SKIP_QC=1`, `SKIP_APPLY=1`, `SKIP_EXPORT=1`, `SKIP_ABUNDANCE=1` —
when skipping a stage whose output a later stage needs, point the pipeline at the existing output
via `CALIBRATION_FRAMES_CSV`, `CALIB_DIR`, `MAIN_JOB_DIR`, `TRACK_QC_CSV`,
`CALIBRATED_OBJECTS_CSV`, or `DISTANCES_CSV`. See `hpc/scripts/run_full_pipeline.sh` and
`hpc/configs/pipeline.env.example` for the full variable list, and `hpc/README.md` for job-script
and container details (including the new `envs/containers/dap3-sam3.def`, needed for
`--sam3-backend official`).

Outputs land under `OUT_ROOT`: `sign_frames/calibration_frames.csv`, `calibration/calib/`
(per-camera `.npz` + `calibration_summary.csv`), `main/` (dap3 runs, with
`main/calib_applied/calibrated_objects.csv` + `apply_summary.json` and `main/qc/track_qc.csv`),
`distances.csv`, and `abundance/` (CTDS flatfile, activity inputs, and
`abundance_estimates.csv`).

## Full Command Template

```bash
python apps/camera_trap/cli/dap3_cli.py \
  --input-video-dir /data/camera_trap/videos \
  --output-dir /data/camera_trap/results \
  --sam3-model-path weights/sam3/safari_checkpoint_hf.pt \
  --sam3-text-prompts animal deer boar \
  --da3-model-id depth-anything/DA3NESTED-GIANT-LARGE \
  --da3-mode batch \
  --da3-batch-size 4 \
  --target-fps 1.0 \
  --sam3-mode track \
  --conf 0.25 \
  --device auto \
  --mask-storage-format rle
```

## CLI Options (All)

| Option | Type | Required | Default | Description |
|---|---|---|---|---|
| `--input-video-dir` | path | yes | - | Directory containing videos to process. |
| `--output-dir` | path | yes | - | Output root for run manifest and per-video results. |
| `--video-exts` | string | no | `.mp4,.mov,.avi,.mkv` | Comma-separated extensions (with or without leading `.`). |
| `--sam3-model-path` | path | yes | - | Path to SAM3 checkpoint (`.pt`). |
| `--sam3-text-prompts` | list of strings | yes | - | One or more global SAM3 text prompts. |
| `--da3-model-id` | string | no | `depth-anything/DA3NESTED-GIANT-LARGE` | DA3 pretrained model ID. |
| `--da3-mode` | enum | no | `batch` | DA3 execution mode: `batch` (batched DA3 on SAM-positive frames), `stream` (DA3-Streaming on all sampled frames), or `all_frames` (standard DA3 on all sampled frames). |
| `--da3-stream-config` | path | no | `da3_streaming/configs/base_config.yaml` | DA3-Streaming YAML config path (used when `--da3-mode stream`). |
| `--da3-batch-size` | int | yes | - | Positive integer required by CLI. In `batch` mode it is the DA3 batch size; in `stream` and `all_frames` modes it is retained for compatibility/progress accounting. |
| `--target-fps` | float | no | `1.0` | Sampling rate for processing. Must be `> 0`. |
| `--sam3-mode` | enum | no | `track` | `track` (video tracking) or `frame` (per-frame segmentation). |
| `--conf` | float | no | `0.25` | SAM3 confidence threshold (final filter, applied identically to both backends' output). |
| `--sam3-backend` | enum | no | `official` | `official` (facebookresearch/sam3, SA-FARI checkpoints; `--sam3-mode track` only) or `ultralytics` (ultralytics SAM3 port; `track` or `frame`). |
| `--sam3-det-threshold` | float | no | `0.5` | Detection/presence score threshold passed to the official backend's internal `propagate_in_video` call. Ignored by `--sam3-backend ultralytics`. |
| `--device` | enum | no | `auto` | `auto`, `cuda`, or `cpu`. |
| `--half` | flag | no | `false` | Enable FP16 for SAM3 (CUDA only). |
| `--mask-storage-format` | enum | no | `rle` | Mask persistence mode: `raw`, `rle`, or `both`. Use `both` only when you need paired raw+RLE artifacts for validation. |
| `--sam3-track-isolation` | enum | no | `recreate` | Track-mode predictor isolation per video: `recreate`, `reset`, or `both`. |
| `--sam3-track-tail-policy` | enum | no | `warn_and_finalize` | `warn_and_finalize` keeps partial output on SAM3 stream `IndexError`; `fail_fast` marks the video failed. |
| `--overwrite` | flag | no | `false` | Reprocess even if output JSON + NPZ already exist. |
| `--max-videos` | int | no | `None` | Cap number of videos after sorting. Must be `> 0` if provided. |

## Output Structure

Given `--output-dir ./out`, outputs are:
- `./out/run_manifest.json`
- `./out/<video_stem>/<video_stem>.json`
- `./out/<video_stem>/<video_stem>_arrays.npz`

Per-frame status values include: `processed`, `empty_mask`, `sam_error`, `da3_error`, `frame_decode_error`.

## Notes

- `target_fps` is implemented as frame stride (`round(video_fps / target_fps)`, minimum 1).
- Non-overwrite mode skips videos that already have both expected output files.
- `--da3-mode stream` runs DA3-Streaming in memory on all sampled frames; SAM outputs still control which frames are persisted as `processed` in JSON/NPZ.
- `--da3-mode all_frames` runs standard DA3 in memory on all sampled frames; persisted depth storage remains aligned to `processed` (SAM-positive) frame rows.
- If `--sam3-mode track` is unavailable in your ultralytics build, use `--sam3-mode frame` with `--sam3-backend ultralytics` (the default `official` backend supports `track` only).
- Default `--sam3-track-isolation recreate` prevents cross-video tracker state leakage in multi-video runs.
- Use `hpc/scripts/summarize_track_run.py --run-dir <run>` for a quick forensic summary of per-video frame counts and track IDs.

## Additional References

- Camera trap CLI source: [`apps/camera_trap/cli/dap3_cli.py`](./apps/camera_trap/cli/dap3_cli.py)
- Archive index: [`archive/INDEX.md`](./archive/INDEX.md)
- Archived upstream README: [`archive/2026-03/readmes/README.upstream.md`](./archive/2026-03/readmes/README.upstream.md)
