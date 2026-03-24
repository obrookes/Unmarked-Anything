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
- Visualization notebook: [`notebooks/camera_trap/visualize_test_output.ipynb`](./notebooks/camera_trap/visualize_test_output.ipynb) (kept for ad-hoc exploration; CLI is the repeatable/default path)
- Pass `--video-dir` for stem-based auto-resolution and selection from `run_manifest.json`.
- Use `--video-path` only for direct single-video mode.
- Use `--depth-source new|old` to choose which depth maps are rendered (`old` expects merged `*_old` keys in NPZ).

Video preprocessing helper:
- `hpc/scripts/crop_videos.sh` crops a fixed bottom percentage from all videos in a directory (requires `ffmpeg`).
- Example: `hpc/scripts/crop_videos.sh --suffix assets/videos -p 9.75`
- `hpc/scripts/reduce_video_fps.sh` creates reduced-FPS duplicates for all videos in a directory.
- Example: `hpc/scripts/reduce_video_fps.sh assets/videos --fps 5`

Interactive example:

```bash
python apps/camera_trap/cli/visualize_test_output.py \
  --output-root outputs/demo \
  --video-stem demo \
  --video-dir assets/videos \
  --view both \
  --page-size 2
```

Interactive selection example (multiple videos in one run root):

```bash
python apps/camera_trap/cli/visualize_test_output.py \
  --output-root outputs/demo \
  --video-dir assets/videos \
  --view processed
```

List videos and exit:

```bash
python apps/camera_trap/cli/visualize_test_output.py \
  --output-root outputs/demo \
  --video-dir assets/videos \
  --list-videos
```

Full-timeline overlay video export (single selected video):

```bash
python apps/camera_trap/cli/visualize_test_output.py \
  --output-root outputs/demo \
  --video-stem demo \
  --video-dir assets/videos \
  --view processed \
  --write-video outputs/demo/demo_overlay.mp4 \
  --ov-mask --ov-bbox --ov-center --ov-hud
```

Analysis-style depth overlay export (no histogram panel; outline-only masks; per-object mean mask-depth labels; full-height external scale bar):

```bash
python apps/camera_trap/cli/visualize_test_output.py \
  --output-root outputs/demo \
  --video-stem demo \
  --video-dir assets/videos \
  --no-gui \
  --write-video outputs/demo/demo_analysis_overlay.mp4 \
  --export-style analysis-depth
```

Headless export-only example (no GUI windows):

```bash
python apps/camera_trap/cli/visualize_test_output.py \
  --output-root outputs/demo \
  --video-stem demo \
  --video-dir assets/videos \
  --no-gui \
  --write-video outputs/demo/demo_overlay.mp4
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

Depth analysis helpers now live in `apps/camera_trap/scripts` (not `hpc/scripts`) since they are reusable locally and on HPC.

Merge legacy depth maps into a stream NPZ for side-by-side visualization source switching:

```bash
python apps/camera_trap/scripts/merge_old_depth_into_stream_npz.py \
  --old-results-dir da3_streaming/exps/extract_images_/2026-03-04-20-15-48/results_output \
  --new-path hpc/runs/demo_stream \
  --video-name 03240068-crop-5fps
```

This merge also writes per-frame `depth_mask_mean_old` in the per-video JSON, while preserving
existing `depth_mask_mean`.

Then visualize old maps via:

```bash
python apps/camera_trap/cli/visualize_test_output.py \
  --output-root hpc/runs/demo_stream \
  --video-stem 03240068-crop-5fps \
  --video-dir assets/videos \
  --depth-source old \
  --view both
```

Plot `depth_mask_mean` (new) vs `depth_mask_mean_old` (merged old):

```bash
python apps/camera_trap/scripts/plot_depth_metrics.py \
  --video-json hpc/runs/demo_stream/03240068-crop-5fps/03240068-crop-5fps.json \
  --mode mask_means \
  --x-axis frame
```

Plot the mean-depth difference (`depth_mask_mean_old - depth_mask_mean`):

```bash
python apps/camera_trap/scripts/plot_depth_metrics.py \
  --video-json hpc/runs/demo_stream/03240068-crop-5fps/03240068-crop-5fps.json \
  --mode mask_diff \
  --x-axis frame
```

Plot per-track-ID mean depth (from object masks) using new and old merged depth maps:

```bash
python apps/camera_trap/scripts/plot_depth_metrics.py \
  --video-json hpc/runs/demo_stream/03240068-crop-5fps/03240068-crop-5fps.json \
  --mode id_means \
  --id-depth-source both \
  --max-ids 20 \
  --x-axis frame
```

## HPC Execution

- For SLURM/HPC job structure, batch scripts, manifests, and helper utilities, see: [`hpc/README.md`](./hpc/README.md)

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
  --device auto
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
| `--conf` | float | no | `0.25` | SAM3 confidence threshold. |
| `--device` | enum | no | `auto` | `auto`, `cuda`, or `cpu`. |
| `--half` | flag | no | `false` | Enable FP16 for SAM3 (CUDA only). |
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
- If `--sam3-mode track` is unavailable in your ultralytics build, use `--sam3-mode frame`.
- Default `--sam3-track-isolation recreate` prevents cross-video tracker state leakage in multi-video runs.
- Use `hpc/scripts/summarize_track_run.py --run-dir <run>` for a quick forensic summary of per-video frame counts and track IDs.

## Additional References

- Camera trap CLI source: [`apps/camera_trap/cli/dap3_cli.py`](./apps/camera_trap/cli/dap3_cli.py)
- Archive index: [`archive/INDEX.md`](./archive/INDEX.md)
- Archived upstream README: [`archive/2026-03/readmes/README.upstream.md`](./archive/2026-03/readmes/README.upstream.md)
