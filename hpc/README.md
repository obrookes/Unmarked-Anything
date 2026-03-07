# HPC Workflow (SLURM)

This directory contains cluster-oriented job scripts and helpers for running `dap3_cli.py` at scale on GPU nodes.

## Layout

- `jobs/`: SLURM batch scripts.
  - `dap3_predict_ape.sh`: single-job template with sensible defaults.
  - `dap3_predict_array.sh`: array-job script driven by a TSV manifest.
- `configs/`: job configuration inputs.
  - `job_manifest.tsv`: one array task per non-comment row.
- `scripts/`: helper utilities for submit/monitor/result collection.
- `logs/slurm/`: SLURM `.out/.err` logs (`%x-%j` and `%x-%A_%a`).
- `runs/`: per-job output folders.

`logs/` and `runs/` are runtime artifacts and are git-ignored.

## Prerequisites

- Repository cloned on the HPC filesystem.
- Conda environment `dap-3_py3-11` available.
- CUDA module available as `cuda/12.6` (adjust in scripts if your cluster differs).
- SAM3 checkpoint present (default path in scripts: `weights/sam3/safari_checkpoint_hf.pt`).

## Single Job

Submit the included single-job script:

```bash
sbatch hpc/jobs/dap3_predict_ape.sh
```

### Overriding defaults

You can override runtime parameters at submit time via exported env vars:

```bash
sbatch --export=ALL,\
REPO_ROOT=$HOME/Unmarked-Anything,\
INPUT_VIDEO_DIR=$HOME/data/camera_trap/videos,\
SAM3_MODEL_PATH=$HOME/models/safari_checkpoint_hf.pt,\
SAM3_TEXT_PROMPTS=ape,baboon,DA3_MODEL_ID=depth-anything/DA3NESTED-GIANT-LARGE,\
DA3_MODE=stream,DA3_STREAM_CONFIG=$HOME/Unmarked-Anything/da3_streaming/configs/base_config.yaml,\
TARGET_FPS=1.0,SAM3_MODE=track,DEVICE=auto,USE_HALF=1,OVERWRITE=0,MAX_VIDEOS=,DA3_BATCH_SIZE=4,\
SAM3_TRACK_ISOLATION=recreate,SAM3_TRACK_TAIL_POLICY=warn_and_finalize,\
OUTPUT_ROOT=$HOME/Unmarked-Anything/hpc/runs,USE_SCRATCH=1 \
  hpc/jobs/dap3_predict_ape.sh
```

## Array Jobs (manifest-driven)

### 1) Edit manifest

`hpc/configs/job_manifest.tsv` is tab-separated with columns:

1. `input_video_dir`
2. `sam3_model_path`
3. `sam3_text_prompts_csv`
4. `da3_model_id`
5. `da3_mode` (optional: `batch`, `stream`; default `batch`)
6. `da3_stream_config` (optional path; used in stream mode)
7. `target_fps`
8. `sam3_mode`
9. `device`
10. `conf`
11. `use_half` (`0`/`1`)
12. `overwrite` (`0`/`1`)
13. `max_videos` (optional)
14. `da3_batch_size` (required, positive integer)
15. `sam3_track_isolation` (optional: `recreate`, `reset`, `both`; default `recreate`)
16. `sam3_track_tail_policy` (optional: `warn_and_finalize`, `fail_fast`; default `warn_and_finalize`)

Notes:
- Comment lines start with `#`.
- Blank lines are ignored.
- Prompts are comma-separated (for example `ape,baboon`).
- Relative paths are resolved from `REPO_ROOT`.
- `job_tag` is generated automatically from SAM3 model, DA3 model, prompts, fps, and mode.
- For model-based tags, only model identifiers/basenames are used (not full filesystem paths).
- `da3_batch_size` is passed through directly to the CLI and must be a positive integer.
- `da3_mode=stream` runs in-memory DA3-Streaming on all sampled frames and requires a valid `da3_stream_config`.
- `sam3_track_isolation` and `sam3_track_tail_policy` are only relevant when `sam3_mode=track`.

### 2) Submit array

```bash
hpc/scripts/submit_array.sh
```

Or provide explicit paths:

```bash
hpc/scripts/submit_array.sh hpc/jobs/dap3_predict_array.sh hpc/configs/job_manifest.tsv
```

The helper auto-counts runnable manifest rows and submits `--array=1-N`.

## Helper Scripts

### `hpc/scripts/submit_all.sh`

Submits all non-array scripts in `hpc/jobs/*.sh`.

```bash
hpc/scripts/submit_all.sh
```

### `hpc/scripts/submit_array.sh`

Submits `dap3_predict_array.sh` with size inferred from manifest.

```bash
hpc/scripts/submit_array.sh
```

### `hpc/scripts/monitor_jobs.sh`

Shows:
- `squeue -u $USER`
- tail of the 3 most recent SLURM logs in `hpc/logs/slurm`

```bash
hpc/scripts/monitor_jobs.sh
# optional
TAIL_LINES=80 hpc/scripts/monitor_jobs.sh
```

### `hpc/scripts/collect_results.sh`

Manual rsync-based collection of outputs from a source directory (for example from scratch).

```bash
hpc/scripts/collect_results.sh <from_dir> [to_dir]
```

If `<from_dir>` is omitted and `SLURM_TMPDIR` is set, it uses `SLURM_TMPDIR`.

### `hpc/scripts/clean_outputs.sh`

Removes runtime artifacts with explicit flags:
- `--logs`: delete `*.out` and `*.err` from `hpc/logs/slurm`
- `--runs`: delete all entries under `hpc/runs`
- `--all`: equivalent to `--logs --runs`

Safety options:
- `--dry-run`: show what would be deleted
- `--yes`: skip confirmation prompt

```bash
hpc/scripts/clean_outputs.sh --logs
hpc/scripts/clean_outputs.sh --runs --yes
hpc/scripts/clean_outputs.sh --all --dry-run
```

### `hpc/scripts/crop_videos.sh`

Batch-crops a fixed bottom percentage from all videos in a directory using `ffmpeg`.
Useful for removing camera overlays/timestamps before running the camera trap pipeline.

```bash
hpc/scripts/crop_videos.sh --suffix assets/videos -p 9.75
# or overwrite in place:
hpc/scripts/crop_videos.sh --overwrite assets/videos -p 9.75
```

### `hpc/scripts/summarize_track_run.py`

Summarizes per-video JSON outputs to highlight tracking anomalies (sampled frame counts, first track IDs, warnings).

```bash
python hpc/scripts/summarize_track_run.py --run-dir hpc/runs/<run_tag>
```

### Depth analysis helpers (`apps/camera_trap/scripts`)

These scripts are not HPC-specific and are shared under `apps/camera_trap/scripts`.

`merge_old_depth_into_stream_npz.py` merges legacy `da3_streaming` depth maps
(`results_output/frame_<idx>.npz`) into a stream-mode `*_arrays.npz` using keys
`<existing_depth_key>_old`, so visualization can toggle depth source.
It also computes per-frame `depth_mask_mean_old` in the per-video JSON while
preserving existing `depth_mask_mean`.

```bash
python apps/camera_trap/scripts/merge_old_depth_into_stream_npz.py \
  --old-results-dir da3_streaming/exps/extract_images_/2026-03-04-20-15-48/results_output \
  --new-path hpc/runs/demo_stream \
  --video-name 03240068-crop-5fps
```

Use with visualization:

```bash
python apps/camera_trap/cli/visualize_test_output.py \
  --output-root hpc/runs/demo_stream \
  --video-stem 03240068-crop-5fps \
  --video-dir assets/videos \
  --depth-source old
```

`compare_depth_outputs.py` computes quantitative overlap metrics and writes plots/panels:

```bash
python apps/camera_trap/scripts/compare_depth_outputs.py \
  --old-results-dir da3_streaming/exps/extract_images_/2026-03-04-20-15-48/results_output \
  --new-path hpc/runs/demo_stream \
  --video-name 03240068-crop-5fps \
  --output-dir hpc/runs/demo_stream/depth_compare_03240068
```

Plot old vs new depth-mask mean time series:

```bash
python apps/camera_trap/scripts/plot_depth_metrics.py \
  --video-json hpc/runs/demo_stream/03240068-crop-5fps/03240068-crop-5fps.json \
  --mode mask_means \
  --x-axis frame
```

Plot only the difference (`depth_mask_mean_old - depth_mask_mean`):

```bash
python apps/camera_trap/scripts/plot_depth_metrics.py \
  --video-json hpc/runs/demo_stream/03240068-crop-5fps/03240068-crop-5fps.json \
  --mode mask_diff \
  --x-axis frame
```

Plot per-track-ID mask-mean depth (`new`, `old`, or `both`):

```bash
python apps/camera_trap/scripts/plot_depth_metrics.py \
  --video-json hpc/runs/demo_stream/03240068-crop-5fps/03240068-crop-5fps.json \
  --mode id_means \
  --id-depth-source both \
  --max-ids 20 \
  --x-axis frame
```

## Output conventions

- Single job run dir: `hpc/runs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}`
- Array run dir: `hpc/runs/${SLURM_JOB_NAME}-${sam3basename}-${da3model}-${da3mode}-${prompt}-fps${fps}-${mode}-${SLURM_ARRAY_TASK_ID}-${SLURM_JOB_ID}`
- Logs:
  - single: `hpc/logs/slurm/%x-%j.out|err`
  - array: `hpc/logs/slurm/%x-%A_%a.out|err`

## Scratch mode

Both job scripts support optional scratch execution:

- `USE_SCRATCH=1` and `SLURM_TMPDIR` set:
  - run writes into scratch first,
  - then `rsync` copies to final output dir.

Set `USE_SCRATCH=0` to write directly to final output path.

## Common checks

- Missing module: update `module load cuda/12.6` for your site.
- Wrong repo path: override `REPO_ROOT` at submit time.
- Wrong checkpoint path: set `SAM3_MODEL_PATH` (single job) or manifest field (array job).
- No jobs in array: confirm manifest has non-comment, non-empty rows.
