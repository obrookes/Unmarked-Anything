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
SAM3_TEXT_PROMPTS=ape,baboon,\
TARGET_FPS=1.0,SAM3_MODE=track,DEVICE=auto,USE_HALF=1,OVERWRITE=0,MAX_VIDEOS=,\
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
5. `target_fps`
6. `sam3_mode`
7. `device`
8. `conf`
9. `use_half` (`0`/`1`)
10. `overwrite` (`0`/`1`)
11. `max_videos` (optional)

Notes:
- Comment lines start with `#`.
- Blank lines are ignored.
- Prompts are comma-separated (for example `ape,baboon`).
- Relative paths are resolved from `REPO_ROOT`.
- `job_tag` is generated automatically from SAM3 model, DA3 model, prompts, fps, and mode.
- For model-based tags, only model identifiers/basenames are used (not full filesystem paths).

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

## Output conventions

- Single job run dir: `hpc/runs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}`
- Array run dir: `hpc/runs/${SLURM_JOB_NAME}-${sam3basename}-${da3model}-${prompt}-fps${fps}-${mode}-${SLURM_ARRAY_TASK_ID}-${SLURM_JOB_ID}`
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
