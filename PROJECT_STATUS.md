# Project Status — 2026-09-23

Branch `feat/timmh-calibration` reworks PSS P3 distance calibration from the earlier linear
per-camera fit onto the Haucke et al. (2022, *Ecological Informatics* 68:101536) disparity-based
method, ported from `timmh/distance-estimation`:

- New calibration modules: `apps/camera_trap/calibration/find_sign_frames.py` (VLM agent scans
  reference videos for person-holding-a-sign frames, array job `hpc/jobs/find_sign_frames.sh`,
  producing `calibration_frames.csv`), `apps/camera_trap/calibration/build_reference.py` (SAM3
  person mask + DA3 depth per-camera fit, job `hpc/jobs/build_reference.sh`, producing
  `calib/<transect_cam>.npz` + `calibration_summary.csv` with a `loo_mae_m` QC column), and
  `apps/camera_trap/calibration/apply.py` (RANSAC disparity alignment to each camera's reference
  anchor + piecewise-linear calibration, array job `hpc/jobs/apply_calibration.sh`, producing
  `calibrated_objects.csv` + `apply_summary.json`).
- `dap3_cli.py` gained `--depth-interval-seconds` (default 2s): DA3 depth is now computed on a
  time grid rather than every frame, for speed; NPZ depth is stored float16 at native resolution.
- `export_job_distances_csv.py` now takes `--calibrated-objects calibrated_objects.csv
  --track-qc track_qc.csv`, replacing the old `--calibration calibration.json`.
- `hpc/scripts/run_full_pipeline.sh` rewritten around the new stage graph (find_sign_frames →
  build_reference → apply_calibration, in parallel with the main dap3 run and mask QC → export →
  abundance), with matching `SKIP_*`/resume env vars documented in
  `hpc/configs/pipeline.env.example`.
- `apps/camera_trap/calibration/fit.py` and its test have been deleted — superseded by
  `build_reference.py`/`apply.py`.

---

# Project Status — 2026-09-22

Branch `feat/sam3-calib-ctds` (off `dev`) adds the full PSS P3 calibration/CTDS pipeline:

- Pluggable SAM3 backends (`--sam3-backend official|ultralytics`, default `official`) using the
  official `facebookresearch/sam3` package alongside the existing ultralytics port.
- Per-camera linear depth calibration (VLM distance-board reading from reference videos +
  `calibration/fit.py`) and mask-QC (VLM track review, `qc/mask_verify.py`) feeding
  `export_job_distances_csv.py`.
- In-pipeline CTDS density/abundance estimation (`abundance/build_ctds_inputs.py` +
  `abundance/ctds_abundance.R`, porting `wcf-pps-p3/scripts/PSS_P3_Chimp_CTDS_all.R`).
- `envs/containers/dap3-sam3.def`: a new Apptainer image (NGC PyTorch 25.06 base, Python 3.12/
  torch 2.7.x) for the official SAM3 backend, whose Python/torch floor
  `envs/containers/depth-anything-3.def` (python3.10/torch 2.6.0) cannot satisfy; that def's sam3
  install was reverted accordingly (comment points at the new file).
- `hpc/configs/job_manifest.tsv` / `hpc/jobs/dap3_predict_array.sh` gained `sam3_backend` /
  `sam3_det_threshold` manifest columns (default `official`/`0.5`, backward compatible) and
  container/conda-env selection for the official backend.
- `hpc/scripts/run_full_pipeline.sh`: one entry point that submits the whole six-stage chain
  (reference dap3 → calibration → main dap3 → mask QC → export → abundance) as
  dependency-chained (`sbatch --dependency=afterok:`) jobs, with per-stage `SKIP_*` flags and a
  `DRY_RUN=1` mode. Shared config lives in `hpc/configs/pipeline.env.example`. New CPU job
  scripts: `hpc/jobs/export_distances.sh`, `hpc/jobs/ctds_abundance.sh`.
- Fixed a stale doc note: `export_job_distances_csv.py` is JSON-only and does not need `--npz`
  output (NPZ is needed by mask QC / calibration board reading instead).
- Not verified on the cluster (compute nodes were offline for this branch): the `dap3-sam3.def`
  container build, `--sam3-backend official` end-to-end, and the live `run_full_pipeline.sh`
  submission (dry-run only, locally).

Earlier snapshot below.

---

# Project Status — 2026-07-02

Consolidation snapshot: all active development work has been merged onto the
`dev` branch and the feature worktrees have been removed.

## What was merged

- **feat/export-distances** — Distance CSV exporter overhaul:
  - NPZ output is now opt-in via `--npz` (JSON-only depth output is the default
    export path; `write_npz=1` is forced only for the export workflow rows in
    the HPC manifest).
  - `export_job_distances_csv.py` rewritten to read depth from the per-video
    JSON (`depth_mask_mean`) instead of NPZ arrays.
  - Sample points are aligned to a global grid (0, interval, 2×interval, …)
    rather than per-object first appearance.
  - Bbox in-frame filter: detections touching the left/right frame edge are
    always excluded from distance windows.
  - Quality-control filters (all disableable via `--no-filter`):
    min track frames (default 5), static-track displacement (default 5 px),
    min confidence (default 0.5), min bbox area fraction (default 1% of
    frame), and IoU-based duplicate-track dedup (default 0.5, shorter track
    dropped). Missing or negative depth values are always skipped.

- **feat/depth-compression** — Depth-consistency fixes:
  - `depth_mask_mean` is computed per object/mask (was previously shared).
  - Depth averaging is now consistent between JSON output and video overlay.
  - BGR→RGB fix for frames passed to DA3.
  - HPC job runtime increased; a later "reinstating hpc job changes" commit
    was reverted (branch tip is that revert).

## What was removed

- **agent-refactor** worktree and branch. It had no commits beyond `dev`; its
  uncommitted work-in-progress (a `PipelineConfig` dataclass refactor of
  `dap3_cli.py`, ~378 changed lines) is preserved in the git stash:
  `git stash list` → "agent-refactor WIP: PipelineConfig dataclass refactor".
  Apply with `git stash apply` if the refactor is picked up again.

## Current state

- `dev` is the single development branch; `main` lags behind it
  (last merged at the visualiser/notebook-helpers stage).
- Mask storage defaults to COCO RLE (PR #6).
- Test suite: 24 passing (`pytest tests/`, run with the `dap-3_py3-11`
  conda env).
- Untracked local artifacts in the repo root (test CSVs, `estimate_activity.R`,
  frame-extraction assets) are working data and are not committed.

## Likely next steps

- Merge `dev` into `main` / open a PR once the exporter QC filters are
  validated on real job output.
- Decide whether to resume the `PipelineConfig` refactor from the stash.
- Clean up or `.gitignore` the untracked analysis artifacts in the repo root.
