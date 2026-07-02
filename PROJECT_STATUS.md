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
