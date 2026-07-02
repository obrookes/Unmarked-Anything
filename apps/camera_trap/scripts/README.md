# Camera Trap Analysis Scripts

These utilities operate on existing camera-trap outputs (`<video>.json` and `<video>_arrays.npz`).
Scripts that read depth maps or masks from NPZ files require the pipeline to have been run with `--npz`
(or `write_npz=1` in the TSV manifest), since NPZ output is not written by default.

## Scripts

### `validate_mask_rle_roundtrip.py`

Validates transitional `both`-mode camera-trap outputs by comparing each stored raw mask
against the mask recovered from its paired COCO RLE payload. Writes per-video CSV detail
reports, a run-level summary JSON, and optional visual comparison videos.

Example:

```bash
python apps/camera_trap/scripts/validate_mask_rle_roundtrip.py \
  --run-root hpc/runs/dap3_predict_array-sam3-safari-neg-parents-da3nested-giant-large-batch-ape-fps6p0-track-1-3324174 \
  --output-dir hpc/runs/dap3_predict_array-sam3-safari-neg-parents-da3nested-giant-large-batch-ape-fps6p0-track-1-3324174/mask_rle_validation \
  --allow-missing-video
```

Notes:
- Exit code is non-zero if any raw/RLE pair fails exact equality.
- Visual comparison videos render processed frames only.
- The right-hand panel in each comparison video shows the union of all per-entry diffs for that frame.
- This script is intended for pre-cutover validation runs written with `--mask-storage-format both`;
  default pipeline outputs now use RLE-only storage.

### `plot_depth_metrics.py`

Plots depth metrics from per-video outputs.

Modes:
- `mask_means`: plot `depth_mask_mean` and `depth_mask_mean_old`
- `mask_diff`: plot `depth_mask_mean_old - depth_mask_mean`
- `id_means`: plot per-track-ID mean depth over time/frame

Behavior:
- By default, displays the plot with Matplotlib.
- Pass `--output <path>` to save an image.
- Pass `--no-show` for headless use.

Examples:

```bash
python apps/camera_trap/scripts/plot_depth_metrics.py \
  --video-json hpc/runs/demo_stream/03240068-crop-5fps/03240068-crop-5fps.json \
  --mode mask_means
```

```bash
python apps/camera_trap/scripts/plot_depth_metrics.py \
  --video-json hpc/runs/demo_stream/03240068-crop-5fps/03240068-crop-5fps.json \
  --mode id_means \
  --id-depth-source both \
  --max-ids 20 \
  --min-duration-sec 0.5 \
  --min-mean-confidence 0.25 \
  --smooth-method savgol \
  --smooth-window 7 \
  --savgol-polyorder 2 \
  --overlay-raw
```

`id_means` options:
- `--min-duration-sec`: drop IDs with short observed duration
- `--min-mean-confidence`: drop IDs with low mean confidence
- `--smooth-window`: smoothing window in observations
- `--smooth-method`: `moving_average` or `savgol`
- `--savgol-polyorder`: polynomial order for Savitzky-Golay
- `--overlay-raw`: draw raw retained series underneath smoothed series

### `merge_old_depth_into_stream_npz.py`

Merges legacy `da3_streaming` depth maps into a stream-mode `*_arrays.npz` and writes
parallel JSON fields such as `depth_mask_mean_old`.

Example:

```bash
python apps/camera_trap/scripts/merge_old_depth_into_stream_npz.py \
  --old-results-dir da3_streaming/exps/extract_images_/2026-03-04-20-15-48/results_output \
  --new-path hpc/runs/demo_stream \
  --video-name 03240068-crop-5fps
```

### `compare_depth_outputs.py`

Compares legacy `da3_streaming` depth outputs with pipeline outputs and writes quantitative
summaries, plots, and panel images.

Example:

```bash
python apps/camera_trap/scripts/compare_depth_outputs.py \
  --old-results-dir da3_streaming/exps/extract_images_/2026-03-04-20-15-48/results_output \
  --new-path hpc/runs/demo_stream \
  --video-name 03240068-crop-5fps \
  --output-dir hpc/runs/demo_stream/depth_compare_03240068
```
