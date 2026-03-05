#!/usr/bin/env python3
"""Compare DA3 depth outputs between legacy da3_streaming and DAP3 stream mode."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

OLD_FRAME_RE = re.compile(r"^frame_(\d+)\.npz$")


@dataclass(frozen=True)
class ResolvedNewVideo:
    video_name: str
    video_dir: Path
    json_path: Path
    arrays_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare depth maps from legacy da3_streaming results_output with "
            "DAP3 stream-mode outputs."
        )
    )
    parser.add_argument(
        "--old-results-dir",
        required=True,
        help="Path to legacy results_output directory containing frame_<idx>.npz files.",
    )
    parser.add_argument(
        "--new-path",
        required=True,
        help=(
            "Path to new output root (run dir containing per-video subdirs) or a single "
            "per-video directory containing one .json and one *_arrays.npz."
        ),
    )
    parser.add_argument(
        "--video-name",
        default=None,
        help=(
            "Optional video folder/json stem to select when --new-path is a run dir with "
            "multiple videos."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write summary JSON, per-frame CSV, plots, and panel images.",
    )
    parser.add_argument(
        "--resize-to",
        choices=["old", "new"],
        default="old",
        help="Resize depth maps to match old or new spatial size before comparison.",
    )
    parser.add_argument(
        "--top-k-worst",
        type=int,
        default=8,
        help="Number of highest-MAE frames to render as panels.",
    )
    parser.add_argument(
        "--sample-k",
        type=int,
        default=6,
        help="Number of evenly-spaced frames to render as panels.",
    )
    parser.add_argument(
        "--pixel-scatter-samples",
        type=int,
        default=250_000,
        help="Max sampled pixels for old-vs-new scatter plot.",
    )
    parser.add_argument(
        "--hist-bins",
        type=int,
        default=512,
        help="Number of bins for signed/absolute error histograms.",
    )
    return parser.parse_args()


def find_single_json(dir_path: Path) -> Path:
    candidates = sorted(p for p in dir_path.glob("*.json") if p.name != "run_manifest.json")
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one per-video JSON in {dir_path}, found {len(candidates)}."
        )
    return candidates[0]


def find_single_arrays(dir_path: Path) -> Path:
    candidates = sorted(dir_path.glob("*_arrays.npz"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one *_arrays.npz in {dir_path}, found {len(candidates)}."
        )
    return candidates[0]


def resolve_new_video(new_path: Path, video_name: str | None) -> ResolvedNewVideo:
    if not new_path.is_dir():
        raise FileNotFoundError(f"new-path is not a directory: {new_path}")

    direct_json = sorted(p for p in new_path.glob("*.json") if p.name != "run_manifest.json")
    direct_npz = sorted(new_path.glob("*_arrays.npz"))
    if len(direct_json) == 1 and len(direct_npz) == 1:
        json_path = direct_json[0]
        arrays_path = direct_npz[0]
        return ResolvedNewVideo(
            video_name=json_path.stem,
            video_dir=new_path,
            json_path=json_path,
            arrays_path=arrays_path,
        )

    if video_name:
        video_dir = new_path / video_name
        if not video_dir.is_dir():
            raise FileNotFoundError(f"Video directory not found: {video_dir}")
    else:
        video_dirs = sorted(
            d
            for d in new_path.iterdir()
            if d.is_dir() and (d / "run_manifest.json").exists() is False
        )
        if len(video_dirs) != 1:
            raise RuntimeError(
                "Multiple video directories found. Pass --video-name to select one."
            )
        video_dir = video_dirs[0]

    json_path = find_single_json(video_dir)
    arrays_path = find_single_arrays(video_dir)
    return ResolvedNewVideo(
        video_name=json_path.stem,
        video_dir=video_dir,
        json_path=json_path,
        arrays_path=arrays_path,
    )


def index_old_depths(old_results_dir: Path) -> dict[int, Path]:
    if not old_results_dir.is_dir():
        raise FileNotFoundError(f"old-results-dir not found: {old_results_dir}")
    mapping: dict[int, Path] = {}
    for path in sorted(old_results_dir.glob("frame_*.npz")):
        match = OLD_FRAME_RE.match(path.name)
        if not match:
            continue
        mapping[int(match.group(1))] = path
    if not mapping:
        raise RuntimeError(f"No frame_<idx>.npz files found in {old_results_dir}")
    return mapping


def index_new_depth_keys(video_json: dict[str, Any]) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for frame in video_json.get("frames") or []:
        frame_index = frame.get("frame_index")
        npz_keys = frame.get("npz_keys") or {}
        depth_key = npz_keys.get("depth")
        if frame_index is None or not depth_key:
            continue
        mapping[int(frame_index)] = str(depth_key)
    if not mapping:
        raise RuntimeError("No per-frame depth keys found in new video JSON.")
    return mapping


def align_depths(old_depth: np.ndarray, new_depth: np.ndarray, resize_to: str) -> tuple[np.ndarray, np.ndarray]:
    if resize_to == "old":
        target_h, target_w = old_depth.shape
        if new_depth.shape != (target_h, target_w):
            new_depth = cv2.resize(new_depth, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        return old_depth, new_depth

    target_h, target_w = new_depth.shape
    if old_depth.shape != (target_h, target_w):
        old_depth = cv2.resize(old_depth, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return old_depth, new_depth


def finite_positive_mask(old_depth: np.ndarray, new_depth: np.ndarray) -> np.ndarray:
    return np.isfinite(old_depth) & np.isfinite(new_depth) & (old_depth > 0.0) & (new_depth > 0.0)


def hist_quantile(hist: np.ndarray, edges: np.ndarray, q: float) -> float:
    if hist.sum() == 0:
        return float("nan")
    target = q * int(hist.sum())
    cdf = np.cumsum(hist)
    idx = int(np.searchsorted(cdf, target, side="left"))
    idx = max(0, min(idx, len(edges) - 2))
    return float(edges[idx])


def sample_from_array(values: np.ndarray, max_samples: int, rng: np.random.Generator) -> np.ndarray:
    if max_samples <= 0 or values.size == 0:
        return np.empty((0,), dtype=np.float32)
    if values.size <= max_samples:
        return values.astype(np.float32, copy=False)
    idx = rng.choice(values.size, size=max_samples, replace=False)
    return values[idx].astype(np.float32, copy=False)


def write_per_frame_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "frame_index",
        "valid_pixels",
        "mean_old",
        "mean_new",
        "bias_new_minus_old",
        "mae",
        "rmse",
        "median_abs",
        "p95_abs",
        "max_abs",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def plot_per_frame_metrics(per_frame: list[dict[str, Any]], out_dir: Path) -> None:
    frame_idx = [row["frame_index"] for row in per_frame]
    mae = [row["mae"] for row in per_frame]
    rmse = [row["rmse"] for row in per_frame]
    bias = [row["bias_new_minus_old"] for row in per_frame]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    axes[0].plot(frame_idx, mae, color="#136f63", linewidth=1.5)
    axes[0].set_ylabel("MAE")
    axes[0].set_title("Per-frame Error Metrics")
    axes[0].grid(alpha=0.2)

    axes[1].plot(frame_idx, rmse, color="#0b4f6c", linewidth=1.5)
    axes[1].set_ylabel("RMSE")
    axes[1].grid(alpha=0.2)

    axes[2].plot(frame_idx, bias, color="#b02e0c", linewidth=1.5)
    axes[2].set_ylabel("Bias (new-old)")
    axes[2].set_xlabel("Frame Index")
    axes[2].grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_dir / "per_frame_metrics.png", dpi=180)
    plt.close(fig)

    means_old = [row["mean_old"] for row in per_frame]
    means_new = [row["mean_new"] for row in per_frame]
    low = min(min(means_old), min(means_new))
    high = max(max(means_old), max(means_new))
    pad = 0.02 * (high - low if high > low else 1.0)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(means_old, means_new, s=18, alpha=0.7, color="#1f77b4", edgecolor="none")
    ax.plot([low - pad, high + pad], [low - pad, high + pad], linestyle="--", color="black", linewidth=1)
    ax.set_xlabel("Old depth mean")
    ax.set_ylabel("New depth mean")
    ax.set_title("Per-frame Mean Depth: Old vs New")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "mean_depth_scatter.png", dpi=180)
    plt.close(fig)


def plot_histograms(
    abs_edges: np.ndarray,
    abs_hist: np.ndarray,
    signed_edges: np.ndarray,
    signed_hist: np.ndarray,
    out_dir: Path,
) -> None:
    abs_centers = 0.5 * (abs_edges[:-1] + abs_edges[1:])
    signed_centers = 0.5 * (signed_edges[:-1] + signed_edges[1:])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(abs_centers, abs_hist, color="#0b4f6c", linewidth=1.5)
    axes[0].set_title("Absolute Error Histogram")
    axes[0].set_xlabel("|new-old|")
    axes[0].set_ylabel("Pixel Count")
    axes[0].grid(alpha=0.2)

    axes[1].plot(signed_centers, signed_hist, color="#b02e0c", linewidth=1.5)
    axes[1].set_title("Signed Error Histogram")
    axes[1].set_xlabel("new-old")
    axes[1].set_ylabel("Pixel Count")
    axes[1].grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_dir / "error_histograms.png", dpi=180)
    plt.close(fig)


def plot_pixel_scatter(pixel_old: np.ndarray, pixel_new: np.ndarray, out_dir: Path) -> None:
    if pixel_old.size == 0:
        return
    low = float(min(pixel_old.min(), pixel_new.min()))
    high = float(max(pixel_old.max(), pixel_new.max()))
    pad = 0.02 * (high - low if high > low else 1.0)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(pixel_old, pixel_new, s=1.0, alpha=0.12, color="#136f63", edgecolor="none")
    ax.plot([low - pad, high + pad], [low - pad, high + pad], linestyle="--", color="black", linewidth=1)
    ax.set_xlabel("Old depth (sampled pixels)")
    ax.set_ylabel("New depth (sampled pixels)")
    ax.set_title(f"Pixel Scatter (n={pixel_old.size:,})")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "pixel_scatter.png", dpi=180)
    plt.close(fig)


def render_frame_panel(
    frame_index: int,
    old_npz_path: Path,
    new_arrays: np.lib.npyio.NpzFile,
    new_depth_key: str,
    resize_to: str,
    per_frame_row: dict[str, Any],
    out_path: Path,
) -> None:
    with np.load(old_npz_path) as old_npz:
        old_depth = old_npz["depth"].astype(np.float32)
        old_image = old_npz["image"] if "image" in old_npz.files else None

    new_depth = new_arrays[new_depth_key].astype(np.float32)
    old_depth, new_depth = align_depths(old_depth, new_depth, resize_to=resize_to)
    valid = finite_positive_mask(old_depth, new_depth)
    if not np.any(valid):
        return

    diff_abs = np.abs(new_depth - old_depth)
    depth_vals = np.concatenate([old_depth[valid], new_depth[valid]])
    vmin = float(np.percentile(depth_vals, 2))
    vmax = float(np.percentile(depth_vals, 98))
    vmax = max(vmax, vmin + 1e-6)
    dmax = float(np.percentile(diff_abs[valid], 99))
    dmax = max(dmax, 1e-6)

    panels = 4 if old_image is not None else 3
    fig, axes = plt.subplots(1, panels, figsize=(4.8 * panels, 4.5))
    ax_idx = 0

    if old_image is not None:
        axes[ax_idx].imshow(old_image)
        axes[ax_idx].set_title(f"RGB (frame {frame_index})")
        axes[ax_idx].axis("off")
        ax_idx += 1

    im0 = axes[ax_idx].imshow(old_depth, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[ax_idx].set_title("Old depth")
    axes[ax_idx].axis("off")
    fig.colorbar(im0, ax=axes[ax_idx], fraction=0.046, pad=0.02)
    ax_idx += 1

    im1 = axes[ax_idx].imshow(new_depth, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[ax_idx].set_title("New depth")
    axes[ax_idx].axis("off")
    fig.colorbar(im1, ax=axes[ax_idx], fraction=0.046, pad=0.02)
    ax_idx += 1

    im2 = axes[ax_idx].imshow(diff_abs, cmap="magma", vmin=0.0, vmax=dmax)
    axes[ax_idx].set_title("|new-old|")
    axes[ax_idx].axis("off")
    fig.colorbar(im2, ax=axes[ax_idx], fraction=0.046, pad=0.02)

    fig.suptitle(
        f"Frame {frame_index} | MAE={per_frame_row['mae']:.4f} RMSE={per_frame_row['rmse']:.4f}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def choose_sampled_frames(common_indices: list[int], k: int) -> list[int]:
    if k <= 0 or not common_indices:
        return []
    if len(common_indices) <= k:
        return common_indices[:]
    positions = np.linspace(0, len(common_indices) - 1, num=k, dtype=int)
    return [common_indices[i] for i in positions]


def main() -> None:
    args = parse_args()
    old_results_dir = Path(args.old_results_dir).resolve()
    new_path = Path(args.new_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    plots_dir = output_dir / "plots"
    frames_dir = output_dir / "frames"
    plots_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    resolved = resolve_new_video(new_path, args.video_name)
    video_json = json.loads(resolved.json_path.read_text(encoding="utf-8"))
    old_depths = index_old_depths(old_results_dir)
    new_depth_keys = index_new_depth_keys(video_json)

    common_indices = sorted(set(old_depths) & set(new_depth_keys))
    if not common_indices:
        raise RuntimeError("No overlapping frame indices found between old and new outputs.")

    rng = np.random.default_rng(7)
    new_arrays = np.load(resolved.arrays_path)
    per_frame_rows: list[dict[str, Any]] = []

    sum_old = 0.0
    sum_new = 0.0
    sum_old_sq = 0.0
    sum_new_sq = 0.0
    sum_old_new = 0.0
    sum_diff = 0.0
    sum_abs = 0.0
    sum_sq = 0.0
    sum_rel_abs = 0.0
    total_pixels = 0
    global_abs_max = 0.0
    global_signed_min = math.inf
    global_signed_max = -math.inf

    pixel_old_samples: list[np.ndarray] = []
    pixel_new_samples: list[np.ndarray] = []
    per_frame_sample_budget = max(1, args.pixel_scatter_samples // len(common_indices))

    for frame_index in common_indices:
        old_npz_path = old_depths[frame_index]
        with np.load(old_npz_path) as old_npz:
            old_depth = old_npz["depth"].astype(np.float32)
        new_depth = new_arrays[new_depth_keys[frame_index]].astype(np.float32)
        old_depth, new_depth = align_depths(old_depth, new_depth, resize_to=args.resize_to)

        valid = finite_positive_mask(old_depth, new_depth)
        if not np.any(valid):
            continue

        old_vals = old_depth[valid].astype(np.float64, copy=False)
        new_vals = new_depth[valid].astype(np.float64, copy=False)
        diff = new_vals - old_vals
        abs_diff = np.abs(diff)
        rel_abs = abs_diff / np.maximum(np.abs(old_vals), 1e-6)

        count = int(old_vals.size)
        total_pixels += count

        sum_old += float(old_vals.sum())
        sum_new += float(new_vals.sum())
        sum_old_sq += float(np.square(old_vals).sum())
        sum_new_sq += float(np.square(new_vals).sum())
        sum_old_new += float((old_vals * new_vals).sum())
        sum_diff += float(diff.sum())
        sum_abs += float(abs_diff.sum())
        sum_sq += float(np.square(diff).sum())
        sum_rel_abs += float(rel_abs.sum())

        if abs_diff.size:
            global_abs_max = max(global_abs_max, float(abs_diff.max()))
            global_signed_min = min(global_signed_min, float(diff.min()))
            global_signed_max = max(global_signed_max, float(diff.max()))

        frame_row = {
            "frame_index": frame_index,
            "valid_pixels": count,
            "mean_old": float(old_vals.mean()),
            "mean_new": float(new_vals.mean()),
            "bias_new_minus_old": float(diff.mean()),
            "mae": float(abs_diff.mean()),
            "rmse": float(np.sqrt(np.square(diff).mean())),
            "median_abs": float(np.median(abs_diff)),
            "p95_abs": float(np.percentile(abs_diff, 95)),
            "max_abs": float(abs_diff.max()),
        }
        per_frame_rows.append(frame_row)

        sampled_old = sample_from_array(old_vals.astype(np.float32, copy=False), per_frame_sample_budget, rng)
        sampled_new = sample_from_array(new_vals.astype(np.float32, copy=False), per_frame_sample_budget, rng)
        if sampled_old.size > 0:
            pixel_old_samples.append(sampled_old)
            pixel_new_samples.append(sampled_new)

    if total_pixels == 0 or not per_frame_rows:
        raise RuntimeError("No valid overlapping depth pixels found for comparison.")

    per_frame_rows.sort(key=lambda row: row["frame_index"])
    mean_old = sum_old / total_pixels
    mean_new = sum_new / total_pixels
    bias = sum_diff / total_pixels
    mae = sum_abs / total_pixels
    rmse = math.sqrt(sum_sq / total_pixels)
    rel_mae = sum_rel_abs / total_pixels

    var_old = (sum_old_sq / total_pixels) - (mean_old**2)
    var_new = (sum_new_sq / total_pixels) - (mean_new**2)
    cov = (sum_old_new / total_pixels) - (mean_old * mean_new)
    corr = float("nan")
    if var_old > 0 and var_new > 0:
        corr = cov / math.sqrt(var_old * var_new)

    abs_hist = np.zeros(args.hist_bins, dtype=np.int64)
    signed_hist = np.zeros(args.hist_bins, dtype=np.int64)
    abs_max = max(global_abs_max, 1e-6)
    if not math.isfinite(global_signed_min):
        global_signed_min = -1e-6
    if not math.isfinite(global_signed_max):
        global_signed_max = 1e-6
    if abs(global_signed_max - global_signed_min) < 1e-9:
        global_signed_min -= 1e-6
        global_signed_max += 1e-6

    abs_edges = np.linspace(0.0, abs_max, num=args.hist_bins + 1)
    signed_edges = np.linspace(global_signed_min, global_signed_max, num=args.hist_bins + 1)

    for frame_index in common_indices:
        old_npz_path = old_depths[frame_index]
        with np.load(old_npz_path) as old_npz:
            old_depth = old_npz["depth"].astype(np.float32)
        new_depth = new_arrays[new_depth_keys[frame_index]].astype(np.float32)
        old_depth, new_depth = align_depths(old_depth, new_depth, resize_to=args.resize_to)

        valid = finite_positive_mask(old_depth, new_depth)
        if not np.any(valid):
            continue
        diff = (new_depth[valid] - old_depth[valid]).astype(np.float64, copy=False)
        abs_diff = np.abs(diff)
        abs_hist += np.histogram(abs_diff, bins=abs_edges)[0]
        signed_hist += np.histogram(diff, bins=signed_edges)[0]

    median_abs_approx = hist_quantile(abs_hist, abs_edges, 0.50)
    p95_abs_approx = hist_quantile(abs_hist, abs_edges, 0.95)

    pixel_old = (
        np.concatenate(pixel_old_samples)[: args.pixel_scatter_samples]
        if pixel_old_samples
        else np.empty((0,), dtype=np.float32)
    )
    pixel_new = (
        np.concatenate(pixel_new_samples)[: args.pixel_scatter_samples]
        if pixel_new_samples
        else np.empty((0,), dtype=np.float32)
    )

    write_per_frame_csv(output_dir / "per_frame_metrics.csv", per_frame_rows)
    plot_per_frame_metrics(per_frame_rows, plots_dir)
    plot_histograms(abs_edges, abs_hist, signed_edges, signed_hist, plots_dir)
    plot_pixel_scatter(pixel_old, pixel_new, plots_dir)

    sorted_worst = sorted(per_frame_rows, key=lambda row: row["mae"], reverse=True)
    worst_indices = [row["frame_index"] for row in sorted_worst[: max(0, args.top_k_worst)]]
    sampled_indices = choose_sampled_frames(common_indices, args.sample_k)
    rendered = []
    per_frame_lookup = {row["frame_index"]: row for row in per_frame_rows}
    for frame_index in worst_indices + sampled_indices:
        if frame_index in rendered:
            continue
        out_path = frames_dir / f"frame_{frame_index:04d}_panel.png"
        render_frame_panel(
            frame_index=frame_index,
            old_npz_path=old_depths[frame_index],
            new_arrays=new_arrays,
            new_depth_key=new_depth_keys[frame_index],
            resize_to=args.resize_to,
            per_frame_row=per_frame_lookup[frame_index],
            out_path=out_path,
        )
        rendered.append(frame_index)

    summary = {
        "old_results_dir": str(old_results_dir),
        "new_video_dir": str(resolved.video_dir),
        "new_video_json": str(resolved.json_path),
        "new_video_arrays": str(resolved.arrays_path),
        "video_name": resolved.video_name,
        "resize_to": args.resize_to,
        "counts": {
            "old_depth_frames": len(old_depths),
            "new_depth_frames": len(new_depth_keys),
            "overlap_frames": len(common_indices),
            "compared_frames_with_valid_pixels": len(per_frame_rows),
            "compared_pixels": int(total_pixels),
        },
        "metrics": {
            "mean_old_depth": mean_old,
            "mean_new_depth": mean_new,
            "bias_new_minus_old": bias,
            "mae": mae,
            "rmse": rmse,
            "relative_mae": rel_mae,
            "pearson_corr": corr,
            "max_abs_error": global_abs_max,
            "median_abs_error_approx": median_abs_approx,
            "p95_abs_error_approx": p95_abs_approx,
        },
        "frame_selection": {
            "worst_by_mae": worst_indices,
            "evenly_sampled": sampled_indices,
            "rendered_panels": rendered,
        },
        "artifacts": {
            "summary_json": str(output_dir / "summary.json"),
            "per_frame_csv": str(output_dir / "per_frame_metrics.csv"),
            "plots_dir": str(plots_dir),
            "frames_dir": str(frames_dir),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
