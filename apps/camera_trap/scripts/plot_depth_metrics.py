#!/usr/bin/env python3
"""Plot merged depth statistics from per-video camera-trap outputs.

Modes:
- mask_means: plot depth_mask_mean and depth_mask_mean_old
- mask_diff: plot (depth_mask_mean_old - depth_mask_mean)
- id_means: plot per-track-id mask mean depth over frame/time
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot merged depth metrics from a per-video JSON (+NPZ for id_means mode)."
    )
    parser.add_argument("--video-json", type=Path, required=True, help="Path to per-video JSON file.")
    parser.add_argument(
        "--mode",
        choices=["mask_means", "mask_diff", "id_means"],
        default="mask_means",
        help="Plot mode.",
    )
    parser.add_argument(
        "--x-axis",
        choices=["frame", "time"],
        default="frame",
        help="Use frame_index or timestamp_sec on x-axis.",
    )
    parser.add_argument(
        "--new-key",
        type=str,
        default="depth_mask_mean",
        help="JSON key for new depth mask mean.",
    )
    parser.add_argument(
        "--old-key",
        type=str,
        default="depth_mask_mean_old",
        help="JSON key for old depth mask mean.",
    )
    parser.add_argument(
        "--npz-path",
        type=Path,
        default=None,
        help="Optional NPZ path; defaults to sibling <video_stem>_arrays.npz.",
    )
    parser.add_argument(
        "--id-depth-source",
        choices=["new", "old", "both"],
        default="both",
        help="Depth source(s) for id_means mode.",
    )
    parser.add_argument(
        "--max-ids",
        type=int,
        default=20,
        help="Maximum unique track IDs to plot in id_means mode (top by #points).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (defaults to mode-specific filename next to JSON).",
    )
    return parser.parse_args()


def _to_float_or_nan(value: object) -> float:
    if value is None:
        return float("nan")
    try:
        val = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return val if np.isfinite(val) else float("nan")


def _resolve_output_path(args: argparse.Namespace, video_json: Path) -> Path:
    if args.output is not None:
        out = args.output.resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        return out
    stem = video_json.stem
    out = video_json.with_name(f"{stem}_{args.mode}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def _sorted_frames(payload: dict[str, Any]) -> list[dict[str, Any]]:
    frames = list(payload.get("frames") or [])
    frames.sort(key=lambda row: int(row.get("frame_index", -1)))
    return frames


def _x_value(row: dict[str, Any], x_axis: str) -> float:
    if x_axis == "time":
        return _to_float_or_nan(row.get("timestamp_sec"))
    return _to_float_or_nan(row.get("frame_index"))


def _plot_mask_means(
    *,
    frames: list[dict[str, Any]],
    x_axis: str,
    new_key: str,
    old_key: str,
    output: Path,
) -> dict[str, Any]:
    x_vals: list[float] = []
    y_new: list[float] = []
    y_old: list[float] = []
    for row in frames:
        x = _x_value(row, x_axis)
        if not np.isfinite(x):
            continue
        x_vals.append(x)
        y_new.append(_to_float_or_nan(row.get(new_key)))
        y_old.append(_to_float_or_nan(row.get(old_key)))

    if not x_vals:
        raise RuntimeError("No valid x-axis values found in frames.")
    x = np.asarray(x_vals, dtype=np.float64)
    y_new_arr = np.asarray(y_new, dtype=np.float64)
    y_old_arr = np.asarray(y_old, dtype=np.float64)
    finite_new = np.isfinite(y_new_arr)
    finite_old = np.isfinite(y_old_arr)
    if not finite_new.any() and not finite_old.any():
        raise RuntimeError(f"No finite values found for '{new_key}' or '{old_key}'.")

    fig, ax = plt.subplots(figsize=(13, 5.5))
    if finite_new.any():
        ax.plot(
            x[finite_new],
            y_new_arr[finite_new],
            color="#136f63",
            linewidth=1.4,
            label=f"{new_key} (n={int(finite_new.sum())})",
        )
    if finite_old.any():
        ax.plot(
            x[finite_old],
            y_old_arr[finite_old],
            color="#b02e0c",
            linewidth=1.4,
            label=f"{old_key} (n={int(finite_old.sum())})",
        )
    ax.set_title("Depth Mask Means")
    ax.set_xlabel("Frame index" if x_axis == "frame" else "Timestamp (sec)")
    ax.set_ylabel("Mean depth")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)

    return {
        "x_points": int(x.size),
        "new_finite": int(finite_new.sum()),
        "old_finite": int(finite_old.sum()),
    }


def _plot_mask_diff(
    *,
    frames: list[dict[str, Any]],
    x_axis: str,
    new_key: str,
    old_key: str,
    output: Path,
) -> dict[str, Any]:
    x_vals: list[float] = []
    diff_vals: list[float] = []
    for row in frames:
        x = _x_value(row, x_axis)
        new_val = _to_float_or_nan(row.get(new_key))
        old_val = _to_float_or_nan(row.get(old_key))
        if not np.isfinite(x) or not np.isfinite(new_val) or not np.isfinite(old_val):
            continue
        x_vals.append(x)
        diff_vals.append(old_val - new_val)

    if not x_vals:
        raise RuntimeError(f"No comparable finite values for '{old_key}' and '{new_key}'.")
    x = np.asarray(x_vals, dtype=np.float64)
    diff = np.asarray(diff_vals, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    ax.plot(x, diff, color="#0b4f6c", linewidth=1.4, label=f"{old_key} - {new_key}")
    ax.axhline(0.0, color="#b02e0c", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_title("Depth Mask Mean Difference")
    ax.set_xlabel("Frame index" if x_axis == "frame" else "Timestamp (sec)")
    ax.set_ylabel("Mean depth difference")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)

    return {
        "points": int(diff.size),
        "mean_diff": float(np.mean(diff)),
        "median_diff": float(np.median(diff)),
        "p05_diff": float(np.percentile(diff, 5)),
        "p95_diff": float(np.percentile(diff, 95)),
    }


def _resolve_npz_path(video_json_path: Path, npz_path_arg: Path | None) -> Path:
    if npz_path_arg is not None:
        return npz_path_arg.resolve()
    parent = video_json_path.parent
    stem = video_json_path.stem
    candidate = parent / f"{stem}_arrays.npz"
    if not candidate.is_file():
        raise FileNotFoundError(
            f"Could not resolve NPZ automatically from JSON stem. Expected: {candidate}. "
            "Pass --npz-path explicitly."
        )
    return candidate


def _mean_depth_for_mask(depth: np.ndarray, mask: np.ndarray) -> float | None:
    if depth.shape != mask.shape:
        depth = cv2.resize(depth, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_CUBIC)
    vals = depth[mask]
    if vals.size == 0:
        return None
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return None
    return float(np.mean(finite))


def _plot_id_means(
    *,
    frames: list[dict[str, Any]],
    x_axis: str,
    npz_path: Path,
    id_depth_source: str,
    max_ids: int,
    output: Path,
) -> dict[str, Any]:
    if max_ids <= 0:
        raise ValueError("--max-ids must be > 0")

    series: dict[tuple[int, str], list[tuple[float, float]]] = defaultdict(list)
    counts_by_id: dict[int, int] = defaultdict(int)
    requested_sources = ["new", "old"] if id_depth_source == "both" else [id_depth_source]

    with np.load(npz_path, allow_pickle=False) as npz_data:
        for row in frames:
            x = _x_value(row, x_axis)
            if not np.isfinite(x):
                continue
            npz_keys = row.get("npz_keys") or {}
            depth_key_base = npz_keys.get("depth")
            if not depth_key_base:
                continue
            objects = row.get("objects") or []
            for obj in objects:
                track_id = obj.get("track_id")
                if track_id is None:
                    continue
                try:
                    tid = int(track_id)
                except (TypeError, ValueError):
                    continue
                mask_key = obj.get("key")
                if not mask_key or mask_key not in npz_data:
                    continue
                mask = np.asarray(npz_data[mask_key]).astype(bool)
                if mask.size == 0 or not np.any(mask):
                    continue
                for source in requested_sources:
                    depth_key = str(depth_key_base) if source == "new" else f"{depth_key_base}_old"
                    if depth_key not in npz_data:
                        continue
                    depth = np.asarray(npz_data[depth_key], dtype=np.float32)
                    mean_depth = _mean_depth_for_mask(depth, mask)
                    if mean_depth is None or not np.isfinite(mean_depth):
                        continue
                    series[(tid, source)].append((float(x), float(mean_depth)))
                    counts_by_id[tid] += 1

    if not series:
        raise RuntimeError("No valid per-ID depth series found. Check npz keys and object masks.")

    top_ids = [tid for tid, _ in sorted(counts_by_id.items(), key=lambda kv: kv[1], reverse=True)[:max_ids]]
    top_ids_set = set(top_ids)

    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(14, 6))
    plotted_lines = 0
    for color_idx, tid in enumerate(top_ids):
        color = cmap(color_idx % 20)
        for source in requested_sources:
            key = (tid, source)
            points = series.get(key) or []
            if not points:
                continue
            points.sort(key=lambda p: p[0])
            x = np.asarray([p[0] for p in points], dtype=np.float64)
            y = np.asarray([p[1] for p in points], dtype=np.float64)
            linestyle = "-" if source == "new" else "--"
            ax.plot(
                x,
                y,
                color=color,
                linestyle=linestyle,
                linewidth=1.3,
                label=f"id={tid} ({source})",
            )
            plotted_lines += 1

    if plotted_lines == 0:
        raise RuntimeError("No ID series remained after --max-ids filtering.")

    ax.set_title("Per-Track-ID Mean Depth")
    ax.set_xlabel("Frame index" if x_axis == "frame" else "Timestamp (sec)")
    ax.set_ylabel("Mean depth")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0, fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return {
        "npz_path": str(npz_path),
        "requested_sources": requested_sources,
        "unique_ids_total": len(counts_by_id),
        "unique_ids_plotted": len(top_ids_set),
        "series_total": len(series),
        "series_plotted": int(plotted_lines),
    }


def main() -> int:
    args = parse_args()
    video_json_path = args.video_json.resolve()
    if not video_json_path.is_file():
        raise FileNotFoundError(f"Video JSON not found: {video_json_path}")

    payload = json.loads(video_json_path.read_text(encoding="utf-8"))
    frames = _sorted_frames(payload)
    if not frames:
        raise RuntimeError(f"No frames found in {video_json_path}")

    output_path = _resolve_output_path(args, video_json_path)
    if args.mode == "mask_means":
        stats = _plot_mask_means(
            frames=frames,
            x_axis=args.x_axis,
            new_key=args.new_key,
            old_key=args.old_key,
            output=output_path,
        )
    elif args.mode == "mask_diff":
        stats = _plot_mask_diff(
            frames=frames,
            x_axis=args.x_axis,
            new_key=args.new_key,
            old_key=args.old_key,
            output=output_path,
        )
    else:
        npz_path = _resolve_npz_path(video_json_path, args.npz_path)
        stats = _plot_id_means(
            frames=frames,
            x_axis=args.x_axis,
            npz_path=npz_path,
            id_depth_source=args.id_depth_source,
            max_ids=args.max_ids,
            output=output_path,
        )

    summary = {
        "video_json": str(video_json_path),
        "output_plot": str(output_path),
        "mode": args.mode,
        "x_axis": args.x_axis,
        "new_key": args.new_key,
        "old_key": args.old_key,
        "stats": stats,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
