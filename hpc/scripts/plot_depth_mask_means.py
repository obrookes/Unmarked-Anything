#!/usr/bin/env python3
"""Plot per-frame depth_mask_mean series for new vs merged old depth maps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot depth_mask_mean (existing) and depth_mask_mean_old (merged) from a per-video JSON."
        )
    )
    parser.add_argument("--video-json", type=Path, required=True, help="Path to per-video JSON file.")
    parser.add_argument(
        "--new-key",
        type=str,
        default="depth_mask_mean",
        help="JSON key for the existing/new depth mask mean series.",
    )
    parser.add_argument(
        "--old-key",
        type=str,
        default="depth_mask_mean_old",
        help="JSON key for the merged old depth mask mean series.",
    )
    parser.add_argument(
        "--x-axis",
        choices=["frame", "time"],
        default="frame",
        help="Use frame_index or timestamp_sec on x-axis.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output plot path (PNG). Default: <video_json_stem>_depth_mask_means.png",
    )
    return parser.parse_args()


def _to_float_or_nan(value: object) -> float:
    if value is None:
        return float("nan")
    try:
        f = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return f if np.isfinite(f) else float("nan")


def main() -> int:
    args = parse_args()
    video_json_path = args.video_json.resolve()
    if not video_json_path.is_file():
        raise FileNotFoundError(f"Video JSON not found: {video_json_path}")

    payload = json.loads(video_json_path.read_text(encoding="utf-8"))
    frames = list(payload.get("frames") or [])
    if not frames:
        raise RuntimeError(f"No frames found in {video_json_path}")

    frames.sort(key=lambda row: int(row.get("frame_index", -1)))
    x_vals: list[float] = []
    y_new: list[float] = []
    y_old: list[float] = []

    for row in frames:
        if args.x_axis == "frame":
            x = _to_float_or_nan(row.get("frame_index"))
        else:
            x = _to_float_or_nan(row.get("timestamp_sec"))
        if not np.isfinite(x):
            continue
        x_vals.append(x)
        y_new.append(_to_float_or_nan(row.get(args.new_key)))
        y_old.append(_to_float_or_nan(row.get(args.old_key)))

    if not x_vals:
        raise RuntimeError("No valid x-axis values found in frames.")

    x = np.asarray(x_vals, dtype=np.float64)
    y_new_arr = np.asarray(y_new, dtype=np.float64)
    y_old_arr = np.asarray(y_old, dtype=np.float64)
    finite_new = np.isfinite(y_new_arr)
    finite_old = np.isfinite(y_old_arr)

    if not finite_new.any() and not finite_old.any():
        raise RuntimeError(
            f"No finite values found for keys '{args.new_key}' or '{args.old_key}' in {video_json_path}"
        )

    output_path = (
        args.output.resolve()
        if args.output is not None
        else video_json_path.with_name(f"{video_json_path.stem}_depth_mask_means.png")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13, 5.5))
    if finite_new.any():
        ax.plot(
            x[finite_new],
            y_new_arr[finite_new],
            color="#136f63",
            linewidth=1.4,
            label=f"{args.new_key} (n={int(finite_new.sum())})",
        )
    if finite_old.any():
        ax.plot(
            x[finite_old],
            y_old_arr[finite_old],
            color="#b02e0c",
            linewidth=1.4,
            label=f"{args.old_key} (n={int(finite_old.sum())})",
        )

    ax.set_title("Depth Mask Mean Comparison")
    ax.set_xlabel("Frame index" if args.x_axis == "frame" else "Timestamp (sec)")
    ax.set_ylabel("Mean depth")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    print(
        json.dumps(
            {
                "video_json": str(video_json_path),
                "output_plot": str(output_path),
                "new_key": args.new_key,
                "old_key": args.old_key,
                "x_axis": args.x_axis,
                "counts": {
                    "frames_total": len(frames),
                    "x_points": len(x_vals),
                    "new_finite": int(finite_new.sum()),
                    "old_finite": int(finite_old.sum()),
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
