#!/usr/bin/env python3
"""Plot depth-mask mean difference (old - new) from a per-video JSON."""

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
            "Visualize depth_mask_mean difference after merge: "
            "(depth_mask_mean_old - depth_mask_mean) over frame/time."
        )
    )
    parser.add_argument("--video-json", type=Path, required=True, help="Path to per-video JSON file.")
    parser.add_argument(
        "--new-key",
        type=str,
        default="depth_mask_mean",
        help="JSON key for new depth mask mean (default: depth_mask_mean).",
    )
    parser.add_argument(
        "--old-key",
        type=str,
        default="depth_mask_mean_old",
        help="JSON key for old depth mask mean (default: depth_mask_mean_old).",
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
        help="Output PNG path. Default: <video_json_stem>_depth_mask_mean_diff.png",
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


def main() -> int:
    args = parse_args()
    video_json = args.video_json.resolve()
    if not video_json.is_file():
        raise FileNotFoundError(f"Video JSON not found: {video_json}")

    payload = json.loads(video_json.read_text(encoding="utf-8"))
    frames = list(payload.get("frames") or [])
    if not frames:
        raise RuntimeError(f"No frames found in {video_json}")

    frames.sort(key=lambda row: int(row.get("frame_index", -1)))
    x_vals: list[float] = []
    diffs: list[float] = []

    for row in frames:
        if args.x_axis == "frame":
            x = _to_float_or_nan(row.get("frame_index"))
        else:
            x = _to_float_or_nan(row.get("timestamp_sec"))
        new_val = _to_float_or_nan(row.get(args.new_key))
        old_val = _to_float_or_nan(row.get(args.old_key))
        if not np.isfinite(x) or not np.isfinite(new_val) or not np.isfinite(old_val):
            continue
        x_vals.append(x)
        diffs.append(old_val - new_val)

    if not x_vals:
        raise RuntimeError(
            f"No comparable finite values for '{args.old_key}' and '{args.new_key}' in {video_json}"
        )

    x = np.asarray(x_vals, dtype=np.float64)
    diff = np.asarray(diffs, dtype=np.float64)

    output = (
        args.output.resolve()
        if args.output is not None
        else video_json.with_name(f"{video_json.stem}_depth_mask_mean_diff.png")
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    ax.plot(x, diff, color="#0b4f6c", linewidth=1.4, label=f"{args.old_key} - {args.new_key}")
    ax.axhline(0.0, color="#b02e0c", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_title("Depth Mask Mean Difference")
    ax.set_xlabel("Frame index" if args.x_axis == "frame" else "Timestamp (sec)")
    ax.set_ylabel("Mean depth difference")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)

    summary = {
        "video_json": str(video_json),
        "output_plot": str(output),
        "x_axis": args.x_axis,
        "new_key": args.new_key,
        "old_key": args.old_key,
        "counts": {"points": int(diff.size)},
        "stats": {
            "mean_diff": float(np.mean(diff)),
            "median_diff": float(np.median(diff)),
            "p05_diff": float(np.percentile(diff, 5)),
            "p95_diff": float(np.percentile(diff, 95)),
        },
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
