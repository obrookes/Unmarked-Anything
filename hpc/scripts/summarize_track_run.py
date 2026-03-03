#!/usr/bin/env python3
"""Summarize per-video tracking outputs to spot cross-video leakage patterns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize dap3 track-mode run outputs.")
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to run directory containing run_manifest.json and per-video output folders.",
    )
    return parser.parse_args()


def iter_video_jsons(run_dir: Path) -> list[Path]:
    files = []
    for child in sorted(run_dir.iterdir()):
        if not child.is_dir():
            continue
        video_jsons = sorted(child.glob("*.json"))
        if len(video_jsons) != 1:
            continue
        files.append(video_jsons[0])
    return files


def summarize_video(video_json: dict[str, Any]) -> dict[str, Any]:
    frames = video_json.get("frames") or []
    processed = [f for f in frames if f.get("status") == "processed"]
    empty = [f for f in frames if f.get("status") == "empty_mask"]
    errors = [f for f in frames if (f.get("status") or "").endswith("_error")]
    first_processed = processed[0] if processed else None
    first_track_ids = []
    if first_processed:
        tracks = (first_processed.get("track_summary") or {}).get("tracks") or []
        first_track_ids = [t.get("track_id") for t in tracks if t.get("track_id") is not None]

    return {
        "video_name": video_json.get("video_name"),
        "sampled_frames": len(frames),
        "processed_frames": len(processed),
        "empty_frames": len(empty),
        "error_frames": len(errors),
        "sam3_tracker_num_frames": video_json.get("sam3_tracker_num_frames"),
        "first_processed_frame_index": first_processed.get("frame_index") if first_processed else None,
        "first_processed_track_ids": first_track_ids,
        "warnings": len(video_json.get("warnings") or []),
    }


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    manifest_path = run_dir / "run_manifest.json"
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"run_manifest.json not found in: {run_dir}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print("Run manifest settings:")
    print(
        json.dumps(
            {
                "sam3_mode": manifest.get("sam3_mode"),
                "sam3_track_isolation": manifest.get("sam3_track_isolation"),
                "sam3_track_tail_policy": manifest.get("sam3_track_tail_policy"),
                "max_videos": manifest.get("max_videos"),
                "videos_discovered": manifest.get("videos_discovered"),
                "summary": manifest.get("summary"),
            },
            indent=2,
        )
    )
    print()

    json_files = iter_video_jsons(run_dir)
    if not json_files:
        print("No per-video JSON files found.")
        return

    print("Per-video summary:")
    for path in json_files:
        video = json.loads(path.read_text(encoding="utf-8"))
        summary = summarize_video(video)
        print(f"- {summary['video_name']}")
        print(
            "  sampled={sampled_frames} processed={processed_frames} empty={empty_frames} "
            "errors={error_frames} tracker_num_frames={sam3_tracker_num_frames} "
            "first_idx={first_processed_frame_index} first_track_ids={first_processed_track_ids} "
            "warnings={warnings}".format(**summary)
        )


if __name__ == "__main__":
    main()
