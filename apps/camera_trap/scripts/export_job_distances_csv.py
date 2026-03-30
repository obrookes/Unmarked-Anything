#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from depth_anything_3.utils.camera_trap_masks import load_mask_bool


VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv")
CSV_COLUMNS = [
    "video_name",
    "starting_date",
    "year",
    "month",
    "day",
    "hour",
    "minute",
    "second",
    "ind_no",
    "distance",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export per-object interval-sampled distances from camera-trap job outputs "
            "(JSON + NPZ) into a single CSV."
        )
    )
    parser.add_argument(
        "--job-dir",
        type=Path,
        required=True,
        help="Job directory containing per-video output folders (e.g., hpc/runs/job).",
    )
    parser.add_argument("--output-csv", type=Path, required=True, help="Destination CSV path.")
    parser.add_argument(
        "--interval-seconds",
        type=float,
        required=True,
        help="Sampling interval in seconds, applied per object from first appearance.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Optional FPS override. If omitted, uses video_fps from each video JSON.",
    )
    args = parser.parse_args()
    if args.interval_seconds <= 0:
        parser.error("--interval-seconds must be > 0.")
    if args.fps is not None and args.fps <= 0:
        parser.error("--fps must be > 0 when provided.")
    return args


def discover_video_jsons(job_dir: Path) -> list[Path]:
    json_paths: list[Path] = []
    skipped: list[Path] = []
    for path in sorted(job_dir.rglob("*.json")):
        if path.name == "run_manifest.json":
            continue
        npz_path = path.with_name(f"{path.stem}_arrays.npz")
        if npz_path.is_file():
            json_paths.append(path)
        else:
            skipped.append(path)
    if skipped:
        print(
            f"Warning: skipped {len(skipped)} JSON file(s) with no paired *_arrays.npz "
            f"(re-run the pipeline with --npz to generate NPZ output):",
            file=sys.stderr,
        )
        for p in skipped:
            print(f"  {p}", file=sys.stderr)
    return json_paths


def parse_creation_time(raw: str) -> datetime | None:
    text = raw.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def ffprobe_creation_time(video_path: Path) -> datetime | None:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format_tags=creation_time:stream_tags=creation_time",
        "-of",
        "json",
        str(video_path),
    ]
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

    try:
        payload = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        return None

    fmt_tags = (payload.get("format") or {}).get("tags") or {}
    fmt_creation = fmt_tags.get("creation_time")
    dt = parse_creation_time(fmt_creation) if isinstance(fmt_creation, str) else None
    if dt is not None:
        return dt

    for stream in payload.get("streams") or []:
        tags = stream.get("tags") or {}
        stream_creation = tags.get("creation_time")
        dt = parse_creation_time(stream_creation) if isinstance(stream_creation, str) else None
        if dt is not None:
            return dt
    return None


def build_video_lookup(repo_root: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    videos_dir = repo_root / "assets" / "videos"
    by_name: dict[str, Path] = {}
    by_stem: dict[str, Path] = {}
    if not videos_dir.is_dir():
        return by_name, by_stem

    for p in videos_dir.iterdir():
        if not p.is_file():
            continue
        by_name[p.name.lower()] = p
        by_stem[p.stem.lower()] = p
    return by_name, by_stem


def resolve_video_path(
    video_json: dict[str, Any],
    json_path: Path,
    by_name: dict[str, Path],
    by_stem: dict[str, Path],
) -> Path | None:
    raw = video_json.get("video_path")
    if isinstance(raw, str) and raw.strip():
        path = Path(raw)
        if path.is_file():
            return path

        from_name = by_name.get(path.name.lower())
        if from_name is not None:
            return from_name

        from_stem = by_stem.get(path.stem.lower())
        if from_stem is not None:
            return from_stem

    video_name = video_json.get("video_name")
    if isinstance(video_name, str) and video_name:
        from_stem = by_stem.get(video_name.lower())
        if from_stem is not None:
            return from_stem

        for ext in VIDEO_EXTS:
            from_name = by_name.get(f"{video_name}{ext}".lower())
            if from_name is not None:
                return from_name

    run_id = json_path.stem.lower()
    from_stem = by_stem.get(run_id)
    if from_stem is not None:
        return from_stem

    for ext in VIDEO_EXTS:
        from_name = by_name.get(f"{run_id}{ext}".lower())
        if from_name is not None:
            return from_name

    return None


def collect_track_samples(frames: list[dict[str, Any]]) -> dict[Any, dict[int, dict[str, Any]]]:
    track_samples: dict[Any, dict[int, dict[str, Any]]] = {}
    for frame in frames:
        frame_idx = frame.get("frame_index")
        if not isinstance(frame_idx, int):
            continue
        for obj in frame.get("objects") or []:
            track_id = obj.get("track_id")
            key = obj.get("key")
            if track_id is None or not isinstance(key, str) or not key:
                continue
            track_samples.setdefault(track_id, {})
            # Keep the first object entry observed for a track at a given frame.
            track_samples[track_id].setdefault(frame_idx, dict(obj))
    return track_samples


def compute_distance(depth: np.ndarray, mask: np.ndarray) -> float | None:
    if depth.shape != mask.shape:
        return None
    valid = mask > 0
    if not np.any(valid):
        return None
    depth_vals = depth[valid]
    depth_vals = depth_vals[np.isfinite(depth_vals)]
    if depth_vals.size == 0:
        return None
    return float(np.mean(depth_vals))


def build_rows_for_video(
    video_json: dict[str, Any],
    json_path: Path,
    npz_path: Path,
    interval_seconds: float,
    fps_override: float | None,
    creation_dt: datetime | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    video_name = str(video_json.get("video_name") or json_path.stem)
    fps_value = float(fps_override if fps_override is not None else (video_json.get("video_fps") or 0.0))
    if fps_value <= 0:
        return rows

    step_frames = max(1, int(round(interval_seconds * fps_value)))
    frames_per_second = max(1, int(round(fps_value)))
    frames = video_json.get("frames") or []
    track_samples = collect_track_samples(frames)

    starting_date = ""
    year = month = day = hour = minute = ""
    if creation_dt is not None:
        starting_date = creation_dt.date().isoformat()
        year = creation_dt.year
        month = creation_dt.month
        day = creation_dt.day
        hour = creation_dt.hour
        minute = creation_dt.minute

    with np.load(npz_path) as arrays:
        for track_id, frame_to_entry in sorted(track_samples.items(), key=lambda item: str(item[0])):
            frame_indices = sorted(frame_to_entry.keys())
            if not frame_indices:
                continue
            first_idx = frame_indices[0]
            last_idx = frame_indices[-1]
            present_indices = set(frame_indices)

            sampled_idx = first_idx
            while sampled_idx <= last_idx:
                window_distances: list[float] = []
                window_end = sampled_idx + frames_per_second
                for frame_idx in range(sampled_idx, window_end):
                    if frame_idx not in present_indices:
                        continue
                    depth_key = f"f{frame_idx}_depth"
                    mask_entry = frame_to_entry[frame_idx]
                    if depth_key not in arrays:
                        continue
                    try:
                        mask = load_mask_bool(npz_data=arrays, entry=mask_entry)
                    except KeyError:
                        continue
                    distance = compute_distance(arrays[depth_key], mask)
                    if distance is not None:
                        window_distances.append(distance)

                if window_distances:
                    rows.append(
                        {
                            "video_name": video_name,
                            "starting_date": starting_date,
                            "year": year,
                            "month": month,
                            "day": day,
                            "hour": hour,
                            "minute": minute,
                            "second": round((sampled_idx - first_idx) / fps_value, 6),
                            "ind_no": track_id,
                            "distance": float(np.mean(window_distances)),
                        }
                    )
                sampled_idx += step_frames

    return rows


def main() -> None:
    args = parse_args()
    job_dir = args.job_dir
    if not job_dir.is_dir():
        raise FileNotFoundError(f"Job directory not found: {job_dir}")

    json_paths = discover_video_jsons(job_dir)
    if not json_paths:
        raise FileNotFoundError(
            f"No per-video JSON+NPZ pairs found in {job_dir}. "
            "Expected files like <video>/<video>.json and <video>/<video>_arrays.npz."
        )

    by_name, by_stem = build_video_lookup(REPO_ROOT)

    all_rows: list[dict[str, Any]] = []
    skipped_missing_fps = 0
    for json_path in json_paths:
        npz_path = json_path.with_name(f"{json_path.stem}_arrays.npz")
        video_json = json.loads(json_path.read_text(encoding="utf-8"))

        if args.fps is None and float(video_json.get("video_fps") or 0.0) <= 0:
            skipped_missing_fps += 1
            continue

        video_path = resolve_video_path(video_json=video_json, json_path=json_path, by_name=by_name, by_stem=by_stem)
        creation_dt = ffprobe_creation_time(video_path) if video_path is not None else None

        all_rows.extend(
            build_rows_for_video(
                video_json=video_json,
                json_path=json_path,
                npz_path=npz_path,
                interval_seconds=args.interval_seconds,
                fps_override=args.fps,
                creation_dt=creation_dt,
            )
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Wrote {len(all_rows)} rows to {args.output_csv}")
    if skipped_missing_fps:
        print(
            f"Skipped {skipped_missing_fps} videos because FPS was unavailable "
            "(set --fps to override)."
        )


if __name__ == "__main__":
    main()
