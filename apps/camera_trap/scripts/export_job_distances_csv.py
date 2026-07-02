#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


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
    "confidence",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export per-object interval-sampled distances from camera-trap job outputs "
            "(JSON) into a single CSV."
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
        help="Sampling interval in seconds; sample points are aligned to the global grid 0, interval, 2*interval, ...",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=1.0,
        help="Duration in seconds of the averaging window at each sample point (default: 1.0).",
    )
    # --- Quality-control filters ---
    parser.add_argument(
        "--no-filter",
        action="store_true",
        default=False,
        help="Disable all quality-control filters and emit every detected track.",
    )
    parser.add_argument(
        "--min-track-frames",
        type=int,
        default=5,
        help="Exclude tracks detected in fewer than this many frames (default: 5). Set 0 to disable.",
    )
    parser.add_argument(
        "--max-static-displacement",
        type=float,
        default=5.0,
        help=(
            "Exclude tracks whose bbox centre never moves more than this many pixels total "
            "(default: 5.0). Catches stationary false-positive detections. Set 0.0 to disable."
        ),
    )
    parser.add_argument(
        "--iou-dedup-threshold",
        type=float,
        default=0.5,
        help=(
            "IoU threshold above which two concurrent detections are considered duplicates; "
            "the shorter track is dropped (default: 0.5). Use > 1.0 to disable."
        ),
    )
    parser.add_argument(
        "--min-bbox-area-frac",
        type=float,
        default=0.01,
        help=(
            "Exclude per-frame detections whose bbox area is smaller than this fraction of the "
            "frame area (default: 0.01 = 1%%). Set 0.0 to disable."
        ),
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Exclude tracks whose confidence score is below this threshold (default: 0.5). Set 0.0 to disable.",
    )
    args = parser.parse_args()
    if args.interval_seconds <= 0:
        parser.error("--interval-seconds must be > 0.")
    if args.window_seconds <= 0:
        parser.error("--window-seconds must be > 0.")
    if not args.no_filter:
        if args.min_track_frames < 0:
            parser.error("--min-track-frames must be >= 0.")
        if args.max_static_displacement < 0.0:
            parser.error("--max-static-displacement must be >= 0.0.")
        if not (0.0 <= args.min_bbox_area_frac < 1.0):
            parser.error("--min-bbox-area-frac must be in [0.0, 1.0).")
        if not (0.0 <= args.min_confidence <= 1.0):
            parser.error("--min-confidence must be in [0.0, 1.0].")
        if args.iou_dedup_threshold < 0.0:
            parser.error("--iou-dedup-threshold must be >= 0.0 (use > 1.0 to disable).")
    return args


def discover_video_jsons(job_dir: Path) -> list[Path]:
    json_paths: list[Path] = []
    for subdir in sorted(job_dir.iterdir()):
        if not subdir.is_dir():
            continue
        candidate = subdir / f"{subdir.name}.json"
        if candidate.is_file():
            json_paths.append(candidate)
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
            if track_id is None:
                continue
            track_samples.setdefault(track_id, {})
            # Keep the first object entry observed for a track at a given frame.
            track_samples[track_id].setdefault(frame_idx, dict(obj))
    return track_samples


def compute_iou(a: list, b: list) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def build_rows_for_video(
    video_json: dict[str, Any],
    json_path: Path,
    interval_seconds: float,
    window_seconds: float,
    creation_dt: datetime | None,
    *,
    apply_filters: bool = True,
    min_track_frames: int = 5,
    max_static_displacement: float = 5.0,
    min_bbox_area_frac: float = 0.01,
    iou_dedup_threshold: float = 0.5,
    min_confidence: float = 0.5,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    video_name = str(video_json.get("video_name") or json_path.stem)
    fps_value = float(video_json.get("target_fps") or video_json.get("video_fps") or 0.0)
    if fps_value <= 0:
        return rows

    frame_width = int(video_json.get("frame_width") or 0)
    frame_height = int(video_json.get("frame_height") or 0)
    frame_area = frame_width * frame_height
    step_frames = max(1, int(round(interval_seconds * fps_value)))
    window_frames = max(1, int(round(window_seconds * fps_value)))
    frames = video_json.get("frames") or []
    track_samples = collect_track_samples(frames)

    # --- Per-track and per-frame quality filters ---
    excluded_tracks: set = set()
    if apply_filters:
        # Step A: per-track exclusions (short tracks, stationary objects, low confidence)
        for track_id, frame_to_entry in track_samples.items():
            if min_track_frames > 0 and len(frame_to_entry) < min_track_frames:
                excluded_tracks.add(track_id)
                continue
            if max_static_displacement > 0.0:
                centers = [e.get("center_xy") or [0, 0] for e in frame_to_entry.values()]
                xs = [c[0] for c in centers]
                ys = [c[1] for c in centers]
                if max(max(xs) - min(xs), max(ys) - min(ys)) < max_static_displacement:
                    excluded_tracks.add(track_id)
                    continue
            if min_confidence > 0.0:
                conf = next(iter(frame_to_entry.values())).get("confidence")
                if conf is None or float(conf) < min_confidence:
                    excluded_tracks.add(track_id)

        # Step B: IoU duplicate exclusion (pre-scan all frames once)
        if iou_dedup_threshold <= 1.0:
            frame_to_active: dict[int, list] = {}
            for tid, f2e in track_samples.items():
                if tid in excluded_tracks:
                    continue
                for fidx, entry in f2e.items():
                    bbox = entry.get("bbox_xyxy") or []
                    if len(bbox) >= 4:
                        frame_to_active.setdefault(fidx, []).append((tid, bbox))

            track_frame_count = {tid: len(f2e) for tid, f2e in track_samples.items()}
            iou_excluded: set = set()
            for active in frame_to_active.values():
                for i in range(len(active)):
                    for j in range(i + 1, len(active)):
                        tid_a, bbox_a = active[i]
                        tid_b, bbox_b = active[j]
                        if tid_a in iou_excluded or tid_b in iou_excluded:
                            continue
                        if compute_iou(bbox_a, bbox_b) > iou_dedup_threshold:
                            ca, cb = track_frame_count[tid_a], track_frame_count[tid_b]
                            drop = tid_a if (ca < cb or (ca == cb and str(tid_a) > str(tid_b))) else tid_b
                            iou_excluded.add(drop)
            excluded_tracks |= iou_excluded

    starting_date = ""
    year = month = day = hour = minute = ""
    if creation_dt is not None:
        starting_date = creation_dt.date().isoformat()
        year = creation_dt.year
        month = creation_dt.month
        day = creation_dt.day
        hour = creation_dt.hour
        minute = creation_dt.minute

    for track_id, frame_to_entry in sorted(track_samples.items(), key=lambda item: str(item[0])):
        if track_id in excluded_tracks:
            continue
        frame_indices = sorted(frame_to_entry.keys())
        if not frame_indices:
            continue
        first_idx = frame_indices[0]
        last_idx = frame_indices[-1]
        present_indices = set(frame_indices)

        sampled_idx = math.ceil(first_idx / step_frames) * step_frames
        while sampled_idx <= last_idx:
            window_distances: list[float] = []
            window_confidences: list[float] = []
            for frame_idx in range(sampled_idx, sampled_idx + window_frames):
                if frame_idx not in present_indices:
                    continue
                entry = frame_to_entry[frame_idx]
                bbox = entry.get("bbox_xyxy") or []
                # Bbox left/right edge clip check (always active)
                if frame_width > 0 and len(bbox) >= 3 and (bbox[0] <= 0 or bbox[2] >= frame_width):
                    continue
                # Tiny bbox area check (filter mode only)
                if apply_filters and min_bbox_area_frac > 0.0 and frame_area > 0 and len(bbox) >= 4:
                    if (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) / frame_area < min_bbox_area_frac:
                        continue
                # Depth: always skip missing or physically impossible values
                distance = entry.get("depth_mask_mean")
                if distance is None or float(distance) < 0.0:
                    continue
                window_distances.append(float(distance))
                conf = entry.get("confidence")
                if conf is not None:
                    window_confidences.append(float(conf))

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
                        "second": round(sampled_idx / fps_value, 6),
                        "ind_no": track_id,
                        "distance": sum(window_distances) / len(window_distances),
                        "confidence": sum(window_confidences) / len(window_confidences) if window_confidences else "",
                    }
                )
            sampled_idx += step_frames

    return rows


REPO_ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    args = parse_args()
    job_dir = args.job_dir
    if not job_dir.is_dir():
        raise FileNotFoundError(f"Job directory not found: {job_dir}")

    json_paths = discover_video_jsons(job_dir)
    if not json_paths:
        raise FileNotFoundError(
            f"No per-video JSON files found in {job_dir}. "
            "Expected subdirectories each containing <subdir>/<subdir>.json."
        )

    by_name, by_stem = build_video_lookup(REPO_ROOT)

    all_rows: list[dict[str, Any]] = []
    skipped_missing_fps = 0
    for json_path in json_paths:
        video_json = json.loads(json_path.read_text(encoding="utf-8"))

        fps_value = float(video_json.get("target_fps") or video_json.get("video_fps") or 0.0)
        if fps_value <= 0:
            skipped_missing_fps += 1
            continue

        video_path = resolve_video_path(video_json=video_json, json_path=json_path, by_name=by_name, by_stem=by_stem)
        creation_dt = ffprobe_creation_time(video_path) if video_path is not None else None

        all_rows.extend(
            build_rows_for_video(
                video_json=video_json,
                json_path=json_path,
                interval_seconds=args.interval_seconds,
                window_seconds=args.window_seconds,
                creation_dt=creation_dt,
                apply_filters=not args.no_filter,
                min_track_frames=args.min_track_frames,
                max_static_displacement=args.max_static_displacement,
                min_bbox_area_frac=args.min_bbox_area_frac,
                iou_dedup_threshold=args.iou_dedup_threshold,
                min_confidence=args.min_confidence,
            )
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Wrote {len(all_rows)} rows to {args.output_csv}")
    if skipped_missing_fps:
        print(f"Skipped {skipped_missing_fps} videos because FPS was unavailable.")


if __name__ == "__main__":
    main()
