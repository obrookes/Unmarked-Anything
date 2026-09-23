#!/usr/bin/env python3
"""Export per-object interval-sampled distances from camera-trap job outputs (JSON) to CSV.

Distances are computed from `depth_mask_mean` averaged over a forward window at each
sample point on a global time grid. All frame/second conversions use the video's native
fps (`video_fps` from the JSON) since `frame_index` is a native frame index regardless of
any `target_fps` subsampling used at inference time.

Optional post-processing:
- `--calibrated-objects CSV`: look up per-(video, frame, track) calibrated distances from an
  `apply.py`-produced `calibrated_objects.csv` (Haucke et al. 2022 method). For each sampled
  frame, the row is matched at the exact grid frame (`sampled_idx`), since calibrated-objects
  rows only exist on the DA3 2s-grid. `distance` becomes the row's `distance_m`, `calib_method`
  and `raw_depth_at_point` are taken from the row, and `raw_distance` remains the raw
  `depth_mask_mean` window-averaged value. Sampled rows with no matching calibrated-objects row,
  or with `calib_method == "failed"`, are dropped (see `--dropped-log`).
- `--track-qc PATH`: drop tracks flagged `keep=False` in a track_qc.csv (columns
  `video_name, track_id, n_checked, n_reject, reject_frac, issues, keep`), matched by
  video name (extension-insensitive) and track id.
- `--video-dir PATH`: override the directory searched for source videos (for ffprobe
  creation_time lookups) in place of `assets/videos`, e.g. when paths moved on a cluster.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
from datetime import datetime, timedelta
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
    "transect_cam",
    "raw_distance",
    "raw_depth_at_point",
    "calib_method",
    "detection_datetime",
]

# Patterns to extract a normalised "<transect>_cam<NNN>" identifier from a video/filename.
# Ported from wcf-pps-p3/notebooks/03_reconcile_annotations.ipynb (cell 9), with the second
# pattern's spurious leading "-" dropped so 'T44-Cam_077' (no leading hyphen before the T)
# matches as documented.
TRANSECT_CAM_PATTERNS = [
    r"(\w+)_Cam(\d+)",
    r"T_?(\d+)-Cam_?(\d+)",
    r"T(\d+)-\d+cam_?(\d+)",
]


def extract_transect_cam(filename: str) -> str | None:
    """Extract and normalise a transect/cam identifier from a filename.

    Normalises the cam number to a minimum of 3 digits. Tries, in order:
    - '{transect}_Cam{number}'   e.g. '171_Cam0122'       -> '171_cam122'
    - 'T{id}-Cam_{number}'       e.g. 'T44-Cam_077'        -> '44_cam077'
    - 'T{id}-{id}cam_{number}'   e.g. 'T17-17cam_82'       -> '17_cam082'
    """
    for pattern in TRANSECT_CAM_PATTERNS:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            transect = match.group(1).lstrip("T")
            cam_normalised = match.group(2).lstrip("0").zfill(3)
            return f"{transect}_cam{cam_normalised}"
    return None


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
        default=2.0,
        help="Sampling interval in seconds; sample points are aligned to the global grid 0, interval, 2*interval, ... (default: 2.0, matching the DA3 depth grid).",
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
    parser.add_argument(
        "--calibrated-objects",
        type=Path,
        default=None,
        help=(
            "Path to a calibrated_objects.csv produced by apps/camera_trap/calibration/apply.py "
            "(columns: video_name, frame_index, track_id, transect_cam, distance_m, "
            "raw_depth_at_point, calib_method, align_inlier_frac, homography_used). When given, "
            "'distance' and 'calib_method' come from the matching row at the sample's grid "
            "frame, 'raw_depth_at_point' is also emitted, and 'raw_distance' remains the raw "
            "depth_mask_mean window average. Sampled rows with no matching calibrated-objects "
            "row, or with calib_method == 'failed', are dropped."
        ),
    )
    parser.add_argument(
        "--track-qc",
        type=Path,
        default=None,
        help=(
            "Path to a track_qc.csv (columns: video_name, track_id, n_checked, n_reject, "
            "reject_frac, issues, keep) identifying manually-reviewed tracks to drop "
            "(keep == False)."
        ),
    )
    parser.add_argument(
        "--dropped-log",
        type=Path,
        default=None,
        help="Optional CSV path to write dropped (video_name, track_id, reason) rows for --track-qc and --calibrated-objects drops.",
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=None,
        help="Directory to search for source videos (for ffprobe creation_time) instead of assets/videos.",
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


def build_video_lookup(repo_root: Path, video_dir: Path | None = None) -> tuple[dict[str, Path], dict[str, Path]]:
    videos_dir = video_dir if video_dir is not None else repo_root / "assets" / "videos"
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


def normalize_video_name(name: str) -> str:
    """Normalise a video name for matching against track_qc.csv, ignoring extension/case."""
    return Path(str(name)).stem.lower()


def load_calibrated_objects(path: Path) -> dict[tuple[str, int, str], dict[str, Any]]:
    """Load a calibration/apply.py calibrated_objects.csv into a lookup keyed by
    (normalized_video_name, frame_index, track_id_str) -> row dict."""
    lookup: dict[tuple[str, int, str], dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            video_name = row.get("video_name")
            frame_index = row.get("frame_index")
            track_id = row.get("track_id")
            if not video_name or frame_index is None or track_id is None:
                continue
            try:
                frame_idx_int = int(frame_index)
            except ValueError:
                continue
            lookup[(normalize_video_name(video_name), frame_idx_int, str(track_id))] = row
    return lookup


def load_track_qc(path: Path) -> dict[tuple[str, str], bool]:
    """Load track_qc.csv into a {(normalized_video_name, track_id_str): keep} map."""
    keep_map: dict[tuple[str, str], bool] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            video_name = row.get("video_name")
            track_id = row.get("track_id")
            if not video_name or track_id is None:
                continue
            keep_raw = str(row.get("keep", "")).strip().lower()
            keep = keep_raw in ("true", "1", "yes")
            keep_map[(normalize_video_name(video_name), str(track_id))] = keep
    return keep_map


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
    calibrated_objects: dict[tuple[str, int, str], dict[str, Any]] | None = None,
    track_qc: dict[tuple[str, str], bool] | None = None,
    dropped_log: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    video_name = str(video_json.get("video_name") or json_path.stem)
    # frame_index is a native video frame index regardless of any target_fps subsampling
    # used at inference time, so all frame/second conversions must use video_fps.
    fps_value = float(video_json.get("video_fps") or video_json.get("target_fps") or 0.0)
    if fps_value <= 0:
        return rows

    transect_cam = extract_transect_cam(video_name)
    norm_video_name_for_calib = normalize_video_name(video_name)

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

    # Step C: external manual track QC (runs regardless of apply_filters)
    if track_qc:
        norm_video_name = normalize_video_name(video_name)
        for track_id in track_samples:
            if track_id in excluded_tracks:
                continue
            keep = track_qc.get((norm_video_name, str(track_id)))
            if keep is False:
                excluded_tracks.add(track_id)
                if dropped_log is not None:
                    dropped_log.append(
                        {"video_name": video_name, "track_id": track_id, "reason": "track_qc_reject"}
                    )

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
            window_raw_distances: list[float] = []
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
                # Depth: always skip missing or physically impossible values (checked on raw depth)
                distance = entry.get("depth_mask_mean")
                if distance is None or float(distance) < 0.0:
                    continue
                raw_val = float(distance)
                window_raw_distances.append(raw_val)
                conf = entry.get("confidence")
                if conf is not None:
                    window_confidences.append(float(conf))

            if window_raw_distances:
                raw_distance = sum(window_raw_distances) / len(window_raw_distances)
                distance = raw_distance
                calib_method = ""
                raw_depth_at_point: Any = ""

                if calibrated_objects is not None:
                    # The calibrated-objects lookup is always at the exact grid frame
                    # (sampled_idx), independent of --window-seconds.
                    calib_row = calibrated_objects.get(
                        (norm_video_name_for_calib, sampled_idx, str(track_id))
                    )
                    if calib_row is None:
                        if dropped_log is not None:
                            dropped_log.append(
                                {"video_name": video_name, "track_id": track_id, "reason": "calib_missing"}
                            )
                        sampled_idx += step_frames
                        continue
                    if calib_row.get("calib_method") == "failed":
                        if dropped_log is not None:
                            dropped_log.append(
                                {"video_name": video_name, "track_id": track_id, "reason": "calib_failed"}
                            )
                        sampled_idx += step_frames
                        continue
                    distance = float(calib_row["distance_m"])
                    calib_method = calib_row.get("calib_method") or ""
                    raw_depth_at_point = calib_row.get("raw_depth_at_point", "")

                second = round(sampled_idx / fps_value, 6)
                detection_datetime = ""
                if creation_dt is not None:
                    detection_datetime = (creation_dt + timedelta(seconds=second)).isoformat()
                rows.append(
                    {
                        "video_name": video_name,
                        "starting_date": starting_date,
                        "year": year,
                        "month": month,
                        "day": day,
                        "hour": hour,
                        "minute": minute,
                        "second": second,
                        "ind_no": track_id,
                        "distance": distance,
                        "confidence": sum(window_confidences) / len(window_confidences) if window_confidences else "",
                        "transect_cam": transect_cam or "",
                        "raw_distance": raw_distance,
                        "raw_depth_at_point": raw_depth_at_point,
                        "calib_method": calib_method,
                        "detection_datetime": detection_datetime,
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

    by_name, by_stem = build_video_lookup(REPO_ROOT, video_dir=args.video_dir)

    calibrated_objects: dict[tuple[str, int, str], dict[str, Any]] | None = None
    if args.calibrated_objects is not None:
        calibrated_objects = load_calibrated_objects(args.calibrated_objects)

    track_qc: dict[tuple[str, str], bool] | None = None
    if args.track_qc is not None:
        track_qc = load_track_qc(args.track_qc)

    # Always collected (for the drop-count summary); only written to disk if --dropped-log is given.
    dropped_log: list[dict[str, Any]] | None = (
        [] if (track_qc is not None or calibrated_objects is not None) else None
    )

    all_rows: list[dict[str, Any]] = []
    skipped_missing_fps = 0
    missing_creation_dt = 0
    for json_path in json_paths:
        video_json = json.loads(json_path.read_text(encoding="utf-8"))

        fps_value = float(video_json.get("video_fps") or video_json.get("target_fps") or 0.0)
        if fps_value <= 0:
            skipped_missing_fps += 1
            continue

        video_path = resolve_video_path(video_json=video_json, json_path=json_path, by_name=by_name, by_stem=by_stem)
        creation_dt = ffprobe_creation_time(video_path) if video_path is not None else None
        if creation_dt is None:
            missing_creation_dt += 1

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
                calibrated_objects=calibrated_objects,
                track_qc=track_qc,
                dropped_log=dropped_log,
            )
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)

    if args.dropped_log is not None:
        args.dropped_log.parent.mkdir(parents=True, exist_ok=True)
        with args.dropped_log.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["video_name", "track_id", "reason"])
            writer.writeheader()
            writer.writerows(dropped_log or [])

    print(f"Wrote {len(all_rows)} rows to {args.output_csv}")
    if skipped_missing_fps:
        print(f"Skipped {skipped_missing_fps} videos because FPS was unavailable.")
    if missing_creation_dt:
        print(f"Warning: {missing_creation_dt} videos had no ffprobe creation_time; date/time columns left empty.")
    if track_qc is not None:
        dropped_tracks = len(dropped_log) if dropped_log is not None else "?"
        print(f"track-qc: dropped {dropped_tracks} track(s) flagged keep=False.")
    if calibrated_objects is not None:
        dropped_log_rows = dropped_log or []
        dropped_failed = sum(1 for row in dropped_log_rows if row["reason"] == "calib_failed")
        dropped_missing = sum(1 for row in dropped_log_rows if row["reason"] == "calib_missing")
        print(f"dropped {dropped_failed} row(s): failed calib_method")
        print(f"dropped {dropped_missing} row(s): no calibrated-objects match")


if __name__ == "__main__":
    main()
