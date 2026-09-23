"""PSS P3 reference-video map lookups.

Ported from calibration/read_boards.py's camera-mapping logic (load_reference_map,
build_reference_index, match_transect_cam, the video-cam override CSV, and its "missing"/
"relabelled" status handling), so it can be shared by other calibration tooling.

find_sign_frames.py discovers its input videos directly from the map (`discover_reference_videos`)
rather than by resolving already-processed dap3 video paths, since it runs on the raw reference
videos, not on dap3 output. `match_transect_cam` / `build_reference_index` remain here for anything
that instead needs to map an arbitrary (possibly moved) video path back to a transect_cam.

Map CSV columns: transect, cam, transect_cam, mission, n_distance_annotations, reference_dir,
n_reference_clips, status, note (see configs/pss_p3/reference_video_map.csv). `reference_dir` is
relative to a `--reference-root`. Rows with status "missing" are dropped; a "relabelled" row's
transect_cam is authoritative (not derivable from the folder path).
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

VIDEO_EXTS = (".mp4", ".mov")


def load_reference_map(path: Path) -> list[dict[str, Any]]:
    """Rows of --reference-map, dropping status == 'missing' (no video ever synced for those)."""
    with open(path, newline="") as f:
        return [row for row in csv.DictReader(f) if row.get("status") != "missing"]


def build_reference_index(rows: list[dict[str, Any]], reference_root: Path) -> list[tuple[str, str]]:
    """(normalised absolute reference_dir, transect_cam), longest dir first for prefix matching."""
    entries = [(os.path.normpath(str(reference_root / row["reference_dir"])), row["transect_cam"]) for row in rows]
    entries.sort(key=lambda e: -len(e[0]))
    return entries


def match_transect_cam(video_path: Path, ref_index: list[tuple[str, str]], ref_rows: list[dict[str, Any]]) -> tuple[str | None, str]:
    """Longest-prefix match of the video's directory against reference_root/reference_dir,
    falling back to an unambiguous match on the leaf directory name. Returns (transect_cam,
    method) with transect_cam None and method a reason string when nothing matched."""
    video_dir = os.path.normpath(str(video_path.parent))
    for abs_dir, cam in ref_index:
        if video_dir == abs_dir or video_dir.startswith(abs_dir + os.sep):
            return cam, "prefix"
    base = video_path.parent.name
    candidates = {row["transect_cam"] for row in ref_rows if Path(row["reference_dir"]).name == base}
    if len(candidates) == 1:
        return next(iter(candidates)), "basename"
    return None, "ambiguous_basename" if len(candidates) > 1 else "no_match"


def load_video_cam_csv(path: Path) -> dict[str, str]:
    """Optional CSV (columns: video, transect_cam) overriding the reference-map lookup by stem."""
    with open(path, newline="") as f:
        return {row["video"]: row["transect_cam"] for row in csv.DictReader(f)}


def discover_reference_videos(ref_rows: list[dict[str, Any]], reference_root: Path) -> list[tuple[Path, str]]:
    """Walk every non-missing map row's reference_dir for video files that sit directly in it
    (.MP4/.MOV, case-insensitive; not recursive), returning (video_path, transect_cam) pairs.

    transect_cam always comes straight from the row -- including for a "relabelled" row, whose
    transect_cam is the corrected id, not derivable from the folder path -- never re-derived by
    matching the path, so relabelled rows are authoritative. Sorted by video path for a stable,
    order-independent scan/shard order.
    """
    videos: list[tuple[Path, str]] = []
    for row in ref_rows:
        ref_dir = reference_root / row["reference_dir"]
        if not ref_dir.is_dir():
            continue
        for p in ref_dir.iterdir():
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
                videos.append((p, row["transect_cam"]))
    videos.sort(key=lambda vc: str(vc[0]))
    return videos
