#!/usr/bin/env python3
"""Read distance-board numbers from PSS P3 reference videos to build DA3 depth calibration points.

Field teams recorded, per camera, reference videos of a person walking out holding a board
showing their distance (in metres) from the camera. Those videos are run through dap3_cli.py
with `--sam3-text-prompts "person holding sign" --npz` (same DA3 model/mode as the main run),
producing per-video `<job-dir>/<stem>/<stem>.json` + `<stem>_arrays.npz` (see
apps/camera_trap/qc/mask_verify.py, whose discovery/video-resolution/track-sampling helpers this
script reuses). This script samples frames per tracked person, asks a local VLM to read the
board number in each, and writes `calib_points.csv` for apps/camera_trap/calibration/fit.py.

Camera identification: each reference video's transect/cam ("transect_cam") is looked up from
--reference-map (a CSV of transect, cam, transect_cam, reference_dir, status, ...; see
configs/pss_p3/reference_video_map.csv), by matching the video's resolved path against
`--reference-root / reference_dir` (longest directory-prefix match, falling back to an
unambiguous match on the leaf directory name). Rows with status "missing" are skipped. An
optional --video-cam-csv (columns: video, transect_cam) overrides the map for individual videos
by stem.

Outputs (in --out-dir):
    calib_points.csv     transect_cam, video, frame_idx, track_id, depth_mask_mean,
                          board_distance_m, vlm_conf, vlm_distance_raw, legible, outlier,
                          raw_text -- rebuilt from the full read_boards.jsonl on every run.
    read_boards.jsonl     one line per (track, frame) request: video_name, transect_cam,
                           track_id, frame_index, depth_mask_mean, and either "verdict" (the raw
                           VLM JSON) or "error".
    unmapped_videos.csv   videos whose transect_cam could not be determined.
    contact_sheet.html    (with --contact-sheet N) N sampled readings per camera, for spot-checks.

    python apps/camera_trap/calibration/read_boards.py --job-dir hpc/runs/ref_job --out-dir hpc/runs/ref_job/calib \
        --reference-map configs/pss_p3/reference_video_map.csv --reference-root /path/to/wcf-pps-p3
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apps.camera_trap.qc.mask_verify import (
    discover_video_jsons,
    group_tracks,
    load_done_videos,
    resolve_video_path,
    sample_track_frames,
)
from apps.camera_trap.vlm import render
from apps.camera_trap.vlm.engine import VLMEngine, VLMRequest
from apps.camera_trap.vlm.frames import iter_frames_at_indices

# Board may be held out away from the body, so dilate generously before cropping.
CROP_PAD_FRAC = 0.6
CROP_TARGET_SIDE = 768
FULL_FRAME_MAX_SIDE = 1024

CONF_MAP = {"high": 1.0, "medium": 0.6, "low": 0.3}

# Static prompt text first, no variable parts, so every request in a run shares one prefix
# (prefix caching). Camera-trap specific: IR/night frames are often greyscale.
BOARD_PROMPT = (
    "You are reading a distance-calibration board in a camera-trap reference video. A person "
    "walks out in front of the camera holding a board or sign showing a number: the person's "
    "distance from the camera, in metres. You are given two images of the same frame: the FIRST "
    "is the full raw frame for context, the SECOND is a zoomed-in crop around that person so the "
    "board text is legible. Frames may be greyscale/monochrome night-vision (IR) footage; that "
    "alone is not a defect. "
    'Set "board_visible" to true if the person is holding a board/sign in the crop. Set '
    '"legible" to true only if you can read the number on it with confidence. Read the number '
    "exactly as written; it is a distance in metres, typically between 0 and 50. If there is no "
    'board, or the number cannot be read, set "distance_m" to null and "legible" to false. '
    'Set "confidence" to your confidence in the reading: "high", "medium", or "low". Set '
    '"raw_text" to the exact characters you read on the board (or empty string if none). '
    "Respond ONLY with a JSON object matching this schema: "
    '{"board_visible": bool, "distance_m": number|null, "legible": bool, '
    '"confidence": "high|medium|low", "raw_text": "<=40 chars"}'
)

BOARD_SCHEMA = {
    "type": "object",
    "properties": {
        "board_visible": {"type": "boolean"},
        "distance_m": {"type": ["number", "null"], "minimum": 0, "maximum": 50},
        "legible": {"type": "boolean"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "raw_text": {"type": "string", "maxLength": 40},
    },
    "required": ["board_visible", "distance_m", "legible", "confidence", "raw_text"],
    "additionalProperties": False,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--job-dir", type=Path, required=True, help="dap3 output dir for the reference videos.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Where calib_points.csv / read_boards.jsonl are written.")
    parser.add_argument("--reference-map", type=Path, required=True,
                         help="CSV with transect, cam, transect_cam, reference_dir, status columns (see configs/pss_p3/reference_video_map.csv).")
    parser.add_argument("--reference-root", type=Path, required=True,
                         help="Root directory that each row's reference_dir is relative to.")
    parser.add_argument("--video-dir", type=Path, default=None, help="Optional dir to resolve moved video files by stem.")
    parser.add_argument("--video-cam-csv", type=Path, default=None,
                         help="Optional CSV with 'video' (stem), 'transect_cam' columns overriding the reference-map lookup.")
    parser.add_argument("--model", default="qwen", help="VLM alias ('qwen') or HF model id (default: qwen).")
    parser.add_argument("--tp", type=int, default=None, help="VLM tensor_parallel_size override.")
    parser.add_argument("--max-per-track", type=int, default=20, help="Max sampled frames per track (default: 20).")
    parser.add_argument("--min-sam-conf", type=float, default=0.0, help="Skip person detections below this SAM3 confidence (default: 0.0).")
    parser.add_argument("--dry-run", action="store_true", help="Write crop images only; no VLM calls, no jsonl/csv output.")
    parser.add_argument("--overwrite", action="store_true", help="Reprocess videos even if already in read_boards.jsonl.")
    parser.add_argument("--contact-sheet", type=int, default=0, help="Write up to N sampled readings per camera to contact_sheet.html (default: 0, off).")
    parser.add_argument("--max-videos", type=int, default=None, help="Cap the number of videos processed this run.")
    args = parser.parse_args(argv)
    if args.max_per_track <= 0:
        parser.error("--max-per-track must be > 0.")
    if args.contact_sheet < 0:
        parser.error("--contact-sheet must be >= 0.")
    return args


# --------------------------------------------------------------------------------------
# camera mapping
# --------------------------------------------------------------------------------------

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
    with open(path, newline="") as f:
        return {row["video"]: row["transect_cam"] for row in csv.DictReader(f)}


def write_unmapped_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video_name", "video_path", "reason"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# --------------------------------------------------------------------------------------
# per-video processing
# --------------------------------------------------------------------------------------

def filter_finite_depth_tracks(tracks: dict[Any, dict[int, dict[str, Any]]], min_sam_conf: float) -> dict[Any, dict[int, dict[str, Any]]]:
    """Drop objects without a finite depth_mask_mean or below --min-sam-conf confidence."""
    filtered: dict[Any, dict[int, dict[str, Any]]] = {}
    for track_id, frame_map in tracks.items():
        good = {}
        for frame_idx, obj in frame_map.items():
            depth = obj.get("depth_mask_mean")
            try:
                depth = float(depth)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(depth):
                continue
            conf = obj.get("confidence")
            if conf is not None and float(conf) < min_sam_conf:
                continue
            good[frame_idx] = obj
        if good:
            filtered[track_id] = good
    return filtered


def resize_long_side(img: np.ndarray, target: int) -> np.ndarray:
    """Resize `img` so its longer side is `target` (upscales as well as downscales)."""
    h, w = img.shape[:2]
    if max(h, w) == 0:
        return img
    scale = target / max(h, w)
    interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=interp)


def build_request_images(frame_bgr: np.ndarray, bbox_xyxy) -> list[np.ndarray]:
    full_ds = render.downscale_max_side(frame_bgr, max_side=FULL_FRAME_MAX_SIDE)
    crop = render.crop_around(frame_bgr, bbox_xyxy, pad_frac=CROP_PAD_FRAC)
    crop = resize_long_side(crop, CROP_TARGET_SIDE)
    return [full_ds, crop]


def process_video(
    json_path: Path,
    video_json: dict[str, Any],
    video_path: Path,
    transect_cam: str,
    args: argparse.Namespace,
    engine,
    contact_counts: dict[str, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (read_boards.jsonl records, contact-sheet entries) for one video."""
    video_name = str(video_json.get("video_name") or json_path.stem)
    tracks = filter_finite_depth_tracks(group_tracks(video_json), args.min_sam_conf)
    if not tracks:
        return [], []

    per_track_frames = {
        tid: sample_track_frames(list(frame_map.keys()), args.max_per_track) for tid, frame_map in tracks.items()
    }
    wanted_frames = sorted({fi for fis in per_track_frames.values() for fi in fis})

    decoded_frames: dict[int, np.ndarray] = {}
    for frame_idx, frame_bgr in iter_frames_at_indices(video_path, wanted_frames):
        decoded_frames[frame_idx] = frame_bgr

    requests: list[VLMRequest | None] = []
    meta: list[tuple[Any, int, float]] = []  # track_id, frame_idx, depth_mask_mean
    dry_run_images: list[tuple[Any, int, list[np.ndarray]]] = []
    for track_id, frame_map in per_track_frames.items():
        for frame_idx in frame_map:
            obj = tracks[track_id][frame_idx]
            depth = float(obj["depth_mask_mean"])
            frame_bgr = decoded_frames.get(frame_idx)
            if frame_bgr is None or obj.get("bbox_xyxy") is None:
                requests.append(None)
                meta.append((track_id, frame_idx, depth))
                continue
            images = build_request_images(frame_bgr, obj["bbox_xyxy"])
            if args.dry_run:
                dry_run_images.append((track_id, frame_idx, images))
                requests.append(None)
                meta.append((track_id, frame_idx, depth))
                continue
            requests.append(VLMRequest(images=images, prompt=BOARD_PROMPT, color="bgr"))
            meta.append((track_id, frame_idx, depth))

    if args.dry_run:
        out_dir = args.out_dir / "dry_run" / video_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for track_id, frame_idx, images in dry_run_images:
            cv2.imwrite(str(out_dir / f"track{track_id}_f{frame_idx}_full.jpg"), images[0])
            cv2.imwrite(str(out_dir / f"track{track_id}_f{frame_idx}_crop.jpg"), images[1])
        print(f"{video_name}: wrote {len(dry_run_images)} dry-run image pairs to {out_dir}")
        return [], []

    real_indices = [i for i, r in enumerate(requests) if r is not None]
    real_requests = [requests[i] for i in real_indices]
    results = engine.run_json(real_requests, schema=BOARD_SCHEMA, max_tokens=256) if real_requests else []
    errors = list(engine.last_errors) if real_requests else []
    result_by_index: dict[int, tuple[dict | None, str | None]] = {}
    for pos, i in enumerate(real_indices):
        result_by_index[i] = (results[pos], errors[pos])

    jsonl_records: list[dict[str, Any]] = []
    contact_entries: list[dict[str, Any]] = []
    for i, (track_id, frame_idx, depth) in enumerate(meta):
        rec: dict[str, Any] = {
            "video_name": video_name,
            "transect_cam": transect_cam,
            "track_id": track_id,
            "frame_index": frame_idx,
            "depth_mask_mean": depth,
        }
        if i not in result_by_index:
            rec["error"] = "frame decode failed or missing bbox_xyxy"
        else:
            verdict, error = result_by_index[i]
            if verdict is not None:
                rec["verdict"] = verdict
            else:
                rec["error"] = error or "vlm call failed"
        jsonl_records.append(rec)

        if args.contact_sheet and "verdict" in rec and contact_counts[transect_cam] < args.contact_sheet:
            obj = tracks[track_id].get(frame_idx)
            frame_bgr = decoded_frames.get(frame_idx)
            if frame_bgr is not None and obj is not None and obj.get("bbox_xyxy") is not None:
                crop = render.crop_around(frame_bgr, obj["bbox_xyxy"], pad_frac=CROP_PAD_FRAC)
                img_dir = args.out_dir / "contact_sheet_imgs"
                img_dir.mkdir(parents=True, exist_ok=True)
                img_name = f"{transect_cam}_{video_name}_t{track_id}_f{frame_idx}.jpg"
                cv2.imwrite(str(img_dir / img_name), crop)
                contact_counts[transect_cam] += 1
                contact_entries.append(
                    {
                        "transect_cam": transect_cam,
                        "video_name": video_name,
                        "track_id": track_id,
                        "frame_index": frame_idx,
                        "depth_mask_mean": depth,
                        "img_path": f"contact_sheet_imgs/{img_name}",
                        **rec["verdict"],
                    }
                )

    return jsonl_records, contact_entries


# --------------------------------------------------------------------------------------
# calib_points.csv (rebuilt from read_boards.jsonl on every run)
# --------------------------------------------------------------------------------------

def build_calib_row(rec: dict[str, Any]) -> dict[str, Any]:
    depth = rec.get("depth_mask_mean")
    depth = float(depth) if depth is not None else float("nan")
    base = {
        "transect_cam": rec["transect_cam"],
        "video": rec["video_name"],
        "frame_idx": rec["frame_index"],
        "track_id": rec["track_id"],
        "depth_mask_mean": depth,
    }
    verdict = rec.get("verdict")
    if verdict is None:
        return {
            **base,
            "board_distance_m": float("nan"),
            "vlm_conf": float("nan"),
            "vlm_distance_raw": float("nan"),
            "legible": False,
            "outlier": False,
            "raw_text": "",
        }
    distance_m = verdict.get("distance_m")
    reading = float(distance_m) if distance_m is not None else float("nan")
    legible = bool(verdict.get("legible", False))
    board_visible = bool(verdict.get("board_visible", False))
    vlm_conf = CONF_MAP.get(verdict.get("confidence"), float("nan"))
    board_distance_m = reading if (legible and board_visible and not math.isnan(reading)) else float("nan")
    return {
        **base,
        "board_distance_m": board_distance_m,
        "vlm_conf": vlm_conf,
        "vlm_distance_raw": reading,
        "legible": legible,
        "outlier": False,
        "raw_text": str(verdict.get("raw_text", ""))[:40],
    }


def flag_outliers(rows: list[dict[str, Any]]) -> None:
    """A person stands still at each marker, so within a track, a reading that differs from
    both its (temporally) nearest valid neighbours by more than 3 m is likely a misread; keep the
    row but null out board_distance_m (raw reading stays in vlm_distance_raw). Rows sorted by
    frame_idx are expected; the first/last valid reading in a track is never flagged (no second
    neighbour to compare against)."""
    valid = [i for i, r in enumerate(rows) if not math.isnan(r["board_distance_m"])]
    for pos, i in enumerate(valid):
        if pos == 0 or pos == len(valid) - 1:
            continue
        cur = rows[i]["board_distance_m"]
        prev_v = rows[valid[pos - 1]]["board_distance_m"]
        next_v = rows[valid[pos + 1]]["board_distance_m"]
        if abs(cur - prev_v) > 3.0 and abs(cur - next_v) > 3.0:
            rows[i]["outlier"] = True
            rows[i]["board_distance_m"] = float("nan")


CALIB_CSV_COLUMNS = [
    "transect_cam", "video", "frame_idx", "track_id", "depth_mask_mean",
    "board_distance_m", "vlm_conf", "vlm_distance_raw", "legible", "outlier", "raw_text",
]


def rebuild_calib_csv(jsonl_path: Path, csv_path: Path) -> int:
    per_track: dict[tuple[str, Any], list[dict[str, Any]]] = defaultdict(list)
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            per_track[(rec["video_name"], rec["track_id"])].append(rec)

    all_rows: list[dict[str, Any]] = []
    for key in sorted(per_track, key=lambda k: (str(k[0]), str(k[1]))):
        recs = sorted(per_track[key], key=lambda r: r["frame_index"])
        rows = [build_calib_row(r) for r in recs]
        flag_outliers(rows)
        all_rows.extend(rows)

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CALIB_CSV_COLUMNS)
        writer.writeheader()
        for row in all_rows:
            out = dict(row)
            for k in ("depth_mask_mean", "board_distance_m", "vlm_conf", "vlm_distance_raw"):
                if isinstance(out[k], float) and math.isnan(out[k]):
                    out[k] = ""
            writer.writerow(out)
    return len(all_rows)


def write_contact_sheet(out_dir: Path, entries: list[dict[str, Any]]) -> None:
    entries = sorted(entries, key=lambda e: (str(e["transect_cam"]), e["video_name"], str(e["track_id"]), e["frame_index"]))
    rows_html = []
    for e in entries:
        rows_html.append(
            "<tr>"
            f"<td>{e['transect_cam']}</td><td>{e['video_name']}</td><td>{e['track_id']}</td>"
            f"<td>{e['frame_index']}</td><td><img src=\"{e['img_path']}\" height=\"200\"></td>"
            f"<td>{e.get('distance_m')}</td><td>{round(float(e['depth_mask_mean']), 3)}</td>"
            f"<td>{e.get('confidence')}</td><td>{e.get('legible')}</td><td>{e.get('raw_text', '')}</td>"
            "</tr>"
        )
    html = (
        "<html><head><meta charset='utf-8'><title>Board reading contact sheet</title>"
        "<style>table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:4px;font-family:sans-serif;font-size:13px}</style>"
        "</head><body><table><tr><th>transect_cam</th><th>video</th><th>track</th><th>frame</th>"
        "<th>crop</th><th>vlm distance_m</th><th>depth_mask_mean</th><th>confidence</th>"
        "<th>legible</th><th>raw_text</th></tr>" + "".join(rows_html) + "</table></body></html>"
    )
    (out_dir / "contact_sheet.html").write_text(html)


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.out_dir / "read_boards.jsonl"

    if args.overwrite and jsonl_path.exists():
        jsonl_path.unlink()

    ref_rows = load_reference_map(args.reference_map)
    ref_index = build_reference_index(ref_rows, args.reference_root)
    video_cam_override = load_video_cam_csv(args.video_cam_csv) if args.video_cam_csv else {}

    all_jsons = discover_video_jsons(args.job_dir)
    done = load_done_videos(jsonl_path) if not args.dry_run else set()

    todo = []
    for json_path in all_jsons:
        video_json = json.loads(json_path.read_text())
        video_name = str(video_json.get("video_name") or json_path.stem)
        if video_name in done:
            continue
        todo.append((json_path, video_json, video_name))
    if args.max_videos is not None:
        todo = todo[: args.max_videos]

    print(f"{len(all_jsons)} videos found, {len(done)} already done, {len(todo)} to do")

    engine = None
    if not args.dry_run:
        engine = VLMEngine(model_id=args.model, tensor_parallel_size=args.tp)

    contact_counts: dict[str, int] = defaultdict(int)
    contact_entries: list[dict[str, Any]] = []
    unmapped_rows: list[dict[str, Any]] = []
    n_processed = 0

    for json_path, video_json, video_name in todo:
        video_path = resolve_video_path(video_json, json_path, args.video_dir)
        if video_path is None:
            unmapped_rows.append({"video_name": video_name, "video_path": video_json.get("video_path", ""), "reason": "video_not_found"})
            continue

        transect_cam = video_cam_override.get(video_path.stem) or video_cam_override.get(video_name)
        if transect_cam is None:
            transect_cam, reason = match_transect_cam(video_path, ref_index, ref_rows)
        else:
            reason = "override"
        if transect_cam is None:
            unmapped_rows.append({"video_name": video_name, "video_path": str(video_path), "reason": reason})
            continue

        try:
            jsonl_records, entries = process_video(json_path, video_json, video_path, transect_cam, args, engine, contact_counts)
        except FileNotFoundError as e:
            print(f"skipping {json_path}: {e}", file=sys.stderr)
            continue

        if args.dry_run:
            continue

        with jsonl_path.open("a") as f:
            for rec in jsonl_records:
                f.write(json.dumps(rec) + "\n")
        contact_entries.extend(entries)
        n_processed += 1
        print(f"{video_name} ({transect_cam}): {len(jsonl_records)} readings written")

    if unmapped_rows:
        write_unmapped_csv(args.out_dir / "unmapped_videos.csv", unmapped_rows)
        print(f"{len(unmapped_rows)} unmapped videos logged to {args.out_dir / 'unmapped_videos.csv'}")

    if args.dry_run:
        return 0

    n_points = rebuild_calib_csv(jsonl_path, args.out_dir / "calib_points.csv") if jsonl_path.exists() else 0
    print(f"wrote {n_points} calibration points to {args.out_dir / 'calib_points.csv'}")

    if args.contact_sheet and contact_entries:
        write_contact_sheet(args.out_dir, contact_entries)
        print(f"wrote contact sheet to {args.out_dir / 'contact_sheet.html'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
