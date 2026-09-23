#!/usr/bin/env python3
"""Find distance-calibration sign frames in PSS P3 reference videos, for DA3 depth calibration.

Field teams recorded, per camera, reference videos of a person walking out and holding up a
board/sign showing their distance (in metres) from the camera (Haucke et al. / timmh's
distance-estimation calibration protocol). This script scans each reference video at a low frame
rate, asks a local VLM whether each sampled frame shows a person holding a legible sign and what
number it reads, groups agreeing consecutive readings into "calibration events" (one per
sign-holding instance), and writes one representative frame per event to `calibration_frames.csv`.
Downstream, SAM3 ("person") + DA3 depth are run on those frames (not this script's job) to
produce the actual depth-vs-board-distance calibration points.

Camera identification: videos are discovered by walking --reference-map's non-missing rows'
reference_dir folders directly (see calibration/refmap.py) -- no dap3 job-dir dependency, and no
flattening of the folder structure. Each row's transect_cam is authoritative for videos found
under it.

Outputs (in --out-dir):
    scan.jsonl / scan.shard{i}.jsonl
                            One record per sampled frame's final (possibly tile-fallback-merged)
                            reading: video_path, transect_cam, fps, frame_idx, person_with_sign,
                            distance_m, legible, confidence, sign_box, source ("vlm_whole" or
                            "vlm_tile"), or "error". Sharded by --shard-index/--num-shards; resume
                            skips videos already present in the shard's own file (unless
                            --overwrite).
    calibration_frames.csv Rebuilt from all scan*.jsonl (and --override-csv, if given) every run.
                            Columns: transect_cam, video_path, event_id, frame_idx, distance_m,
                            sign_box_norm ("x0;y0;x1;y1"), n_agree, vlm_conf, source ("vlm" or
                            "override").
    events_summary.csv     Per camera: transect_cam, n_events, distances, n_videos.
    no_event_log.csv       Videos that were scanned but produced zero events.
    contact_sheet.html + contact_sheet_imgs/*.jpg
                            One image per event: the representative frame with its sign box drawn,
                            annotated with the reading / n_agree / confidence.

    python apps/camera_trap/calibration/find_sign_frames.py \\
        --reference-map configs/pss_p3/reference_video_map.csv --reference-root /path/to/wcf-pps-p3 \\
        --out-dir hpc/runs/ref_job/sign_frames
"""

from __future__ import annotations

import argparse
import csv
import json
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

from apps.camera_trap.calibration import refmap
from apps.camera_trap.vlm import render
from apps.camera_trap.vlm.engine import VLMEngine, VLMRequest
from apps.camera_trap.vlm.frames import iter_frames_at_indices

CONF_MAP = {"high": 1.0, "medium": 0.6, "low": 0.3}
CONF_RANK = {"high": 2, "medium": 1, "low": 0}

# 2x2 overlapping tiles (fractions of the full frame: x0, y0, x1, y1), used as a fallback when the
# whole-frame query misses the sign (e.g. it's small in a wide shot).
TILE_BOXES = [
    (0.0, 0.0, 0.6, 0.6),
    (0.4, 0.0, 1.0, 0.6),
    (0.0, 0.4, 0.6, 1.0),
    (0.4, 0.4, 1.0, 1.0),
]

# Static prompt text first, no variable parts, so every request in a run shares one prefix
# (prefix caching). Camera-trap specific: IR/night frames are often greyscale.
SIGN_PROMPT = (
    "You are scanning a camera-trap distance-calibration reference video for calibration events. "
    "In these videos, a person walks out in front of the camera and holds up a board or sign "
    "showing a number: their distance from the camera, in metres. Footage may be greyscale/"
    "monochrome night-vision (IR) footage; that alone is not a defect. "
    'Set "person_with_sign" to true only if a person holding a board/sign is visible in this '
    'frame. Set "legible" to true only if you can read the number on the sign with confidence. '
    "Read the number exactly as written; it is a distance in metres, typically between 0 and 60. "
    'If there is no person holding a sign, or the number cannot be read, set "distance_m" to '
    'null and "legible" to false. Set "confidence" to your confidence in the reading: "high", '
    '"medium", or "low". Set "sign_box" to the sign\'s bounding box as [x0, y0, x1, y1], '
    "normalised to this image's width/height (0 to 1), or null if there is no sign. "
    "Respond ONLY with a JSON object matching this schema: "
    '{"person_with_sign": bool, "distance_m": number|null, "legible": bool, '
    '"confidence": "high|medium|low", "sign_box": [number, number, number, number]|null}'
)

SIGN_SCHEMA = {
    "type": "object",
    "properties": {
        "person_with_sign": {"type": "boolean"},
        "distance_m": {"type": ["number", "null"], "minimum": 0, "maximum": 60},
        "legible": {"type": "boolean"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "sign_box": {
            "type": ["array", "null"],
            "items": {"type": "number"},
            "minItems": 4,
            "maxItems": 4,
        },
    },
    "required": ["person_with_sign", "distance_m", "legible", "confidence", "sign_box"],
    "additionalProperties": False,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference-map", type=Path, required=True,
                         help="CSV with transect, cam, transect_cam, reference_dir, status columns (see configs/pss_p3/reference_video_map.csv).")
    parser.add_argument("--reference-root", type=Path, required=True,
                         help="Root directory that each row's reference_dir is relative to.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Where scan.jsonl / calibration_frames.csv are written.")
    parser.add_argument("--model", default="qwen", help="VLM alias ('qwen') or HF model id (default: qwen).")
    parser.add_argument("--tp", type=int, default=None, help="VLM tensor_parallel_size override.")
    parser.add_argument("--scan-fps", type=float, default=2.0, help="Sampling rate in frames/second of video (default: 2.0).")
    parser.add_argument("--max-side", type=int, default=1536, help="Downscale sampled frames so the longer side is at most this (default: 1536).")
    parser.add_argument("--tile-fallback", action="store_true", help="Re-query 2x2 overlapping tiles when the whole-frame query misses the sign.")
    parser.add_argument("--min-agree", type=int, default=2, help="Minimum agreeing frames to keep an event (default: 2).")
    parser.add_argument("--max-gap-s", type=float, default=1.0, help="Max seconds of null/illegible frames tolerated within one event (default: 1.0).")
    parser.add_argument("--cameras", default=None, help="Comma-separated transect_cam allowlist (default: all).")
    parser.add_argument("--max-videos", type=int, default=None, help="Cap the number of videos processed this run.")
    parser.add_argument("--override-csv", type=Path, default=None,
                         help="CSV (transect_cam, video_path, frame_idx, distance_m, action[, sign_box_norm]) applied when writing calibration_frames.csv.")
    parser.add_argument("--dry-run", action="store_true", help="Sample and write frame JPEGs only; no VLM calls, no scan.jsonl/csv output.")
    parser.add_argument("--overwrite", action="store_true", help="Reprocess videos even if already in this shard's scan jsonl.")
    parser.add_argument("--shard-index", type=int, default=0, help="0-based shard index for array-job sharding (default: 0).")
    parser.add_argument("--num-shards", type=int, default=1, help="Total shards; a video is processed iff index %% num_shards == shard_index (default: 1).")
    parser.add_argument("--build-only", action="store_true", help="Skip scanning; just rebuild calibration_frames.csv/events_summary.csv/contact_sheet.html from existing scan*.jsonl.")
    args = parser.parse_args(argv)
    if args.scan_fps <= 0:
        parser.error("--scan-fps must be > 0.")
    if args.max_side <= 0:
        parser.error("--max-side must be > 0.")
    if args.min_agree <= 0:
        parser.error("--min-agree must be > 0.")
    if args.max_gap_s < 0:
        parser.error("--max-gap-s must be >= 0.")
    if args.num_shards <= 0:
        parser.error("--num-shards must be > 0.")
    if not (0 <= args.shard_index < args.num_shards):
        parser.error("--shard-index must be in [0, num_shards).")
    return args


# --------------------------------------------------------------------------------------
# sampling
# --------------------------------------------------------------------------------------

def sample_frame_indices(fps: float, frame_count: int, scan_fps: float) -> list[int]:
    """Frame indices at scan_fps using the video's native fps: round(k * fps / scan_fps)."""
    if fps <= 0 or frame_count <= 0 or scan_fps <= 0:
        return []
    step = fps / scan_fps
    indices = []
    k = 0
    while True:
        idx = round(k * step)
        if idx >= frame_count:
            break
        indices.append(idx)
        k += 1
    return indices


def crop_frac(frame: np.ndarray, box_frac: tuple[float, float, float, float]) -> np.ndarray:
    h, w = frame.shape[:2]
    x0f, y0f, x1f, y1f = box_frac
    x0, y0 = int(round(x0f * w)), int(round(y0f * h))
    x1, y1 = int(round(x1f * w)), int(round(y1f * h))
    x1, y1 = max(x1, x0 + 1), max(y1, y0 + 1)
    return frame[y0:y1, x0:x1]


def map_tile_box(box_norm: list[float], tile_box_frac: tuple[float, float, float, float]) -> list[float]:
    """Map a box normalised within a tile back to full-frame normalised coordinates."""
    tx0, ty0, tx1, ty1 = tile_box_frac
    x0, y0, x1, y1 = box_norm
    return [
        tx0 + x0 * (tx1 - tx0),
        ty0 + y0 * (ty1 - ty0),
        tx0 + x1 * (tx1 - tx0),
        ty0 + y1 * (ty1 - ty0),
    ]


def extract_fields(verdict: dict[str, Any]) -> dict[str, Any]:
    distance_m = verdict.get("distance_m")
    sign_box = verdict.get("sign_box")
    return {
        "person_with_sign": bool(verdict.get("person_with_sign", False)),
        "distance_m": float(distance_m) if distance_m is not None else None,
        "legible": bool(verdict.get("legible", False)),
        "confidence": verdict.get("confidence"),
        "sign_box": [float(v) for v in sign_box] if sign_box else None,
    }


def pick_best_tile(candidates: list[tuple[tuple[float, float, float, float], dict[str, Any]]]):
    """Best (tile_box, verdict) among tile candidates with person_with_sign true: prefer a
    legible non-null reading, then higher confidence. None if there are no candidates."""
    if not candidates:
        return None
    legible = [c for c in candidates if c[1].get("legible") and c[1].get("distance_m") is not None]
    pool = legible or candidates
    return max(pool, key=lambda c: CONF_RANK.get(c[1].get("confidence"), -1))


# --------------------------------------------------------------------------------------
# per-video scanning
# --------------------------------------------------------------------------------------

def process_video(video_path: Path, transect_cam: str, args: argparse.Namespace, engine, scan_f) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"skipping {video_path}: cannot open video", file=sys.stderr)
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if fps <= 0 or frame_count <= 0:
        print(f"skipping {video_path}: invalid fps/frame_count", file=sys.stderr)
        return

    indices = sample_frame_indices(fps, frame_count, args.scan_fps)
    if not indices:
        return
    decoded = dict(iter_frames_at_indices(video_path, indices))

    if args.dry_run:
        out_dir = args.out_dir / "dry_run" / video_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for idx in indices:
            frame = decoded.get(idx)
            if frame is None:
                continue
            cv2.imwrite(str(out_dir / f"f{idx}.jpg"), frame)
            n += 1
        print(f"{video_path.name}: wrote {n} dry-run frames to {out_dir}")
        return

    requests: list[VLMRequest] = []
    meta: list[int] = []
    for idx in indices:
        frame = decoded.get(idx)
        if frame is None:
            continue
        img = render.downscale_max_side(frame, max_side=args.max_side)
        requests.append(VLMRequest(images=[img], prompt=SIGN_PROMPT, color="bgr"))
        meta.append(idx)

    results = engine.run_json(requests, schema=SIGN_SCHEMA, max_tokens=256) if requests else []
    errors = list(engine.last_errors) if requests else []

    records: list[dict[str, Any]] = []
    pending_tiles: list[tuple[int, dict[str, Any]]] = []
    for i, idx in enumerate(meta):
        rec: dict[str, Any] = {"video_path": str(video_path), "transect_cam": transect_cam, "fps": fps, "frame_idx": idx}
        verdict, error = results[i], errors[i]
        if verdict is None:
            rec["error"] = error or "vlm call failed"
            records.append(rec)
            continue
        rec.update(extract_fields(verdict))
        rec["source"] = "vlm_whole"
        if args.tile_fallback and not rec["person_with_sign"]:
            pending_tiles.append((idx, rec))
        else:
            records.append(rec)

    if pending_tiles:
        tile_requests: list[VLMRequest] = []
        tile_info: list[tuple[int, tuple[float, float, float, float]]] = []
        for idx, _rec in pending_tiles:
            frame = decoded[idx]
            for tile_box in TILE_BOXES:
                crop = crop_frac(frame, tile_box)
                img = render.downscale_max_side(crop, max_side=args.max_side)
                tile_requests.append(VLMRequest(images=[img], prompt=SIGN_PROMPT, color="bgr"))
                tile_info.append((idx, tile_box))

        tile_results = engine.run_json(tile_requests, schema=SIGN_SCHEMA, max_tokens=256) if tile_requests else []
        candidates_by_idx: dict[int, list] = defaultdict(list)
        for (idx, tile_box), verdict in zip(tile_info, tile_results):
            if verdict is not None and verdict.get("person_with_sign"):
                candidates_by_idx[idx].append((tile_box, verdict))

        for idx, rec in pending_tiles:
            best = pick_best_tile(candidates_by_idx.get(idx, []))
            if best is not None:
                tile_box, verdict = best
                fields = extract_fields(verdict)
                if fields["sign_box"] is not None:
                    fields["sign_box"] = map_tile_box(fields["sign_box"], tile_box)
                rec.update(fields)
                rec["source"] = "vlm_tile"
            records.append(rec)

    for rec in records:
        scan_f.write(json.dumps(rec) + "\n")
    scan_f.flush()
    print(f"{video_path.name} ({transect_cam}): {len(records)} frame readings written")


def load_done_videos(jsonl_path: Path) -> set[str]:
    done: set[str] = set()
    if not jsonl_path.exists():
        return done
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            vp = rec.get("video_path")
            if vp:
                done.add(vp)
    return done


# --------------------------------------------------------------------------------------
# event grouping
# --------------------------------------------------------------------------------------

def group_events(records: list[dict[str, Any]], fps: float, max_gap_s: float, min_agree: int) -> list[dict[str, Any]]:
    """Group a video's sorted-by-frame_idx scan records into calibration events: a maximal run
    of frames agreeing on the same legible, non-null distance_m, allowing null/illegible gaps up
    to max_gap_s (measured from the run's last agreeing frame); a differing distance, or a gap
    exceeding max_gap_s, ends the run. Only runs with >= min_agree agreeing frames are kept."""
    usable = [r for r in records if "error" not in r]
    usable = sorted(usable, key=lambda r: r["frame_idx"])

    events: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    def finalize(cur: dict[str, Any]) -> None:
        if len(cur["frames"]) >= min_agree:
            events.append(cur)

    for rec in usable:
        idx = rec["frame_idx"]
        valid = bool(rec.get("legible")) and rec.get("distance_m") is not None

        if current is not None:
            gap_s = (idx - current["last_idx"]) / fps
            if gap_s > max_gap_s:
                finalize(current)
                current = None

        if valid:
            distance = rec["distance_m"]
            if current is not None and abs(distance - current["distance"]) < 1e-6:
                current["frames"].append(rec)
                current["last_idx"] = idx
            else:
                if current is not None:
                    finalize(current)
                current = {"distance": distance, "frames": [rec], "last_idx": idx}

    if current is not None:
        finalize(current)

    return events


def events_to_rows(video_path: str, transect_cam: str, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    video_stem = Path(video_path).stem
    for k, ev in enumerate(events, start=1):
        frames = ev["frames"]
        rep = frames[len(frames) // 2]
        conf_vals = [CONF_MAP.get(f.get("confidence"), 0.0) for f in frames]
        vlm_conf = sum(conf_vals) / len(conf_vals) if conf_vals else 0.0
        sign_box = rep.get("sign_box")
        sign_box_norm = ";".join(f"{v:.6f}" for v in sign_box) if sign_box else ""
        rows.append({
            "transect_cam": transect_cam,
            "video_path": video_path,
            "event_id": f"{video_stem}_e{k}",
            "frame_idx": rep["frame_idx"],
            "distance_m": ev["distance"],
            "sign_box_norm": sign_box_norm,
            "n_agree": len(frames),
            "vlm_conf": round(vlm_conf, 4),
            "source": "vlm",
        })
    return rows


# --------------------------------------------------------------------------------------
# override CSV
# --------------------------------------------------------------------------------------

def load_overrides(path: Path) -> list[dict[str, Any]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def apply_overrides(rows: list[dict[str, Any]], overrides: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = list(rows)
    by_key = {(r["transect_cam"], r["video_path"], int(r["frame_idx"])): i for i, r in enumerate(rows)}
    manual_counts: dict[str, int] = defaultdict(int)

    for ov in overrides:
        key = (ov["transect_cam"], ov["video_path"], int(ov["frame_idx"]))
        action = ov["action"]
        if action == "drop":
            if key in by_key:
                rows[by_key[key]] = None
        elif action == "fix":
            if key in by_key:
                r = rows[by_key[key]]
                r["distance_m"] = float(ov["distance_m"])
                if ov.get("sign_box_norm"):
                    r["sign_box_norm"] = ov["sign_box_norm"]
                r["source"] = "override"
        elif action == "add":
            video_stem = Path(ov["video_path"]).stem
            manual_counts[video_stem] += 1
            new_row = {
                "transect_cam": ov["transect_cam"],
                "video_path": ov["video_path"],
                "event_id": f"{video_stem}_manual{manual_counts[video_stem]}",
                "frame_idx": int(ov["frame_idx"]),
                "distance_m": float(ov["distance_m"]),
                "sign_box_norm": ov.get("sign_box_norm") or "",
                "n_agree": 1,
                "vlm_conf": 1.0,
                "source": "override",
            }
            rows.append(new_row)
            by_key[key] = len(rows) - 1
        else:
            raise ValueError(f"unknown override action: {action!r}")

    return [r for r in rows if r is not None]


# --------------------------------------------------------------------------------------
# outputs
# --------------------------------------------------------------------------------------

CALIB_CSV_COLUMNS = ["transect_cam", "video_path", "event_id", "frame_idx", "distance_m", "sign_box_norm", "n_agree", "vlm_conf", "source"]


def write_calibration_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    rows_sorted = sorted(rows, key=lambda r: (str(r["transect_cam"]), str(r["video_path"]), r["frame_idx"]))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CALIB_CSV_COLUMNS)
        writer.writeheader()
        for r in rows_sorted:
            writer.writerow({k: r.get(k, "") for k in CALIB_CSV_COLUMNS})


def write_events_summary(path: Path, cams_videos: dict[str, set[str]], rows_by_video: dict[str, list[dict[str, Any]]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["transect_cam", "n_events", "distances", "n_videos"])
        writer.writeheader()
        for cam in sorted(cams_videos):
            distances = []
            for video_path in cams_videos[cam]:
                distances.extend(r["distance_m"] for r in rows_by_video.get(video_path, []))
            writer.writerow({
                "transect_cam": cam,
                "n_events": len(distances),
                "distances": ";".join(str(d) for d in sorted(distances)),
                "n_videos": len(cams_videos[cam]),
            })


def write_no_event_log(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video_path", "transect_cam", "n_frames_scanned"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_contact_sheet(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_video[r["video_path"]].append(r)

    img_dir = out_dir / "contact_sheet_imgs"
    entries = []
    for video_path, vrows in sorted(by_video.items()):
        indices = sorted({r["frame_idx"] for r in vrows})
        try:
            decoded = dict(iter_frames_at_indices(video_path, indices))
        except OSError:
            continue
        for r in sorted(vrows, key=lambda r: r["frame_idx"]):
            frame = decoded.get(r["frame_idx"])
            if frame is None:
                continue
            img = frame.copy()
            box = r.get("sign_box_norm")
            if box:
                h, w = img.shape[:2]
                x0, y0, x1, y1 = (float(v) for v in box.split(";"))
                cv2.rectangle(img, (int(x0 * w), int(y0 * h)), (int(x1 * w), int(y1 * h)), (0, 255, 0), 2)
            label = f"{r['distance_m']}m  n={r['n_agree']}  conf={r['vlm_conf']}"
            cv2.putText(img, label, (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, label, (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            img_dir.mkdir(parents=True, exist_ok=True)
            img_name = f"{r['event_id']}.jpg"
            cv2.imwrite(str(img_dir / img_name), img)
            entries.append({**r, "img_path": f"contact_sheet_imgs/{img_name}"})

    rows_html = []
    for e in entries:
        rows_html.append(
            "<tr>"
            f"<td>{e['transect_cam']}</td><td>{e['event_id']}</td><td>{e['frame_idx']}</td>"
            f"<td><img src=\"{e['img_path']}\" height=\"200\"></td>"
            f"<td>{e['distance_m']}</td><td>{e['n_agree']}</td><td>{e['vlm_conf']}</td><td>{e['source']}</td>"
            "</tr>"
        )
    html = (
        "<html><head><meta charset='utf-8'><title>Sign frame contact sheet</title>"
        "<style>table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:4px;font-family:sans-serif;font-size:13px}</style>"
        "</head><body><table><tr><th>transect_cam</th><th>event_id</th><th>frame</th>"
        "<th>image</th><th>distance_m</th><th>n_agree</th><th>vlm_conf</th><th>source</th></tr>"
        + "".join(rows_html) + "</table></body></html>"
    )
    (out_dir / "contact_sheet.html").write_text(html)


def load_all_scan_records(out_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for p in sorted(out_dir.glob("scan*.jsonl")):
        with p.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
    return records


def build_outputs(args: argparse.Namespace) -> None:
    scan_records = load_all_scan_records(args.out_dir)
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    video_cam: dict[str, str] = {}
    for rec in scan_records:
        by_video[rec["video_path"]].append(rec)
        video_cam[rec["video_path"]] = rec["transect_cam"]

    all_rows: list[dict[str, Any]] = []
    no_event_rows: list[dict[str, Any]] = []
    for video_path, recs in sorted(by_video.items()):
        fps = recs[0].get("fps") or 0.0
        cam = video_cam[video_path]
        events = group_events(recs, fps, args.max_gap_s, args.min_agree) if fps > 0 else []
        rows = events_to_rows(video_path, cam, events)
        all_rows.extend(rows)
        if not rows:
            no_event_rows.append({"video_path": video_path, "transect_cam": cam, "n_frames_scanned": len(recs)})

    if args.override_csv:
        overrides = load_overrides(args.override_csv)
        all_rows = apply_overrides(all_rows, overrides)
        for ov in overrides:
            if ov["action"] == "add":
                video_cam.setdefault(ov["video_path"], ov["transect_cam"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_calibration_csv(args.out_dir / "calibration_frames.csv", all_rows)

    rows_by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in all_rows:
        rows_by_video[r["video_path"]].append(r)
    cams_videos: dict[str, set[str]] = defaultdict(set)
    for video_path, cam in video_cam.items():
        cams_videos[cam].add(video_path)
    write_events_summary(args.out_dir / "events_summary.csv", cams_videos, rows_by_video)

    if no_event_rows:
        write_no_event_log(args.out_dir / "no_event_log.csv", no_event_rows)

    write_contact_sheet(args.out_dir, all_rows)
    print(f"wrote {len(all_rows)} calibration frames ({len(scan_records)} scan records, {len(by_video)} videos) to {args.out_dir / 'calibration_frames.csv'}")


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.build_only:
        build_outputs(args)
        return 0

    ref_rows = refmap.load_reference_map(args.reference_map)
    videos = refmap.discover_reference_videos(ref_rows, args.reference_root)

    if args.cameras:
        wanted = {c.strip() for c in args.cameras.split(",") if c.strip()}
        videos = [(p, cam) for p, cam in videos if cam in wanted]

    if args.num_shards > 1:
        videos = [v for i, v in enumerate(videos) if i % args.num_shards == args.shard_index]
        scan_path = args.out_dir / f"scan.shard{args.shard_index}.jsonl"
    else:
        scan_path = args.out_dir / "scan.jsonl"

    if args.max_videos is not None:
        videos = videos[: args.max_videos]

    if args.dry_run:
        for video_path, transect_cam in videos:
            process_video(video_path, transect_cam, args, engine=None, scan_f=None)
        return 0

    if args.overwrite and scan_path.exists():
        scan_path.unlink()

    done = load_done_videos(scan_path)
    todo = [(p, cam) for p, cam in videos if str(p) not in done]

    print(f"{len(videos)} videos in scope, {len(done)} already done, {len(todo)} to do")

    engine = VLMEngine(model_id=args.model, tensor_parallel_size=args.tp)

    with scan_path.open("a") as scan_f:
        for video_path, transect_cam in todo:
            process_video(video_path, transect_cam, args, engine, scan_f)

    build_outputs(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
