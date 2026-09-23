#!/usr/bin/env python3
"""Local-VLM "agent check" for SAM3 tracked masks: per-track accept/reject QC after dap3_cli.py.

For each video's dap3 output (`<job-dir>/<stem>/<stem>.json` + `<stem>_arrays.npz`, the NPZ
requires the run to have used --npz), sample up to --frames-per-track frames evenly across each
track's lifetime, and for each sampled frame ask a local VLM (via apps/camera_trap/vlm) whether
the highlighted mask tightly covers exactly one whole animal. Tracks whose reject rate is too
high are flagged so a later export step (e.g. export_job_distances_csv.py) can drop them.

Outputs (in --out-dir):
    mask_qc.jsonl   one line per (track, frame) request: video_name, track_id, frame_index,
                    verdict fields (ok/issue/animal_visible/note), or an "error" field if the
                    VLM call failed.
    track_qc.csv    one row per track: video_name, track_id, n_checked, n_reject, reject_frac,
                    issues, keep. Rebuilt from the full mask_qc.jsonl on every run.

    python apps/camera_trap/qc/mask_verify.py --job-dir hpc/runs/some_job --out-dir hpc/runs/some_job/qc
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

from apps.camera_trap.vlm import render
from apps.camera_trap.vlm.engine import VLMEngine, VLMRequest
from apps.camera_trap.vlm.frames import iter_frames_at_indices

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv")

# Static prompt text first, no variable parts, so every request in a run shares one prefix
# (prefix caching). Camera-trap specific: single animal per mask, IR/night frames are greyscale.
MASK_QC_PROMPT = (
    "You are an expert camera-trap annotator checking an animal segmentation mask. You are given "
    "two images of the same camera-trap frame: the FIRST is the full raw frame for context, the "
    "SECOND is a crop around one tracked animal with its proposed mask drawn as a coloured "
    "semi-transparent fill with a 2px outline. Camera-trap frames are often greyscale/monochrome "
    "night-vision (IR) footage; that alone is not a defect. Animals are often partly occluded by "
    "vegetation or partly cut off by the frame edge; that alone is not a defect either -- only "
    "the parts of the animal actually visible in the frame matter. "
    "Decide whether the highlighted mask tightly covers the visible parts of exactly one animal: "
    "not background, not vegetation, not more than one animal, and not a non-animal object. "
    'Set "ok" to true only if the mask is accurate by that standard. If not, set "issue" to the '
    'single best match: "loose" (includes background/vegetation beyond the animal), "fragment" '
    "(misses a substantial visible part of the animal, e.g. only a limb is masked while the body "
    'is visible), "merged" (covers more than one animal), "wrong_object" (not an animal), or '
    '"background" (no animal is actually there). Set "issue" to "none" iff ok '
    'is true. Set "animal_visible" to true if any animal is visible anywhere in the crop, '
    "regardless of whether the mask is correct. "
    "Respond ONLY with a JSON object matching this schema: "
    '{"ok": bool, "issue": "none|loose|fragment|merged|wrong_object|background", '
    '"animal_visible": bool, "note": "<=160 chars of evidence"}'
)

MASK_QC_SCHEMA = {
    "type": "object",
    "properties": {
        "ok": {"type": "boolean"},
        "issue": {
            "type": "string",
            "enum": ["none", "loose", "fragment", "merged", "wrong_object", "background"],
        },
        "animal_visible": {"type": "boolean"},
        "note": {"type": "string", "maxLength": 160},
    },
    "required": ["ok", "issue", "animal_visible", "note"],
    "additionalProperties": False,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--job-dir", type=Path, required=True, help="dap3 output dir (contains <stem>/<stem>.json).")
    parser.add_argument("--out-dir", type=Path, required=True, help="Where mask_qc.jsonl / track_qc.csv are written.")
    parser.add_argument("--model", default="qwen", help="VLM alias ('qwen') or HF model id (default: qwen).")
    parser.add_argument("--frames-per-track", type=int, default=5, help="Max sampled frames per track (default: 5).")
    parser.add_argument("--reject-frac-max", type=float, default=0.5, help="Drop a track above this reject fraction (default: 0.5).")
    parser.add_argument("--min-checked", type=int, default=1, help="Minimum successful checks required to keep a track (default: 1).")
    parser.add_argument("--video-dir", type=Path, default=None, help="Optional dir to resolve moved video files by stem.")
    parser.add_argument("--tp", type=int, default=None, help="VLM tensor_parallel_size override.")
    parser.add_argument("--max-videos", type=int, default=None, help="Cap the number of videos processed this run.")
    parser.add_argument("--overwrite", action="store_true", help="Reprocess videos even if already in mask_qc.jsonl.")
    parser.add_argument("--dry-run", action="store_true", help="Render QC images only; no VLM calls, no jsonl/csv output.")
    parser.add_argument("--gallery", type=int, default=0, help="Write this many sample overlay JPEGs of rejected masks.")
    parser.add_argument("--shard-index", type=int, default=0, help="0-based shard index for array-job sharding (default: 0).")
    parser.add_argument("--num-shards", type=int, default=1, help="Total shards; a video is processed iff index %% num_shards == shard_index (default: 1).")
    args = parser.parse_args(argv)
    if args.frames_per_track <= 0:
        parser.error("--frames-per-track must be > 0.")
    if not (0.0 <= args.reject_frac_max <= 1.0):
        parser.error("--reject-frac-max must be in [0, 1].")
    if args.min_checked < 0:
        parser.error("--min-checked must be >= 0.")
    if args.num_shards <= 0:
        parser.error("--num-shards must be > 0.")
    if not (0 <= args.shard_index < args.num_shards):
        parser.error("--shard-index must be in [0, num_shards).")
    return args


def discover_video_jsons(job_dir: Path) -> list[Path]:
    """Same discovery convention as export_job_distances_csv.py: <job_dir>/<subdir>/<subdir>.json."""
    json_paths = []
    for subdir in sorted(job_dir.iterdir()):
        if not subdir.is_dir():
            continue
        candidate = subdir / f"{subdir.name}.json"
        if candidate.is_file():
            json_paths.append(candidate)
    return json_paths


def resolve_video_path(video_json: dict[str, Any], json_path: Path, video_dir: Path | None) -> Path | None:
    raw = video_json.get("video_path")
    if isinstance(raw, str) and raw:
        p = Path(raw)
        if p.is_file():
            return p
    stem = str(video_json.get("video_name") or json_path.stem)
    # Same directory as the JSON (common for small local runs).
    for ext in VIDEO_EXTS:
        cand = json_path.parent / f"{stem}{ext}"
        if cand.is_file():
            return cand
    if video_dir is not None and video_dir.is_dir():
        for ext in VIDEO_EXTS:
            cand = video_dir / f"{stem}{ext}"
            if cand.is_file():
                return cand
        for p in video_dir.iterdir():
            if p.is_file() and p.stem == stem:
                return p
    return None


def group_tracks(video_json: dict[str, Any]) -> dict[Any, dict[int, dict[str, Any]]]:
    """track_id -> {frame_index: object_entry}, only over frames with status 'processed' or
    'tracked' (SAM3-only frames off the DA3 depth grid; see dap3_cli.py --depth-interval-seconds)."""
    tracks: dict[Any, dict[int, dict[str, Any]]] = defaultdict(dict)
    for frame in video_json.get("frames") or []:
        if frame.get("status") not in ("processed", "tracked"):
            continue
        frame_index = frame.get("frame_index")
        if not isinstance(frame_index, int):
            continue
        for obj in frame.get("objects") or []:
            track_id = obj.get("track_id")
            if track_id is None:
                continue
            tracks[track_id].setdefault(frame_index, obj)
    return tracks


def sample_track_frames(frame_indices: list[int], k: int) -> list[int]:
    """Up to k frame indices, evenly spaced across the (sorted) lifetime."""
    frame_indices = sorted(frame_indices)
    n = len(frame_indices)
    if n <= k:
        return frame_indices
    if k <= 1:
        return [frame_indices[n // 2]]
    positions = sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})
    return [frame_indices[p] for p in positions]


def decode_object_mask(npz_data, entry: dict[str, Any]) -> np.ndarray:
    """Decode one object's mask, preferring pycocotools (via camera_trap_masks) when available,
    falling back to the pure-numpy RLE decoder (for environments without pycocotools)."""
    encoding = entry.get("encoding")
    key = entry.get("key")
    if encoding == "raw" or encoding is None:
        return np.asarray(npz_data[key]) > 0
    if encoding == "coco_rle":
        try:
            from depth_anything_3.utils.camera_trap_masks import load_mask_bool

            return load_mask_bool(npz_data, entry)
        except RuntimeError:
            payload = npz_data[key]
            counts = str(payload.item()) if payload.shape == () else str(payload.reshape(-1)[0])
            return render.rle_decode({"size": entry["size"], "counts": counts})
    raise ValueError(f"unsupported mask encoding: {encoding!r}")


def build_qc_images(frame_bgr: np.ndarray, mask: np.ndarray, bbox_xyxy) -> list[np.ndarray]:
    raw_ds = render.downscale_max_side(frame_bgr, max_side=1024)
    overlay = render.render_overlay(frame_bgr, [mask], labels=["0"])
    crop = render.crop_around(overlay, bbox_xyxy, pad_frac=0.25)
    return [raw_ds, crop]


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
            vn = rec.get("video_name")
            if vn:
                done.add(vn)
    return done


def process_video(json_path: Path, args: argparse.Namespace, engine) -> list[dict[str, Any]]:
    """Return the mask_qc.jsonl records for one video (empty list if it has no tracks/NPZ)."""
    video_json = json.loads(json_path.read_text())
    video_name = str(video_json.get("video_name") or json_path.stem)
    tracks = group_tracks(video_json)
    if not tracks:
        return []

    npz_path = json_path.with_name(f"{json_path.stem}_arrays.npz")
    if not npz_path.exists():
        raise FileNotFoundError(
            f"{video_name}: missing {npz_path.name}; mask QC requires the dap3 run to have used --npz."
        )
    video_path = resolve_video_path(video_json, json_path, args.video_dir)
    if video_path is None:
        raise FileNotFoundError(f"{video_name}: could not resolve source video (try --video-dir).")

    per_track_frames: dict[Any, list[int]] = {
        tid: sample_track_frames(list(frame_map.keys()), args.frames_per_track) for tid, frame_map in tracks.items()
    }
    wanted_frames = sorted({fi for fis in per_track_frames.values() for fi in fis})

    decoded_frames: dict[int, np.ndarray] = {}
    for frame_idx, frame_bgr in iter_frames_at_indices(video_path, wanted_frames):
        decoded_frames[frame_idx] = frame_bgr

    requests: list[VLMRequest] = []
    meta: list[tuple[Any, int]] = []
    dry_run_images: list[tuple[Any, int, list[np.ndarray]]] = []
    with np.load(npz_path, allow_pickle=False) as npz_data:
        for track_id, frame_map in per_track_frames.items():
            for frame_idx in frame_map:
                frame_bgr = decoded_frames.get(frame_idx)
                if frame_bgr is None:
                    meta.append((track_id, frame_idx))
                    requests.append(None)  # placeholder; filled as an error below
                    continue
                obj = tracks[track_id][frame_idx]
                bbox = obj.get("bbox_xyxy")
                if bbox is None:
                    meta.append((track_id, frame_idx))
                    requests.append(None)
                    continue
                mask = decode_object_mask(npz_data, obj)
                images = build_qc_images(frame_bgr, mask, bbox)
                if args.dry_run:
                    dry_run_images.append((track_id, frame_idx, images))
                    continue
                requests.append(VLMRequest(images=images, prompt=MASK_QC_PROMPT, color="bgr"))
                meta.append((track_id, frame_idx))

    if args.dry_run:
        out_dir = args.out_dir / "dry_run" / video_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for track_id, frame_idx, images in dry_run_images:
            cv2.imwrite(str(out_dir / f"track{track_id}_f{frame_idx}_raw.jpg"), images[0])
            cv2.imwrite(str(out_dir / f"track{track_id}_f{frame_idx}_overlay.jpg"), images[1])
        print(f"{video_name}: wrote {len(dry_run_images)} dry-run image pairs to {out_dir}")
        return []

    # Requests may contain None placeholders for decode/bbox failures; run only the real ones.
    real_indices = [i for i, r in enumerate(requests) if r is not None]
    real_requests = [requests[i] for i in real_indices]
    results = engine.run_json(real_requests, schema=MASK_QC_SCHEMA, max_tokens=384) if real_requests else []
    errors = list(engine.last_errors) if real_requests else []

    records: list[dict[str, Any]] = []
    result_by_index: dict[int, tuple[dict | None, str | None]] = {}
    for pos, i in enumerate(real_indices):
        result_by_index[i] = (results[pos], errors[pos])

    for i, (track_id, frame_idx) in enumerate(meta):
        rec: dict[str, Any] = {"video_name": video_name, "track_id": track_id, "frame_index": frame_idx}
        if i not in result_by_index:
            rec["error"] = "frame decode failed or missing bbox_xyxy"
        else:
            verdict, error = result_by_index[i]
            if verdict is not None:
                rec.update(verdict)
            else:
                rec["error"] = error or "vlm call failed"
        records.append(rec)

    if args.gallery > 0:
        write_gallery(args.out_dir, video_name, records, tracks, decoded_frames, npz_path, args.gallery)

    return records


def write_gallery(out_dir: Path, video_name: str, records: list[dict[str, Any]], tracks, decoded_frames, npz_path: Path, n: int) -> None:
    rejects = [r for r in records if r.get("ok") is False][:n]
    if not rejects:
        return
    gallery_dir = out_dir / "gallery"
    gallery_dir.mkdir(parents=True, exist_ok=True)
    with np.load(npz_path, allow_pickle=False) as npz_data:
        for rec in rejects:
            track_id, frame_idx = rec["track_id"], rec["frame_index"]
            frame_bgr = decoded_frames.get(frame_idx)
            obj = tracks.get(track_id, {}).get(frame_idx)
            if frame_bgr is None or obj is None or obj.get("bbox_xyxy") is None:
                continue
            mask = decode_object_mask(npz_data, obj)
            overlay = render.render_overlay(frame_bgr, [mask], labels=["0"])
            crop = render.crop_around(overlay, obj["bbox_xyxy"], pad_frac=0.25)
            issue = rec.get("issue", "unknown")
            cv2.imwrite(str(gallery_dir / f"{video_name}_track{track_id}_f{frame_idx}_{issue}.jpg"), crop)


def aggregate_track_qc(jsonl_path: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    per_track: dict[tuple[str, Any], dict[str, Any]] = {}
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = (rec["video_name"], rec["track_id"])
            entry = per_track.setdefault(key, {"n_checked": 0, "n_reject": 0, "issues": defaultdict(int)})
            if "error" in rec:
                continue
            entry["n_checked"] += 1
            if rec.get("ok") is False:
                entry["n_reject"] += 1
                issue = rec.get("issue") or "unknown"
                entry["issues"][issue] += 1

    rows = []
    for (video_name, track_id), entry in sorted(per_track.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        n_checked, n_reject = entry["n_checked"], entry["n_reject"]
        reject_frac = (n_reject / n_checked) if n_checked > 0 else 0.0
        if n_checked == 0:
            keep = True  # infrastructure failure: don't drop a track for want of data
        else:
            keep = n_checked >= args.min_checked and reject_frac <= args.reject_frac_max
        issues_str = ";".join(f"{k}:{v}" for k, v in sorted(entry["issues"].items()))
        rows.append(
            {
                "video_name": video_name,
                "track_id": track_id,
                "n_checked": n_checked,
                "n_reject": n_reject,
                "reject_frac": round(reject_frac, 4),
                "issues": issues_str,
                "keep": keep,
            }
        )
    return rows


def write_track_qc_csv(csv_path: Path, rows: list[dict[str, Any]]) -> None:
    columns = ["video_name", "track_id", "n_checked", "n_reject", "reject_frac", "issues", "keep"]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.out_dir / "mask_qc.jsonl"

    if args.overwrite and jsonl_path.exists():
        jsonl_path.unlink()

    all_jsons = discover_video_jsons(args.job_dir)
    sharded = [p for i, p in enumerate(all_jsons) if i % args.num_shards == args.shard_index]

    done = load_done_videos(jsonl_path) if not args.dry_run else set()
    todo = [p for p in sharded if json.loads(p.read_text()).get("video_name", p.stem) not in done]
    if args.max_videos is not None:
        todo = todo[: args.max_videos]

    print(f"{len(all_jsons)} videos found, shard has {len(sharded)}, {len(done)} already done, {len(todo)} to do")

    engine = None
    if not args.dry_run:
        engine = VLMEngine(model_id=args.model, tensor_parallel_size=args.tp)

    for json_path in todo:
        try:
            records = process_video(json_path, args, engine)
        except FileNotFoundError as e:
            print(f"skipping {json_path}: {e}", file=sys.stderr)
            continue
        if args.dry_run:
            continue
        with jsonl_path.open("a") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        print(f"{json_path.parent.name}: {len(records)} checks written")

    if args.dry_run:
        return 0

    rows = aggregate_track_qc(jsonl_path, args)
    write_track_qc_csv(args.out_dir / "track_qc.csv", rows)
    print(f"wrote {len(rows)} tracks to {args.out_dir / 'track_qc.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
