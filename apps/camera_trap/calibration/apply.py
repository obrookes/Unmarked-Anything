#!/usr/bin/env python3
"""Apply per-camera Haucke et al. 2022 (timmh/distance-estimation) depth calibration to a dap3
job's tracked animal detections.

For each frame with a depth map (the 2s DA3 grid; see dap3_cli.py --depth-interval-seconds),
the frame's disparity is aligned onto the camera's anchor disparity (built by build_reference.py)
with RANSAC, excluding far-background and animal pixels, then the camera's piecewise-linear
calibration curve converts the aligned disparity to calibrated depth. Each object's distance is
read from that calibrated depth map at its mask's interior point (farthest-from-boundary pixel).

Cameras without a saved anchor (unknown transect_cam, or --force-pooled) fall back to a single
pooled calibration curve fit across all reference cameras, with no per-frame alignment.

Each row's `calib_method` reflects how its distance was derived, read from the camera's saved
NPZ (build_reference.py; defaults to "per_camera" if the NPZ predates the `calib_method` key):
  per_camera    camera has its own anchor and its own fitted curve (>=2 distinct reference
                distances).
  pooled_anchor camera has its own anchor (so per-frame alignment still happens), but the curve
                is the cross-camera pooled one, borrowed for lack of >=2 distinct reference
                distances.
  pooled        camera has no saved anchor at all; no per-frame alignment, pooled curve applied
                directly to the raw disparity.
  failed        alignment or mask lookup failed for this row.

Outputs (in --out-dir):
    calibrated_objects.csv       one row per (video, frame, track); see CSV_COLUMNS below.
    apply_summary.json           counts per calib_method, per camera, and failures by reason.

    python apps/camera_trap/calibration/apply.py --job-dir hpc/runs/main_job \\
        --calib-dir hpc/runs/ref_job/calib --out-dir hpc/runs/main_job/calib_applied
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
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

from apps.camera_trap.calibration import extrinsic
from apps.camera_trap.calibration.timmh import (
    align_disparity,
    depth_from_disparity,
    disparity_from_depth,
    interior_point,
    piecewise_from_knots,
    resize_to,
)
from apps.camera_trap.qc.mask_verify import decode_object_mask, resolve_video_path
from apps.camera_trap.scripts.export_job_distances_csv import extract_transect_cam
from apps.camera_trap.vlm.frames import iter_frames_at_indices

CSV_COLUMNS = [
    "video_name",
    "frame_index",
    "track_id",
    "transect_cam",
    "distance_m",
    "raw_depth_at_point",
    "calib_method",
    "align_inlier_frac",
    "homography_used",
]

FAR_EPS = 1e-6
ANIMAL_MASK_DILATE_ITER = 2


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--job-dir", type=Path, required=True, help="dap3 output dir (contains <stem>/<stem>.json + _arrays.npz).")
    parser.add_argument("--calib-dir", type=Path, required=True, help="build_reference.py output dir (calib/<transect_cam>.npz + _pooled.npz).")
    parser.add_argument("--out-dir", type=Path, required=True, help="Where calibrated_objects.csv / apply_summary.json are written.")
    parser.add_argument("--align-method", default="ransac", choices=["ransac", "leastsquares"], help="Disparity alignment regression method (default: ransac).")
    parser.add_argument("--align-blur", action="store_true", help="Blur/downsample fields before fitting the alignment (matches timmh calibrate_blur).")
    parser.add_argument("--min-align-pixels", type=int, default=500, help="Minimum non-excluded pixels required to fit alignment (default: 500).")
    parser.add_argument("--force-pooled", action="store_true", help="Always use the pooled calibration curve, even for cameras with a saved anchor.")
    parser.add_argument("--extrinsic-recalibration", action="store_true", help="Warp each frame onto the anchor image via feature-matched homography before alignment (default off).")
    parser.add_argument("--lightglue-weights", type=Path, default=None, help="Local LightGlue-ONNX weights file. Required with --extrinsic-recalibration.")
    parser.add_argument("--video-dir", type=Path, default=None, help="Override dir to resolve source videos (for --extrinsic-recalibration frame decoding).")
    parser.add_argument("--shard-index", type=int, default=0, help="0-based shard index for array-job sharding (default: 0).")
    parser.add_argument("--num-shards", type=int, default=1, help="Total shards; a video is processed iff index %% num_shards == shard_index (default: 1).")
    parser.add_argument("--merge", action="store_true", help="Concatenate calibrated_objects.shard*.csv / apply_summary.shard*.json in --out-dir instead of processing.")
    parser.add_argument("--overwrite", action="store_true", help="Reprocess videos even if already present in the output CSV.")
    parser.add_argument("--max-videos", type=int, default=None, help="Cap the number of videos processed this run.")
    args = parser.parse_args(argv)
    if args.num_shards <= 0:
        parser.error("--num-shards must be > 0.")
    if not (0 <= args.shard_index < args.num_shards):
        parser.error("--shard-index must be in [0, num_shards).")
    if args.min_align_pixels <= 0:
        parser.error("--min-align-pixels must be > 0.")
    if args.extrinsic_recalibration and args.lightglue_weights is None:
        parser.error("--extrinsic-recalibration requires --lightglue-weights.")
    return args


# --------------------------------------------------------------------------------------
# discovery / calibration loading
# --------------------------------------------------------------------------------------

def discover_video_jsons(job_dir: Path) -> list[Path]:
    json_paths = []
    for subdir in sorted(job_dir.iterdir()):
        if not subdir.is_dir():
            continue
        candidate = subdir / f"{subdir.name}.json"
        if candidate.is_file():
            json_paths.append(candidate)
    return json_paths


def load_camera_calib(npz_path: Path) -> dict[str, Any]:
    with np.load(npz_path, allow_pickle=False) as data:
        entry = {
            "anchor_disp_raw": np.asarray(data["anchor_disp_raw"], dtype=np.float64),
            "anchor_img": np.asarray(data["anchor_img"]),
            "anchor_person_mask": np.asarray(data["anchor_person_mask"], dtype=bool),
            "knots_x": np.asarray(data["knots_x"], dtype=np.float64),
            "knots_y": np.asarray(data["knots_y"], dtype=np.float64),
            "max_depth": float(data["max_depth"]),
            "min_depth": float(data["min_depth"]),
            "calib_method": str(data["calib_method"].item()) if "calib_method" in data.files else "per_camera",
        }
    entry["curve"] = piecewise_from_knots(entry["knots_x"], entry["knots_y"])
    anchor_cal_depth = depth_from_disparity(entry["curve"](entry["anchor_disp_raw"]), entry["min_depth"], entry["max_depth"])
    entry["far_mask"] = anchor_cal_depth >= (entry["max_depth"] - FAR_EPS)
    return entry


def load_pooled_calib(npz_path: Path) -> dict[str, Any]:
    with np.load(npz_path, allow_pickle=False) as data:
        entry = {
            "knots_x": np.asarray(data["knots_x"], dtype=np.float64),
            "knots_y": np.asarray(data["knots_y"], dtype=np.float64),
            "max_depth": float(data["max_depth"]),
            "min_depth": float(data["min_depth"]),
        }
    entry["curve"] = piecewise_from_knots(entry["knots_x"], entry["knots_y"])
    return entry


def discover_calibrations(calib_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    by_cam: dict[str, dict[str, Any]] = {}
    for npz_path in sorted(calib_dir.glob("*.npz")):
        if npz_path.stem == "_pooled":
            continue
        by_cam[npz_path.stem] = load_camera_calib(npz_path)
    pooled_path = calib_dir / "_pooled.npz"
    if not pooled_path.is_file():
        raise FileNotFoundError(f"missing pooled calibration: {pooled_path}")
    pooled = load_pooled_calib(pooled_path)
    return by_cam, pooled


# --------------------------------------------------------------------------------------
# per-frame depth / mask helpers
# --------------------------------------------------------------------------------------

def load_frame_depth(npz_data, frame_rec: dict[str, Any]) -> np.ndarray | None:
    npz_keys = frame_rec.get("npz_keys") or {}
    depth_key = npz_keys.get("depth") or f"f{frame_rec['frame_index']}_depth"
    if depth_key not in npz_data:
        return None
    depth = np.asarray(npz_data[depth_key])
    depth_shape = frame_rec.get("depth_shape")
    if depth_shape is not None:
        depth_shape = tuple(int(v) for v in depth_shape)
        if depth.shape != depth_shape and depth.size == depth_shape[0] * depth_shape[1]:
            depth = depth.reshape(depth_shape)
    return depth.astype(np.float32, copy=False)


def map_point(point: tuple[int, int], from_shape, to_shape) -> tuple[int, int]:
    """Map an (row, col) point from from_shape resolution to to_shape resolution."""
    if from_shape[0] == to_shape[0] and from_shape[1] == to_shape[1]:
        return point
    row = int(round(point[0] * to_shape[0] / from_shape[0]))
    col = int(round(point[1] * to_shape[1] / from_shape[1]))
    row = max(0, min(to_shape[0] - 1, row))
    col = max(0, min(to_shape[1] - 1, col))
    return row, col


# --------------------------------------------------------------------------------------
# per-video processing
# --------------------------------------------------------------------------------------

def process_video(
    json_path: Path,
    by_cam: dict[str, dict[str, Any]],
    pooled: dict[str, Any],
    args: argparse.Namespace,
    failures: Counter,
) -> list[dict[str, Any]]:
    video_json = json.loads(json_path.read_text())
    video_name = str(video_json.get("video_name") or json_path.stem)
    npz_path = json_path.with_name(f"{json_path.stem}_arrays.npz")
    rows: list[dict[str, Any]] = []
    if not npz_path.exists():
        return rows

    transect_cam = extract_transect_cam(video_name)
    cam_entry = None if args.force_pooled else by_cam.get(transect_cam) if transect_cam else None

    recalibrator = None
    video_path = None
    if args.extrinsic_recalibration and cam_entry is not None:
        recalibrator = extrinsic.ExtrinsicRecalibrator(lightglue_weights=args.lightglue_weights)
        video_path = resolve_video_path(video_json, json_path, args.video_dir)

    with np.load(npz_path, allow_pickle=False) as npz_data:
        for frame in video_json.get("frames") or []:
            if frame.get("status") != "processed":
                continue
            objs = frame.get("objects") or []
            if not objs:
                continue
            depth_raw = load_frame_depth(npz_data, frame)
            if depth_raw is None:
                continue
            frame_idx = frame["frame_index"]

            masks_native = [decode_object_mask(npz_data, obj) for obj in objs]

            if cam_entry is None:
                rows.extend(
                    _process_pooled_frame(video_name, frame_idx, objs, masks_native, depth_raw, pooled, transect_cam, failures)
                )
                continue

            rows.extend(
                _process_per_camera_frame(
                    video_name, frame_idx, objs, masks_native, depth_raw, cam_entry, transect_cam,
                    args, failures, recalibrator, video_path,
                )
            )

    return rows


def _process_pooled_frame(video_name, frame_idx, objs, masks_native, depth_raw, pooled, transect_cam, failures) -> list[dict[str, Any]]:
    disp = disparity_from_depth(depth_raw)
    depth_cal = depth_from_disparity(pooled["curve"](disp), pooled["min_depth"], pooled["max_depth"])
    rows = []
    for obj, mask in zip(objs, masks_native):
        pt = interior_point(mask)
        if pt is None:
            failures["empty_mask"] += 1
            rows.append(_row(video_name, frame_idx, obj, transect_cam, float("nan"), float("nan"), "failed", None, False))
            continue
        distance_m = float(depth_cal[pt])
        raw_depth = float(depth_raw[pt])
        rows.append(_row(video_name, frame_idx, obj, transect_cam, distance_m, raw_depth, "pooled", None, False))
    return rows


def _process_per_camera_frame(
    video_name, frame_idx, objs, masks_native, depth_raw, cam_entry, transect_cam, args, failures,
    recalibrator, video_path,
) -> list[dict[str, Any]]:
    anchor_shape = cam_entry["anchor_disp_raw"].shape[:2]
    disp = disparity_from_depth(depth_raw)
    disp_anchor = resize_to(disp, anchor_shape)
    masks_anchor = [resize_to(m.astype(bool), anchor_shape) for m in masks_native]

    homography_used = False
    if recalibrator is not None and video_path is not None:
        homography_used, disp_anchor, masks_anchor = _apply_extrinsic(
            recalibrator, video_path, frame_idx, cam_entry, anchor_shape, disp_anchor, masks_anchor
        )

    kernel = np.ones((3, 3), np.uint8)
    if masks_anchor:
        animal_union = np.zeros(anchor_shape, dtype=np.uint8)
        for m in masks_anchor:
            animal_union |= m.astype(np.uint8)
        animal_union = cv2.dilate(animal_union, kernel, iterations=ANIMAL_MASK_DILATE_ITER).astype(bool)
    else:
        animal_union = np.zeros(anchor_shape, dtype=bool)

    exclude = cam_entry["far_mask"] | animal_union | cam_entry["anchor_person_mask"]

    aligned, info = align_disparity(
        disp_anchor, cam_entry["anchor_disp_raw"], exclude,
        method=args.align_method, min_pixels=args.min_align_pixels, blur=args.align_blur,
    )

    rows = []
    if aligned is None:
        failures[info["status"]] += 1
        for obj, mask in zip(objs, masks_native):
            pt = interior_point(mask)
            raw_depth = float(depth_raw[pt]) if pt is not None else float("nan")
            rows.append(_row(video_name, frame_idx, obj, transect_cam, float("nan"), raw_depth, "failed", None, homography_used))
        return rows

    depth_cal = depth_from_disparity(cam_entry["curve"](aligned), cam_entry["min_depth"], cam_entry["max_depth"])
    for obj, mask_native, mask_anchor in zip(objs, masks_native, masks_anchor):
        pt_anchor = interior_point(mask_anchor)
        if pt_anchor is None:
            failures["empty_mask"] += 1
            rows.append(_row(video_name, frame_idx, obj, transect_cam, float("nan"), float("nan"), "failed", info["inlier_frac"], homography_used))
            continue
        distance_m = float(depth_cal[pt_anchor])
        pt_raw = map_point(pt_anchor, anchor_shape, depth_raw.shape[:2])
        raw_depth = float(depth_raw[pt_raw])
        rows.append(_row(video_name, frame_idx, obj, transect_cam, distance_m, raw_depth, cam_entry["calib_method"], info["inlier_frac"], homography_used))
    return rows


def _apply_extrinsic(recalibrator, video_path, frame_idx, cam_entry, anchor_shape, disp_anchor, masks_anchor):
    """Best-effort: on any decode/estimate failure, fall back to no warp (homography_used=False)."""
    try:
        frames = dict(iter_frames_at_indices(video_path, [frame_idx]))
        frame_bgr = frames.get(frame_idx)
        if frame_bgr is None:
            return False, disp_anchor, masks_anchor
        baseline_img = cv2.resize(
            cam_entry["anchor_img"], (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_LINEAR
        )
        estimate = recalibrator.estimate(baseline_img, frame_bgr)
        if estimate is None:
            return False, disp_anchor, masks_anchor
        h_target = extrinsic.rescale_homography(estimate.homography, frame_bgr.shape[:2], anchor_shape)
        disp_warped = extrinsic.warp_depth(disp_anchor, h_target, anchor_shape)
        masks_warped = [extrinsic.warp_mask(m, h_target, anchor_shape) for m in masks_anchor]
        return True, disp_warped, masks_warped
    except Exception:  # noqa: BLE001 - extrinsic step is best-effort
        return False, disp_anchor, masks_anchor


def _row(video_name, frame_idx, obj, transect_cam, distance_m, raw_depth, calib_method, inlier_frac, homography_used) -> dict[str, Any]:
    return {
        "video_name": video_name,
        "frame_index": frame_idx,
        "track_id": obj.get("track_id"),
        "transect_cam": transect_cam or "",
        "distance_m": distance_m,
        "raw_depth_at_point": raw_depth,
        "calib_method": calib_method,
        "align_inlier_frac": inlier_frac if inlier_frac is not None else "",
        "homography_used": bool(homography_used),
    }


# --------------------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------------------

def build_summary(rows: list[dict[str, Any]], failures: Counter) -> dict[str, Any]:
    by_method: Counter = Counter()
    by_camera: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        by_method[row["calib_method"]] += 1
        by_camera[row["transect_cam"] or "unknown"][row["calib_method"]] += 1
    return {
        "n_objects": len(rows),
        "by_method": dict(by_method),
        "by_camera": {cam: dict(counts) for cam, counts in by_camera.items()},
        "failures_by_reason": dict(failures),
    }


def merge_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    by_method: Counter = Counter()
    by_camera: dict[str, Counter] = defaultdict(Counter)
    failures: Counter = Counter()
    n_objects = 0
    for s in summaries:
        n_objects += s.get("n_objects", 0)
        by_method.update(s.get("by_method") or {})
        for cam, counts in (s.get("by_camera") or {}).items():
            by_camera[cam].update(counts)
        failures.update(s.get("failures_by_reason") or {})
    return {
        "n_objects": n_objects,
        "by_method": dict(by_method),
        "by_camera": {cam: dict(counts) for cam, counts in by_camera.items()},
        "failures_by_reason": dict(failures),
    }


# --------------------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------------------

def csv_path_for_shard(out_dir: Path, shard_index: int, num_shards: int) -> Path:
    if num_shards > 1:
        return out_dir / f"calibrated_objects.shard{shard_index}.csv"
    return out_dir / "calibrated_objects.csv"


def summary_path_for_shard(out_dir: Path, shard_index: int, num_shards: int) -> Path:
    if num_shards > 1:
        return out_dir / f"apply_summary.shard{shard_index}.json"
    return out_dir / "apply_summary.json"


def load_done_videos(csv_path: Path) -> set[str]:
    done: set[str] = set()
    if not csv_path.exists():
        return done
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vn = row.get("video_name")
            if vn:
                done.add(vn)
    return done


def write_rows(csv_path: Path, rows: list[dict[str, Any]], append: bool) -> None:
    write_header = not (append and csv_path.exists())
    mode = "a" if append and csv_path.exists() else "w"
    with csv_path.open(mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def do_merge(out_dir: Path) -> None:
    shard_csvs = sorted(out_dir.glob("calibrated_objects.shard*.csv"))
    merged_csv = out_dir / "calibrated_objects.csv"
    if not shard_csvs:
        # A --num-shards 1 run writes calibrated_objects.csv directly; nothing to merge.
        if merged_csv.is_file():
            print(f"no shard files; {merged_csv} already written by a single-shard run")
            return
        raise FileNotFoundError(f"no calibrated_objects.shard*.csv found in {out_dir}")
    with merged_csv.open("w", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for shard_csv in shard_csvs:
            with shard_csv.open(newline="") as in_f:
                reader = csv.DictReader(in_f)
                writer.writerows(reader)

    summaries = []
    for summary_path in sorted(out_dir.glob("apply_summary.shard*.json")):
        summaries.append(json.loads(summary_path.read_text()))
    if summaries:
        merged_summary = merge_summaries(summaries)
        (out_dir / "apply_summary.json").write_text(json.dumps(merged_summary, indent=2))
    print(f"merged {len(shard_csvs)} shard(s) into {merged_csv}")


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.extrinsic_recalibration and args.lightglue_weights is not None and not Path(args.lightglue_weights).is_file():
        raise FileNotFoundError(f"--lightglue-weights not found: {args.lightglue_weights}")

    if args.merge:
        do_merge(args.out_dir)
        return 0

    by_cam, pooled = discover_calibrations(args.calib_dir)

    all_jsons = discover_video_jsons(args.job_dir)
    sharded = [p for i, p in enumerate(all_jsons) if i % args.num_shards == args.shard_index]

    csv_path = csv_path_for_shard(args.out_dir, args.shard_index, args.num_shards)
    summary_path = summary_path_for_shard(args.out_dir, args.shard_index, args.num_shards)

    if args.overwrite:
        csv_path.unlink(missing_ok=True)
        summary_path.unlink(missing_ok=True)
        done: set[str] = set()
    else:
        done = load_done_videos(csv_path)

    todo = [p for p in sharded if json.loads(p.read_text()).get("video_name", p.stem) not in done]
    if args.max_videos is not None:
        todo = todo[: args.max_videos]

    print(f"{len(all_jsons)} videos found, shard has {len(sharded)}, {len(done)} already done, {len(todo)} to do")

    failures: Counter = Counter()
    existing_summary = json.loads(summary_path.read_text()) if (not args.overwrite and summary_path.exists()) else None
    all_new_rows: list[dict[str, Any]] = []

    for json_path in todo:
        rows = process_video(json_path, by_cam, pooled, args, failures)
        write_rows(csv_path, rows, append=True)
        all_new_rows.extend(rows)
        print(f"{json_path.parent.name}: {len(rows)} objects written")

    new_summary = build_summary(all_new_rows, failures)
    summary = merge_summaries([existing_summary, new_summary]) if existing_summary else new_summary
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"wrote {len(all_new_rows)} new rows to {csv_path}; summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
