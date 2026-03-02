#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

# Allow running directly from repository root without requiring editable install.
REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from depth_anything_3.api import DepthAnything3
from ultralytics.models.sam import SAM3SemanticPredictor
try:
    from ultralytics.models.sam import SAM3VideoSemanticPredictor
except ImportError:
    SAM3VideoSemanticPredictor = None


DEFAULT_VIDEO_EXTS = ".mp4,.mov,.avi,.mkv"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch pipeline: sample frames from videos, run SAM3 segmentation with text prompts, "
            "then run Depth Anything 3 only on SAM3-positive frames."
        )
    )
    parser.add_argument(
        "--input-video-dir",
        required=True,
        help="Directory containing videos to process.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Root directory where per-video outputs and run manifest will be written.",
    )
    parser.add_argument(
        "--video-exts",
        default=DEFAULT_VIDEO_EXTS,
        help='Comma-separated video extensions, e.g. ".mp4,.mov,.avi,.mkv".',
    )
    parser.add_argument("--sam3-model-path", required=True, help="Path to SAM3 checkpoint (.pt).")
    parser.add_argument(
        "--sam3-text-prompts",
        nargs="+",
        required=True,
        help='One or more global SAM3 text prompts, e.g. --sam3-text-prompts person car "traffic light".',
    )
    parser.add_argument(
        "--da3-model-id",
        default="depth-anything/DA3NESTED-GIANT-LARGE",
        help="Depth Anything 3 model ID.",
    )
    parser.add_argument("--target-fps", type=float, default=1.0, help="Sampling rate for processing.")
    parser.add_argument(
        "--sam3-mode",
        choices=["track", "frame"],
        default="track",
        help="SAM3 inference mode: native video tracking ('track') or per-frame segmentation ('frame').",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="SAM3 confidence threshold.")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Inference device.",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Enable FP16 for SAM3 (recommended on CUDA).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reprocess videos even if output JSON and NPZ already exist.",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Optional cap on number of videos to process after sorting.",
    )
    parser.add_argument(
        "--da3-batch-size",
        type=int,
        required=True,
        help="DA3 batch size for SAM-positive sampled frames.",
    )

    args = parser.parse_args()
    if args.target_fps <= 0:
        parser.error("--target-fps must be > 0.")
    if args.max_videos is not None and args.max_videos <= 0:
        parser.error("--max-videos must be > 0 when provided.")
    if args.da3_batch_size <= 0:
        parser.error("--da3-batch-size must be > 0.")
    return args


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def parse_video_exts(raw_exts: str) -> tuple[str, ...]:
    exts = []
    for part in raw_exts.split(","):
        ext = part.strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = f".{ext}"
        exts.append(ext)
    unique = tuple(sorted(set(exts)))
    if not unique:
        raise ValueError("At least one valid extension must be provided in --video-exts.")
    return unique


def discover_videos(input_dir: Path, video_exts: tuple[str, ...]) -> list[Path]:
    videos = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in video_exts]
    videos.sort(key=lambda p: p.name.lower())
    return videos


def slugify_prompt(prompt: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", prompt.strip().lower())
    slug = slug.strip("_")
    return slug or "prompt"


def build_prompt_slugs(prompts: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    slugs: list[str] = []
    for prompt in prompts:
        base = slugify_prompt(prompt)
        seen = counts.get(base, 0)
        if seen == 0:
            slug = base
        else:
            slug = f"{base}_{seen + 1}"
        counts[base] = seen + 1
        slugs.append(slug)
    return slugs


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{path.name}.tmp"
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    os.replace(tmp_path, path)


def write_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_base = path.parent / f".{path.name}.tmp"
    np.savez_compressed(str(tmp_base), **arrays)
    tmp_npz = Path(f"{tmp_base}.npz")
    os.replace(tmp_npz, path)


def build_manifest_summary(video_entries: list[dict[str, Any]]) -> dict[str, int]:
    summary = {
        "processed_videos": 0,
        "failed_videos": 0,
        "skipped_videos": 0,
        "sampled_frames": 0,
        "processed_frames": 0,
        "empty_mask_frames": 0,
        "error_frames": 0,
    }
    for entry in video_entries:
        status = entry.get("status")
        if status == "success":
            summary["processed_videos"] += 1
        elif status == "failed":
            summary["failed_videos"] += 1
        elif status == "skipped_existing":
            summary["skipped_videos"] += 1
        counts = entry.get("counts") or {}
        summary["sampled_frames"] += int(counts.get("sampled_frames", 0))
        summary["processed_frames"] += int(counts.get("processed_frames", 0))
        summary["empty_mask_frames"] += int(counts.get("empty_mask_frames", 0))
        summary["error_frames"] += int(counts.get("error_frames", 0))
    return summary


def update_and_write_manifest(manifest_path: Path, manifest: dict[str, Any]) -> None:
    manifest["finished_at_utc"] = utc_now_iso()
    manifest["summary"] = build_manifest_summary(manifest["videos"])
    write_json_atomic(manifest_path, manifest)


def compute_sample_interval(video_fps: float, target_fps: float) -> int:
    if video_fps <= 0:
        return 1
    return max(1, round(video_fps / target_fps))


def _find_prompt_idx_by_label(label: str, prompts: list[str]) -> int | None:
    if not label:
        return None
    norm_label = label.strip().casefold()
    for idx, prompt in enumerate(prompts):
        if prompt.strip().casefold() == norm_label:
            return idx
    return None


def extract_prompt_masks(
    sam_result: Any,
    prompts: list[str],
    frame_shape_hw: tuple[int, int],
) -> list[np.ndarray]:
    height, width = frame_shape_hw
    prompt_masks = [np.zeros((height, width), dtype=bool) for _ in prompts]

    if sam_result.masks is None or sam_result.masks.data is None:
        return [mask.astype(np.uint8) for mask in prompt_masks]

    masks_data = sam_result.masks.data.detach().cpu().numpy()
    if masks_data.size == 0:
        return [mask.astype(np.uint8) for mask in prompt_masks]
    det_masks = masks_data > 0
    num_dets = det_masks.shape[0]

    cls_ids = None
    if sam_result.boxes is not None and sam_result.boxes.cls is not None:
        cls_ids = sam_result.boxes.cls.detach().cpu().numpy().astype(np.int64)
    names = sam_result.names if hasattr(sam_result, "names") else None

    assigned_any = False
    if cls_ids is not None and cls_ids.shape[0] == num_dets:
        for det_idx in range(num_dets):
            cls_id = int(cls_ids[det_idx])
            prompt_idx = None
            if 0 <= cls_id < len(prompts):
                prompt_idx = cls_id
            elif isinstance(names, dict) and cls_id in names:
                prompt_idx = _find_prompt_idx_by_label(str(names[cls_id]), prompts)
            if prompt_idx is not None:
                prompt_masks[prompt_idx] |= det_masks[det_idx]
                assigned_any = True

    # Fallback for unknown class mapping: put union mask into first prompt.
    if not assigned_any and len(prompts) > 0:
        prompt_masks[0] |= det_masks.max(axis=0)

    return [mask.astype(np.uint8) for mask in prompt_masks]


def run_da3_inference_batch(da3: DepthAnything3, frames_bgr: list[np.ndarray]) -> list[np.ndarray]:
    depth_pred = da3.inference(frames_bgr, use_ray_pose=False, infer_gs=False, export_dir=None)
    return [np.asarray(depth_map, dtype=np.float32) for depth_map in depth_pred.depth]


def base_frame_record(frame_index: int, timestamp_sec: float | None) -> dict[str, Any]:
    return {
        "frame_index": int(frame_index),
        "timestamp_sec": timestamp_sec,
        "status": None,
        "error": None,
        "sam3_mode": None,
        "track_summary": None,
        "mask_nonzero_pixels": 0,
        "mask_area_fraction": 0.0,
        "bbox_xyxy": None,
        "center_xy": None,
        "depth_mask_mean": None,
        "depth_center_value": None,
        "npz_keys": None,
        "timing_ms": {"sam_ms": None, "da3_ms": None, "total_ms": None},
    }


def compute_mask_geometry(union_mask_bool: np.ndarray) -> tuple[int, float, list[int] | None, list[int] | None]:
    nonzero_pixels = int(union_mask_bool.sum())
    height, width = union_mask_bool.shape
    area_fraction = float(nonzero_pixels / float(height * width)) if height > 0 and width > 0 else 0.0
    if nonzero_pixels == 0:
        return nonzero_pixels, area_fraction, None, None

    ys, xs = np.where(union_mask_bool)
    xmin, xmax = int(xs.min()), int(xs.max())
    ymin, ymax = int(ys.min()), int(ys.max())
    cx = int((xmin + xmax) // 2)
    cy = int((ymin + ymax) // 2)
    return nonzero_pixels, area_fraction, [xmin, ymin, xmax, ymax], [cx, cy]


def extract_track_summary(sam_result: Any) -> dict[str, Any]:
    summary = {"active_track_count": 0, "tracks": []}
    if sam_result is None or sam_result.boxes is None or len(sam_result.boxes) == 0:
        return summary

    boxes = sam_result.boxes
    xyxy = boxes.xyxy.detach().cpu().numpy() if boxes.xyxy is not None else np.zeros((0, 4))
    confs = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else np.zeros((len(boxes),))
    classes = boxes.cls.detach().cpu().numpy().astype(np.int64) if boxes.cls is not None else np.zeros((len(boxes),), dtype=np.int64)
    ids = (
        boxes.id.detach().cpu().numpy().astype(np.int64)
        if getattr(boxes, "is_track", False) and boxes.id is not None
        else np.full((len(boxes),), -1, dtype=np.int64)
    )

    names = sam_result.names if hasattr(sam_result, "names") else {}
    det_mask_pixels = np.zeros((len(boxes),), dtype=np.int64)
    if sam_result.masks is not None and sam_result.masks.data is not None:
        det_masks = sam_result.masks.data.detach().cpu().numpy() > 0
        if det_masks.shape[0] == len(boxes):
            det_mask_pixels = det_masks.reshape(det_masks.shape[0], -1).sum(axis=1).astype(np.int64)

    track_ids_non_null: set[int] = set()
    tracks: list[dict[str, Any]] = []
    for i in range(len(boxes)):
        cls_id = int(classes[i]) if i < len(classes) else -1
        track_id = int(ids[i]) if i < len(ids) and int(ids[i]) >= 0 else None
        if track_id is not None:
            track_ids_non_null.add(track_id)

        label = str(cls_id)
        if isinstance(names, dict) and cls_id in names:
            label = str(names[cls_id])

        bbox = [int(round(v)) for v in xyxy[i].tolist()] if i < len(xyxy) else None
        tracks.append(
            {
                "track_id": track_id,
                "label": label,
                "confidence": float(confs[i]) if i < len(confs) else None,
                "bbox_xyxy": bbox,
                "mask_pixels": int(det_mask_pixels[i]) if i < len(det_mask_pixels) else 0,
            }
        )

    summary["tracks"] = tracks
    summary["active_track_count"] = len(track_ids_non_null) if track_ids_non_null else len(tracks)
    return summary


def process_video(
    *,
    video_path: Path,
    output_root: Path,
    args: argparse.Namespace,
    sam3_frame: SAM3SemanticPredictor | None,
    sam3_track: Any,
    da3: DepthAnything3,
    prompt_slugs: list[str],
    device: torch.device,
    da3_batch_size: int,
) -> dict[str, Any]:
    video_stem = video_path.stem
    video_out_dir = output_root / video_stem
    json_path = video_out_dir / f"{video_stem}.json"
    npz_path = video_out_dir / f"{video_stem}_arrays.npz"

    if not args.overwrite and json_path.exists() and npz_path.exists():
        return {
            "video_name": video_stem,
            "video_path": str(video_path.resolve()),
            "status": "skipped_existing",
            "error": None,
            "json_path": str(json_path.resolve()),
            "npz_path": str(npz_path.resolve()),
            "counts": {
                "sampled_frames": 0,
                "processed_frames": 0,
                "empty_mask_frames": 0,
                "error_frames": 0,
            },
            "duration_sec": 0.0,
        }

    video_out_dir.mkdir(parents=True, exist_ok=True)
    started_at = utc_now_iso()
    started_perf = time.perf_counter()
    frame_rows: list[dict[str, Any]] = []
    npz_arrays: dict[str, np.ndarray] = {}
    counts = {
        "sampled_frames": 0,
        "processed_frames": 0,
        "empty_mask_frames": 0,
        "error_frames": 0,
    }

    video_json: dict[str, Any] = {
        "video_name": video_stem,
        "video_path": str(video_path.resolve()),
        "video_fps": 0.0,
        "target_fps": float(args.target_fps),
        "sam3_mode": args.sam3_mode,
        "sample_interval_frames": 1,
        "frame_width": 0,
        "frame_height": 0,
        "sam3_prompts": list(args.sam3_text_prompts),
        "sam3_conf": float(args.conf),
        "da3_model_id": args.da3_model_id,
        "da3_batch_size": int(da3_batch_size),
        "device": str(device),
        "started_at_utc": started_at,
        "finished_at_utc": None,
        "status": "failed",
        "error": None,
        "frames": frame_rows,
    }

    def finalize_record(
        rec: dict[str, Any],
        *,
        frame_idx: int,
        depth: np.ndarray,
        union_mask_bool: np.ndarray,
        center_xy: list[int] | None,
        prompt_masks: list[np.ndarray],
        frame_start: float,
        da3_ms: float,
    ) -> None:
        if depth.shape != union_mask_bool.shape:
            depth = cv2.resize(
                depth,
                (union_mask_bool.shape[1], union_mask_bool.shape[0]),
                interpolation=cv2.INTER_CUBIC,
            )

        depth_values = depth[union_mask_bool]
        rec["depth_mask_mean"] = float(np.nanmean(depth_values)) if depth_values.size > 0 else None
        if center_xy is not None:
            cx, cy = center_xy
            rec["depth_center_value"] = float(depth[cy, cx])
        rec["status"] = "processed"
        rec["timing_ms"]["da3_ms"] = da3_ms

        frame_prefix = f"f{frame_idx}"
        depth_key = f"{frame_prefix}_depth"
        prompt_order_key = f"{frame_prefix}_prompt_order"
        npz_arrays[depth_key] = np.asarray(depth, dtype=np.float32)
        npz_arrays[prompt_order_key] = np.asarray(args.sam3_text_prompts, dtype=np.str_)

        mask_key_rows = []
        for prompt_idx, (prompt, prompt_slug, mask) in enumerate(
            zip(args.sam3_text_prompts, prompt_slugs, prompt_masks)
        ):
            mask_key = f"{frame_prefix}_mask_{prompt_slug}"
            npz_arrays[mask_key] = np.asarray(mask, dtype=np.uint8)
            mask_key_rows.append(
                {
                    "prompt_index": int(prompt_idx),
                    "prompt": prompt,
                    "slug": prompt_slug,
                    "key": mask_key,
                }
            )
        rec["npz_keys"] = {
            "depth": depth_key,
            "prompt_order": prompt_order_key,
            "masks": mask_key_rows,
        }
        rec["timing_ms"]["total_ms"] = (time.perf_counter() - frame_start) * 1000.0
        frame_rows.append(rec)
        counts["processed_frames"] += 1

    pending_da3: list[dict[str, Any]] = []

    def flush_pending_da3() -> None:
        nonlocal pending_da3
        if not pending_da3:
            return
        batch_start = time.perf_counter()
        frames_bgr = [item["frame_bgr"] for item in pending_da3]
        try:
            depth_batch = run_da3_inference_batch(da3, frames_bgr)
            if len(depth_batch) != len(pending_da3):
                raise RuntimeError(
                    f"DA3 returned {len(depth_batch)} depth maps for {len(pending_da3)} frames."
                )
            batch_ms = (time.perf_counter() - batch_start) * 1000.0
            per_frame_da3_ms = batch_ms / max(1, len(pending_da3))
            for item, depth in zip(pending_da3, depth_batch):
                finalize_record(
                    item["rec"],
                    frame_idx=item["frame_idx"],
                    depth=depth,
                    union_mask_bool=item["union_mask_bool"],
                    center_xy=item["center_xy"],
                    prompt_masks=item["prompt_masks"],
                    frame_start=item["frame_start"],
                    da3_ms=per_frame_da3_ms,
                )
        except Exception as exc:
            for item in pending_da3:
                rec = item["rec"]
                rec["status"] = "da3_error"
                rec["error"] = f"{type(exc).__name__}: {exc}"
                rec["timing_ms"]["total_ms"] = (time.perf_counter() - item["frame_start"]) * 1000.0
                frame_rows.append(rec)
                counts["error_frames"] += 1
        finally:
            pending_da3 = []

    meta_cap = cv2.VideoCapture(str(video_path))
    try:
        if not meta_cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        video_fps = float(meta_cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_width = int(meta_cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        frame_height = int(meta_cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        frame_count_est = int(meta_cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        meta_cap.release()

    sample_interval = compute_sample_interval(video_fps, args.target_fps)
    estimated_sampled_frames = (
        int(math.ceil(frame_count_est / sample_interval)) if frame_count_est > 0 else None
    )
    estimated_sampled_batches = (
        int(math.ceil(estimated_sampled_frames / da3_batch_size))
        if estimated_sampled_frames is not None and estimated_sampled_frames > 0
        else None
    )
    batch_progress_done = 0
    batch_progress = (
        tqdm(
            total=estimated_sampled_batches,
            desc=f"{video_stem} sampled-batches",
            unit="batch",
            leave=False,
            dynamic_ncols=True,
        )
        if tqdm is not None
        else None
    )

    def update_batch_progress() -> None:
        nonlocal batch_progress_done
        if batch_progress is None:
            return
        completed_batches = counts["sampled_frames"] // da3_batch_size
        delta = completed_batches - batch_progress_done
        if delta > 0:
            batch_progress.update(delta)
            batch_progress_done = completed_batches

    def finalize_batch_progress() -> None:
        nonlocal batch_progress_done
        if batch_progress is None:
            return
        total_batches = int(math.ceil(counts["sampled_frames"] / da3_batch_size)) if counts["sampled_frames"] > 0 else 0
        delta = total_batches - batch_progress_done
        if delta > 0:
            batch_progress.update(delta)
            batch_progress_done = total_batches
        batch_progress.close()

    video_json["video_fps"] = video_fps
    video_json["sample_interval_frames"] = int(sample_interval)
    video_json["frame_width"] = frame_width
    video_json["frame_height"] = frame_height

    try:
        if args.sam3_mode == "frame":
            if sam3_frame is None:
                raise RuntimeError("SAM3 frame predictor is not initialized.")
            cap = cv2.VideoCapture(str(video_path))
            try:
                frame_idx = 0
                while True:
                    ret, frame_bgr = cap.read()
                    if not ret:
                        if frame_count_est > 0 and frame_idx < frame_count_est - 1:
                            timestamp = float(frame_idx / video_fps) if video_fps > 0 else None
                            rec = base_frame_record(frame_idx, timestamp)
                            rec["sam3_mode"] = "frame"
                            rec["track_summary"] = None
                            rec["status"] = "frame_decode_error"
                            rec["timing_ms"]["total_ms"] = 0.0
                            frame_rows.append(rec)
                            counts["error_frames"] += 1
                        break

                    if frame_idx % sample_interval != 0:
                        frame_idx += 1
                        continue

                    counts["sampled_frames"] += 1
                    update_batch_progress()
                    timestamp = float(frame_idx / video_fps) if video_fps > 0 else None
                    rec = base_frame_record(frame_idx, timestamp)
                    rec["sam3_mode"] = "frame"
                    rec["track_summary"] = None
                    frame_start = time.perf_counter()

                    try:
                        sam_start = time.perf_counter()
                        sam_results = sam3_frame(source=frame_bgr, text=args.sam3_text_prompts)
                        sam_result = sam_results[0]
                        prompt_masks = extract_prompt_masks(
                            sam_result,
                            args.sam3_text_prompts,
                            frame_bgr.shape[:2],
                        )
                        rec["timing_ms"]["sam_ms"] = (time.perf_counter() - sam_start) * 1000.0
                    except Exception as exc:
                        rec["status"] = "sam_error"
                        rec["timing_ms"]["total_ms"] = (time.perf_counter() - frame_start) * 1000.0
                        rec["error"] = f"{type(exc).__name__}: {exc}"
                        frame_rows.append(rec)
                        counts["error_frames"] += 1
                        frame_idx += 1
                        continue

                    union_mask_bool = np.zeros(frame_bgr.shape[:2], dtype=bool)
                    for mask in prompt_masks:
                        union_mask_bool |= mask.astype(bool)
                    nonzero_pixels, area_fraction, bbox_xyxy, center_xy = compute_mask_geometry(union_mask_bool)
                    rec["mask_nonzero_pixels"] = nonzero_pixels
                    rec["mask_area_fraction"] = area_fraction
                    rec["bbox_xyxy"] = bbox_xyxy
                    rec["center_xy"] = center_xy

                    if nonzero_pixels == 0:
                        rec["status"] = "empty_mask"
                        rec["timing_ms"]["total_ms"] = (time.perf_counter() - frame_start) * 1000.0
                        frame_rows.append(rec)
                        counts["empty_mask_frames"] += 1
                        frame_idx += 1
                        continue

                    pending_da3.append(
                        {
                            "frame_idx": frame_idx,
                            "rec": rec,
                            "frame_bgr": frame_bgr,
                            "union_mask_bool": union_mask_bool,
                            "center_xy": center_xy,
                            "prompt_masks": prompt_masks,
                            "frame_start": frame_start,
                        }
                    )
                    if len(pending_da3) >= da3_batch_size:
                        flush_pending_da3()
                    frame_idx += 1
            finally:
                cap.release()
        else:
            if sam3_track is None:
                raise RuntimeError("SAM3 track predictor is not initialized.")

            sampled_idx = 0
            track_stream = sam3_track(
                source=str(video_path),
                text=args.sam3_text_prompts,
                stream=True,
                vid_stride=sample_interval,
            )
            for sam_result in track_stream:
                default_frame_idx = sampled_idx * sample_interval
                dataset_frame = getattr(getattr(sam3_track, "dataset", None), "frame", None)
                frame_idx = int(dataset_frame) - 1 if isinstance(dataset_frame, int) and dataset_frame > 0 else default_frame_idx
                timestamp = float(frame_idx / video_fps) if video_fps > 0 else None
                rec = base_frame_record(frame_idx, timestamp)
                rec["sam3_mode"] = "track"
                frame_start = time.perf_counter()
                counts["sampled_frames"] += 1
                update_batch_progress()
                sampled_idx += 1

                try:
                    rec["track_summary"] = extract_track_summary(sam_result)
                    speed = getattr(sam_result, "speed", None)
                    if isinstance(speed, dict):
                        inference_ms = speed.get("inference")
                        rec["timing_ms"]["sam_ms"] = float(inference_ms) if inference_ms is not None else None

                    frame_bgr = getattr(sam_result, "orig_img", None)
                    if frame_bgr is None:
                        raise RuntimeError("SAM3 track result did not include orig_img.")

                    prompt_masks = extract_prompt_masks(
                        sam_result,
                        args.sam3_text_prompts,
                        frame_bgr.shape[:2],
                    )
                except Exception as exc:
                    rec["status"] = "sam_error"
                    rec["timing_ms"]["total_ms"] = (time.perf_counter() - frame_start) * 1000.0
                    rec["error"] = f"{type(exc).__name__}: {exc}"
                    if rec["track_summary"] is None:
                        rec["track_summary"] = {"active_track_count": 0, "tracks": []}
                    frame_rows.append(rec)
                    counts["error_frames"] += 1
                    continue

                union_mask_bool = np.zeros(frame_bgr.shape[:2], dtype=bool)
                for mask in prompt_masks:
                    union_mask_bool |= mask.astype(bool)
                nonzero_pixels, area_fraction, bbox_xyxy, center_xy = compute_mask_geometry(union_mask_bool)
                rec["mask_nonzero_pixels"] = nonzero_pixels
                rec["mask_area_fraction"] = area_fraction
                rec["bbox_xyxy"] = bbox_xyxy
                rec["center_xy"] = center_xy

                if nonzero_pixels == 0:
                    rec["status"] = "empty_mask"
                    rec["timing_ms"]["total_ms"] = (time.perf_counter() - frame_start) * 1000.0
                    frame_rows.append(rec)
                    counts["empty_mask_frames"] += 1
                    continue

                pending_da3.append(
                    {
                        "frame_idx": frame_idx,
                        "rec": rec,
                        "frame_bgr": frame_bgr,
                        "union_mask_bool": union_mask_bool,
                        "center_xy": center_xy,
                        "prompt_masks": prompt_masks,
                        "frame_start": frame_start,
                    }
                )
                if len(pending_da3) >= da3_batch_size:
                    flush_pending_da3()

        flush_pending_da3()
        write_npz_atomic(npz_path, npz_arrays)
        video_json["status"] = "success"
    except Exception as exc:
        video_json["status"] = "failed"
        video_json["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        finalize_batch_progress()

    finished_at = utc_now_iso()
    video_json["finished_at_utc"] = finished_at
    if video_json["status"] == "success":
        video_json["error"] = None
    write_json_atomic(json_path, video_json)

    duration_sec = time.perf_counter() - started_perf
    return {
        "video_name": video_stem,
        "video_path": str(video_path.resolve()),
        "status": video_json["status"],
        "error": video_json["error"],
        "json_path": str(json_path.resolve()),
        "npz_path": str(npz_path.resolve()),
        "counts": counts,
        "duration_sec": float(duration_sec),
    }

def main() -> None:
    args = parse_args()

    input_video_dir = Path(args.input_video_dir)
    if not input_video_dir.is_dir():
        raise FileNotFoundError(f"Input video directory not found: {input_video_dir}")
    if not os.path.isfile(args.sam3_model_path):
        raise FileNotFoundError(f"SAM3 model not found: {args.sam3_model_path}")

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "run_manifest.json"

    video_exts = parse_video_exts(args.video_exts)
    all_videos = discover_videos(input_video_dir, video_exts)
    if args.max_videos is not None:
        all_videos = all_videos[: args.max_videos]

    device = resolve_device(args.device)
    use_half = args.half and device.type == "cuda"
    prompt_slugs = build_prompt_slugs(args.sam3_text_prompts)
    da3_batch_size = int(args.da3_batch_size)

    da3 = DepthAnything3.from_pretrained(args.da3_model_id).to(device)
    sam3_overrides = dict(
        conf=args.conf,
        task="segment",
        mode="predict",
        model=args.sam3_model_path,
        half=use_half,
        save=False,
        verbose=False,
    )
    sam3_frame: SAM3SemanticPredictor | None = None
    sam3_track: Any = None
    if args.sam3_mode == "frame":
        sam3_frame = SAM3SemanticPredictor(overrides=sam3_overrides)
    else:
        if SAM3VideoSemanticPredictor is None:
            raise ImportError(
                "SAM3VideoSemanticPredictor is not available in this ultralytics build. "
                "Upgrade ultralytics or run with --sam3-mode frame."
            )
        sam3_track = SAM3VideoSemanticPredictor(overrides=sam3_overrides)

    manifest: dict[str, Any] = {
        "started_at_utc": utc_now_iso(),
        "finished_at_utc": None,
        "input_video_dir": str(input_video_dir.resolve()),
        "output_dir": str(output_root.resolve()),
        "video_exts": list(video_exts),
        "target_fps": float(args.target_fps),
        "sam3_mode": args.sam3_mode,
        "sam3_model_path": str(Path(args.sam3_model_path).resolve()),
        "sam3_prompts": list(args.sam3_text_prompts),
        "da3_model_id": args.da3_model_id,
        "da3_batch_size": int(da3_batch_size),
        "device": str(device),
        "overwrite": bool(args.overwrite),
        "max_videos": args.max_videos,
        "videos_discovered": len(all_videos),
        "videos": [],
        "summary": {},
    }
    update_and_write_manifest(manifest_path, manifest)

    if not all_videos:
        print("No videos found for processing.")
        return

    for idx, video_path in enumerate(all_videos, start=1):
        print(f"[{idx}/{len(all_videos)}] Processing {video_path.name} ...")
        entry = process_video(
            video_path=video_path,
            output_root=output_root,
            args=args,
            sam3_frame=sam3_frame,
            sam3_track=sam3_track,
            da3=da3,
            prompt_slugs=prompt_slugs,
            device=device,
            da3_batch_size=da3_batch_size,
        )
        manifest["videos"].append(entry)
        update_and_write_manifest(manifest_path, manifest)
        print(
            f"  -> {entry['status']} | sampled={entry['counts']['sampled_frames']} "
            f"processed={entry['counts']['processed_frames']} empty={entry['counts']['empty_mask_frames']} "
            f"errors={entry['counts']['error_frames']}"
        )

    summary = manifest["summary"]
    print(
        "Done. "
        f"videos: success={summary['processed_videos']} failed={summary['failed_videos']} "
        f"skipped={summary['skipped_videos']} | "
        f"frames: sampled={summary['sampled_frames']} processed={summary['processed_frames']} "
        f"empty={summary['empty_mask_frames']} errors={summary['error_frames']}"
    )


if __name__ == "__main__":
    main()
