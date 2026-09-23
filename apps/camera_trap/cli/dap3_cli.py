#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
import shutil
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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.camera_trap_masks import (
    MASK_ENCODING_COCO_RLE,
    MASK_ENCODING_RAW,
    MASK_STORAGE_FORMAT_CHOICES,
    encode_mask_to_rle_npz_payload,
    normalize_binary_mask,
)
from apps.camera_trap import sam3_backends
from apps.camera_trap.sam3_backends import (
    TRACK_ISOLATION_CHOICES,
    TRACK_TAIL_POLICY_CHOICES,
    OfficialSam3Backend,
    UltralyticsBackend,
    create_sam3_track_predictor,
    extract_prompt_masks_and_objects,
    extract_track_summary,
    handle_track_stream_index_error,
    prepare_sam3_track_predictor_for_video,
    reset_sam3_track_predictor_state,
    validate_track_state_num_frames,
)


DEFAULT_VIDEO_EXTS = ".mp4,.mov,.avi,.mkv"
SAM3_BACKEND_CHOICES = ("official", "ultralytics")
DA3_MODE_CHOICES = ("batch", "stream", "all_frames")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch pipeline: sample frames from videos, run SAM3 segmentation with text prompts, "
            "then run Depth Anything 3 with either batch inference or DA3-Streaming."
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
    parser.add_argument(
        "--da3-mode",
        choices=DA3_MODE_CHOICES,
        default="batch",
        help=(
            "DA3 execution mode: standard batched inference on SAM-positive frames ('batch'), "
            "DA3-Streaming on all sampled frames ('stream'), or standard DA3 on all sampled "
            "frames ('all_frames')."
        ),
    )
    parser.add_argument(
        "--da3-stream-config",
        default=str(REPO_ROOT / "da3_streaming" / "configs" / "base_config.yaml"),
        help="Path to DA3-Streaming config YAML used when --da3-mode stream.",
    )
    parser.add_argument("--target-fps", type=float, default=1.0, help="Sampling rate for processing.")
    parser.add_argument(
        "--depth-interval-seconds",
        type=float,
        default=2.0,
        help=(
            "Run DA3 depth only on frames at this wall-clock interval (a grid aligned with "
            "export_job_distances_csv.py's sampling rule), instead of on every sampled "
            "non-empty-mask frame. 0 disables the grid (old behaviour)."
        ),
    )
    parser.add_argument(
        "--sam3-mode",
        choices=["track", "frame"],
        default="track",
        help="SAM3 inference mode: native video tracking ('track') or per-frame segmentation ('frame').",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="SAM3 confidence threshold.")
    parser.add_argument(
        "--sam3-backend",
        choices=SAM3_BACKEND_CHOICES,
        default="official",
        help=(
            "SAM3 backend: 'official' uses the facebookresearch/sam3 package (SA-FARI "
            "checkpoints; --sam3-mode track only), 'ultralytics' uses the ultralytics SAM3 "
            "port (track or frame)."
        ),
    )
    parser.add_argument(
        "--sam3-det-threshold",
        type=float,
        default=0.5,
        help=(
            "Detection/presence score threshold passed to the official SAM3 backend's "
            "propagate_in_video call. Ignored by --sam3-backend ultralytics, which uses --conf."
        ),
    )
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
        help=(
            "Positive integer required by CLI. In batch mode this is DA3 batch size for "
            "SAM-positive sampled frames; in stream mode it is retained for compatibility/progress."
        ),
    )
    parser.add_argument(
        "--sam3-track-isolation",
        choices=TRACK_ISOLATION_CHOICES,
        default="recreate",
        help=(
            "Track-mode predictor isolation strategy per video: "
            "'recreate' (fresh predictor each video), "
            "'reset' (reuse predictor but clear state), "
            "'both' (reset previous predictor then recreate)."
        ),
    )
    parser.add_argument(
        "--sam3-track-tail-policy",
        choices=TRACK_TAIL_POLICY_CHOICES,
        default="warn_and_finalize",
        help=(
            "Behavior when SAM3 track stream raises IndexError: "
            "'warn_and_finalize' keeps partial results, "
            "'fail_fast' marks the video failed."
        ),
    )
    parser.add_argument(
        "--mask-storage-format",
        choices=MASK_STORAGE_FORMAT_CHOICES,
        default="rle",
        help=(
            "Mask persistence format in *_arrays.npz: "
            "'raw' writes dense uint8 masks, "
            "'rle' writes COCO RLE payloads, "
            "'both' writes both and keeps the dense key as the canonical reader key."
        ),
    )
    parser.add_argument(
        "--npz",
        action="store_true",
        help="Also write the *_arrays.npz file alongside the JSON output.",
    )

    args = parser.parse_args(argv)
    if args.target_fps <= 0:
        parser.error("--target-fps must be > 0.")
    if args.max_videos is not None and args.max_videos <= 0:
        parser.error("--max-videos must be > 0 when provided.")
    if args.da3_batch_size <= 0:
        parser.error("--da3-batch-size must be > 0.")
    if args.depth_interval_seconds < 0:
        parser.error("--depth-interval-seconds must be >= 0.")
    if args.sam3_backend == "official" and args.sam3_mode == "frame":
        parser.error("--sam3-backend official currently supports --sam3-mode track only.")
    return args


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def build_mask_storage_entry(
    *,
    npz_arrays: dict[str, np.ndarray],
    key_prefix: str,
    mask: np.ndarray,
    base_entry: dict[str, Any],
    storage_format: str,
) -> dict[str, Any]:
    mask_u8 = normalize_binary_mask(mask)
    entry = dict(base_entry)
    entry["size"] = [int(mask_u8.shape[0]), int(mask_u8.shape[1])]

    raw_key = f"{key_prefix}_raw"
    rle_key = f"{key_prefix}_rle"

    if storage_format in {"raw", "both"}:
        npz_arrays[raw_key] = mask_u8
        entry["raw_key"] = raw_key
    if storage_format in {"rle", "both"}:
        rle_payload, size = encode_mask_to_rle_npz_payload(mask_u8)
        npz_arrays[rle_key] = rle_payload
        entry["rle_key"] = rle_key
        entry["size"] = size

    if storage_format == "raw":
        entry["encoding"] = MASK_ENCODING_RAW
        entry["key"] = raw_key
    elif storage_format == "rle":
        entry["encoding"] = MASK_ENCODING_COCO_RLE
        entry["key"] = rle_key
    elif storage_format == "both":
        entry["encoding"] = MASK_ENCODING_RAW
        entry["key"] = raw_key
    else:
        raise ValueError(f"Unsupported mask storage format: {storage_format!r}")

    return entry


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
        "tracked_frames": 0,
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
        summary["tracked_frames"] += int(counts.get("tracked_frames", 0))
    return summary


def update_and_write_manifest(manifest_path: Path, manifest: dict[str, Any]) -> None:
    manifest["finished_at_utc"] = utc_now_iso()
    manifest["summary"] = build_manifest_summary(manifest["videos"])
    write_json_atomic(manifest_path, manifest)


def compute_sample_interval(video_fps: float, target_fps: float) -> int:
    if video_fps <= 0:
        return 1
    return max(1, round(video_fps / target_fps))


def compute_depth_grid_step(video_fps: float, depth_interval_seconds: float) -> int | None:
    """None means depth-grid disabled (old behaviour: DA3 on every sampled non-empty-mask frame).

    Mirrors export_job_distances_csv.py's `step_frames = round(interval_seconds * video_fps)`
    (min 1) so grid frame indices (`frame_idx % step == 0`) line up with what that exporter
    expects.
    """
    if depth_interval_seconds <= 0:
        return None
    if video_fps <= 0:
        return 1
    return max(1, round(video_fps * depth_interval_seconds))


def run_da3_inference_batch(da3: DepthAnything3, frames_bgr: list[np.ndarray]) -> list[np.ndarray]:
    frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]
    depth_pred = da3.inference(frames_rgb, use_ray_pose=False, infer_gs=False, export_dir=None)
    return [np.asarray(depth_map, dtype=np.float32) for depth_map in depth_pred.depth]


def run_da3_inference_stream(
    *,
    frames_bgr: list[np.ndarray],
    stream_config_path: Path,
    video_out_dir: Path,
    video_stem: str,
) -> list[np.ndarray]:
    if not frames_bgr:
        return []

    # DA3 expects ndarray inputs in RGB order, while OpenCV/SAM frames are BGR.
    frames_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_bgr]

    stream_root = REPO_ROOT / "da3_streaming"
    if str(stream_root) not in sys.path:
        sys.path.insert(0, str(stream_root))

    import da3_streaming as da3_streaming_module
    from loop_utils.config_utils import load_config as load_stream_config

    if not stream_config_path.is_file():
        raise FileNotFoundError(f"DA3-Streaming config not found: {stream_config_path}")

    config = load_stream_config(str(stream_config_path))
    config["Model"]["save_depth_conf_result"] = False

    def _resolve_stream_path(path_value: str) -> str:
        raw = str(path_value)
        candidate = Path(raw)
        if candidate.is_absolute() and candidate.exists():
            return str(candidate)
        normalized = raw[2:] if raw.startswith("./") else raw
        candidates = [
            stream_root / normalized,
            stream_config_path.parent / normalized,
            REPO_ROOT / normalized,
        ]
        for item in candidates:
            if item.exists():
                return str(item)
        return raw

    for weight_key in ("DA3", "DA3_CONFIG", "SALAD"):
        if weight_key in config.get("Weights", {}):
            config["Weights"][weight_key] = _resolve_stream_path(config["Weights"][weight_key])

    stream_output_dir = video_out_dir / "_da3_streaming_tmp"
    if stream_output_dir.exists():
        shutil.rmtree(stream_output_dir)
    stream_output_dir.mkdir(parents=True, exist_ok=True)

    runner = None
    try:
        runner = da3_streaming_module.DA3_Streaming(
            image_dir=f"in_memory/{video_stem}",
            save_dir=str(stream_output_dir),
            config=config,
            image_arrays=frames_rgb,
            image_arrays_color_order="rgb",
            collect_depth_only=True,
        )
        runner.run()
        depth_batch = runner.get_collected_depths()
    finally:
        if runner is not None:
            try:
                runner.close()
            finally:
                del runner
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        shutil.rmtree(stream_output_dir, ignore_errors=True)

    return [np.asarray(depth_map, dtype=np.float32) for depth_map in depth_batch]


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
        "depth_shape": None,
        "objects": [],
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


def filter_objects_by_conf(
    prompt_masks: list[np.ndarray],
    object_rows: list[dict[str, Any]] | None,
    conf: float,
    frame_shape_hw: tuple[int, int],
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    """Final confidence filter applied identically to both backends' output.

    A no-op for UltralyticsBackend (its predictor already filters by `--conf` internally, so
    every row already clears the threshold); this is the primary confidence filter for
    OfficialSam3Backend, whose `--sam3-det-threshold` only controls its own internal
    propagate_in_video call.
    """
    rows = object_rows or []
    kept = [row for row in rows if row.get("confidence") is None or float(row["confidence"]) >= conf]
    if len(kept) == len(rows):
        return prompt_masks, rows

    height, width = frame_shape_hw
    filtered_masks = [np.zeros((height, width), dtype=np.uint8) for _ in prompt_masks]
    for row in kept:
        prompt_idx = row.get("prompt_index")
        if prompt_idx is not None and 0 <= prompt_idx < len(filtered_masks):
            filtered_masks[prompt_idx] |= np.asarray(row["mask"], dtype=np.uint8)
    return filtered_masks, kept


def build_object_and_mask_npz_entries(
    *,
    npz_arrays: dict[str, np.ndarray],
    frame_prefix: str,
    prompt_masks: list[np.ndarray],
    prompt_slugs: list[str],
    object_entries: list[dict[str, Any]],
    args: argparse.Namespace,
    depth: np.ndarray | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Serialize prompt-union masks and per-object masks to npz, returning
    (mask_key_rows, object_key_rows) for the frame record.

    `depth` is the frame-resolution depth map used for per-object depth stats. When `depth` is
    None (non-grid "tracked" frames, which never run DA3), the per-object
    `depth_mask_mean`/`depth_center_value` fields are set to None without touching any depth
    array.
    """
    mask_key_rows = []
    for prompt_idx, (prompt, prompt_slug, mask) in enumerate(
        zip(args.sam3_text_prompts, prompt_slugs, prompt_masks)
    ):
        mask_key_rows.append(
            build_mask_storage_entry(
                npz_arrays=npz_arrays,
                key_prefix=f"{frame_prefix}_mask_{prompt_slug}",
                mask=mask,
                base_entry={
                    "prompt_index": int(prompt_idx),
                    "prompt": prompt,
                    "slug": prompt_slug,
                },
                storage_format=args.mask_storage_format,
            )
        )

    object_key_rows = []
    for obj in object_entries:
        obj_idx = int(obj.get("object_index", len(object_key_rows)))
        obj_mask = np.asarray(obj.get("mask"), dtype=np.uint8)
        obj_depth_mask_mean = None
        obj_depth_center_value = None
        if depth is not None:
            obj_mask_bool = obj_mask > 0
            if obj_mask_bool.shape != depth.shape:
                obj_mask_bool = cv2.resize(
                    obj_mask_bool.astype(np.uint8),
                    (depth.shape[1], depth.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            obj_depth_values = depth[obj_mask_bool]
            finite_obj_depth_values = obj_depth_values[np.isfinite(obj_depth_values)]
            obj_depth_mask_mean = float(np.mean(finite_obj_depth_values)) if finite_obj_depth_values.size > 0 else None
            obj_center_xy = obj.get("center_xy")
            if obj_center_xy is not None:
                cx, cy = obj_center_xy
                obj_depth_center_value = float(depth[cy, cx])
        object_key_rows.append(
            build_mask_storage_entry(
                npz_arrays=npz_arrays,
                key_prefix=f"{frame_prefix}_obj_{obj_idx}_mask",
                mask=obj_mask,
                base_entry={
                    "object_index": obj_idx,
                    "track_id": obj.get("track_id"),
                    "label": obj.get("label"),
                    "confidence": obj.get("confidence"),
                    "prompt_index": obj.get("prompt_index"),
                    "prompt": obj.get("prompt"),
                    "slug": obj.get("slug"),
                    "bbox_xyxy": obj.get("bbox_xyxy"),
                    "center_xy": obj.get("center_xy"),
                    "mask_nonzero_pixels": int(obj.get("mask_nonzero_pixels") or 0),
                    "depth_mask_mean": obj_depth_mask_mean,
                    "depth_center_value": obj_depth_center_value,
                },
                storage_format=args.mask_storage_format,
            )
        )

    return mask_key_rows, object_key_rows


def process_video(
    *,
    video_path: Path,
    output_root: Path,
    args: argparse.Namespace,
    backend: Any,
    da3: DepthAnything3 | None,
    da3_stream_config: Path,
    prompt_slugs: list[str],
    device: torch.device,
    da3_batch_size: int,
    track_setup_actions: list[str] | None = None,
) -> dict[str, Any]:
    video_stem = video_path.stem
    video_out_dir = output_root / video_stem
    json_path = video_out_dir / f"{video_stem}.json"
    npz_path = video_out_dir / f"{video_stem}_arrays.npz"

    already_done = json_path.exists() and (not args.npz or npz_path.exists())
    if not args.overwrite and already_done:
        return {
            "video_name": video_stem,
            "video_path": str(video_path.resolve()),
            "status": "skipped_existing",
            "error": None,
            "json_path": str(json_path.resolve()),
            "npz_path": None if not args.npz else str(npz_path.resolve()),
            "counts": {
                "sampled_frames": 0,
                "processed_frames": 0,
                "empty_mask_frames": 0,
                "error_frames": 0,
                "tracked_frames": 0,
            },
            "duration_sec": 0.0,
            "depth_grid_step_frames": None,
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
        "tracked_frames": 0,
    }

    video_json: dict[str, Any] = {
        "video_name": video_stem,
        "video_path": str(video_path.resolve()),
        "video_fps": 0.0,
        "target_fps": float(args.target_fps),
        "sam3_mode": args.sam3_mode,
        "sample_interval_frames": 1,
        "depth_interval_seconds": float(args.depth_interval_seconds),
        "depth_grid_step_frames": None,
        "frame_width": 0,
        "frame_height": 0,
        "sam3_prompts": list(args.sam3_text_prompts),
        "sam3_conf": float(args.conf),
        "sam3_backend": args.sam3_backend,
        "sam3_det_threshold": float(args.sam3_det_threshold),
        "sam3_track_isolation": args.sam3_track_isolation if args.sam3_mode == "track" else None,
        "sam3_track_tail_policy": args.sam3_track_tail_policy if args.sam3_mode == "track" else None,
        "sam3_track_setup_actions": list(track_setup_actions or []),
        "sam3_tracker_num_frames": None,
        "da3_mode": args.da3_mode,
        "da3_model_id": args.da3_model_id,
        "da3_stream_config": str(da3_stream_config.resolve()) if args.da3_mode == "stream" else None,
        "da3_batch_size": int(da3_batch_size),
        "device": str(device),
        "started_at_utc": started_at,
        "finished_at_utc": None,
        "status": "failed",
        "error": None,
        "warnings": [],
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
        object_entries: list[dict[str, Any]],
        frame_start: float,
        da3_ms: float,
    ) -> None:
        original_depth = depth
        if depth.shape != union_mask_bool.shape:
            depth = cv2.resize(
                depth,
                (union_mask_bool.shape[1], union_mask_bool.shape[0]),
                interpolation=cv2.INTER_CUBIC,
            )

        depth_values = depth[union_mask_bool]
        finite_depth_values = depth_values[np.isfinite(depth_values)]
        rec["depth_mask_mean"] = float(np.mean(finite_depth_values)) if finite_depth_values.size > 0 else None
        if center_xy is not None:
            cx, cy = center_xy
            rec["depth_center_value"] = float(depth[cy, cx])
        rec["depth_shape"] = [int(original_depth.shape[0]), int(original_depth.shape[1])]
        rec["status"] = "processed"
        rec["timing_ms"]["da3_ms"] = da3_ms

        frame_prefix = f"f{frame_idx}"
        depth_key = f"{frame_prefix}_depth"
        prompt_order_key = f"{frame_prefix}_prompt_order"
        _npz_sink = npz_arrays if args.npz else {}
        if args.npz:
            # Store the DA3-native-resolution depth map (pre-resize) as float16; the resized
            # frame-resolution `depth` above is only used for the mask-mean/center-value stats.
            npz_arrays[depth_key] = np.asarray(original_depth, dtype=np.float16)
            npz_arrays[prompt_order_key] = np.asarray(args.sam3_text_prompts, dtype=np.str_)

        mask_key_rows, object_key_rows = build_object_and_mask_npz_entries(
            npz_arrays=_npz_sink,
            frame_prefix=frame_prefix,
            prompt_masks=prompt_masks,
            prompt_slugs=prompt_slugs,
            object_entries=object_entries,
            args=args,
            depth=depth,
        )
        rec["objects"] = object_key_rows
        rec["npz_keys"] = None if not args.npz else {
            "depth": depth_key,
            "prompt_order": prompt_order_key,
            "masks": mask_key_rows,
            "objects": object_key_rows,
        }
        rec["timing_ms"]["total_ms"] = (time.perf_counter() - frame_start) * 1000.0
        frame_rows.append(rec)
        counts["processed_frames"] += 1

    pending_da3: list[dict[str, Any]] = []
    sampled_frames_bgr: list[np.ndarray] = []

    def flush_pending_da3() -> None:
        nonlocal pending_da3
        if not pending_da3:
            return
        if da3 is None:
            raise RuntimeError("DA3 batch model is not initialized.")
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
                    object_entries=item["object_entries"],
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

    def flush_da3_stream_inference() -> None:
        nonlocal pending_da3
        if not sampled_frames_bgr:
            pending_da3 = []
            return
        batch_start = time.perf_counter()
        try:
            depth_batch = run_da3_inference_stream(
                frames_bgr=sampled_frames_bgr,
                stream_config_path=da3_stream_config,
                video_out_dir=video_out_dir,
                video_stem=video_stem,
            )
            if len(depth_batch) != len(sampled_frames_bgr):
                raise RuntimeError(
                    f"DA3-Streaming returned {len(depth_batch)} depth maps for "
                    f"{len(sampled_frames_bgr)} sampled frames."
                )

            batch_ms = (time.perf_counter() - batch_start) * 1000.0
            per_frame_da3_ms = batch_ms / max(1, len(sampled_frames_bgr))
            for item in pending_da3:
                sample_seq_idx = int(item["sample_seq_idx"])
                if sample_seq_idx < 0 or sample_seq_idx >= len(depth_batch):
                    raise RuntimeError(
                        f"Sample index out of range for DA3-Streaming depth output: {sample_seq_idx}"
                    )
                depth = depth_batch[sample_seq_idx]
                finalize_record(
                    item["rec"],
                    frame_idx=item["frame_idx"],
                    depth=depth,
                    union_mask_bool=item["union_mask_bool"],
                    center_xy=item["center_xy"],
                    prompt_masks=item["prompt_masks"],
                    object_entries=item["object_entries"],
                    frame_start=item["frame_start"],
                    da3_ms=per_frame_da3_ms,
                )
        except Exception as exc:
            if pending_da3:
                for item in pending_da3:
                    rec = item["rec"]
                    rec["status"] = "da3_error"
                    rec["error"] = f"{type(exc).__name__}: {exc}"
                    rec["timing_ms"]["total_ms"] = (time.perf_counter() - item["frame_start"]) * 1000.0
                    frame_rows.append(rec)
                    counts["error_frames"] += 1
            else:
                video_json["warnings"].append(
                    f"DA3-Streaming inference failed with no pending detections: "
                    f"{type(exc).__name__}: {exc}"
                )
        finally:
            pending_da3 = []

    def flush_da3_all_frames_inference() -> None:
        nonlocal pending_da3
        if not sampled_frames_bgr:
            pending_da3 = []
            return
        if da3 is None:
            raise RuntimeError("DA3 batch model is not initialized.")

        batch_start = time.perf_counter()
        try:
            depth_batch = run_da3_inference_batch(da3, sampled_frames_bgr)
            if len(depth_batch) != len(sampled_frames_bgr):
                raise RuntimeError(
                    f"DA3 returned {len(depth_batch)} depth maps for "
                    f"{len(sampled_frames_bgr)} sampled frames."
                )

            batch_ms = (time.perf_counter() - batch_start) * 1000.0
            per_frame_da3_ms = batch_ms / max(1, len(sampled_frames_bgr))
            for item in pending_da3:
                sample_seq_idx = int(item["sample_seq_idx"])
                if sample_seq_idx < 0 or sample_seq_idx >= len(depth_batch):
                    raise RuntimeError(
                        f"Sample index out of range for DA3 all-frames output: {sample_seq_idx}"
                    )
                depth = depth_batch[sample_seq_idx]
                finalize_record(
                    item["rec"],
                    frame_idx=item["frame_idx"],
                    depth=depth,
                    union_mask_bool=item["union_mask_bool"],
                    center_xy=item["center_xy"],
                    prompt_masks=item["prompt_masks"],
                    object_entries=item["object_entries"],
                    frame_start=item["frame_start"],
                    da3_ms=per_frame_da3_ms,
                )
        except Exception as exc:
            if pending_da3:
                for item in pending_da3:
                    rec = item["rec"]
                    rec["status"] = "da3_error"
                    rec["error"] = f"{type(exc).__name__}: {exc}"
                    rec["timing_ms"]["total_ms"] = (time.perf_counter() - item["frame_start"]) * 1000.0
                    frame_rows.append(rec)
                    counts["error_frames"] += 1
            else:
                video_json["warnings"].append(
                    f"DA3 all-frames inference failed with no pending detections: "
                    f"{type(exc).__name__}: {exc}"
                )
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
    depth_grid_step = compute_depth_grid_step(video_fps, args.depth_interval_seconds)
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
    video_json["depth_grid_step_frames"] = depth_grid_step
    video_json["frame_width"] = frame_width
    video_json["frame_height"] = frame_height

    try:
        for result in backend.iter_video(
            video_path=video_path,
            mode=args.sam3_mode,
            prompts=args.sam3_text_prompts,
            prompt_slugs=prompt_slugs,
            sample_interval=sample_interval,
            video_fps=video_fps,
            frame_count_est=frame_count_est,
            estimated_sampled_frames=estimated_sampled_frames,
            warnings=video_json["warnings"],
            depth_grid_step=depth_grid_step,
        ):
            counts["sampled_frames"] += 1
            update_batch_progress()
            timestamp = float(result.frame_idx / video_fps) if video_fps > 0 else None
            rec = base_frame_record(result.frame_idx, timestamp)
            rec["sam3_mode"] = args.sam3_mode
            rec["track_summary"] = result.track_summary
            frame_start = time.perf_counter()

            if result.status == "frame_decode_error":
                rec["status"] = "frame_decode_error"
                rec["timing_ms"]["total_ms"] = 0.0
                frame_rows.append(rec)
                counts["error_frames"] += 1
                continue

            if result.status == "sam_error":
                rec["status"] = "sam_error"
                rec["error"] = result.error
                rec["timing_ms"]["total_ms"] = (time.perf_counter() - frame_start) * 1000.0
                if rec["track_summary"] is None:
                    rec["track_summary"] = {"active_track_count": 0, "tracks": []}
                frame_rows.append(rec)
                counts["error_frames"] += 1
                continue

            rec["timing_ms"]["sam_ms"] = result.timing_ms_sam
            frame_bgr = result.frame_bgr
            prompt_masks, object_entries = filter_objects_by_conf(
                result.prompt_masks, result.object_rows, args.conf, frame_bgr.shape[:2]
            )

            union_mask_bool = np.zeros(frame_bgr.shape[:2], dtype=bool)
            for mask in prompt_masks:
                union_mask_bool |= mask.astype(bool)
            nonzero_pixels, area_fraction, bbox_xyxy, center_xy = compute_mask_geometry(union_mask_bool)
            rec["mask_nonzero_pixels"] = nonzero_pixels
            rec["mask_area_fraction"] = area_fraction
            rec["bbox_xyxy"] = bbox_xyxy
            rec["center_xy"] = center_xy

            is_grid_frame = depth_grid_step is None or (result.frame_idx % depth_grid_step == 0)

            if is_grid_frame:
                if nonzero_pixels == 0:
                    rec["status"] = "empty_mask"
                    rec["timing_ms"]["total_ms"] = (time.perf_counter() - frame_start) * 1000.0
                    frame_rows.append(rec)
                    counts["empty_mask_frames"] += 1
                    continue

                sample_seq_idx = len(sampled_frames_bgr)
                sampled_frames_bgr.append(frame_bgr)
                pending_da3.append(
                    {
                        "frame_idx": result.frame_idx,
                        "sample_seq_idx": sample_seq_idx,
                        "rec": rec,
                        "frame_bgr": frame_bgr if args.da3_mode == "batch" else None,
                        "union_mask_bool": union_mask_bool,
                        "center_xy": center_xy,
                        "prompt_masks": prompt_masks,
                        "object_entries": object_entries,
                        "frame_start": frame_start,
                    }
                )
                if args.da3_mode == "batch" and len(pending_da3) >= da3_batch_size:
                    flush_pending_da3()
            else:
                # Non-grid frame: tracked/masked but no DA3 depth (never sent to DA3).
                frame_prefix = f"f{result.frame_idx}"
                prompt_order_key = f"{frame_prefix}_prompt_order"
                _npz_sink = npz_arrays if args.npz else {}
                if args.npz:
                    npz_arrays[prompt_order_key] = np.asarray(args.sam3_text_prompts, dtype=np.str_)
                mask_key_rows, object_key_rows = build_object_and_mask_npz_entries(
                    npz_arrays=_npz_sink,
                    frame_prefix=frame_prefix,
                    prompt_masks=prompt_masks,
                    prompt_slugs=prompt_slugs,
                    object_entries=object_entries,
                    args=args,
                    depth=None,
                )
                rec["objects"] = object_key_rows
                rec["depth_mask_mean"] = None
                rec["depth_center_value"] = None
                rec["depth_shape"] = None
                rec["status"] = "tracked"
                rec["npz_keys"] = None if not args.npz else {
                    "depth": None,
                    "prompt_order": prompt_order_key,
                    "masks": mask_key_rows,
                    "objects": object_key_rows,
                }
                rec["timing_ms"]["total_ms"] = (time.perf_counter() - frame_start) * 1000.0
                frame_rows.append(rec)
                counts["tracked_frames"] += 1

        if args.sam3_mode == "track":
            video_json["sam3_tracker_num_frames"] = getattr(backend, "last_tracker_num_frames", None)

        if args.da3_mode == "stream":
            flush_da3_stream_inference()
        elif args.da3_mode == "all_frames":
            flush_da3_all_frames_inference()
        else:
            flush_pending_da3()
        if args.npz:
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
        "npz_path": None if not args.npz else str(npz_path.resolve()),
        "counts": counts,
        "duration_sec": float(duration_sec),
        "depth_grid_step_frames": depth_grid_step,
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
    da3_stream_config = Path(args.da3_stream_config)
    if args.da3_mode == "stream" and not da3_stream_config.is_file():
        raise FileNotFoundError(f"DA3-Streaming config not found: {da3_stream_config}")

    da3: DepthAnything3 | None = None
    if args.da3_mode in ("batch", "all_frames"):
        da3 = DepthAnything3.from_pretrained(args.da3_model_id).to(device)

    backend: Any
    if args.sam3_backend == "ultralytics":
        backend = UltralyticsBackend(
            sam3_model_path=args.sam3_model_path,
            conf=args.conf,
            half=use_half,
            track_isolation=args.sam3_track_isolation,
            track_tail_policy=args.sam3_track_tail_policy,
        )
    else:
        backend = OfficialSam3Backend(
            sam3_model_path=args.sam3_model_path,
            det_threshold=args.sam3_det_threshold,
            device=str(device),
        )

    manifest: dict[str, Any] = {
        "started_at_utc": utc_now_iso(),
        "finished_at_utc": None,
        "input_video_dir": str(input_video_dir.resolve()),
        "output_dir": str(output_root.resolve()),
        "video_exts": list(video_exts),
        "target_fps": float(args.target_fps),
        "depth_interval_seconds": float(args.depth_interval_seconds),
        "sam3_mode": args.sam3_mode,
        "sam3_track_isolation": args.sam3_track_isolation if args.sam3_mode == "track" else None,
        "sam3_track_tail_policy": args.sam3_track_tail_policy if args.sam3_mode == "track" else None,
        "sam3_model_path": str(Path(args.sam3_model_path).resolve()),
        "sam3_prompts": list(args.sam3_text_prompts),
        "sam3_backend": args.sam3_backend,
        "sam3_det_threshold": float(args.sam3_det_threshold),
        "da3_mode": args.da3_mode,
        "da3_model_id": args.da3_model_id,
        "da3_stream_config": str(da3_stream_config.resolve()) if args.da3_mode == "stream" else None,
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
        track_setup_actions: list[str] = []
        video_stem = video_path.stem
        video_out_dir = output_root / video_stem
        output_exists = (
            (video_out_dir / f"{video_stem}.json").exists()
            and (not args.npz or (video_out_dir / f"{video_stem}_arrays.npz").exists())
        )
        needs_processing = bool(args.overwrite) or not output_exists

        if needs_processing:
            track_setup_actions = backend.prepare_for_video(mode=args.sam3_mode)
            if track_setup_actions:
                print(
                    "  -> track setup: "
                    f"{', '.join(track_setup_actions)}"
                )

        entry = process_video(
            video_path=video_path,
            output_root=output_root,
            args=args,
            backend=backend,
            da3=da3,
            da3_stream_config=da3_stream_config,
            prompt_slugs=prompt_slugs,
            device=device,
            da3_batch_size=da3_batch_size,
            track_setup_actions=track_setup_actions,
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
