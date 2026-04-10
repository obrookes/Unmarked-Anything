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

from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.camera_trap_masks import (
    MASK_ENCODING_COCO_RLE,
    MASK_ENCODING_RAW,
    MASK_STORAGE_FORMAT_CHOICES,
    encode_mask_to_rle_npz_payload,
    normalize_binary_mask,
)
from ultralytics.models.sam import SAM3SemanticPredictor
try:
    from ultralytics.models.sam import SAM3VideoSemanticPredictor
except ImportError:
    SAM3VideoSemanticPredictor = None


DEFAULT_VIDEO_EXTS = ".mp4,.mov,.avi,.mkv"
TRACK_ISOLATION_CHOICES = ("recreate", "reset", "both")
TRACK_TAIL_POLICY_CHOICES = ("warn_and_finalize", "fail_fast")
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


def create_sam3_track_predictor(sam3_overrides: dict[str, Any]) -> Any:
    if SAM3VideoSemanticPredictor is None:
        raise ImportError(
            "SAM3VideoSemanticPredictor is not available in this ultralytics build. "
            "Upgrade ultralytics or run with --sam3-mode frame."
        )
    return SAM3VideoSemanticPredictor(overrides=sam3_overrides)


def reset_sam3_track_predictor_state(sam3_track: Any) -> list[str]:
    actions: list[str] = []
    if sam3_track is None:
        return actions

    # Do not call reset_prompts() in reset mode.
    # Some ultralytics builds expect prompt-side model internals (for example language feature caches)
    # to persist across calls after model setup; clearing them here can cause KeyError on next video.
    if callable(getattr(sam3_track, "reset_prompts", None)):
        actions.append("reset_prompts_skipped")

    reset_image = getattr(sam3_track, "reset_image", None)
    if callable(reset_image):
        reset_image()
        actions.append("reset_image")

    if hasattr(sam3_track, "inference_state"):
        inference_state = getattr(sam3_track, "inference_state")
        if isinstance(inference_state, dict):
            inference_state.clear()
            actions.append("inference_state.clear")
        else:
            setattr(sam3_track, "inference_state", {})
            actions.append("inference_state={}")
    else:
        setattr(sam3_track, "inference_state", {})
        actions.append("inference_state={}")

    for attr_name in ("dataset", "batch", "results"):
        if hasattr(sam3_track, attr_name):
            setattr(sam3_track, attr_name, None)
            actions.append(f"{attr_name}=None")
    if hasattr(sam3_track, "seen"):
        setattr(sam3_track, "seen", 0)
        actions.append("seen=0")

    tracker = getattr(sam3_track, "tracker", None)
    if tracker is None:
        return actions

    tracker_reset_image = getattr(tracker, "reset_image", None)
    if callable(tracker_reset_image):
        tracker_reset_image()
        actions.append("tracker.reset_image")

    if hasattr(tracker, "inference_state"):
        tracker_inference_state = getattr(tracker, "inference_state")
        if isinstance(tracker_inference_state, dict):
            tracker_inference_state.clear()
            actions.append("tracker.inference_state.clear")
        else:
            setattr(tracker, "inference_state", {})
            actions.append("tracker.inference_state={}")

    return actions


def prepare_sam3_track_predictor_for_video(
    *,
    sam3_track: Any,
    isolation_mode: str,
    sam3_overrides: dict[str, Any],
) -> tuple[Any, list[str]]:
    actions: list[str] = []
    if isolation_mode not in TRACK_ISOLATION_CHOICES:
        raise ValueError(
            f"Unknown --sam3-track-isolation '{isolation_mode}'. "
            f"Expected one of: {', '.join(TRACK_ISOLATION_CHOICES)}."
        )

    if isolation_mode == "recreate":
        return create_sam3_track_predictor(sam3_overrides), ["recreate"]

    if isolation_mode == "reset":
        if sam3_track is None:
            sam3_track = create_sam3_track_predictor(sam3_overrides)
            actions.append("create")
        reset_actions = reset_sam3_track_predictor_state(sam3_track)
        actions.append("reset")
        actions.extend([f"reset:{name}" for name in reset_actions])
        return sam3_track, actions

    # isolation_mode == "both"
    if sam3_track is not None:
        reset_actions = reset_sam3_track_predictor_state(sam3_track)
        actions.append("reset_previous")
        actions.extend([f"reset:{name}" for name in reset_actions])
    sam3_track = create_sam3_track_predictor(sam3_overrides)
    actions.append("recreate")
    return sam3_track, actions


def validate_track_state_num_frames(
    *,
    sam3_track: Any,
    video_stem: str,
    expected_total_frames: int | None,
    expected_sampled_frames: int | None,
    sample_interval: int,
) -> int | None:
    inference_state = getattr(sam3_track, "inference_state", None)
    num_frames = (
        int(inference_state.get("num_frames"))
        if isinstance(inference_state, dict) and inference_state.get("num_frames") is not None
        else None
    )
    # SAM3VideoSemanticPredictor tracks full video length (`dataset.frames`), not sampled frame count.
    # Validate against total input frames and keep sampled-frame info for diagnostics only.
    if (
        expected_total_frames is not None
        and num_frames is not None
        and abs(num_frames - expected_total_frames) > 1
    ):
        raise RuntimeError(
            "SAM3 track predictor state mismatch for video "
            f"'{video_stem}': inference_state.num_frames={num_frames}, "
            f"expected_total_frames={expected_total_frames}, "
            f"expected_sampled_frames={expected_sampled_frames}, sample_interval={sample_interval}. "
            "This usually indicates cross-video state leakage."
        )
    return num_frames


def handle_track_stream_index_error(
    *,
    exc: Exception,
    video_stem: str,
    sampled_idx: int,
    dataset_frame: Any,
    sample_interval: int,
    isolation_mode: str,
    tail_policy: str,
    warnings: list[str],
) -> bool:
    context = (
        f"video={video_stem} sampled_idx={sampled_idx} dataset_frame={dataset_frame} "
        f"sample_interval={sample_interval} isolation={isolation_mode}"
    )
    if tail_policy == "warn_and_finalize":
        warning = (
            "SAM3 track stream ended with IndexError; finalizing partial results "
            f"({context}): {type(exc).__name__}: {exc}"
        )
        warnings.append(warning)
        print(f"Warning [{video_stem}]: {warning}")
        return True

    if tail_policy == "fail_fast":
        raise RuntimeError(
            "SAM3 track stream raised IndexError and fail-fast policy is enabled "
            f"({context}): {type(exc).__name__}: {exc}"
        ) from exc

    raise ValueError(
        f"Unknown --sam3-track-tail-policy '{tail_policy}'. "
        f"Expected one of: {', '.join(TRACK_TAIL_POLICY_CHOICES)}."
    )


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


def extract_prompt_masks_and_objects(
    sam_result: Any,
    prompts: list[str],
    prompt_slugs: list[str],
    frame_shape_hw: tuple[int, int],
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    height, width = frame_shape_hw
    prompt_masks = [np.zeros((height, width), dtype=bool) for _ in prompts]
    object_rows: list[dict[str, Any]] = []

    if sam_result.masks is None or sam_result.masks.data is None:
        return [mask.astype(np.uint8) for mask in prompt_masks], object_rows

    masks_data = sam_result.masks.data.detach().cpu().numpy()
    if masks_data.size == 0:
        return [mask.astype(np.uint8) for mask in prompt_masks], object_rows
    det_masks = masks_data > 0
    num_dets = det_masks.shape[0]

    boxes = sam_result.boxes
    xyxy = (
        boxes.xyxy.detach().cpu().numpy()
        if boxes is not None and boxes.xyxy is not None
        else np.zeros((num_dets, 4), dtype=np.float32)
    )
    confs = (
        boxes.conf.detach().cpu().numpy()
        if boxes is not None and boxes.conf is not None
        else np.zeros((num_dets,), dtype=np.float32)
    )

    cls_ids = None
    if boxes is not None and boxes.cls is not None:
        cls_ids = boxes.cls.detach().cpu().numpy().astype(np.int64)
    track_ids = (
        boxes.id.detach().cpu().numpy().astype(np.int64)
        if boxes is not None and getattr(boxes, "is_track", False) and boxes.id is not None
        else np.full((num_dets,), -1, dtype=np.int64)
    )
    names = sam_result.names if hasattr(sam_result, "names") else None

    for det_idx in range(num_dets):
        det_mask = det_masks[det_idx]
        if det_mask.shape != (height, width):
            det_mask = cv2.resize(
                det_mask.astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        cls_id = int(cls_ids[det_idx]) if cls_ids is not None and det_idx < len(cls_ids) else -1
        prompt_idx = None
        if len(prompts) > 0:
            if 0 <= cls_id < len(prompts):
                prompt_idx = cls_id
            elif isinstance(names, dict) and cls_id in names:
                prompt_idx = _find_prompt_idx_by_label(str(names[cls_id]), prompts)

            # Fallback for unknown class mapping: assign to first prompt.
            if prompt_idx is None:
                prompt_idx = 0
            prompt_masks[prompt_idx] |= det_mask

        mask_pixels = int(det_mask.sum())
        bbox_xyxy: list[int] | None = None
        center_xy: list[int] | None = None
        if mask_pixels > 0:
            ys, xs = np.where(det_mask)
            xmin, xmax = int(xs.min()), int(xs.max())
            ymin, ymax = int(ys.min()), int(ys.max())
            bbox_xyxy = [xmin, ymin, xmax, ymax]
            center_xy = [int((xmin + xmax) // 2), int((ymin + ymax) // 2)]
        elif det_idx < len(xyxy):
            x1, y1, x2, y2 = [int(round(v)) for v in xyxy[det_idx].tolist()]
            x1 = max(0, min(x1, width - 1))
            x2 = max(0, min(x2, width - 1))
            y1 = max(0, min(y1, height - 1))
            y2 = max(0, min(y2, height - 1))
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1
            bbox_xyxy = [x1, y1, x2, y2]
            center_xy = [int((x1 + x2) // 2), int((y1 + y2) // 2)]

        label = str(cls_id)
        if isinstance(names, dict) and cls_id in names:
            label = str(names[cls_id])

        object_rows.append(
            {
                "object_index": int(det_idx),
                "track_id": int(track_ids[det_idx]) if det_idx < len(track_ids) and int(track_ids[det_idx]) >= 0 else None,
                "prompt_index": int(prompt_idx) if prompt_idx is not None else None,
                "prompt": prompts[prompt_idx] if prompt_idx is not None and prompt_idx < len(prompts) else None,
                "slug": prompt_slugs[prompt_idx] if prompt_idx is not None and prompt_idx < len(prompt_slugs) else None,
                "label": label,
                "confidence": float(confs[det_idx]) if det_idx < len(confs) else None,
                "bbox_xyxy": bbox_xyxy,
                "center_xy": center_xy,
                "mask_nonzero_pixels": mask_pixels,
                "mask": det_mask.astype(np.uint8),
            }
        )

    return [mask.astype(np.uint8) for mask in prompt_masks], object_rows


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
        _npz_sink = npz_arrays if args.npz else {}
        if args.npz:
            npz_arrays[depth_key] = np.asarray(depth, dtype=np.float32)
            npz_arrays[prompt_order_key] = np.asarray(args.sam3_text_prompts, dtype=np.str_)

        mask_key_rows = []
        for prompt_idx, (prompt, prompt_slug, mask) in enumerate(
            zip(args.sam3_text_prompts, prompt_slugs, prompt_masks)
        ):
            mask_key_rows.append(
                build_mask_storage_entry(
                    npz_arrays=_npz_sink,
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
            obj_mask_bool = obj_mask > 0
            if obj_mask_bool.shape != depth.shape:
                obj_mask_bool = cv2.resize(
                    obj_mask_bool.astype(np.uint8),
                    (depth.shape[1], depth.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            obj_depth_values = depth[obj_mask_bool]
            obj_depth_mask_mean = float(np.nanmean(obj_depth_values)) if obj_depth_values.size > 0 else None
            obj_center_xy = obj.get("center_xy")
            obj_depth_center_value = None
            if obj_center_xy is not None:
                cx, cy = obj_center_xy
                obj_depth_center_value = float(depth[cy, cx])
            object_key_rows.append(
                build_mask_storage_entry(
                    npz_arrays=_npz_sink,
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
                        "center_xy": obj_center_xy,
                        "mask_nonzero_pixels": int(obj.get("mask_nonzero_pixels") or 0),
                        "depth_mask_mean": obj_depth_mask_mean,
                        "depth_center_value": obj_depth_center_value,
                    },
                    storage_format=args.mask_storage_format,
                )
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
                    sample_seq_idx = len(sampled_frames_bgr)
                    sampled_frames_bgr.append(frame_bgr)
                    timestamp = float(frame_idx / video_fps) if video_fps > 0 else None
                    rec = base_frame_record(frame_idx, timestamp)
                    rec["sam3_mode"] = "frame"
                    rec["track_summary"] = None
                    frame_start = time.perf_counter()

                    try:
                        sam_start = time.perf_counter()
                        sam_results = sam3_frame(source=frame_bgr, text=args.sam3_text_prompts)
                        sam_result = sam_results[0]
                        prompt_masks, object_entries = extract_prompt_masks_and_objects(
                            sam_result,
                            args.sam3_text_prompts,
                            prompt_slugs,
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
                    frame_idx += 1
            finally:
                cap.release()
        else:
            if sam3_track is None:
                raise RuntimeError("SAM3 track predictor is not initialized.")

            sampled_idx = 0
            validated_track_state = False
            track_stream = sam3_track(
                source=str(video_path),
                text=args.sam3_text_prompts,
                stream=True,
                vid_stride=sample_interval,
            )
            track_iter = iter(track_stream)
            while True:
                try:
                    sam_result = next(track_iter)
                except StopIteration:
                    break
                except IndexError as exc:
                    dataset_frame = getattr(getattr(sam3_track, "dataset", None), "frame", None)
                    should_finalize = handle_track_stream_index_error(
                        exc=exc,
                        video_stem=video_stem,
                        sampled_idx=sampled_idx,
                        dataset_frame=dataset_frame,
                        sample_interval=sample_interval,
                        isolation_mode=args.sam3_track_isolation,
                        tail_policy=args.sam3_track_tail_policy,
                        warnings=video_json["warnings"],
                    )
                    if should_finalize:
                        break

                default_frame_idx = sampled_idx * sample_interval
                dataset_frame = getattr(getattr(sam3_track, "dataset", None), "frame", None)
                frame_idx = (
                    int(dataset_frame) - 1
                    if isinstance(dataset_frame, int) and dataset_frame > 0
                    else default_frame_idx
                )
                timestamp = float(frame_idx / video_fps) if video_fps > 0 else None
                rec = base_frame_record(frame_idx, timestamp)
                rec["sam3_mode"] = "track"
                frame_start = time.perf_counter()
                counts["sampled_frames"] += 1
                update_batch_progress()
                sampled_idx += 1
                if not validated_track_state:
                    video_json["sam3_tracker_num_frames"] = validate_track_state_num_frames(
                        sam3_track=sam3_track,
                        video_stem=video_stem,
                        expected_total_frames=frame_count_est if frame_count_est > 0 else None,
                        expected_sampled_frames=estimated_sampled_frames,
                        sample_interval=sample_interval,
                    )
                    validated_track_state = True

                try:
                    rec["track_summary"] = extract_track_summary(sam_result)
                    speed = getattr(sam_result, "speed", None)
                    if isinstance(speed, dict):
                        inference_ms = speed.get("inference")
                        rec["timing_ms"]["sam_ms"] = float(inference_ms) if inference_ms is not None else None

                    frame_bgr = getattr(sam_result, "orig_img", None)
                    if frame_bgr is None:
                        raise RuntimeError("SAM3 track result did not include orig_img.")
                    sample_seq_idx = len(sampled_frames_bgr)
                    sampled_frames_bgr.append(frame_bgr)

                    prompt_masks, object_entries = extract_prompt_masks_and_objects(
                        sam_result,
                        args.sam3_text_prompts,
                        prompt_slugs,
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
        sam3_track = None

    manifest: dict[str, Any] = {
        "started_at_utc": utc_now_iso(),
        "finished_at_utc": None,
        "input_video_dir": str(input_video_dir.resolve()),
        "output_dir": str(output_root.resolve()),
        "video_exts": list(video_exts),
        "target_fps": float(args.target_fps),
        "sam3_mode": args.sam3_mode,
        "sam3_track_isolation": args.sam3_track_isolation if args.sam3_mode == "track" else None,
        "sam3_track_tail_policy": args.sam3_track_tail_policy if args.sam3_mode == "track" else None,
        "sam3_model_path": str(Path(args.sam3_model_path).resolve()),
        "sam3_prompts": list(args.sam3_text_prompts),
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
        current_sam3_track = sam3_track
        video_stem = video_path.stem
        video_out_dir = output_root / video_stem
        output_exists = (
            (video_out_dir / f"{video_stem}.json").exists()
            and (video_out_dir / f"{video_stem}_arrays.npz").exists()
        )
        needs_processing = bool(args.overwrite) or not output_exists

        if args.sam3_mode == "track" and needs_processing:
            current_sam3_track, track_setup_actions = prepare_sam3_track_predictor_for_video(
                sam3_track=sam3_track,
                isolation_mode=args.sam3_track_isolation,
                sam3_overrides=sam3_overrides,
            )
            sam3_track = current_sam3_track
            print(
                "  -> track setup: "
                f"{', '.join(track_setup_actions) if track_setup_actions else 'none'}"
            )

        entry = process_video(
            video_path=video_path,
            output_root=output_root,
            args=args,
            sam3_frame=sam3_frame,
            sam3_track=current_sam3_track,
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
