#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from depth_anything_3.utils.camera_trap_masks import (
    decode_coco_rle_to_mask,
    normalize_binary_mask,
)


@dataclass(frozen=True)
class ResolvedVideo:
    video_name: str
    video_dir: Path
    json_path: Path
    npz_path: Path


@dataclass
class PairValidationResult:
    video_name: str
    frame_index: int
    mask_kind: str
    mask_label: str
    key: str
    raw_key: str
    rle_key: str
    size: list[int]
    raw_pixels: int
    rle_pixels: int
    diff_pixels: int
    iou: float
    exact_match: bool
    raw_member_bytes: int | None
    rle_member_bytes: int | None
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate that COCO RLE mask payloads in camera-trap outputs decode back to the "
            "stored full-size raw masks, and emit reports plus visual comparison videos."
        )
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-root", type=Path, default=None, help="Run directory containing per-video output folders.")
    group.add_argument("--video-dir", type=Path, default=None, help="Single per-video output directory.")
    parser.add_argument("--video-name", type=str, default=None, help="Optional video name when selecting from --run-root.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Destination for reports and visual artifacts. Defaults under the selected run/video directory.",
    )
    parser.add_argument("--max-videos", type=int, default=None, help="Optional cap on number of videos to validate.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on processed frames validated per video, starting from the earliest processed frames.",
    )
    parser.add_argument(
        "--visual-fps",
        type=float,
        default=2.0,
        help="FPS for processed-frame comparison videos. Applies to written comparison videos only.",
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.45,
        help="Mask alpha used in the raw/RLE visual overlays.",
    )
    parser.add_argument(
        "--skip-visuals",
        action="store_true",
        help="Disable comparison video generation and only write numeric reports.",
    )
    parser.add_argument(
        "--allow-missing-video",
        action="store_true",
        help="Do not fail validation if the source video path is unavailable; write mask-only comparison videos instead.",
    )
    parser.add_argument(
        "--strict-pairs",
        action="store_true",
        default=True,
        help="Treat missing raw/rle pair metadata or NPZ members as validation failures (default: enabled).",
    )
    parser.add_argument(
        "--no-strict-pairs",
        dest="strict_pairs",
        action="store_false",
        help="Allow entries with missing raw/rle pair metadata to be skipped instead of failing.",
    )
    args = parser.parse_args()
    if args.max_videos is not None and args.max_videos <= 0:
        parser.error("--max-videos must be > 0 when provided")
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be > 0 when provided")
    if args.visual_fps <= 0:
        parser.error("--visual-fps must be > 0")
    if not (0.0 <= args.overlay_alpha <= 1.0):
        parser.error("--overlay-alpha must be in [0, 1]")
    return args


def _find_single_json(dir_path: Path) -> Path:
    candidates = sorted(p for p in dir_path.glob("*.json") if p.name != "run_manifest.json")
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one per-video JSON in {dir_path}, found {len(candidates)}.")
    return candidates[0]


def _find_single_arrays(dir_path: Path) -> Path:
    candidates = sorted(dir_path.glob("*_arrays.npz"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one *_arrays.npz in {dir_path}, found {len(candidates)}.")
    return candidates[0]


def resolve_videos(
    *,
    run_root: Path | None,
    video_dir: Path | None,
    video_name: str | None,
    max_videos: int | None,
) -> list[ResolvedVideo]:
    if video_dir is not None:
        json_path = _find_single_json(video_dir)
        npz_path = _find_single_arrays(video_dir)
        return [ResolvedVideo(video_name=json_path.stem, video_dir=video_dir, json_path=json_path, npz_path=npz_path)]

    assert run_root is not None
    if not run_root.is_dir():
        raise FileNotFoundError(f"Run root not found: {run_root}")

    candidates: list[ResolvedVideo] = []
    for child in sorted(p for p in run_root.iterdir() if p.is_dir()):
        if video_name and child.name != video_name:
            continue
        json_candidates = sorted(p for p in child.glob("*.json") if p.name != "run_manifest.json")
        npz_candidates = sorted(child.glob("*_arrays.npz"))
        if len(json_candidates) == 1 and len(npz_candidates) == 1:
            candidates.append(
                ResolvedVideo(
                    video_name=json_candidates[0].stem,
                    video_dir=child,
                    json_path=json_candidates[0],
                    npz_path=npz_candidates[0],
                )
            )

    if not candidates:
        raise RuntimeError(f"No per-video JSON+NPZ pairs found under {run_root}")
    if max_videos is not None:
        candidates = candidates[:max_videos]
    return candidates


def _entry_label(entry: dict[str, Any], mask_kind: str) -> str:
    if mask_kind == "prompt":
        prompt = entry.get("prompt")
        slug = entry.get("slug")
        if isinstance(prompt, str) and prompt:
            return prompt
        if isinstance(slug, str) and slug:
            return slug
        return f"prompt-{entry.get('prompt_index', '?')}"

    obj_idx = entry.get("object_index", "?")
    track_id = entry.get("track_id")
    label = entry.get("label")
    parts = [f"obj{obj_idx}"]
    if track_id is not None:
        parts.append(f"track={track_id}")
    if label not in (None, ""):
        parts.append(f"label={label}")
    return " ".join(parts)


def _member_size_map(npz_path: Path) -> dict[str, int]:
    size_map: dict[str, int] = {}
    with zipfile.ZipFile(npz_path) as zf:
        for info in zf.infolist():
            if not info.filename.endswith(".npy"):
                continue
            size_map[Path(info.filename).stem] = int(info.compress_size)
    return size_map


def _counts_text_from_payload(payload: np.ndarray) -> str:
    if payload.shape == ():
        return str(payload.item())
    return str(payload.reshape(-1)[0])


def _safe_iou(raw_mask: np.ndarray, rle_mask: np.ndarray) -> float:
    raw_bool = raw_mask.astype(bool)
    rle_bool = rle_mask.astype(bool)
    union = np.logical_or(raw_bool, rle_bool)
    if not np.any(union):
        return 1.0
    inter = np.logical_and(raw_bool, rle_bool)
    return float(inter.sum() / union.sum())


def validate_mask_pair(
    *,
    video_name: str,
    frame_index: int,
    npz_data: np.lib.npyio.NpzFile,
    entry: dict[str, Any],
    mask_kind: str,
    member_sizes: dict[str, int],
    strict_pairs: bool,
) -> PairValidationResult | None:
    raw_key = entry.get("raw_key")
    rle_key = entry.get("rle_key")
    size = entry.get("size")
    label = _entry_label(entry, mask_kind)
    canonical_key = str(entry.get("key") or "")

    if not isinstance(raw_key, str) or not raw_key or not isinstance(rle_key, str) or not rle_key:
        if strict_pairs:
            return PairValidationResult(
                video_name=video_name,
                frame_index=frame_index,
                mask_kind=mask_kind,
                mask_label=label,
                key=canonical_key,
                raw_key=str(raw_key or ""),
                rle_key=str(rle_key or ""),
                size=list(size) if isinstance(size, (list, tuple)) else [],
                raw_pixels=0,
                rle_pixels=0,
                diff_pixels=0,
                iou=0.0,
                exact_match=False,
                raw_member_bytes=member_sizes.get(str(raw_key)) if isinstance(raw_key, str) else None,
                rle_member_bytes=member_sizes.get(str(rle_key)) if isinstance(rle_key, str) else None,
                error="missing raw_key or rle_key metadata",
            )
        return None

    if raw_key not in npz_data or rle_key not in npz_data:
        return PairValidationResult(
            video_name=video_name,
            frame_index=frame_index,
            mask_kind=mask_kind,
            mask_label=label,
            key=canonical_key,
            raw_key=raw_key,
            rle_key=rle_key,
            size=list(size) if isinstance(size, (list, tuple)) else [],
            raw_pixels=0,
            rle_pixels=0,
            diff_pixels=0,
            iou=0.0,
            exact_match=False,
            raw_member_bytes=member_sizes.get(raw_key),
            rle_member_bytes=member_sizes.get(rle_key),
            error="missing raw or rle member in npz",
        )

    try:
        raw_mask = normalize_binary_mask(np.asarray(npz_data[raw_key]))
        if not isinstance(size, (list, tuple)):
            raise ValueError("missing size metadata")
        rle_mask = decode_coco_rle_to_mask(
            counts=_counts_text_from_payload(np.asarray(npz_data[rle_key])),
            size=size,
        )
        if raw_mask.shape != rle_mask.shape:
            raise ValueError(f"shape mismatch {raw_mask.shape} vs {rle_mask.shape}")
    except Exception as exc:
        return PairValidationResult(
            video_name=video_name,
            frame_index=frame_index,
            mask_kind=mask_kind,
            mask_label=label,
            key=canonical_key,
            raw_key=raw_key,
            rle_key=rle_key,
            size=list(size) if isinstance(size, (list, tuple)) else [],
            raw_pixels=0,
            rle_pixels=0,
            diff_pixels=0,
            iou=0.0,
            exact_match=False,
            raw_member_bytes=member_sizes.get(raw_key),
            rle_member_bytes=member_sizes.get(rle_key),
            error=str(exc),
        )

    diff_pixels = int(np.count_nonzero(raw_mask != rle_mask))
    exact_match = diff_pixels == 0
    return PairValidationResult(
        video_name=video_name,
        frame_index=frame_index,
        mask_kind=mask_kind,
        mask_label=label,
        key=canonical_key,
        raw_key=raw_key,
        rle_key=rle_key,
        size=[int(v) for v in size],
        raw_pixels=int(raw_mask.sum()),
        rle_pixels=int(rle_mask.sum()),
        diff_pixels=diff_pixels,
        iou=_safe_iou(raw_mask, rle_mask),
        exact_match=exact_match,
        raw_member_bytes=member_sizes.get(raw_key),
        rle_member_bytes=member_sizes.get(rle_key),
        error=None,
    )


def _colorize_mask(mask: np.ndarray, color_rgb: tuple[int, int, int], alpha: float) -> np.ndarray:
    mask_bool = mask.astype(bool)
    rgb = np.zeros(mask.shape + (3,), dtype=np.uint8)
    color_arr = np.asarray(color_rgb, dtype=np.uint8)
    rgb[mask_bool] = color_arr
    if alpha >= 1.0:
        return rgb
    return (rgb.astype(np.float32) * alpha).astype(np.uint8)


def _overlay_mask(frame_rgb: np.ndarray, mask: np.ndarray, color_rgb: tuple[int, int, int], alpha: float) -> np.ndarray:
    out = frame_rgb.copy()
    mask_bool = mask.astype(bool)
    if not np.any(mask_bool):
        return out
    color = np.asarray(color_rgb, dtype=np.float32)
    base = out.astype(np.float32)
    base[mask_bool] = (1.0 - alpha) * base[mask_bool] + alpha * color
    return np.clip(base, 0, 255).astype(np.uint8)


def _draw_text(img_rgb: np.ndarray, lines: list[str]) -> np.ndarray:
    out_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    y = 24
    for line in lines:
        cv2.putText(out_bgr, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out_bgr, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (24, 24, 24), 1, cv2.LINE_AA)
        y += 22
    return cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)


def _mask_only_canvas(raw_union: np.ndarray, rle_union: np.ndarray, diff_union: np.ndarray) -> np.ndarray:
    h, w = raw_union.shape
    black = np.zeros((h, w, 3), dtype=np.uint8)
    raw_panel = _overlay_mask(black, raw_union, (0, 255, 255), alpha=1.0)
    rle_panel = _overlay_mask(black, rle_union, (255, 0, 255), alpha=1.0)
    diff_panel = _overlay_mask(black, diff_union, (255, 64, 64), alpha=1.0)
    return np.concatenate([raw_panel, rle_panel, diff_panel], axis=1)


def build_visual_frame(
    *,
    frame_rgb: np.ndarray | None,
    frame_index: int,
    raw_union: np.ndarray,
    rle_union: np.ndarray,
    diff_union: np.ndarray,
    mismatch_entries: int,
    total_entries: int,
    diff_pixels: int,
    overlay_alpha: float,
) -> np.ndarray:
    if frame_rgb is None:
        canvas = _mask_only_canvas(raw_union, rle_union, diff_union)
    else:
        raw_panel = _overlay_mask(frame_rgb, raw_union, (0, 255, 255), alpha=overlay_alpha)
        rle_panel = _overlay_mask(frame_rgb, rle_union, (255, 0, 255), alpha=overlay_alpha)
        diff_panel = _overlay_mask(frame_rgb, diff_union, (255, 64, 64), alpha=overlay_alpha)
        canvas = np.concatenate([raw_panel, rle_panel, diff_panel], axis=1)

    return _draw_text(
        canvas,
        [
            f"frame={frame_index} entries={total_entries} mismatched={mismatch_entries} diff_px={diff_pixels}",
            "left=raw overlay  middle=decoded RLE overlay  right=union of per-entry diffs",
        ],
    )


def _resolve_source_video(video_json: dict[str, Any]) -> Path | None:
    raw = video_json.get("video_path")
    if isinstance(raw, str) and raw.strip():
        path = Path(raw)
        if path.is_file():
            return path
    return None


def _load_frame_rgb(cap: cv2.VideoCapture, frame_index: int) -> np.ndarray | None:
    ok = cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    if not ok:
        return None
    ret, frame_bgr = cap.read()
    if not ret or frame_bgr is None:
        return None
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def _write_comparison_video(
    *,
    resolved: ResolvedVideo,
    video_json: dict[str, Any],
    npz_data: np.lib.npyio.NpzFile,
    processed_frames: list[dict[str, Any]],
    frame_results: dict[int, list[PairValidationResult]],
    output_path: Path,
    visual_fps: float,
    overlay_alpha: float,
    allow_missing_video: bool,
) -> dict[str, Any]:
    source_path = _resolve_source_video(video_json)
    cap: cv2.VideoCapture | None = None
    frame_size: tuple[int, int] | None = None
    source_mode = "mask_only"

    try:
        if source_path is not None:
            cap = cv2.VideoCapture(str(source_path))
            if not cap.isOpened():
                cap.release()
                cap = None
            else:
                source_mode = "source_video"

        if cap is None and not allow_missing_video:
            raise FileNotFoundError(
                f"Source video unavailable for {resolved.video_name}; pass --allow-missing-video for mask-only visuals."
            )

        writer: cv2.VideoWriter | None = None
        written = 0
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            for rec in processed_frames:
                frame_index = int(rec.get("frame_index", -1))
                results = frame_results.get(frame_index) or []
                if not results:
                    continue

                raw_union: np.ndarray | None = None
                rle_union: np.ndarray | None = None
                diff_union: np.ndarray | None = None
                mismatch_entries = 0
                total_entries = 0

                for entry in (rec.get("npz_keys") or {}).get("masks") or []:
                    pair = validate_mask_pair(
                        video_name=resolved.video_name,
                        frame_index=frame_index,
                        npz_data=npz_data,
                        entry=entry,
                        mask_kind="prompt",
                        member_sizes={},
                        strict_pairs=False,
                    )
                    if pair is None or pair.error:
                        continue
                    raw_mask = normalize_binary_mask(np.asarray(npz_data[pair.raw_key]))
                    rle_mask = decode_coco_rle_to_mask(
                        counts=_counts_text_from_payload(np.asarray(npz_data[pair.rle_key])),
                        size=pair.size,
                    )
                    diff_mask = raw_mask != rle_mask
                    raw_union = raw_mask if raw_union is None else np.logical_or(raw_union, raw_mask)
                    rle_union = rle_mask if rle_union is None else np.logical_or(rle_union, rle_mask)
                    diff_union = diff_mask if diff_union is None else np.logical_or(diff_union, diff_mask)
                    total_entries += 1
                    mismatch_entries += 0 if pair.exact_match else 1

                for entry in (rec.get("npz_keys") or {}).get("objects") or []:
                    pair = validate_mask_pair(
                        video_name=resolved.video_name,
                        frame_index=frame_index,
                        npz_data=npz_data,
                        entry=entry,
                        mask_kind="object",
                        member_sizes={},
                        strict_pairs=False,
                    )
                    if pair is None or pair.error:
                        continue
                    raw_mask = normalize_binary_mask(np.asarray(npz_data[pair.raw_key]))
                    rle_mask = decode_coco_rle_to_mask(
                        counts=_counts_text_from_payload(np.asarray(npz_data[pair.rle_key])),
                        size=pair.size,
                    )
                    diff_mask = raw_mask != rle_mask
                    raw_union = raw_mask if raw_union is None else np.logical_or(raw_union, raw_mask)
                    rle_union = rle_mask if rle_union is None else np.logical_or(rle_union, rle_mask)
                    diff_union = diff_mask if diff_union is None else np.logical_or(diff_union, diff_mask)
                    total_entries += 1
                    mismatch_entries += 0 if pair.exact_match else 1

                if raw_union is None or rle_union is None or diff_union is None:
                    continue

                frame_rgb = _load_frame_rgb(cap, frame_index) if cap is not None else None
                canvas = build_visual_frame(
                    frame_rgb=frame_rgb,
                    frame_index=frame_index,
                    raw_union=raw_union.astype(bool),
                    rle_union=rle_union.astype(bool),
                    diff_union=diff_union.astype(bool),
                    mismatch_entries=mismatch_entries,
                    total_entries=total_entries,
                    diff_pixels=int(np.count_nonzero(diff_union)),
                    overlay_alpha=overlay_alpha,
                )

                if writer is None:
                    height, width = canvas.shape[:2]
                    frame_size = (width, height)
                    writer = cv2.VideoWriter(
                        str(output_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        visual_fps,
                        frame_size,
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"Failed to open VideoWriter for: {output_path}")

                writer.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
                written += 1
        finally:
            if writer is not None:
                writer.release()

        return {
            "video_path": str(output_path),
            "written_frames": written,
            "mode": source_mode,
        }
    finally:
        if cap is not None:
            cap.release()


def _write_detail_csv(results: list[PairValidationResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "video_name",
                "frame_index",
                "mask_kind",
                "mask_label",
                "key",
                "raw_key",
                "rle_key",
                "size",
                "raw_pixels",
                "rle_pixels",
                "diff_pixels",
                "iou",
                "exact_match",
                "raw_member_bytes",
                "rle_member_bytes",
                "error",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "video_name": r.video_name,
                    "frame_index": r.frame_index,
                    "mask_kind": r.mask_kind,
                    "mask_label": r.mask_label,
                    "key": r.key,
                    "raw_key": r.raw_key,
                    "rle_key": r.rle_key,
                    "size": json.dumps(r.size),
                    "raw_pixels": r.raw_pixels,
                    "rle_pixels": r.rle_pixels,
                    "diff_pixels": r.diff_pixels,
                    "iou": r.iou,
                    "exact_match": r.exact_match,
                    "raw_member_bytes": r.raw_member_bytes,
                    "rle_member_bytes": r.rle_member_bytes,
                    "error": r.error,
                }
            )


def validate_video(
    *,
    resolved: ResolvedVideo,
    output_dir: Path,
    max_frames: int | None,
    skip_visuals: bool,
    allow_missing_video: bool,
    visual_fps: float,
    overlay_alpha: float,
    strict_pairs: bool,
) -> dict[str, Any]:
    video_json = json.loads(resolved.json_path.read_text(encoding="utf-8"))
    processed_frames = [f for f in (video_json.get("frames") or []) if f.get("status") == "processed"]
    processed_frames.sort(key=lambda rec: int(rec.get("frame_index", -1)))
    if max_frames is not None:
        processed_frames = processed_frames[:max_frames]

    member_sizes = _member_size_map(resolved.npz_path)
    results: list[PairValidationResult] = []
    frame_results: dict[int, list[PairValidationResult]] = {}

    with np.load(resolved.npz_path, allow_pickle=False) as npz_data:
        for rec in processed_frames:
            frame_index = int(rec.get("frame_index", -1))
            frame_pairs: list[PairValidationResult] = []
            for entry in ((rec.get("npz_keys") or {}).get("masks") or []):
                result = validate_mask_pair(
                    video_name=resolved.video_name,
                    frame_index=frame_index,
                    npz_data=npz_data,
                    entry=entry,
                    mask_kind="prompt",
                    member_sizes=member_sizes,
                    strict_pairs=strict_pairs,
                )
                if result is not None:
                    results.append(result)
                    frame_pairs.append(result)
            for entry in ((rec.get("npz_keys") or {}).get("objects") or []):
                result = validate_mask_pair(
                    video_name=resolved.video_name,
                    frame_index=frame_index,
                    npz_data=npz_data,
                    entry=entry,
                    mask_kind="object",
                    member_sizes=member_sizes,
                    strict_pairs=strict_pairs,
                )
                if result is not None:
                    results.append(result)
                    frame_pairs.append(result)
            frame_results[frame_index] = frame_pairs

        detail_csv = output_dir / f"{resolved.video_name}_mask_rle_validation.csv"
        _write_detail_csv(results, detail_csv)

        visual_summary: dict[str, Any] | None = None
        if not skip_visuals:
            visual_summary = _write_comparison_video(
                resolved=resolved,
                video_json=video_json,
                npz_data=npz_data,
                processed_frames=processed_frames,
                frame_results=frame_results,
                output_path=output_dir / "videos" / f"{resolved.video_name}_mask_rle_validation.mp4",
                visual_fps=visual_fps,
                overlay_alpha=overlay_alpha,
                allow_missing_video=allow_missing_video,
            )

    exact_matches = sum(1 for r in results if r.exact_match)
    mismatches = [r for r in results if not r.exact_match]
    raw_member_total = sum(int(r.raw_member_bytes or 0) for r in results if r.raw_member_bytes is not None)
    rle_member_total = sum(int(r.rle_member_bytes or 0) for r in results if r.rle_member_bytes is not None)
    total_diff_pixels = sum(int(r.diff_pixels) for r in results)
    summary = {
        "video_name": resolved.video_name,
        "video_dir": str(resolved.video_dir),
        "json_path": str(resolved.json_path),
        "npz_path": str(resolved.npz_path),
        "processed_frames_checked": len(processed_frames),
        "entries_checked": len(results),
        "exact_match_entries": exact_matches,
        "mismatched_entries": len(mismatches),
        "failed": len(mismatches) > 0,
        "total_diff_pixels": total_diff_pixels,
        "raw_member_total_bytes": raw_member_total,
        "rle_member_total_bytes": rle_member_total,
        "compression_ratio_raw_over_rle": (float(raw_member_total / rle_member_total) if rle_member_total > 0 else None),
        "detail_csv": str(detail_csv),
        "visual": visual_summary,
        "first_mismatch": (
            {
                "frame_index": mismatches[0].frame_index,
                "mask_kind": mismatches[0].mask_kind,
                "mask_label": mismatches[0].mask_label,
                "diff_pixels": mismatches[0].diff_pixels,
                "error": mismatches[0].error,
            }
            if mismatches
            else None
        ),
    }
    return summary


def default_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    if args.video_dir is not None:
        return args.video_dir / "mask_rle_validation"
    assert args.run_root is not None
    return args.run_root / "mask_rle_validation"


def main() -> int:
    args = parse_args()
    videos = resolve_videos(
        run_root=args.run_root.resolve() if args.run_root is not None else None,
        video_dir=args.video_dir.resolve() if args.video_dir is not None else None,
        video_name=args.video_name,
        max_videos=args.max_videos,
    )
    output_dir = default_output_dir(args).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    video_summaries: list[dict[str, Any]] = []
    failed_videos = 0

    for resolved in videos:
        summary = validate_video(
            resolved=resolved,
            output_dir=output_dir / resolved.video_name,
            max_frames=args.max_frames,
            skip_visuals=args.skip_visuals,
            allow_missing_video=args.allow_missing_video,
            visual_fps=args.visual_fps,
            overlay_alpha=args.overlay_alpha,
            strict_pairs=args.strict_pairs,
        )
        video_summaries.append(summary)
        failed_videos += 1 if summary["failed"] else 0
        print(
            f"{resolved.video_name}: entries={summary['entries_checked']} "
            f"mismatched={summary['mismatched_entries']} diff_px={summary['total_diff_pixels']} "
            f"ratio={summary['compression_ratio_raw_over_rle']}"
        )

    total_entries = sum(int(v["entries_checked"]) for v in video_summaries)
    total_mismatches = sum(int(v["mismatched_entries"]) for v in video_summaries)
    total_diff_pixels = sum(int(v["total_diff_pixels"]) for v in video_summaries)
    raw_total = sum(int(v["raw_member_total_bytes"]) for v in video_summaries)
    rle_total = sum(int(v["rle_member_total_bytes"]) for v in video_summaries)
    run_summary = {
        "run_root": str(args.run_root.resolve()) if args.run_root is not None else None,
        "video_dir": str(args.video_dir.resolve()) if args.video_dir is not None else None,
        "videos_checked": len(video_summaries),
        "failed_videos": failed_videos,
        "entries_checked": total_entries,
        "mismatched_entries": total_mismatches,
        "total_diff_pixels": total_diff_pixels,
        "raw_member_total_bytes": raw_total,
        "rle_member_total_bytes": rle_total,
        "compression_ratio_raw_over_rle": (float(raw_total / rle_total) if rle_total > 0 else None),
        "strict_pairs": bool(args.strict_pairs),
        "video_summaries": video_summaries,
    }
    summary_path = output_dir / "mask_rle_validation_summary.json"
    summary_path.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    print(f"Wrote summary: {summary_path}")
    return 1 if total_mismatches > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
