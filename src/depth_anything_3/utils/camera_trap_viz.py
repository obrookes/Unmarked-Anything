from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from depth_anything_3.utils.camera_trap_masks import load_mask_bool


def resolve_video_stem(output_root: Path, video_stem: str | None) -> tuple[str, dict[str, Any] | None]:
    manifest_path = output_root / "run_manifest.json"
    manifest = None
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())

    selected_stem = video_stem
    if selected_stem is None:
        if not manifest:
            raise FileNotFoundError(f"VIDEO_STEM is None and manifest not found: {manifest_path}")
        successes = [v for v in manifest.get("videos", []) if v.get("status") == "success"]
        if not successes:
            raise RuntimeError("No successful videos found in run_manifest.json")
        selected_stem = str(successes[0]["video_name"])

    return selected_stem, manifest


def load_artifacts(output_root: Path, video_stem: str | None) -> tuple[dict[str, Any], np.lib.npyio.NpzFile]:
    selected_stem, manifest = resolve_video_stem(output_root=output_root, video_stem=video_stem)

    video_dir = output_root / selected_stem
    json_path = video_dir / f"{selected_stem}.json"
    npz_path = video_dir / f"{selected_stem}_arrays.npz"

    if not json_path.is_file():
        raise FileNotFoundError(f"Missing JSON output: {json_path}")
    if not npz_path.is_file():
        raise FileNotFoundError(f"Missing NPZ output: {npz_path}")

    video_json: dict[str, Any] = json.loads(json_path.read_text())
    npz_data = np.load(npz_path, allow_pickle=False)

    frames = video_json.get("frames", [])
    processed_count = sum(1 for f in frames if f.get("status") == "processed")

    print(f"Loaded video: {selected_stem}")
    print(f"  JSON status: {video_json.get('status')} | SAM mode: {video_json.get('sam3_mode')}")
    print(f"  Frames total: {len(frames)} | Processed detections: {processed_count}")
    print(f"  NPZ members: {len(npz_data.files)}")

    if manifest is not None:
        summary = manifest.get("summary", {})
        print("  Manifest summary:", summary)

    return video_json, npz_data


def select_detected_frames(video_json: dict[str, Any]) -> list[dict[str, Any]]:
    frames = video_json.get("frames", [])
    processed = [f for f in frames if f.get("status") == "processed"]
    processed.sort(key=lambda f: int(f.get("frame_index", -1)))

    validated: list[dict[str, Any]] = []
    missing_rows = 0
    for f in processed:
        npz_keys = f.get("npz_keys") or {}
        if not npz_keys.get("depth") or not npz_keys.get("masks"):
            missing_rows += 1
            continue
        validated.append(f)

    if missing_rows:
        print(f"Warning: skipped {missing_rows} processed rows with missing npz_keys references.")

    if not validated:
        raise RuntimeError("No valid processed detection frames found in JSON output.")

    print(f"Validated processed frames: {len(validated)}")
    return validated


def open_video_capture(video_path: Path) -> cv2.VideoCapture:
    if str(video_path).startswith("/absolute/or/local/path"):
        raise FileNotFoundError(
            "Please set VIDEO_PATH in the config cell to a local source video path before continuing."
        )
    if not video_path.is_file():
        raise FileNotFoundError(f"Video path not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    print(f"Opened source video: {video_path}")
    return cap


def read_video_frame(cap: cv2.VideoCapture, frame_index: int) -> np.ndarray:
    ok = cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    if not ok:
        raise RuntimeError(f"Failed to seek to frame {frame_index}")
    ret, frame_bgr = cap.read()
    if not ret or frame_bgr is None:
        raise RuntimeError(f"Failed to decode frame {frame_index}")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def build_union_mask(npz_data: np.lib.npyio.NpzFile, frame_row: dict[str, Any]) -> np.ndarray:
    npz_keys = frame_row.get("npz_keys") or {}
    mask_entries = npz_keys.get("masks") or []
    if not mask_entries:
        raise KeyError(f"Frame {frame_row.get('frame_index')}: no mask entries in npz_keys")

    union_mask: np.ndarray | None = None
    missing_keys: list[str] = []

    for entry in mask_entries:
        key = entry.get("key")
        if not key:
            continue
        try:
            mask_bool = load_mask_bool(npz_data=npz_data, entry=entry)
        except KeyError:
            missing_keys.append(key)
            continue
        if union_mask is None:
            union_mask = mask_bool.copy()
        else:
            if union_mask.shape != mask_bool.shape:
                raise ValueError(
                    f"Frame {frame_row.get('frame_index')}: mask shape mismatch {union_mask.shape} vs {mask_bool.shape}"
                )
            union_mask |= mask_bool

    if union_mask is None:
        raise KeyError(
            f"Frame {frame_row.get('frame_index')}: no valid mask arrays found. Missing keys: {missing_keys}"
        )

    return union_mask


def build_object_masks(npz_data: np.lib.npyio.NpzFile, frame_row: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    npz_keys = frame_row.get("npz_keys") or {}
    object_entries = npz_keys.get("objects") or frame_row.get("objects") or []
    objects: list[dict[str, Any]] = []
    missing_keys: list[str] = []

    for obj in object_entries:
        key = obj.get("key")
        if not key:
            continue
        try:
            mask_arr = load_mask_bool(npz_data=npz_data, entry=obj)
        except KeyError:
            missing_keys.append(str(key))
            continue
        objects.append(
            {
                "object_index": int(obj.get("object_index", len(objects))),
                "track_id": obj.get("track_id"),
                "label": obj.get("label"),
                "confidence": obj.get("confidence"),
                "bbox_xyxy": obj.get("bbox_xyxy"),
                "center_xy": obj.get("center_xy"),
                "mask": mask_arr,
                "mask_nonzero_pixels": int(obj.get("mask_nonzero_pixels") or int(mask_arr.sum())),
            }
        )

    return objects, missing_keys


def get_depth_array(
    npz_data: np.lib.npyio.NpzFile,
    frame_row: dict[str, Any],
    frame_shape_hw: tuple[int, int] | None = None,
    depth_source: str = "new",
) -> np.ndarray:
    depth_key = ((frame_row.get("npz_keys") or {}).get("depth"))
    if not depth_key:
        raise KeyError(f"Frame {frame_row.get('frame_index')}: missing depth key reference in npz_keys")

    if depth_source == "new":
        resolved_depth_key = depth_key
    elif depth_source == "old":
        resolved_depth_key = f"{depth_key}_old"
    else:
        raise ValueError(f"Unsupported depth_source: {depth_source!r}. Expected 'new' or 'old'.")

    if resolved_depth_key not in npz_data:
        raise KeyError(
            f"Frame {frame_row.get('frame_index')}: missing depth key {resolved_depth_key} "
            f"(source={depth_source}, base={depth_key})"
        )
    depth = np.asarray(npz_data[resolved_depth_key], dtype=np.float32)
    if frame_shape_hw is not None and depth.shape != frame_shape_hw:
        depth = cv2.resize(depth, (frame_shape_hw[1], frame_shape_hw[0]), interpolation=cv2.INTER_CUBIC)
    return depth


def _compute_depth_stats_for_objects(
    valid_objects: list[dict[str, Any]],
    depth: np.ndarray,
) -> list[dict[str, Any]]:
    depth_stats: list[dict[str, Any]] = []
    for obj in valid_objects:
        obj_mask = np.asarray(obj["mask"], dtype=bool)
        if obj_mask.shape != depth.shape or not np.any(obj_mask):
            continue
        bbox = obj.get("bbox_xyxy")
        center = obj.get("center_xy")
        if (not bbox or len(bbox) != 4):
            ys, xs = np.where(obj_mask)
            bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        if (not center or len(center) != 2) and bbox and len(bbox) == 4:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            center = [int((x1 + x2) // 2), int((y1 + y2) // 2)]
        if not bbox or len(bbox) != 4 or not center or len(center) != 2:
            continue

        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1 = max(0, min(x1, depth.shape[1] - 1))
        x2 = max(0, min(x2, depth.shape[1] - 1))
        y1 = max(0, min(y1, depth.shape[0] - 1))
        y2 = max(0, min(y2, depth.shape[0] - 1))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        bbox_mask = np.zeros_like(obj_mask, dtype=bool)
        bbox_mask[y1 : y2 + 1, x1 : x2 + 1] = True
        cx, cy = [int(v) for v in center]
        cx = max(0, min(cx, depth.shape[1] - 1))
        cy = max(0, min(cy, depth.shape[0] - 1))

        mask_vals = depth[obj_mask]
        bbox_vals = depth[bbox_mask]
        bbox_only_vals = depth[bbox_mask & (~obj_mask)]

        depth_stats.append(
            {
                "object_index": int(obj.get("object_index", len(depth_stats))),
                "track_id": obj.get("track_id"),
                "mask_mean": float(np.nanmean(mask_vals)) if mask_vals.size else float("nan"),
                "bbox_mean": float(np.nanmean(bbox_vals)) if bbox_vals.size else float("nan"),
                "bbox_only_mean": float(np.nanmean(bbox_only_vals)) if bbox_only_vals.size else float("nan"),
                "center_depth": float(depth[cy, cx]),
                "bbox": (x1, y1, x2, y2),
                "center": (cx, cy),
            }
        )

    return depth_stats


def draw_overlay_frame(
    frame_rgb: np.ndarray,
    rec: dict[str, Any],
    npz_obj: np.lib.npyio.NpzFile,
    overlay_alpha: float = 0.45,
    show_bbox: bool = True,
    show_center: bool = True,
    draw_mask: bool = True,
    draw_hud: bool = False,
    compute_depth_stats: bool = False,
    depth_source: str = "new",
) -> tuple[np.ndarray, list[dict[str, Any]], list[str], list[str]]:
    warnings: list[str] = []
    frame_idx = int(rec.get("frame_index", -1))
    overlay = frame_rgb.astype(np.float32) / 255.0

    try:
        union_mask = build_union_mask(npz_obj, rec)
    except Exception as e:
        warnings.append(f"Frame {frame_idx}: mask missing/invalid ({e})")
        union_mask = np.zeros(frame_rgb.shape[:2], dtype=bool)

    object_rows, missing_object_keys = build_object_masks(npz_obj, rec)
    if missing_object_keys:
        warnings.append(f"Frame {frame_idx}: missing object mask keys: {missing_object_keys[:6]}")

    if union_mask.shape != frame_rgb.shape[:2]:
        warnings.append(
            f"Frame {frame_idx}: mask/frame shape mismatch {union_mask.shape} vs {frame_rgb.shape[:2]} (mask ignored)"
        )
        union_mask = np.zeros(frame_rgb.shape[:2], dtype=bool)

    cmap = plt.get_cmap("tab20")
    valid_objects: list[dict[str, Any]] = []
    for obj in object_rows:
        obj_mask = np.asarray(obj["mask"], dtype=bool)
        if obj_mask.shape != frame_rgb.shape[:2]:
            warnings.append(
                f"Frame {frame_idx}: object mask/frame shape mismatch {obj_mask.shape} vs {frame_rgb.shape[:2]}"
            )
            continue
        if not np.any(obj_mask):
            continue
        valid_objects.append(obj)

    if draw_mask:
        if valid_objects:
            for obj in valid_objects:
                color = np.array(cmap(int(obj["object_index"]) % 20)[:3], dtype=np.float32)
                obj_mask = np.asarray(obj["mask"], dtype=bool)
                overlay[obj_mask] = (1.0 - overlay_alpha) * overlay[obj_mask] + overlay_alpha * color
        else:
            overlay_color = np.array([0.0, 1.0, 1.0], dtype=np.float32)
            if np.any(union_mask):
                overlay[union_mask] = (1.0 - overlay_alpha) * overlay[union_mask] + overlay_alpha * overlay_color

    overlay_rgb = np.clip(overlay * 255.0, 0, 255).astype(np.uint8)
    overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)

    if show_bbox or show_center:
        if valid_objects:
            for obj in valid_objects:
                color = np.array(cmap(int(obj["object_index"]) % 20)[:3], dtype=np.float32)
                color_bgr = tuple(int(c) for c in (color[::-1] * 255.0))
                bbox = obj.get("bbox_xyxy")
                center = obj.get("center_xy")
                if (not bbox or len(bbox) != 4) and np.any(obj["mask"]):
                    ys, xs = np.where(np.asarray(obj["mask"], dtype=bool))
                    bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
                if (not center or len(center) != 2) and bbox and len(bbox) == 4:
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    center = [int((x1 + x2) // 2), int((y1 + y2) // 2)]
                if show_bbox and bbox and len(bbox) == 4:
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    cv2.rectangle(overlay_bgr, (x1, y1), (x2, y2), color_bgr, 2)
                if show_center and center and len(center) == 2:
                    cv2.circle(overlay_bgr, (int(center[0]), int(center[1])), 4, color_bgr, -1)
                    cv2.circle(overlay_bgr, (int(center[0]), int(center[1])), 5, (0, 0, 0), 1)
        else:
            if show_bbox:
                bbox = rec.get("bbox_xyxy")
                if bbox and len(bbox) == 4:
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    cv2.rectangle(overlay_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if show_center:
                center = rec.get("center_xy")
                if center and len(center) == 2:
                    cv2.circle(overlay_bgr, (int(center[0]), int(center[1])), 4, (0, 255, 255), -1)
                    cv2.circle(overlay_bgr, (int(center[0]), int(center[1])), 5, (0, 0, 0), 1)

    depth_stats: list[dict[str, Any]] = []
    hud_lines: list[str] = []
    if compute_depth_stats or draw_hud:
        try:
            depth = get_depth_array(
                npz_data=npz_obj,
                frame_row=rec,
                frame_shape_hw=frame_rgb.shape[:2],
                depth_source=depth_source,
            )
            depth_stats = _compute_depth_stats_for_objects(valid_objects=valid_objects, depth=depth)
        except Exception as e:
            warnings.append(f"Frame {frame_idx}: depth stats unavailable ({e})")
            depth_stats = []

    if draw_hud:
        object_count = len(valid_objects)
        hud_lines.append(
            f"frame={frame_idx} obj={object_count} mask_px={int(rec.get('mask_nonzero_pixels', 0))}"
        )
        if depth_stats:
            for stat in depth_stats[:4]:
                bbox_only_txt = "NaN" if np.isnan(stat["bbox_only_mean"]) else f"{stat['bbox_only_mean']:.4f}"
                hud_lines.append(
                    f"obj{stat['object_index']} mask={stat['mask_mean']:.4f} "
                    f"center={stat['center_depth']:.4f} bbox={stat['bbox_mean']:.4f} bbox_only={bbox_only_txt}"
                )

        if hud_lines:
            y = 24
            for line in hud_lines[:6]:
                cv2.putText(
                    overlay_bgr,
                    line,
                    (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    overlay_bgr,
                    line,
                    (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (16, 16, 16),
                    1,
                    cv2.LINE_AA,
                )
                y += 22

    return cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB), depth_stats, hud_lines, warnings


def _format_object_mask_means(obj_stats: list[dict[str, Any]], max_items: int = 4) -> str:
    if not obj_stats:
        return "none"
    parts: list[str] = []
    for o in obj_stats[:max_items]:
        mean_val = o.get("mask_mean", float("nan"))
        mean_txt = "NaN" if np.isnan(mean_val) else f"{mean_val:.4f}"
        parts.append(f"obj{o.get('object_index', '?')}={mean_txt}")
    if len(obj_stats) > max_items:
        parts.append(f"+{len(obj_stats) - max_items} more")
    return ", ".join(parts)


def render_page(
    records: list[dict[str, Any]],
    cap_obj: cv2.VideoCapture,
    npz_obj: np.lib.npyio.NpzFile,
    page: int = 0,
    page_size: int = 4,
    overlay_alpha: float = 0.45,
    depth_cmap: str = "inferno",
    show_bbox_and_center: bool = True,
) -> None:
    if page_size <= 0:
        raise ValueError("page_size must be > 0")
    total = len(records)
    if total == 0:
        raise RuntimeError("No records to render.")

    max_page = max(0, math.ceil(total / page_size) - 1)
    if page < 0 or page > max_page:
        raise ValueError(f"page out of range: {page}. valid range is 0..{max_page}")

    start = page * page_size
    end = min(start + page_size, total)
    page_records = records[start:end]

    n = len(page_records)
    fig, axes = plt.subplots(nrows=n, ncols=2, figsize=(16, max(4 * n, 5)))
    if n == 1:
        axes = np.array([axes])

    warnings: list[str] = []

    for i, rec in enumerate(page_records):
        ax_overlay, ax_depth = axes[i, 0], axes[i, 1]
        frame_idx = int(rec.get("frame_index"))

        try:
            frame_rgb = read_video_frame(cap_obj, frame_idx)
        except Exception as e:
            warnings.append(f"Frame {frame_idx}: decode failed ({e})")
            ax_overlay.text(0.5, 0.5, f"Decode failed\nframe={frame_idx}", ha="center", va="center")
            ax_overlay.axis("off")
            ax_depth.axis("off")
            continue

        overlay_rgb, depth_stats, _, overlay_warnings = draw_overlay_frame(
            frame_rgb=frame_rgb,
            rec=rec,
            npz_obj=npz_obj,
            overlay_alpha=overlay_alpha,
            show_bbox=show_bbox_and_center,
            show_center=show_bbox_and_center,
            draw_mask=True,
            draw_hud=False,
            compute_depth_stats=True,
        )
        warnings.extend(overlay_warnings)
        ax_overlay.imshow(overlay_rgb)

        ts = rec.get("timestamp_sec")
        ts_txt = f"{ts:.2f}s" if isinstance(ts, (int, float)) else "n/a"
        object_count = len(depth_stats) if depth_stats else len((rec.get("npz_keys") or {}).get("objects") or [])
        ax_overlay.set_title(
            f"Frame {frame_idx} @ {ts_txt} | mask_px={rec.get('mask_nonzero_pixels', 0)} "
            f"| area={rec.get('mask_area_fraction', 0.0):.4f} | objects={object_count}"
        )
        ax_overlay.axis("off")

        try:
            depth = get_depth_array(npz_data=npz_obj, frame_row=rec, frame_shape_hw=frame_rgb.shape[:2])
        except Exception as e:
            warnings.append(f"Frame {frame_idx}: depth unavailable ({e})")
            ax_depth.text(0.5, 0.5, f"Missing depth\nframe={frame_idx}", ha="center", va="center")
            ax_depth.axis("off")
            continue

        im = ax_depth.imshow(depth, cmap=depth_cmap)
        plt.colorbar(im, ax=ax_depth, fraction=0.046, pad=0.01)
        if depth_stats:
            depth_title = f"Depth | object mean(mask): {_format_object_mask_means(depth_stats)}"
        else:
            depth_title = (
                f"Depth | mean(mask)={rec.get('depth_mask_mean')} | center={rec.get('depth_center_value')}"
            )
        ax_depth.set_title(depth_title)
        ax_depth.axis("off")

    fig.suptitle(f"Processed detections: page {page + 1}/{max_page + 1} (frames {start}..{end - 1})", y=1.002)
    plt.tight_layout()
    plt.show()

    if warnings:
        print("Warnings:")
        for w in warnings[:30]:
            print("-", w)
        if len(warnings) > 30:
            print(f"... {len(warnings) - 30} more warnings")


def build_frame_record_map(records: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    rec_map: dict[int, dict[str, Any]] = {}
    for rec in records:
        frame_idx = int(rec.get("frame_index", -1))
        if frame_idx < 0:
            continue
        rec_map[frame_idx] = rec
    return rec_map


def write_overlay_video(
    records: list[dict[str, Any]],
    cap_obj: cv2.VideoCapture,
    npz_obj: np.lib.npyio.NpzFile,
    output_path: Path,
    overlay_alpha: float = 0.45,
    show_bbox: bool = True,
    show_center: bool = True,
    draw_mask: bool = True,
    draw_hud: bool = True,
    output_fps: float | None = None,
    fourcc: str = "mp4v",
    depth_source: str = "new",
) -> dict[str, Any]:
    if len(fourcc) != 4:
        raise ValueError("fourcc must be a 4-character code, e.g. 'mp4v'")

    rec_map = build_frame_record_map(records)
    frame_count = int(cap_obj.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap_obj.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_obj.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = float(cap_obj.get(cv2.CAP_PROP_FPS) or 0.0)
    fps = float(output_fps) if output_fps is not None else (source_fps if source_fps > 0 else 30.0)
    if fps <= 0:
        raise ValueError("output fps must be > 0")
    if frame_count <= 0:
        raise RuntimeError("video capture reports no frames")
    if width <= 0 or height <= 0:
        raise RuntimeError("video capture reports invalid dimensions")

    ok = cap_obj.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if not ok:
        raise RuntimeError("failed to seek source capture to frame 0")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*fourcc),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for: {output_path}")

    warnings: list[str] = []
    written = 0
    overlaid = 0
    try:
        for frame_idx in range(frame_count):
            ret, frame_bgr = cap_obj.read()
            if not ret or frame_bgr is None:
                warnings.append(f"Decode failed at source frame {frame_idx}")
                break

            rec = rec_map.get(frame_idx)
            if rec is None:
                writer.write(frame_bgr)
                written += 1
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            overlay_rgb, _, _, frame_warnings = draw_overlay_frame(
                frame_rgb=frame_rgb,
                rec=rec,
                npz_obj=npz_obj,
                overlay_alpha=overlay_alpha,
                show_bbox=show_bbox,
                show_center=show_center,
                draw_mask=draw_mask,
                draw_hud=draw_hud,
                compute_depth_stats=draw_hud,
                depth_source=depth_source,
            )
            warnings.extend(frame_warnings)
            writer.write(cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
            written += 1
            overlaid += 1
    finally:
        writer.release()

    return {
        "output_path": str(output_path),
        "source_frame_count": frame_count,
        "written_frames": written,
        "overlay_frames": overlaid,
        "source_fps": source_fps,
        "output_fps": fps,
        "warnings": warnings,
    }


def select_records_for_analysis(
    records: list[dict[str, Any]],
    mode: str = "manual",
    manual_frame_indices: list[int] | None = None,
    auto_count: int = 3,
) -> list[dict[str, Any]]:
    if not records:
        raise RuntimeError("No processed records available for analysis.")

    manual_frame_indices = manual_frame_indices or []
    rec_by_idx = {int(r.get("frame_index")): r for r in records}

    if mode == "manual":
        selected: list[dict[str, Any]] = []
        missing: list[int] = []
        seen: set[int] = set()
        for idx in manual_frame_indices:
            idx = int(idx)
            if idx in seen:
                continue
            seen.add(idx)
            rec = rec_by_idx.get(idx)
            if rec is None:
                missing.append(idx)
            else:
                selected.append(rec)
        if missing:
            print(f"Warning: frame indices not found in processed detections: {missing}")
        if not selected:
            raise RuntimeError("Manual mode selected but no valid frame indices were found.")
        selected.sort(key=lambda r: int(r.get("frame_index", -1)))
        return selected

    n = len(records)
    count = max(1, min(int(auto_count), n))
    if count == 1:
        picks = [0]
    else:
        picks = sorted(set(round(i * (n - 1) / (count - 1)) for i in range(count)))
    return [records[i] for i in picks]


def compute_depth_comparison_results(
    analysis_records: list[dict[str, Any]],
    cap: cv2.VideoCapture,
    npz_data: np.lib.npyio.NpzFile,
    depth_source: str = "new",
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    cmap = plt.get_cmap("tab20")

    for rec in analysis_records:
        frame_idx = int(rec.get("frame_index"))

        frame_rgb = read_video_frame(cap, frame_idx)
        union_mask = build_union_mask(npz_data, rec).astype(bool)
        object_rows, _ = build_object_masks(npz_data, rec)

        if union_mask.shape != frame_rgb.shape[:2]:
            raise RuntimeError(
                f"Frame {frame_idx}: mask/frame shape mismatch {union_mask.shape} vs {frame_rgb.shape[:2]}"
            )

        depth = get_depth_array(
            npz_data=npz_data,
            frame_row=rec,
            frame_shape_hw=frame_rgb.shape[:2],
            depth_source=depth_source,
        )

        object_results: list[dict[str, Any]] = []
        for obj in object_rows:
            obj_mask = np.asarray(obj.get("mask"), dtype=bool)
            if obj_mask.shape != frame_rgb.shape[:2] or not np.any(obj_mask):
                continue

            bbox = obj.get("bbox_xyxy")
            center = obj.get("center_xy")
            if (not bbox or len(bbox) != 4):
                ys, xs = np.where(obj_mask)
                bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
            if (not center or len(center) != 2) and bbox and len(bbox) == 4:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                center = [int((x1 + x2) // 2), int((y1 + y2) // 2)]
            if not bbox or len(bbox) != 4:
                continue

            x1, y1, x2, y2 = [int(v) for v in bbox]
            x1 = max(0, min(x1, depth.shape[1] - 1))
            x2 = max(0, min(x2, depth.shape[1] - 1))
            y1 = max(0, min(y1, depth.shape[0] - 1))
            y2 = max(0, min(y2, depth.shape[0] - 1))
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1

            bbox_mask = np.zeros_like(obj_mask, dtype=bool)
            bbox_mask[y1 : y2 + 1, x1 : x2 + 1] = True

            cx, cy = [int(v) for v in center]
            cx = max(0, min(cx, depth.shape[1] - 1))
            cy = max(0, min(cy, depth.shape[0] - 1))

            mask_vals = depth[obj_mask]
            bbox_vals = depth[bbox_mask]
            bbox_only_vals = depth[bbox_mask & (~obj_mask)]
            center_depth = float(depth[cy, cx])

            obj_idx = int(obj.get("object_index", len(object_results)))
            object_results.append(
                {
                    "object_index": obj_idx,
                    "track_id": obj.get("track_id"),
                    "label": obj.get("label"),
                    "color": np.array(cmap(obj_idx % 20)[:3], dtype=np.float32),
                    "mask": obj_mask,
                    "bbox": (x1, y1, x2, y2),
                    "bbox_mask": bbox_mask,
                    "center": (cx, cy),
                    "center_depth": center_depth,
                    "mask_pixels": int(obj_mask.sum()),
                    "bbox_pixels": int(bbox_mask.sum()),
                    "mask_mean": float(np.nanmean(mask_vals)) if mask_vals.size else float("nan"),
                    "bbox_mean": float(np.nanmean(bbox_vals)) if bbox_vals.size else float("nan"),
                    "bbox_only_mean": float(np.nanmean(bbox_only_vals)) if bbox_only_vals.size else float("nan"),
                }
            )

        if not object_results and np.any(union_mask):
            bbox = rec.get("bbox_xyxy")
            center = rec.get("center_xy")
            if (not bbox or len(bbox) != 4):
                ys, xs = np.where(union_mask)
                bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
            if (not center or len(center) != 2) and bbox and len(bbox) == 4:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                center = [int((x1 + x2) // 2), int((y1 + y2) // 2)]
            if bbox and len(bbox) == 4 and center and len(center) == 2:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                x1 = max(0, min(x1, depth.shape[1] - 1))
                x2 = max(0, min(x2, depth.shape[1] - 1))
                y1 = max(0, min(y1, depth.shape[0] - 1))
                y2 = max(0, min(y2, depth.shape[0] - 1))
                if x2 < x1:
                    x1, x2 = x2, x1
                if y2 < y1:
                    y1, y2 = y2, y1
                bbox_mask = np.zeros_like(union_mask, dtype=bool)
                bbox_mask[y1 : y2 + 1, x1 : x2 + 1] = True
                cx, cy = [int(v) for v in center]
                cx = max(0, min(cx, depth.shape[1] - 1))
                cy = max(0, min(cy, depth.shape[0] - 1))
                mask_vals = depth[union_mask]
                bbox_vals = depth[bbox_mask]
                bbox_only_vals = depth[bbox_mask & (~union_mask)]
                object_results.append(
                    {
                        "object_index": 0,
                        "track_id": None,
                        "label": "union",
                        "color": np.array(cmap(0)[:3], dtype=np.float32),
                        "mask": union_mask,
                        "bbox": (x1, y1, x2, y2),
                        "bbox_mask": bbox_mask,
                        "center": (cx, cy),
                        "center_depth": float(depth[cy, cx]),
                        "mask_pixels": int(union_mask.sum()),
                        "bbox_pixels": int(bbox_mask.sum()),
                        "mask_mean": float(np.nanmean(mask_vals)) if mask_vals.size else float("nan"),
                        "bbox_mean": float(np.nanmean(bbox_vals)) if bbox_vals.size else float("nan"),
                        "bbox_only_mean": float(np.nanmean(bbox_only_vals)) if bbox_only_vals.size else float("nan"),
                    }
                )

        result = {
            "frame_idx": frame_idx,
            "depth": depth,
            "objects": object_results,
            "union_mask": union_mask,
        }
        results.append(result)

    if not results:
        raise RuntimeError("No analysis results generated.")

    return results


def print_depth_statistics(results: list[dict[str, Any]]) -> None:
    print("=== DAP3-style depth statistics per object (mask vs bbox) ===")
    for r in results:
        frame_idx = r["frame_idx"]
        objects = r.get("objects") or []
        if not objects:
            print(f"frame={frame_idx}: no objects")
            continue
        for o in objects:
            x1, y1, x2, y2 = o["bbox"]
            cx, cy = o["center"]
            bbox_only_txt = "NaN" if np.isnan(o["bbox_only_mean"]) else f"{o['bbox_only_mean']:.6f}"
            print(
                f"frame={frame_idx} obj={o.get('object_index')} track_id={o.get('track_id')}: "
                f"bbox=({x1},{y1},{x2},{y2}) | center=({cx},{cy}) | "
                f"mask_px={o['mask_pixels']} | bbox_px={o['bbox_pixels']} | "
                f"mask_mean={o['mask_mean']:.6f} | bbox_mean={o['bbox_mean']:.6f} | "
                f"center_depth={o['center_depth']:.6f} | bbox_only_mean={bbox_only_txt}"
            )


def plot_depth_comparison(results: list[dict[str, Any]], depth_cmap: str = "inferno") -> None:
    n = len(results)
    fig, axes = plt.subplots(nrows=n, ncols=2, figsize=(16, max(4, 5 * n)), squeeze=False)

    for i, r in enumerate(results):
        ax_depth = axes[i, 0]
        ax_dist = axes[i, 1]
        objects = r.get("objects") or []

        im = ax_depth.imshow(r["depth"], cmap=depth_cmap)
        for o in objects:
            x1, y1, x2, y2 = o["bbox"]
            cx, cy = o["center"]
            color = o.get("color", np.array([0.0, 1.0, 1.0], dtype=np.float32))
            ax_depth.contour(o["mask"].astype(np.uint8), levels=[0.5], colors=[color], linewidths=1.2)
            rect = Rectangle((x1, y1), x2 - x1 + 1, y2 - y1 + 1, fill=False, edgecolor=color, linewidth=1.5)
            ax_depth.add_patch(rect)
            ax_depth.plot(cx, cy, marker="o", markersize=6, color=color, markeredgecolor="black")

        ax_depth.set_title(
            f"Frame {r['frame_idx']} | object mean(mask): {_format_object_mask_means(objects)}"
        )
        ax_depth.axis("off")
        plt.colorbar(im, ax=ax_depth, fraction=0.046, pad=0.01)

        for o in objects:
            mask_vals = r["depth"][o["mask"]]
            finite_mask_vals = mask_vals[np.isfinite(mask_vals)]
            color = o.get("color", np.array([0.0, 1.0, 1.0], dtype=np.float32))
            label = f"obj{o.get('object_index')} mean={o.get('mask_mean', float('nan')):.3f}"
            if finite_mask_vals.size:
                ax_dist.hist(finite_mask_vals, bins=50, alpha=0.45, label=label, color=color)
                ax_dist.axvline(
                    o["center_depth"], color=color, linestyle="--", linewidth=1.2
                )
        ax_dist.set_title(f"Frame {r['frame_idx']} - Depth Distribution")
        ax_dist.set_xlabel("Depth value")
        ax_dist.set_ylabel("Pixel count")
        if objects:
            ax_dist.legend()

    plt.tight_layout()
    plt.show()
