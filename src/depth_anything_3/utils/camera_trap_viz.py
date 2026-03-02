from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


def load_artifacts(output_root: Path, video_stem: str | None) -> tuple[dict[str, Any], np.lib.npyio.NpzFile]:
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
        selected_stem = successes[0]["video_name"]

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
        if key not in npz_data:
            missing_keys.append(key)
            continue
        mask_arr = np.asarray(npz_data[key])
        mask_bool = mask_arr.astype(bool)
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
        if key not in npz_data:
            missing_keys.append(str(key))
            continue
        mask_arr = np.asarray(npz_data[key]).astype(bool)
        objects.append(
            {
                "object_index": int(obj.get("object_index", len(objects))),
                "track_id": obj.get("track_id"),
                "bbox_xyxy": obj.get("bbox_xyxy"),
                "center_xy": obj.get("center_xy"),
                "mask": mask_arr,
                "mask_nonzero_pixels": int(obj.get("mask_nonzero_pixels") or int(mask_arr.sum())),
            }
        )

    return objects, missing_keys


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

        try:
            union_mask = build_union_mask(npz_obj, rec)
        except Exception as e:
            warnings.append(f"Frame {frame_idx}: mask missing/invalid ({e})")
            union_mask = np.zeros(frame_rgb.shape[:2], dtype=bool)
        object_rows, missing_object_keys = build_object_masks(npz_obj, rec)
        if missing_object_keys:
            warnings.append(
                f"Frame {frame_idx}: missing object mask keys: {missing_object_keys[:6]}"
            )

        if union_mask.shape != frame_rgb.shape[:2]:
            warnings.append(
                f"Frame {frame_idx}: mask/frame shape mismatch {union_mask.shape} vs {frame_rgb.shape[:2]} (mask ignored)"
            )
            union_mask = np.zeros(frame_rgb.shape[:2], dtype=bool)

        overlay = frame_rgb.astype(np.float32) / 255.0
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

        if valid_objects:
            for obj in valid_objects:
                color = np.array(cmap(int(obj["object_index"]) % 20)[:3], dtype=np.float32)
                obj_mask = np.asarray(obj["mask"], dtype=bool)
                overlay[obj_mask] = (1.0 - overlay_alpha) * overlay[obj_mask] + overlay_alpha * color
        else:
            overlay_color = np.array([0.0, 1.0, 1.0], dtype=np.float32)
            if np.any(union_mask):
                overlay[union_mask] = (1.0 - overlay_alpha) * overlay[union_mask] + overlay_alpha * overlay_color

        ax_overlay.imshow(overlay)
        if show_bbox_and_center:
            if valid_objects:
                for obj in valid_objects:
                    color = np.array(cmap(int(obj["object_index"]) % 20)[:3], dtype=np.float32)
                    bbox = obj.get("bbox_xyxy")
                    center = obj.get("center_xy")
                    if (not bbox or len(bbox) != 4) and np.any(obj["mask"]):
                        ys, xs = np.where(np.asarray(obj["mask"], dtype=bool))
                        bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
                    if (not center or len(center) != 2) and bbox and len(bbox) == 4:
                        x1, y1, x2, y2 = [int(v) for v in bbox]
                        center = [int((x1 + x2) // 2), int((y1 + y2) // 2)]
                    if bbox and len(bbox) == 4:
                        x1, y1, x2, y2 = [int(v) for v in bbox]
                        ax_overlay.add_patch(
                            Rectangle(
                                (x1, y1),
                                x2 - x1,
                                y2 - y1,
                                fill=False,
                                edgecolor=color,
                                linewidth=1.8,
                            )
                        )
                    if center and len(center) == 2:
                        ax_overlay.plot(
                            int(center[0]),
                            int(center[1]),
                            marker="o",
                            markersize=5,
                            color=color,
                            markeredgecolor="black",
                        )
            else:
                bbox = rec.get("bbox_xyxy")
                if bbox and len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    ax_overlay.add_patch(
                        Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="lime", linewidth=1.5)
                    )
                center = rec.get("center_xy")
                if center and len(center) == 2:
                    ax_overlay.plot(
                        center[0], center[1], marker="o", markersize=5, color="yellow", markeredgecolor="black"
                    )

        ts = rec.get("timestamp_sec")
        ts_txt = f"{ts:.2f}s" if isinstance(ts, (int, float)) else "n/a"
        object_count = len(valid_objects)
        ax_overlay.set_title(
            f"Frame {frame_idx} @ {ts_txt} | mask_px={rec.get('mask_nonzero_pixels', 0)} "
            f"| area={rec.get('mask_area_fraction', 0.0):.4f} | objects={object_count}"
        )
        ax_overlay.axis("off")

        depth_key = ((rec.get("npz_keys") or {}).get("depth"))
        if not depth_key or depth_key not in npz_obj:
            warnings.append(f"Frame {frame_idx}: missing depth key {depth_key}")
            ax_depth.text(0.5, 0.5, f"Missing depth\n{depth_key}", ha="center", va="center")
            ax_depth.axis("off")
            continue

        depth = np.asarray(npz_obj[depth_key], dtype=np.float32)
        im = ax_depth.imshow(depth, cmap=depth_cmap)
        plt.colorbar(im, ax=ax_depth, fraction=0.046, pad=0.01)
        ax_depth.set_title(
            f"Depth | mean(mask)={rec.get('depth_mask_mean')} | center={rec.get('depth_center_value')}"
        )
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
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    for rec in analysis_records:
        frame_idx = int(rec.get("frame_index"))

        frame_rgb = read_video_frame(cap, frame_idx)
        union_mask = build_union_mask(npz_data, rec).astype(bool)

        if union_mask.shape != frame_rgb.shape[:2]:
            raise RuntimeError(
                f"Frame {frame_idx}: mask/frame shape mismatch {union_mask.shape} vs {frame_rgb.shape[:2]}"
            )

        depth_key = ((rec.get("npz_keys") or {}).get("depth"))
        if not depth_key or depth_key not in npz_data:
            raise KeyError(f"Frame {frame_idx}: missing depth key {depth_key}")

        depth = np.asarray(npz_data[depth_key], dtype=np.float32)
        if depth.shape != frame_rgb.shape[:2]:
            depth = cv2.resize(depth, (frame_rgb.shape[1], frame_rgb.shape[0]), interpolation=cv2.INTER_CUBIC)

        bbox = rec.get("bbox_xyxy")
        center = rec.get("center_xy")

        if (not bbox or len(bbox) != 4) and np.any(union_mask):
            ys, xs = np.where(union_mask)
            bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

        if (not center or len(center) != 2) and bbox and len(bbox) == 4:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            center = [int((x1 + x2) // 2), int((y1 + y2) // 2)]

        if not bbox or len(bbox) != 4:
            raise RuntimeError(f"Frame {frame_idx}: no bbox available and mask is empty.")

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

        if not center or len(center) != 2:
            cx, cy = int((x1 + x2) // 2), int((y1 + y2) // 2)
        else:
            cx, cy = [int(v) for v in center]
        cx = max(0, min(cx, depth.shape[1] - 1))
        cy = max(0, min(cy, depth.shape[0] - 1))

        mask_vals = depth[union_mask]
        bbox_vals = depth[bbox_mask]
        bbox_only_vals = depth[bbox_mask & (~union_mask)]
        center_depth = float(depth[cy, cx])

        result = {
            "frame_idx": frame_idx,
            "depth": depth,
            "mask": union_mask,
            "bbox": (x1, y1, x2, y2),
            "bbox_mask": bbox_mask,
            "center": (cx, cy),
            "center_depth": center_depth,
            "mask_pixels": int(union_mask.sum()),
            "bbox_pixels": int(bbox_mask.sum()),
            "mask_mean": float(np.nanmean(mask_vals)) if mask_vals.size else float("nan"),
            "bbox_mean": float(np.nanmean(bbox_vals)) if bbox_vals.size else float("nan"),
            "bbox_only_mean": float(np.nanmean(bbox_only_vals)) if bbox_only_vals.size else float("nan"),
        }
        results.append(result)

    if not results:
        raise RuntimeError("No analysis results generated.")

    return results


def print_depth_statistics(results: list[dict[str, Any]]) -> None:
    print("=== DAP3-style depth statistics (mask vs bbox) ===")
    for r in results:
        x1, y1, x2, y2 = r["bbox"]
        cx, cy = r["center"]
        bbox_only_txt = "NaN" if np.isnan(r["bbox_only_mean"]) else f"{r['bbox_only_mean']:.6f}"
        print(
            f"frame={r['frame_idx']}: bbox=({x1},{y1},{x2},{y2}) | center=({cx},{cy}) | "
            f"mask_px={r['mask_pixels']} | bbox_px={r['bbox_pixels']} | "
            f"mask_mean={r['mask_mean']:.6f} | bbox_mean={r['bbox_mean']:.6f} | "
            f"center_depth={r['center_depth']:.6f} | bbox_only_mean={bbox_only_txt}"
        )


def plot_depth_comparison(results: list[dict[str, Any]], depth_cmap: str = "inferno") -> None:
    n = len(results)
    fig, axes = plt.subplots(nrows=n, ncols=2, figsize=(16, max(4, 5 * n)), squeeze=False)

    for i, r in enumerate(results):
        ax_depth = axes[i, 0]
        ax_dist = axes[i, 1]

        x1, y1, x2, y2 = r["bbox"]
        cx, cy = r["center"]

        im = ax_depth.imshow(r["depth"], cmap=depth_cmap)
        ax_depth.contour(r["mask"].astype(np.uint8), levels=[0.5], colors=["cyan"], linewidths=1.2)
        rect = Rectangle((x1, y1), x2 - x1 + 1, y2 - y1 + 1, fill=False, edgecolor="lime", linewidth=1.5)
        ax_depth.add_patch(rect)
        ax_depth.plot(cx, cy, marker="o", markersize=6, color="white", markeredgecolor="black")

        ax_depth.set_title(
            f"Frame {r['frame_idx']} | mask_mean={r['mask_mean']:.4f}, "
            f"bbox_mean={r['bbox_mean']:.4f}, center={r['center_depth']:.4f}"
        )
        ax_depth.axis("off")
        plt.colorbar(im, ax=ax_depth, fraction=0.046, pad=0.01)

        mask_vals = r["depth"][r["mask"]]
        bbox_vals = r["depth"][r["bbox_mask"]]
        ax_dist.hist(mask_vals[np.isfinite(mask_vals)], bins=50, alpha=0.7, label="Mask")
        ax_dist.hist(bbox_vals[np.isfinite(bbox_vals)], bins=50, alpha=0.5, label="BBox")
        ax_dist.axvline(r["center_depth"], color="black", linestyle="--", linewidth=1.5, label="BBox center depth")
        ax_dist.set_title(f"Frame {r['frame_idx']} - Depth Distribution")
        ax_dist.set_xlabel("Depth value")
        ax_dist.set_ylabel("Pixel count")
        ax_dist.legend()

    plt.tight_layout()
    plt.show()
