#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, Slider, TextBox

# Allow running directly from repository root without requiring editable install.
REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from depth_anything_3.utils.camera_trap_viz import (
    build_object_masks,
    compute_depth_comparison_results,
    draw_overlay_frame,
    get_depth_array,
    load_artifacts,
    open_video_capture,
    read_video_frame,
    select_detected_frames,
    write_overlay_video,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize and export camera-trap test outputs from JSON+NPZ artifacts. "
            "Supports interactive processed/analysis browsing and optional overlay video export."
        )
    )
    parser.add_argument("--output-root", type=Path, required=True, help="Run output root containing manifest/videos.")
    parser.add_argument("--video-path", type=Path, default=None, help="Source video path for direct single-video mode.")
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=None,
        help="Directory containing source videos; used to resolve <video-stem>.<ext> automatically.",
    )
    parser.add_argument(
        "--video-stem",
        type=str,
        default=None,
        help="Video stem inside output root.",
    )
    parser.add_argument(
        "--video-index",
        type=int,
        default=None,
        help="1-based index from discovered successful videos.",
    )
    parser.add_argument(
        "--list-videos",
        action="store_true",
        help="List successful videos discovered from run_manifest.json and exit.",
    )
    parser.add_argument(
        "--view",
        choices=["processed", "analysis", "both"],
        default="processed",
        help="Interactive view mode.",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Disable all interactive matplotlib windows (useful for export-only runs).",
    )
    parser.add_argument("--page-size", type=int, default=1, help="Processed-view page size.")
    parser.add_argument("--overlay-alpha", type=float, default=0.45, help="Mask blend alpha in [0,1].")
    parser.add_argument("--depth-cmap", type=str, default="inferno", help="Matplotlib colormap for depth display.")
    parser.add_argument("--show-bbox", dest="show_bbox", action="store_true", help="Show bounding boxes.")
    parser.add_argument("--no-show-bbox", dest="show_bbox", action="store_false", help="Hide bounding boxes.")
    parser.add_argument("--show-center", dest="show_center", action="store_true", help="Show centers.")
    parser.add_argument("--no-show-center", dest="show_center", action="store_false", help="Hide centers.")
    parser.set_defaults(show_bbox=True, show_center=True)

    parser.add_argument(
        "--write-video",
        type=Path,
        default=None,
        help="If set, write a full-timeline overlay video to this path.",
    )
    parser.add_argument(
        "--write-video-dir",
        type=Path,
        default=None,
        help="Output directory for --export-all; files are written as <video-stem>_overlay.mp4.",
    )
    parser.add_argument(
        "--export-all",
        action="store_true",
        help="Export overlays for all selected/discovered successful videos (headless only).",
    )
    parser.add_argument("--export-fps", type=float, default=None, help="Optional export fps override.")
    parser.add_argument("--export-fourcc", type=str, default="mp4v", help="FourCC code for OpenCV VideoWriter.")
    parser.add_argument(
        "--export-style",
        choices=["rgb", "analysis-depth"],
        default="rgb",
        help="Video export style: standard RGB overlay or analysis-like depth overlay with scale bar.",
    )
    parser.add_argument("--ov-mask", dest="ov_mask", action="store_true", help="Overlay masks in export.")
    parser.add_argument("--no-ov-mask", dest="ov_mask", action="store_false", help="Disable mask overlay in export.")
    parser.add_argument("--ov-bbox", dest="ov_bbox", action="store_true", help="Overlay bboxes in export.")
    parser.add_argument("--no-ov-bbox", dest="ov_bbox", action="store_false", help="Disable bbox overlay in export.")
    parser.add_argument("--ov-center", dest="ov_center", action="store_true", help="Overlay centers in export.")
    parser.add_argument(
        "--no-ov-center", dest="ov_center", action="store_false", help="Disable center overlay in export."
    )
    parser.add_argument("--ov-hud", dest="ov_hud", action="store_true", help="Render text HUD in export.")
    parser.add_argument("--no-ov-hud", dest="ov_hud", action="store_false", help="Disable text HUD in export.")
    parser.set_defaults(ov_mask=True, ov_bbox=True, ov_center=True, ov_hud=True)

    args = parser.parse_args()
    if args.page_size <= 0:
        parser.error("--page-size must be > 0")
    if not (0.0 <= args.overlay_alpha <= 1.0):
        parser.error("--overlay-alpha must be in [0, 1]")
    if args.export_fps is not None and args.export_fps <= 0:
        parser.error("--export-fps must be > 0 when provided")
    if len(args.export_fourcc) != 4:
        parser.error("--export-fourcc must be 4 characters")
    if args.video_path is None and args.video_dir is None:
        parser.error("Provide one of --video-path or --video-dir.")
    if args.video_path is not None and args.video_dir is not None:
        parser.error("Use either --video-path or --video-dir, not both.")
    if args.video_index is not None and args.video_index <= 0:
        parser.error("--video-index must be >= 1 when provided.")
    if args.export_all and args.write_video is not None:
        parser.error("--export-all cannot be used with --write-video; use --write-video-dir.")
    if args.export_all and args.write_video_dir is None:
        parser.error("--export-all requires --write-video-dir.")
    if args.export_all and not args.no_gui:
        parser.error("--export-all requires --no-gui.")
    if args.export_all and args.video_dir is None:
        parser.error("--export-all requires --video-dir.")
    return args


def _load_manifest(output_root: Path) -> dict[str, Any]:
    manifest_path = output_root / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    return json.loads(manifest_path.read_text())


def _successful_videos(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return [v for v in manifest.get("videos", []) if v.get("status") == "success"]


def _candidate_video_extensions(manifest: dict[str, Any]) -> list[str]:
    manifest_exts = [str(e).lower() for e in (manifest.get("video_exts") or []) if str(e).strip()]
    fallback = [".mp4", ".mov", ".avi", ".mkv"]
    all_exts: list[str] = []
    for ext in manifest_exts + fallback:
        ext = ext.lower()
        if not ext.startswith("."):
            ext = f".{ext}"
        if ext not in all_exts:
            all_exts.append(ext)
    return all_exts


def _resolve_video_path_from_dir(video_dir: Path, stem: str, extensions: list[str]) -> Path:
    if not video_dir.is_dir():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")
    lower_map = {p.name.lower(): p for p in video_dir.iterdir() if p.is_file()}
    for ext in extensions:
        candidate = lower_map.get(f"{stem}{ext}".lower())
        if candidate is not None:
            return candidate
    raise FileNotFoundError(
        f"Could not find source video for stem '{stem}' in {video_dir} using extensions {extensions}"
    )


def _print_video_list(videos: list[dict[str, Any]]) -> None:
    if not videos:
        print("No successful videos found in run_manifest.json")
        return
    print(f"Successful videos: {len(videos)}")
    for i, v in enumerate(videos, start=1):
        counts = v.get("counts") or {}
        print(
            f"[{i}] {v.get('video_name')} | sampled={counts.get('sampled_frames', 0)} "
            f"processed={counts.get('processed_frames', 0)}"
        )


def _prompt_video_selection(videos: list[dict[str, Any]]) -> dict[str, Any]:
    while True:
        raw = input(f"Select video index [1-{len(videos)}]: ").strip()
        try:
            idx = int(raw)
        except ValueError:
            print(f"Invalid selection: {raw!r}. Enter an integer.")
            continue
        if idx < 1 or idx > len(videos):
            print(f"Out of range: {idx}. Choose from 1 to {len(videos)}.")
            continue
        return videos[idx - 1]


def _colorize_depth(depth: np.ndarray, cmap_name: str) -> np.ndarray:
    depth_rgb, _, _ = _colorize_depth_with_range(depth=depth, cmap_name=cmap_name)
    return depth_rgb


def _colorize_depth_with_range(depth: np.ndarray, cmap_name: str) -> tuple[np.ndarray, float, float]:
    finite = depth[np.isfinite(depth)]
    if finite.size == 0:
        return np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8), 0.0, 1.0
    lo = float(np.percentile(finite, 2))
    hi = float(np.percentile(finite, 98))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
    if hi <= lo:
        hi = lo + 1e-6
    depth_norm = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
    cmap = plt.get_cmap(cmap_name)
    depth_rgb = (cmap(depth_norm)[..., :3] * 255.0).astype(np.uint8)
    return depth_rgb, lo, hi


def _put_text(img_rgb: np.ndarray, text: str, y: int, color_bgr: tuple[int, int, int] = (255, 255, 255)) -> None:
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.putText(img_bgr, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_bgr, 2, cv2.LINE_AA)
    cv2.putText(img_bgr, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (16, 16, 16), 1, cv2.LINE_AA)
    img_rgb[:, :, :] = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _append_depth_scale_bar(
    frame_rgb: np.ndarray,
    cmap_name: str,
    lo: float | None,
    hi: float | None,
    panel_width: int = 96,
) -> np.ndarray:
    h, _, _ = frame_rgb.shape
    panel = np.full((h, panel_width, 3), 20, dtype=np.uint8)
    bar_w = 24
    x0 = 12
    y0 = 0
    y1 = h

    t = np.linspace(1.0, 0.0, y1 - y0, dtype=np.float32).reshape(-1, 1)
    cmap = plt.get_cmap(cmap_name)
    bar_rgb = (cmap(t)[..., :3] * 255.0).astype(np.uint8)
    bar_rgb = np.repeat(bar_rgb, bar_w, axis=1)
    panel[y0:y1, x0 : x0 + bar_w] = bar_rgb

    panel_bgr = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
    cv2.rectangle(panel_bgr, (x0 - 1, y0), (x0 + bar_w, y1 - 1), (200, 200, 200), 1)
    if lo is None or hi is None:
        cv2.putText(panel_bgr, "n/a", (x0 + bar_w + 8, y0 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
    else:
        mid = (hi + lo) * 0.5
        cv2.putText(panel_bgr, f"{hi:.2f}", (x0 + bar_w + 8, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(panel_bgr, f"{mid:.2f}", (x0 + bar_w + 8, y0 + (y1 - y0) // 2 + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(panel_bgr, f"{lo:.2f}", (x0 + bar_w + 8, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1, cv2.LINE_AA)
    panel = cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2RGB)
    return np.concatenate([frame_rgb, panel], axis=1)


def _draw_mask_outlines_and_depth_labels(
    frame_rgb: np.ndarray,
    rec: dict[str, Any],
    npz_obj: np.lib.npyio.NpzFile,
    depth: np.ndarray,
) -> list[str]:
    warnings: list[str] = []
    h, w = frame_rgb.shape[:2]
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    objects, missing_keys = build_object_masks(npz_data=npz_obj, frame_row=rec)
    if missing_keys:
        warnings.append(f"Frame {rec.get('frame_index')}: missing object mask keys: {missing_keys[:6]}")

    cmap = plt.get_cmap("tab20")
    for obj in objects:
        obj_mask = np.asarray(obj.get("mask"), dtype=bool)
        if obj_mask.shape != (h, w) or not np.any(obj_mask):
            continue
        obj_idx = int(obj.get("object_index", 0))
        color = np.array(cmap(obj_idx % 20)[:3], dtype=np.float32)
        color_bgr = tuple(int(c) for c in (color[::-1] * 255.0))

        mask_u8 = (obj_mask.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(frame_bgr, contours, -1, color_bgr, 2)

        ys, xs = np.where(obj_mask)
        cx = int(np.mean(xs))
        cy = int(np.mean(ys))
        finite_vals = depth[obj_mask]
        finite_vals = finite_vals[np.isfinite(finite_vals)]
        mean_depth = float(np.mean(finite_vals)) if finite_vals.size else float("nan")
        label = f"obj{obj_idx}: {mean_depth:.2f}" if np.isfinite(mean_depth) else f"obj{obj_idx}: n/a"

        tx = max(4, min(cx + 6, w - 180))
        ty = max(14, min(cy - 6, h - 6))
        cv2.putText(frame_bgr, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame_bgr, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 1, cv2.LINE_AA)

    frame_rgb[:, :, :] = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return warnings


def build_processed_page_canvas(
    records: list[dict[str, Any]],
    cap: cv2.VideoCapture,
    npz_data: np.lib.npyio.NpzFile,
    page: int,
    page_size: int,
    overlay_alpha: float,
    depth_cmap: str,
    show_bbox: bool,
    show_center: bool,
) -> tuple[np.ndarray, list[str]]:
    total = len(records)
    max_page = max(0, math.ceil(total / page_size) - 1)
    page = max(0, min(page, max_page))
    start = page * page_size
    end = min(start + page_size, total)
    page_records = records[start:end]

    rows: list[np.ndarray] = []
    warnings: list[str] = []
    for rec in page_records:
        frame_idx = int(rec.get("frame_index", -1))
        try:
            frame_rgb = read_video_frame(cap, frame_idx)
        except Exception as e:
            warnings.append(f"Frame {frame_idx}: decode failed ({e})")
            continue

        overlay_rgb, _, _, overlay_warnings = draw_overlay_frame(
            frame_rgb=frame_rgb,
            rec=rec,
            npz_obj=npz_data,
            overlay_alpha=overlay_alpha,
            show_bbox=show_bbox,
            show_center=show_center,
            draw_mask=True,
            draw_hud=False,
            compute_depth_stats=False,
        )
        warnings.extend(overlay_warnings)

        try:
            depth = get_depth_array(npz_data=npz_data, frame_row=rec, frame_shape_hw=frame_rgb.shape[:2])
            depth_rgb = _colorize_depth(depth, depth_cmap)
        except Exception as e:
            warnings.append(f"Frame {frame_idx}: depth unavailable ({e})")
            depth_rgb = np.zeros_like(frame_rgb)

        ts = rec.get("timestamp_sec")
        ts_txt = f"{ts:.2f}s" if isinstance(ts, (int, float)) else "n/a"
        _put_text(
            overlay_rgb,
            (
                f"frame={frame_idx} @ {ts_txt} "
                f"mask_px={int(rec.get('mask_nonzero_pixels', 0))} "
                f"area={float(rec.get('mask_area_fraction', 0.0)):.4f}"
            ),
            y=24,
        )
        _put_text(depth_rgb, "depth", y=24)

        row = np.concatenate([overlay_rgb, depth_rgb], axis=1)
        rows.append(row)

    if not rows:
        empty = np.zeros((240, 640, 3), dtype=np.uint8)
        _put_text(empty, "No frames rendered on this page", y=120)
        return empty, warnings

    if len(rows) == 1:
        return rows[0], warnings
    pad = np.full((6, rows[0].shape[1], 3), 16, dtype=np.uint8)
    canvas = rows[0]
    for row in rows[1:]:
        canvas = np.concatenate([canvas, pad, row], axis=0)
    return canvas, warnings


class ProcessedBrowser:
    def __init__(
        self,
        records: list[dict[str, Any]],
        cap: cv2.VideoCapture,
        npz_data: np.lib.npyio.NpzFile,
        page_size: int,
        overlay_alpha: float,
        depth_cmap: str,
        show_bbox: bool,
        show_center: bool,
    ) -> None:
        self.records = records
        self.cap = cap
        self.npz_data = npz_data
        self.page_size = page_size
        self.overlay_alpha = overlay_alpha
        self.depth_cmap = depth_cmap
        self.show_bbox = show_bbox
        self.show_center = show_center
        self.max_page = max(0, math.ceil(len(records) / page_size) - 1)
        self.page = 0

        self.fig, self.ax = plt.subplots(figsize=(13, 8))
        self.fig.canvas.manager.set_window_title("Processed Frames")
        plt.subplots_adjust(bottom=0.24)
        self.ax.axis("off")
        self.status = self.fig.text(0.01, 0.01, "", fontsize=10)

        ax_prev = self.fig.add_axes([0.10, 0.10, 0.10, 0.07])
        ax_next = self.fig.add_axes([0.22, 0.10, 0.10, 0.07])
        ax_slider = self.fig.add_axes([0.38, 0.12, 0.34, 0.04])
        ax_text = self.fig.add_axes([0.76, 0.10, 0.12, 0.07])

        self.btn_prev = Button(ax_prev, "Prev")
        self.btn_next = Button(ax_next, "Next")
        self.slider = Slider(ax_slider, "Page", 1, self.max_page + 1, valinit=1, valstep=1)
        self.textbox = TextBox(ax_text, "Go", initial="1")

        self.btn_prev.on_clicked(self._on_prev)
        self.btn_next.on_clicked(self._on_next)
        self.slider.on_changed(self._on_slider)
        self.textbox.on_submit(self._on_submit)
        self._draw()

    def _set_page(self, page: int) -> None:
        clamped = max(0, min(page, self.max_page))
        if clamped == self.page:
            self._draw()
            return
        self.page = clamped
        self.slider.set_val(self.page + 1)

    def _on_prev(self, _event: Any) -> None:
        self._set_page(self.page - 1)

    def _on_next(self, _event: Any) -> None:
        self._set_page(self.page + 1)

    def _on_slider(self, value: float) -> None:
        self._set_page(int(value) - 1)

    def _on_submit(self, text: str) -> None:
        try:
            self._set_page(int(text.strip()) - 1)
        except ValueError:
            self._draw(extra_status=f"Invalid page input: {text!r}")

    def _draw(self, extra_status: str | None = None) -> None:
        canvas, warnings = build_processed_page_canvas(
            records=self.records,
            cap=self.cap,
            npz_data=self.npz_data,
            page=self.page,
            page_size=self.page_size,
            overlay_alpha=self.overlay_alpha,
            depth_cmap=self.depth_cmap,
            show_bbox=self.show_bbox,
            show_center=self.show_center,
        )
        self.ax.clear()
        self.ax.imshow(canvas)
        self.ax.axis("off")
        status = f"Processed page {self.page + 1}/{self.max_page + 1}"
        if warnings:
            status += f" | warnings={len(warnings)}"
        if extra_status:
            status += f" | {extra_status}"
        self.status.set_text(status)
        self.btn_prev.label.set_text("Prev")
        self.btn_next.label.set_text("Next")
        self.btn_prev.ax.set_facecolor("#d0d0d0" if self.page <= 0 else "#f0f0f0")
        self.btn_next.ax.set_facecolor("#d0d0d0" if self.page >= self.max_page else "#f0f0f0")
        self.fig.canvas.draw_idle()


class AnalysisBrowser:
    def __init__(
        self,
        records: list[dict[str, Any]],
        cap: cv2.VideoCapture,
        npz_data: np.lib.npyio.NpzFile,
        depth_cmap: str,
    ) -> None:
        self.records = records
        self.cap = cap
        self.npz_data = npz_data
        self.depth_cmap = depth_cmap
        self.max_idx = max(0, len(records) - 1)
        self.idx = 0

        self.fig = plt.figure(figsize=(14, 6))
        grid = self.fig.add_gridspec(
            nrows=1,
            ncols=3,
            width_ratios=[1.0, 0.05, 1.0],
            left=0.05,
            right=0.97,
            top=0.92,
            bottom=0.24,
            wspace=0.22,
        )
        self.ax_depth = self.fig.add_subplot(grid[0, 0])
        self.cbar_ax = self.fig.add_subplot(grid[0, 1])
        self.ax_dist = self.fig.add_subplot(grid[0, 2])
        self.fig.canvas.manager.set_window_title("Depth Analysis")
        self.status = self.fig.text(0.01, 0.01, "", fontsize=10)

        ax_prev = self.fig.add_axes([0.10, 0.10, 0.10, 0.07])
        ax_next = self.fig.add_axes([0.22, 0.10, 0.10, 0.07])
        ax_slider = self.fig.add_axes([0.38, 0.12, 0.34, 0.04])
        ax_text = self.fig.add_axes([0.76, 0.10, 0.12, 0.07])

        self.btn_prev = Button(ax_prev, "Prev")
        self.btn_next = Button(ax_next, "Next")
        self.slider = Slider(ax_slider, "Index", 1, self.max_idx + 1, valinit=1, valstep=1)
        self.textbox = TextBox(ax_text, "Go", initial="1")

        self.btn_prev.on_clicked(self._on_prev)
        self.btn_next.on_clicked(self._on_next)
        self.slider.on_changed(self._on_slider)
        self.textbox.on_submit(self._on_submit)
        self._draw()

    def _set_idx(self, idx: int) -> None:
        clamped = max(0, min(idx, self.max_idx))
        if clamped == self.idx:
            self._draw()
            return
        self.idx = clamped
        self.slider.set_val(self.idx + 1)

    def _on_prev(self, _event: Any) -> None:
        self._set_idx(self.idx - 1)

    def _on_next(self, _event: Any) -> None:
        self._set_idx(self.idx + 1)

    def _on_slider(self, value: float) -> None:
        self._set_idx(int(value) - 1)

    def _on_submit(self, text: str) -> None:
        try:
            self._set_idx(int(text.strip()) - 1)
        except ValueError:
            self._draw(extra_status=f"Invalid analysis input: {text!r}")

    def _draw(self, extra_status: str | None = None) -> None:
        rec = self.records[self.idx]
        frame_idx = int(rec.get("frame_index", -1))
        result = compute_depth_comparison_results(
            analysis_records=[rec],
            cap=self.cap,
            npz_data=self.npz_data,
        )[0]
        objects = result.get("objects") or []

        self.ax_depth.clear()
        self.ax_dist.clear()
        self.cbar_ax.clear()

        im = self.ax_depth.imshow(result["depth"], cmap=self.depth_cmap)
        for o in objects:
            x1, y1, x2, y2 = o["bbox"]
            cx, cy = o["center"]
            color = o.get("color", np.array([0.0, 1.0, 1.0], dtype=np.float32))
            self.ax_depth.contour(o["mask"].astype(np.uint8), levels=[0.5], colors=[color], linewidths=1.2)
            self.ax_depth.add_patch(Rectangle((x1, y1), x2 - x1 + 1, y2 - y1 + 1, fill=False, edgecolor=color, linewidth=1.5))
            self.ax_depth.plot(cx, cy, marker="o", markersize=6, color=color, markeredgecolor="black")
        self.ax_depth.set_title(f"Frame {frame_idx} | objects={len(objects)}")
        self.ax_depth.axis("off")
        self.fig.colorbar(im, cax=self.cbar_ax)

        for o in objects:
            mask_vals = result["depth"][o["mask"]]
            finite_mask_vals = mask_vals[np.isfinite(mask_vals)]
            color = o.get("color", np.array([0.0, 1.0, 1.0], dtype=np.float32))
            label = f"obj{o.get('object_index')} mean={o.get('mask_mean', float('nan')):.3f}"
            if finite_mask_vals.size:
                self.ax_dist.hist(finite_mask_vals, bins=50, alpha=0.45, label=label, color=color)
                self.ax_dist.axvline(o["center_depth"], color=color, linestyle="--", linewidth=1.2)
        self.ax_dist.set_title("Depth Distribution")
        self.ax_dist.set_xlabel("Depth value")
        self.ax_dist.set_ylabel("Pixel count")
        if objects:
            self.ax_dist.legend()

        status = f"Analysis {self.idx + 1}/{self.max_idx + 1} | frame={frame_idx}"
        if extra_status:
            status += f" | {extra_status}"
        self.status.set_text(status)
        self.btn_prev.ax.set_facecolor("#d0d0d0" if self.idx <= 0 else "#f0f0f0")
        self.btn_next.ax.set_facecolor("#d0d0d0" if self.idx >= self.max_idx else "#f0f0f0")
        self.fig.canvas.draw_idle()


def _select_videos(args: argparse.Namespace, successful_videos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not successful_videos:
        raise RuntimeError("No successful videos found in run_manifest.json")

    by_stem = {str(v.get("video_name")): v for v in successful_videos}
    if args.video_stem is not None:
        selected = by_stem.get(args.video_stem)
        if selected is None:
            raise ValueError(f"--video-stem not found among successful videos: {args.video_stem}")
        return [selected]

    if args.video_index is not None:
        if args.video_index < 1 or args.video_index > len(successful_videos):
            raise ValueError(
                f"--video-index out of range: {args.video_index}. Valid range: 1..{len(successful_videos)}"
            )
        return [successful_videos[args.video_index - 1]]

    if args.video_path is not None:
        inferred_stem = args.video_path.stem
        selected = by_stem.get(inferred_stem)
        if selected is not None:
            return [selected]
        return [{"video_name": inferred_stem}]

    if args.export_all:
        return successful_videos

    if len(successful_videos) == 1:
        return [successful_videos[0]]

    _print_video_list(successful_videos)
    return [_prompt_video_selection(successful_videos)]


def _resolve_video_path_for_entry(
    args: argparse.Namespace,
    entry: dict[str, Any],
    manifest_exts: list[str],
) -> Path:
    stem = str(entry.get("video_name"))
    if args.video_path is not None:
        return args.video_path
    if args.video_dir is None:
        raise RuntimeError("--video-dir is required when --video-path is not provided")
    return _resolve_video_path_from_dir(video_dir=args.video_dir, stem=stem, extensions=manifest_exts)


def _write_analysis_depth_video(
    records: list[dict[str, Any]],
    cap_obj: cv2.VideoCapture,
    npz_obj: np.lib.npyio.NpzFile,
    output_path: Path,
    depth_cmap: str,
    overlay_alpha: float = 0.45,
    show_bbox: bool = True,
    show_center: bool = True,
    draw_mask: bool = True,
    draw_hud: bool = True,
    output_fps: float | None = None,
    fourcc: str = "mp4v",
) -> dict[str, Any]:
    if len(fourcc) != 4:
        raise ValueError("fourcc must be a 4-character code, e.g. 'mp4v'")

    rec_map = {int(rec.get("frame_index", -1)): rec for rec in records if int(rec.get("frame_index", -1)) >= 0}
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

    panel_width = 96
    ok = cap_obj.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if not ok:
        raise RuntimeError("failed to seek source capture to frame 0")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*fourcc),
        fps,
        (width + panel_width, height),
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
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            rec = rec_map.get(frame_idx)
            if rec is None:
                composed = _append_depth_scale_bar(
                    frame_rgb=frame_rgb,
                    cmap_name=depth_cmap,
                    lo=None,
                    hi=None,
                    panel_width=panel_width,
                )
                writer.write(cv2.cvtColor(composed, cv2.COLOR_RGB2BGR))
                written += 1
                continue

            try:
                depth = get_depth_array(npz_data=npz_obj, frame_row=rec, frame_shape_hw=frame_rgb.shape[:2])
                depth_rgb, lo, hi = _colorize_depth_with_range(depth=depth, cmap_name=depth_cmap)
            except Exception as e:
                warnings.append(f"Frame {frame_idx}: depth unavailable ({e})")
                depth_rgb = frame_rgb
                lo, hi = None, None

            overlay_rgb, _, _, frame_warnings = draw_overlay_frame(
                frame_rgb=depth_rgb,
                rec=rec,
                npz_obj=npz_obj,
                overlay_alpha=overlay_alpha,
                show_bbox=show_bbox,
                show_center=show_center,
                draw_mask=False,
                draw_hud=draw_hud,
                compute_depth_stats=draw_hud,
            )
            warnings.extend(frame_warnings)
            warnings.extend(
                _draw_mask_outlines_and_depth_labels(
                    frame_rgb=overlay_rgb,
                    rec=rec,
                    npz_obj=npz_obj,
                    depth=depth,
                )
            )
            composed = _append_depth_scale_bar(
                frame_rgb=overlay_rgb,
                cmap_name=depth_cmap,
                lo=lo,
                hi=hi,
                panel_width=panel_width,
            )
            writer.write(cv2.cvtColor(composed, cv2.COLOR_RGB2BGR))
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


def _run_single_video(
    args: argparse.Namespace,
    manifest_exts: list[str],
    entry: dict[str, Any],
    write_video_path: Path | None,
    allow_gui: bool,
) -> None:
    video_stem = str(entry.get("video_name"))
    video_path = _resolve_video_path_for_entry(args=args, entry=entry, manifest_exts=manifest_exts)
    video_json, npz_data = load_artifacts(output_root=args.output_root, video_stem=video_stem)
    processed_records = select_detected_frames(video_json)
    print(f"Validated processed records: {len(processed_records)}")
    cap = open_video_capture(video_path)
    try:
        if write_video_path is not None:
            if args.export_style == "analysis-depth":
                export = _write_analysis_depth_video(
                    records=processed_records,
                    cap_obj=cap,
                    npz_obj=npz_data,
                    output_path=write_video_path,
                    depth_cmap=args.depth_cmap,
                    overlay_alpha=args.overlay_alpha,
                    show_bbox=args.ov_bbox,
                    show_center=args.ov_center,
                    draw_mask=args.ov_mask,
                    draw_hud=args.ov_hud,
                    output_fps=args.export_fps,
                    fourcc=args.export_fourcc,
                )
            else:
                export = write_overlay_video(
                    records=processed_records,
                    cap_obj=cap,
                    npz_obj=npz_data,
                    output_path=write_video_path,
                    overlay_alpha=args.overlay_alpha,
                    show_bbox=args.ov_bbox,
                    show_center=args.ov_center,
                    draw_mask=args.ov_mask,
                    draw_hud=args.ov_hud,
                    output_fps=args.export_fps,
                    fourcc=args.export_fourcc,
                )
            print("Overlay export complete:")
            print(f"  path: {export['output_path']}")
            print(f"  frames: {export['written_frames']}/{export['source_frame_count']}")
            print(f"  overlaid: {export['overlay_frames']}")
            print(f"  fps: source={export['source_fps']:.3f} output={export['output_fps']:.3f}")
            if export["warnings"]:
                print(f"  warnings: {len(export['warnings'])}")

        if allow_gui:
            browsers: list[Any] = []
            if args.view in ("processed", "both"):
                browsers.append(
                    ProcessedBrowser(
                        records=processed_records,
                        cap=cap,
                        npz_data=npz_data,
                        page_size=args.page_size,
                        overlay_alpha=args.overlay_alpha,
                        depth_cmap=args.depth_cmap,
                        show_bbox=args.show_bbox,
                        show_center=args.show_center,
                    )
                )
            if args.view in ("analysis", "both"):
                browsers.append(
                    AnalysisBrowser(
                        records=processed_records,
                        cap=cap,
                        npz_data=npz_data,
                        depth_cmap=args.depth_cmap,
                    )
                )
            if browsers:
                plt.show()
    finally:
        cap.release()
        npz_data.close()


def main() -> int:
    args = parse_args()
    manifest = _load_manifest(args.output_root)
    successful_videos = _successful_videos(manifest)
    manifest_exts = _candidate_video_extensions(manifest)

    if args.list_videos:
        _print_video_list(successful_videos)
        return 0

    selected = _select_videos(args=args, successful_videos=successful_videos)

    if args.export_all:
        assert args.write_video_dir is not None
        args.write_video_dir.mkdir(parents=True, exist_ok=True)
        ok = 0
        failed = 0
        for entry in selected:
            stem = str(entry.get("video_name"))
            print(f"\n=== Exporting {stem} ===")
            suffix = "_analysis_overlay.mp4" if args.export_style == "analysis-depth" else "_overlay.mp4"
            out_path = args.write_video_dir / f"{stem}{suffix}"
            try:
                _run_single_video(
                    args=args,
                    manifest_exts=manifest_exts,
                    entry=entry,
                    write_video_path=out_path,
                    allow_gui=False,
                )
                ok += 1
            except Exception as e:
                failed += 1
                print(f"Failed: {stem} ({e})")
        print(f"\nBatch export complete: success={ok} failed={failed} total={len(selected)}")
        return 0 if failed == 0 else 1

    chosen = selected[0]
    if args.no_gui and args.write_video is None:
        print("No GUI mode enabled and no --write-video provided. Nothing to display.")
        return 0
    _run_single_video(
        args=args,
        manifest_exts=manifest_exts,
        entry=chosen,
        write_video_path=args.write_video,
        allow_gui=not args.no_gui,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
