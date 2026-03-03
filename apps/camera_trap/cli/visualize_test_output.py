#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
    parser.add_argument("--video-path", type=Path, required=True, help="Source video path.")
    parser.add_argument(
        "--video-stem",
        type=str,
        default=None,
        help="Video stem inside output root. If omitted, first successful manifest entry is used.",
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
    parser.add_argument("--export-fps", type=float, default=None, help="Optional export fps override.")
    parser.add_argument("--export-fourcc", type=str, default="mp4v", help="FourCC code for OpenCV VideoWriter.")
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
    return args


def _colorize_depth(depth: np.ndarray, cmap_name: str) -> np.ndarray:
    finite = depth[np.isfinite(depth)]
    if finite.size == 0:
        return np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)
    lo = float(np.percentile(finite, 2))
    hi = float(np.percentile(finite, 98))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
    if hi <= lo:
        hi = lo + 1e-6
    depth_norm = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
    cmap = plt.get_cmap(cmap_name)
    depth_rgb = (cmap(depth_norm)[..., :3] * 255.0).astype(np.uint8)
    return depth_rgb


def _put_text(img_rgb: np.ndarray, text: str, y: int, color_bgr: tuple[int, int, int] = (255, 255, 255)) -> None:
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.putText(img_bgr, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_bgr, 2, cv2.LINE_AA)
    cv2.putText(img_bgr, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (16, 16, 16), 1, cv2.LINE_AA)
    img_rgb[:, :, :] = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


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

        self.fig, (self.ax_depth, self.ax_dist) = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
        self.fig.canvas.manager.set_window_title("Depth Analysis")
        plt.subplots_adjust(bottom=0.24, right=0.92)
        self.status = self.fig.text(0.01, 0.01, "", fontsize=10)
        self.cbar_ax = self.fig.add_axes([0.93, 0.26, 0.015, 0.62])

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


def main() -> int:
    args = parse_args()
    video_json, npz_data = load_artifacts(output_root=args.output_root, video_stem=args.video_stem)
    processed_records = select_detected_frames(video_json)
    print(f"Validated processed records: {len(processed_records)}")

    cap = open_video_capture(args.video_path)
    try:
        if args.write_video is not None:
            export = write_overlay_video(
                records=processed_records,
                cap_obj=cap,
                npz_obj=npz_data,
                output_path=args.write_video,
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

        if args.no_gui:
            if args.write_video is None:
                print("No GUI mode enabled and no --write-video provided. Nothing to display.")
        else:
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
