#!/usr/bin/env python3
"""Build per-camera distance-calibration references (Haucke et al. 2022 method) from a set of
person-holding-a-distance-sign calibration frames, using SAM3 person masks and DA3 depth.

Ported (disparity-space calibration) from https://github.com/timmh/distance-estimation (run.py:109-197),
via apps/camera_trap/calibration/timmh.py. See that file's docstring/header for MIT license text.

Per camera (`transect_cam`), calibration instances come from `--calibration-frames` (columns:
`transect_cam, video_path, event_id, frame_idx, distance_m, sign_box_norm, n_agree, vlm_conf,
source`). For each instance: decode the frame, pick a SAM3 person mask (highest overlap with the
dilated sign box, falling back to highest score), run DA3 depth, and align that frame's disparity
onto the camera's anchor frame (largest-distance instance) using
apps.camera_trap.calibration.timmh.align_disparity. A piecewise-linear curve is then fit in
disparity space (1/distance vs. aligned disparity) per camera when >=2 distinct calibration
distances are available; cameras with fewer fall back to a pooled (unaligned, cross-camera) curve.

Outputs (in --out-dir):
  calib/<transect_cam>.npz    anchor_disp_raw, anchor_img, anchor_person_mask, knots_x, knots_y,
                               max_depth, min_depth, anchor_distance_m, da3_model_id, calib_method
                               ("per_camera" for this camera's own fitted curve, "pooled_anchor"
                               when the camera has its own anchor/alignment but the knots are
                               borrowed from the cross-camera pooled curve for lack of >=2
                               distinct reference distances)
  calib/_pooled.npz           knots_x, knots_y, max_depth, min_depth, n_instances, n_cameras
  calibration_summary.csv     one row per camera in --cameras (union with cameras that have
                               instances)
  calibration_instances.csv   one row per calibration instance
  calibration_plots/<cam>.png curve + instance points
  instances/<cam>/*.jpg       frame + chosen mask overlay, one per instance, for spot checks
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apps.camera_trap.calibration import timmh
from apps.camera_trap.vlm.frames import iter_frames_at_indices

INSTANCE_STATUS_CHOICES = ("ok", "no_mask", "align_failed", "decode_failed")
INSTANCE_CSV_FIELDS = [
    "transect_cam", "event_id", "video_path", "frame_idx", "distance_m", "status",
    "mask_score", "mask_area_px", "x_aligned", "x_raw", "align_inlier_frac", "pred_depth_insample",
]
SUMMARY_CSV_FIELDS = [
    "transect_cam", "n_instances", "n_valid", "distances", "loo_mae_m", "insample_mae_m",
    "method", "anchor_distance_m", "notes",
]

_ANCHOR_IMG_MAX_SIDE = 1024


# --------------------------------------------------------------------------------------
# calibration_frames.csv parsing
# --------------------------------------------------------------------------------------

def parse_sign_box_norm(raw: str) -> tuple[float, float, float, float]:
    """"x0;y0;x1;y1" (normalised 0-1) -> (x0, y0, x1, y1)."""
    parts = [float(p) for p in str(raw).split(";")]
    if len(parts) != 4:
        raise ValueError(f"invalid sign_box_norm (expected 4 ';'-separated floats): {raw!r}")
    return tuple(parts)  # type: ignore[return-value]


def load_calibration_frames(path, reference_root: Path) -> list[dict[str, Any]]:
    import pandas as pd

    df = pd.read_csv(path)
    required = ["transect_cam", "video_path", "event_id", "frame_idx", "distance_m", "sign_box_norm"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"calibration-frames CSV missing required column: {col}")
    if "vlm_conf" not in df.columns:
        df["vlm_conf"] = np.nan
    if "n_agree" not in df.columns:
        df["n_agree"] = np.nan
    if "source" not in df.columns:
        df["source"] = None

    rows = []
    for _, row in df.iterrows():
        video_path = Path(str(row["video_path"]))
        if not video_path.is_absolute():
            video_path = reference_root / video_path
        rows.append(
            {
                "transect_cam": str(row["transect_cam"]),
                "video_path": video_path,
                "event_id": row["event_id"],
                "frame_idx": int(row["frame_idx"]),
                "distance_m": float(row["distance_m"]),
                "sign_box_norm": parse_sign_box_norm(row["sign_box_norm"]),
                "n_agree": row["n_agree"],
                "vlm_conf": float(row["vlm_conf"]) if not pd.isna(row["vlm_conf"]) else float("nan"),
                "source": row["source"],
            }
        )
    return rows


def load_cameras_list(path) -> list[str]:
    import pandas as pd

    df = pd.read_csv(path)
    if "transect_cam" not in df.columns:
        raise ValueError("cameras CSV missing required column: transect_cam")
    return [str(v) for v in df["transect_cam"].dropna().unique().tolist()]


# --------------------------------------------------------------------------------------
# mask selection (sign box overlap)
# --------------------------------------------------------------------------------------

def dilate_box(box_xyxy: tuple[float, float, float, float], factor: float, shape_hw: tuple[int, int]) -> tuple[int, int, int, int]:
    """Dilate an absolute-pixel xyxy box by `factor` * box size (each side), clamped to shape_hw."""
    x0, y0, x1, y1 = box_xyxy
    w, h = x1 - x0, y1 - y0
    dx, dy = w * factor, h * factor
    x0, y0, x1, y1 = x0 - dx, y0 - dy, x1 + dx, y1 + dy
    height, width = shape_hw
    x0 = max(0, min(width - 1, x0))
    x1 = max(0, min(width, x1))
    y0 = max(0, min(height - 1, y0))
    y1 = max(0, min(height, y1))
    return int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))


def sign_box_norm_to_abs(sign_box_norm, shape_hw: tuple[int, int]) -> tuple[float, float, float, float]:
    height, width = shape_hw
    x0, y0, x1, y1 = sign_box_norm
    return x0 * width, y0 * height, x1 * width, y1 * height


def box_overlap_fraction(dilated_box_xyxy: tuple[int, int, int, int], mask_box_xyxy) -> float:
    """Fraction of the dilated sign box's area covered by mask_box_xyxy (a mask's bbox)."""
    if mask_box_xyxy is None:
        return 0.0
    ax0, ay0, ax1, ay1 = dilated_box_xyxy
    bx0, by0, bx1, by1 = mask_box_xyxy
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter_w, inter_h = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter_area = inter_w * inter_h
    box_area = max(1, (ax1 - ax0) * (ay1 - ay0))
    return inter_area / box_area


def select_person_mask(
    masks: list[tuple[np.ndarray, float, Any]],
    sign_box_norm,
    frame_shape_hw: tuple[int, int],
    dilate_factor: float = 1.0,
) -> Optional[tuple[np.ndarray, float, Any]]:
    """Pick the mask with highest dilated-sign-box overlap; fall back to highest score.

    `masks` is a list of (mask_bool, score, box_xyxy). Returns None if `masks` is empty.
    """
    if not masks:
        return None

    sign_box_abs = sign_box_norm_to_abs(sign_box_norm, frame_shape_hw)
    dilated = dilate_box(sign_box_abs, dilate_factor, frame_shape_hw)

    best = None
    best_overlap = -1.0
    for mask, score, box in masks:
        overlap = box_overlap_fraction(dilated, box)
        if overlap > best_overlap:
            best_overlap = overlap
            best = (mask, score, box)

    if best_overlap <= 0.0:
        best = max(masks, key=lambda m: m[1])
    return best


# --------------------------------------------------------------------------------------
# core pure functions (no GPU / no I/O)
# --------------------------------------------------------------------------------------

def _median_in_mask(arr: np.ndarray, mask: np.ndarray) -> float:
    mask_r = timmh.resize_to(np.asarray(mask, dtype=bool), arr.shape[:2])
    if not mask_r.any():
        return float("nan")
    return float(np.median(np.asarray(arr)[mask_r]))


def select_anchor_index(instances: list[dict[str, Any]]) -> int:
    """Index of the instance with the largest distance_m (ties -> highest vlm_conf)."""
    def key(i_inst):
        _, inst = i_inst
        conf = inst.get("vlm_conf")
        conf = conf if conf is not None and not np.isnan(conf) else float("-inf")
        return (inst["distance_m"], conf)

    return max(enumerate(instances), key=key)[0]


def build_camera_calibration(
    instances: list[dict[str, Any]],
    *,
    transect_cam: str = "",
    min_depth: float = 1.0,
    max_depth: float = 25.0,
    da3_model_id: Optional[str] = None,
) -> tuple[Optional[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Build one camera's calibration from its (already SAM3+DA3-processed) instances.

    Each element of `instances` (only successfully-masked instances; no_mask/decode_failed
    instances must be recorded by the caller and excluded here) is a dict with:
      event_id, video_path, frame_idx, distance_m, vlm_conf, mask_score, mask_area_px,
      disp (2D float array, raw DA3 disparity at DA3 output resolution),
      person_mask (2D bool array, same/resizable resolution),
      frame_bgr (optional; only used if this instance becomes the anchor).

    Returns (npz_dict, instance_rows, summary_row). `npz_dict` is None if there are no
    instances at all; otherwise it always carries anchor_* fields, and knots_x/knots_y are
    populated only when method=="per_camera" (None when method=="pooled", to be filled in by
    the caller from the pooled curve).
    """
    if not instances:
        return None, [], {
            "transect_cam": transect_cam, "n_instances": 0, "n_valid": 0, "distances": "",
            "loo_mae_m": None, "insample_mae_m": None, "method": "pooled",
            "anchor_distance_m": None, "notes": "no_instances",
        }

    anchor_idx = select_anchor_index(instances)
    anchor = instances[anchor_idx]
    anchor_disp_raw = np.asarray(anchor["disp"], dtype=np.float32)
    anchor_mask = timmh.resize_to(np.asarray(anchor["person_mask"], dtype=bool), anchor_disp_raw.shape[:2])

    rows: list[dict[str, Any]] = []
    for inst in instances:
        disp_i = np.asarray(inst["disp"])
        mask_i = np.asarray(inst["person_mask"], dtype=bool)
        x_raw = _median_in_mask(disp_i, mask_i)

        mask_i_anchor_res = timmh.resize_to(mask_i, anchor_disp_raw.shape[:2])
        exclude_mask = mask_i_anchor_res | anchor_mask
        aligned, info = timmh.align_disparity(disp_i, anchor_disp_raw, exclude_mask)

        x_aligned = float("nan")
        align_inlier_frac = info.get("inlier_frac")
        status = "align_failed"
        if aligned is not None:
            x_aligned = _median_in_mask(aligned, mask_i_anchor_res)
            if not np.isnan(x_aligned):
                status = "ok"

        rows.append({
            "transect_cam": transect_cam,
            "event_id": inst.get("event_id"),
            "video_path": inst.get("video_path"),
            "frame_idx": inst.get("frame_idx"),
            "distance_m": float(inst["distance_m"]),
            "status": status,
            "mask_score": inst.get("mask_score"),
            "mask_area_px": inst.get("mask_area_px"),
            "x_aligned": x_aligned,
            "x_raw": x_raw,
            "align_inlier_frac": align_inlier_frac,
            "pred_depth_insample": None,
        })

    valid_idx = [i for i, r in enumerate(rows) if r["status"] == "ok"]
    distinct_distances = sorted({rows[i]["distance_m"] for i in valid_idx})

    method = "pooled"
    knots_x = knots_y = None
    loo_mae_m = insample_mae_m = None

    if len(distinct_distances) >= 2:
        method = "per_camera"

        def fit_curve(idxs):
            by_dist: dict[float, list[float]] = defaultdict(list)
            for i in idxs:
                by_dist[rows[i]["distance_m"]].append(rows[i]["x_aligned"])
            dists = sorted(by_dist)
            x_d = np.array([np.median(by_dist[d]) for d in dists], dtype=np.float64)
            y_d = np.array([1.0 / d for d in dists], dtype=np.float64)
            return timmh.piecewise_linear_calibration(x_d, y_d), len(dists)

        curve, _ = fit_curve(valid_idx)
        knots_x, knots_y = curve.knots_x, curve.knots_y

        insample_errs = []
        for i in valid_idx:
            pred_depth = float(1.0 / curve(np.array([rows[i]["x_aligned"]]))[0])
            rows[i]["pred_depth_insample"] = pred_depth
            insample_errs.append(abs(pred_depth - rows[i]["distance_m"]))
        insample_mae_m = float(np.mean(insample_errs)) if insample_errs else None

        loo_errs = []
        for i in valid_idx:
            remaining = [j for j in valid_idx if j != i]
            remaining_dists = {rows[j]["distance_m"] for j in remaining}
            if len(remaining_dists) < 2:
                continue
            loo_curve, _ = fit_curve(remaining)
            pred_depth = float(1.0 / loo_curve(np.array([rows[i]["x_aligned"]]))[0])
            loo_errs.append(abs(pred_depth - rows[i]["distance_m"]))
        loo_mae_m = float(np.mean(loo_errs)) if loo_errs else None

    anchor_img = anchor.get("frame_bgr")
    if anchor_img is not None:
        anchor_img = _downscale_bgr(anchor_img, _ANCHOR_IMG_MAX_SIDE)
    else:
        anchor_img = np.zeros((1, 1, 3), dtype=np.uint8)

    npz_dict = {
        "anchor_disp_raw": anchor_disp_raw.astype(np.float32),
        "anchor_img": anchor_img.astype(np.uint8),
        "anchor_person_mask": anchor_mask.astype(bool),
        "knots_x": np.asarray(knots_x, dtype=np.float64) if knots_x is not None else None,
        "knots_y": np.asarray(knots_y, dtype=np.float64) if knots_y is not None else None,
        "max_depth": float(max_depth),
        "min_depth": float(min_depth),
        "anchor_distance_m": float(anchor["distance_m"]),
        "da3_model_id": da3_model_id or "",
        # "per_camera": this camera's own fitted curve. "pooled_anchor": this camera still has
        # its own anchor (so per-frame alignment happens), but its knots are borrowed from the
        # cross-camera pooled curve because it had <2 distinct reference distances.
        "calib_method": "per_camera" if method == "per_camera" else "pooled_anchor",
    }

    all_distances = sorted({r["distance_m"] for r in rows})
    summary_row = {
        "transect_cam": transect_cam,
        "n_instances": len(rows),
        "n_valid": len(valid_idx),
        "distances": ";".join(str(d) for d in all_distances),
        "loo_mae_m": loo_mae_m,
        "insample_mae_m": insample_mae_m,
        "method": method,
        "anchor_distance_m": float(anchor["distance_m"]),
        "notes": "",
    }
    return npz_dict, rows, summary_row


def _downscale_bgr(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale >= 1.0:
        return img
    return cv2.resize(img, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)


def build_pooled(
    instance_rows: list[dict[str, Any]],
    *,
    min_depth: float = 1.0,
    max_depth: float = 25.0,
) -> Optional[dict[str, Any]]:
    """Pooled fallback curve from ALL valid (masked) instances across all cameras, using each
    instance's raw (unaligned) disparity median in its person mask."""
    rows = [
        r for r in instance_rows
        if r.get("status") in ("ok", "align_failed") and r.get("x_raw") is not None and not np.isnan(r["x_raw"])
    ]
    if not rows:
        return None

    by_dist: dict[float, list[float]] = defaultdict(list)
    cams_by_dist: dict[float, set] = defaultdict(set)
    for r in rows:
        by_dist[r["distance_m"]].append(r["x_raw"])
        cams_by_dist[r["distance_m"]].add(r["transect_cam"])

    dists = sorted(by_dist)
    x_d = np.array([np.median(by_dist[d]) for d in dists], dtype=np.float64)
    y_d = np.array([1.0 / d for d in dists], dtype=np.float64)
    curve = timmh.piecewise_linear_calibration(x_d, y_d)

    n_cameras = len({r["transect_cam"] for r in rows})
    return {
        "knots_x": np.asarray(curve.knots_x, dtype=np.float64),
        "knots_y": np.asarray(curve.knots_y, dtype=np.float64),
        "max_depth": float(max_depth),
        "min_depth": float(min_depth),
        "n_instances": len(rows),
        "n_cameras": n_cameras,
    }


# --------------------------------------------------------------------------------------
# SAM3 segmenter adapter (real backend constructed lazily; tests inject a fake directly)
# --------------------------------------------------------------------------------------

def _make_segmenter(args) -> Callable[[np.ndarray, str], list[tuple[np.ndarray, float, Any]]]:
    """Build a `segment_image(frame_bgr, prompt) -> list[(mask, score, box_xyxy)]` callable
    around `apps.camera_trap.sam3_backends.OfficialSam3ImageSegmenter`.

    Constructed lazily (only when actually segmenting) so importing this module never requires
    sam3/torch, and tests can pass a fake segmenter into the pipeline functions directly instead
    of going through this adapter.
    """
    from apps.camera_trap.sam3_backends import OfficialSam3ImageSegmenter

    device = args.device
    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

    segmenter = OfficialSam3ImageSegmenter(
        sam3_model_path=args.sam3_model_path,
        device=device,
        confidence_threshold=args.sam3_det_threshold,
    )
    return segmenter.segment_image


# --------------------------------------------------------------------------------------
# overlay / plot rendering
# --------------------------------------------------------------------------------------

def render_mask_overlay(frame_bgr: np.ndarray, mask: np.ndarray, color=(0, 0, 255), alpha: float = 0.5) -> np.ndarray:
    mask_r = timmh.resize_to(np.asarray(mask, dtype=bool), frame_bgr.shape[:2])
    overlay = frame_bgr.copy()
    overlay[mask_r] = (
        overlay[mask_r].astype(np.float32) * (1 - alpha) + np.array(color, dtype=np.float32) * alpha
    ).astype(np.uint8)
    return overlay


def plot_camera_curve(rows: list[dict[str, Any]], npz_dict: dict[str, Any], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    knots_x, knots_y = npz_dict.get("knots_x"), npz_dict.get("knots_y")
    fig, ax = plt.subplots(figsize=(5, 4))
    valid = [r for r in rows if r["status"] == "ok"]
    if valid:
        xs = [r["x_aligned"] for r in valid]
        ys = [1.0 / r["distance_m"] for r in valid]
        ax.scatter(xs, ys, s=16, color="black", label="instances")
    if knots_x is not None and knots_y is not None:
        curve = timmh.piecewise_from_knots(knots_x, knots_y)
        lo = min(knots_x.min(), *(xs if valid else [knots_x.min()]))
        hi = max(knots_x.max(), *(xs if valid else [knots_x.max()]))
        xs_line = np.linspace(lo, hi, 100)
        ax.plot(xs_line, curve(xs_line), color="C0", lw=1, label="curve")
    ax.set_xlabel("aligned disparity")
    ax.set_ylabel("1 / distance (1/m)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------------------
# CSV I/O helpers
# --------------------------------------------------------------------------------------

def append_instance_rows(csv_path: Path, rows: list[dict[str, Any]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INSTANCE_CSV_FIELDS)
        if write_header:
            writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in INSTANCE_CSV_FIELDS})


def read_instance_rows(csv_path: Path) -> list[dict[str, Any]]:
    if not csv_path.exists():
        return []
    rows = []
    with csv_path.open(newline="") as f:
        for r in csv.DictReader(f):
            r["distance_m"] = float(r["distance_m"]) if r["distance_m"] not in (None, "") else None
            for k in ("x_aligned", "x_raw", "align_inlier_frac", "pred_depth_insample"):
                v = r.get(k)
                r[k] = float(v) if v not in (None, "") else float("nan")
            rows.append(r)
    return rows


def write_summary_csv(csv_path: Path, rows: list[dict[str, Any]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_CSV_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in SUMMARY_CSV_FIELDS})


def save_camera_npz(path: Path, npz_dict: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: v for k, v in npz_dict.items() if v is not None}
    tmp_base = path.parent / f".{path.name}.tmp"
    np.savez_compressed(str(tmp_base), **arrays)
    tmp_npz = Path(f"{tmp_base}.npz")
    tmp_npz.rename(path)


# --------------------------------------------------------------------------------------
# GPU-dependent per-camera processing
# --------------------------------------------------------------------------------------

def run_da3_batched(da3, frames_bgr: list[np.ndarray], batch_size: int) -> list[np.ndarray]:
    """Run DA3 depth inference in chunks of `batch_size`, mirroring dap3_cli.run_da3_inference_batch."""
    depths: list[np.ndarray] = []
    for start in range(0, len(frames_bgr), batch_size):
        chunk = frames_bgr[start:start + batch_size]
        frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in chunk]
        depth_pred = da3.inference(frames_rgb, use_ray_pose=False, infer_gs=False, export_dir=None)
        depths.extend(np.asarray(d, dtype=np.float32) for d in depth_pred.depth)
    return depths


def process_camera(
    transect_cam: str,
    events: list[dict[str, Any]],
    *,
    segmenter: Callable[[np.ndarray, str], list[tuple[np.ndarray, float, Any]]],
    sam3_prompt: str,
    da3,
    da3_batch_size: int,
    out_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Decode frames, segment, run DA3, and assemble the `instances` list build_camera_calibration
    expects. Returns (instances, unmasked_instance_rows) where unmasked_instance_rows records
    no_mask/decode_failed instances (for calibration_instances.csv; excluded from calibration).
    """
    by_video: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    for ev in events:
        by_video[ev["video_path"]].append(ev)

    decoded: dict[tuple[Path, int], Optional[np.ndarray]] = {}
    for video_path, evs in by_video.items():
        frame_indices = [ev["frame_idx"] for ev in evs]
        for frame_idx, frame_bgr in iter_frames_at_indices(video_path, frame_indices):
            decoded[(video_path, frame_idx)] = frame_bgr

    candidates: list[dict[str, Any]] = []
    unmasked_rows: list[dict[str, Any]] = []
    for ev in events:
        frame_bgr = decoded.get((ev["video_path"], ev["frame_idx"]))
        if frame_bgr is None:
            unmasked_rows.append(_no_mask_row(transect_cam, ev, "decode_failed"))
            continue

        masks = segmenter(frame_bgr, sam3_prompt)
        chosen = select_person_mask(masks, ev["sign_box_norm"], frame_bgr.shape[:2])
        if chosen is None:
            unmasked_rows.append(_no_mask_row(transect_cam, ev, "no_mask"))
            continue

        mask, score, box = chosen
        instance_dir = out_dir / "instances" / transect_cam
        instance_dir.mkdir(parents=True, exist_ok=True)
        overlay = render_mask_overlay(frame_bgr, mask)
        cv2.imwrite(str(instance_dir / f"{ev['event_id']}_{ev['frame_idx']}.jpg"), overlay)

        candidates.append({
            **ev,
            "frame_bgr": frame_bgr,
            "person_mask": np.asarray(mask, dtype=bool),
            "mask_score": float(score),
            "mask_area_px": int(np.asarray(mask, dtype=bool).sum()),
        })

    if candidates:
        frames_for_da3 = [c["frame_bgr"] for c in candidates]
        depths = run_da3_batched(da3, frames_for_da3, da3_batch_size)
        for c, depth in zip(candidates, depths):
            c["disp"] = timmh.disparity_from_depth(depth)
            c["person_mask"] = timmh.resize_to(c["person_mask"], depth.shape[:2])

    return candidates, unmasked_rows


def _no_mask_row(transect_cam: str, ev: dict[str, Any], status: str) -> dict[str, Any]:
    return {
        "transect_cam": transect_cam,
        "event_id": ev.get("event_id"),
        "video_path": ev.get("video_path"),
        "frame_idx": ev.get("frame_idx"),
        "distance_m": ev.get("distance_m"),
        "status": status,
        "mask_score": None,
        "mask_area_px": None,
        "x_aligned": float("nan"),
        "x_raw": float("nan"),
        "align_inlier_frac": None,
        "pred_depth_insample": None,
    }


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--calibration-frames", required=True, type=Path)
    p.add_argument("--reference-root", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--sam3-model-path", required=True)
    p.add_argument("--sam3-prompt", default="person")
    p.add_argument("--sam3-det-threshold", type=float, default=0.5)
    p.add_argument("--da3-model-id", default="depth-anything/DA3NESTED-GIANT-LARGE")
    p.add_argument("--da3-batch-size", type=int, default=8)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--cameras", type=Path, default=None)
    p.add_argument("--min-depth", type=float, default=1.0)
    p.add_argument("--max-depth", type=float, default=25.0)
    p.add_argument("--only-cams", default=None, help="Comma-separated transect_cam allowlist.")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    calib_dir = args.out_dir / "calib"
    calib_dir.mkdir(parents=True, exist_ok=True)
    instances_csv = args.out_dir / "calibration_instances.csv"

    events = load_calibration_frames(args.calibration_frames, args.reference_root)
    if args.only_cams:
        allow = set(args.only_cams.split(","))
        events = [e for e in events if e["transect_cam"] in allow]

    by_cam: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ev in events:
        by_cam[ev["transect_cam"]].append(ev)

    import torch

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else (
        "cpu" if args.device == "auto" else args.device
    )

    da3 = None
    segmenter = None

    all_instance_rows: list[dict[str, Any]] = read_instance_rows(instances_csv)
    pending_camera_npz: dict[str, tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]] = {}

    for transect_cam, cam_events in sorted(by_cam.items()):
        npz_path = calib_dir / f"{transect_cam}.npz"
        if npz_path.exists() and not args.overwrite:
            print(f"[{transect_cam}] skipping (npz exists): {npz_path}")
            continue

        if segmenter is None:
            segmenter = _make_segmenter(args)
        if da3 is None:
            from depth_anything_3.api import DepthAnything3

            da3 = DepthAnything3.from_pretrained(args.da3_model_id).to(device)

        instances, unmasked_rows = process_camera(
            transect_cam, cam_events,
            segmenter=segmenter, sam3_prompt=args.sam3_prompt,
            da3=da3, da3_batch_size=args.da3_batch_size, out_dir=args.out_dir,
        )

        npz_dict, rows, summary_row = build_camera_calibration(
            instances, transect_cam=transect_cam,
            min_depth=args.min_depth, max_depth=args.max_depth, da3_model_id=args.da3_model_id,
        )
        camera_rows = unmasked_rows + rows
        append_instance_rows(instances_csv, camera_rows)
        all_instance_rows.extend(camera_rows)

        if npz_dict is not None:
            pending_camera_npz[transect_cam] = (npz_dict, rows, summary_row)

    pooled = build_pooled(all_instance_rows, min_depth=args.min_depth, max_depth=args.max_depth)
    if pooled is not None:
        save_camera_npz(calib_dir / "_pooled.npz", pooled)

    summary_rows = []
    cameras_wanted = load_cameras_list(args.cameras) if args.cameras else []
    all_cam_names = sorted(set(cameras_wanted) | set(by_cam.keys()))

    for transect_cam in all_cam_names:
        if transect_cam in pending_camera_npz:
            npz_dict, rows, summary_row = pending_camera_npz[transect_cam]
            if npz_dict.get("knots_x") is None and pooled is not None:
                npz_dict["knots_x"] = pooled["knots_x"]
                npz_dict["knots_y"] = pooled["knots_y"]
            save_camera_npz(calib_dir / f"{transect_cam}.npz", npz_dict)
            plot_camera_curve(rows, npz_dict, args.out_dir / "calibration_plots" / f"{transect_cam}.png")
            summary_rows.append(summary_row)
        else:
            npz_path = calib_dir / f"{transect_cam}.npz"
            if npz_path.exists():
                summary_rows.append({
                    "transect_cam": transect_cam, "n_instances": None, "n_valid": None,
                    "distances": "", "loo_mae_m": None, "insample_mae_m": None,
                    "method": "existing", "anchor_distance_m": None, "notes": "skipped (npz exists)",
                })
            else:
                summary_rows.append({
                    "transect_cam": transect_cam, "n_instances": 0, "n_valid": 0,
                    "distances": "", "loo_mae_m": None, "insample_mae_m": None,
                    "method": "pooled", "anchor_distance_m": None, "notes": "no_instances",
                })

    write_summary_csv(args.out_dir / "calibration_summary.csv", summary_rows)
    print(f"wrote calibration for {len(summary_rows)} cameras to {args.out_dir}")


if __name__ == "__main__":
    main()
