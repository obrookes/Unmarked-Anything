#!/usr/bin/env python3
"""Fit per-camera linear calibration of DA3 mask-mean depth to true distance.

Depth Anything 3 mask-mean depth (metres) is nominally metric but biased short per camera.
This fits, per `transect_cam`, a line `distance = slope * depth + intercept` from reference
calibration points (a person holding a distance board at known distances), falling back to a
pooled (median-of-per-camera) line for cameras with too few/too uniform points, and to a single
global line if no camera has enough points at all. Only linear fits are used: per-video linear
LOOCV MAE (1.26 m) beats raw (2.7 m) in prior benchmarking, and with few points per camera,
higher-order/piecewise fits are unstable (see wcf-mde/scripts/benchmark_calibration.py).

Outputs (written to --out-dir):
  calibration.json          per-camera {slope, intercept, ...} + "_meta"; consumed by the
                             distance exporter via `load_calibration` / `apply_calibration`.
  calib_points_used.csv     points after filtering/overrides, with fitted + LOOCV predictions.
  calibration_summary.csv   one row per camera (same content as calibration.json, tabular).
  calibration_fits.png      scatter + fitted line per camera (paginated if many cameras).
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_POINTS_COLS = ["transect_cam", "video", "frame_idx", "depth_mask_mean", "board_distance_m"]
KEY_COLS = ["transect_cam", "video", "frame_idx"]


# --------------------------------------------------------------------------------------
# loading / filtering
# --------------------------------------------------------------------------------------

def load_points(points_csv, override_csv=None, min_vlm_conf: float = 0.0) -> pd.DataFrame:
    """Load calib_points.csv, apply an optional override CSV, then filter.

    Override rows are applied before the vlm-conf/NaN filtering, so an override can supply a
    previously-missing board_distance_m for an existing (transect_cam, video, frame_idx) row, or
    drop a row entirely (empty board_distance_m in the override).
    """
    df = pd.read_csv(points_csv)
    for col in REQUIRED_POINTS_COLS:
        if col not in df.columns:
            raise ValueError(f"points CSV missing required column: {col}")
    if "vlm_conf" not in df.columns:
        df["vlm_conf"] = np.nan

    if override_csv is not None:
        odf = pd.read_csv(override_csv)
        for col in KEY_COLS + ["board_distance_m"]:
            if col not in odf.columns:
                raise ValueError(f"override CSV missing required column: {col}")

        df = df.set_index(KEY_COLS)
        odf = odf.set_index(KEY_COLS)
        drop_keys = []
        for idx, row in odf.iterrows():
            val = row["board_distance_m"]
            if pd.isna(val) or val == "":
                drop_keys.append(idx)
                continue
            if idx in df.index:
                df.loc[idx, "board_distance_m"] = float(val)
            else:
                new_row = {c: np.nan for c in df.columns}
                new_row["board_distance_m"] = float(val)
                df.loc[idx] = new_row
        drop_keys = [k for k in drop_keys if k in df.index]
        if drop_keys:
            df = df.drop(index=drop_keys)
        df = df.reset_index()

    keep = df["vlm_conf"].isna() | (df["vlm_conf"] >= min_vlm_conf)
    df = df[keep]

    df = df.dropna(subset=["depth_mask_mean", "board_distance_m"])
    df["depth_mask_mean"] = df["depth_mask_mean"].astype(float)
    df["board_distance_m"] = df["board_distance_m"].astype(float)
    return df.reset_index(drop=True)


def load_cameras_list(cameras_csv) -> list:
    cdf = pd.read_csv(cameras_csv)
    if "transect_cam" not in cdf.columns:
        raise ValueError("cameras CSV missing required column: transect_cam")
    return cdf["transect_cam"].dropna().unique().tolist()


# --------------------------------------------------------------------------------------
# fitting
# --------------------------------------------------------------------------------------

def _linear_fit(x: np.ndarray, y: np.ndarray) -> tuple:
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def _loocv_mae(x: np.ndarray, y: np.ndarray) -> float:
    n = len(x)
    idx = np.arange(n)
    preds = np.empty(n, dtype=float)
    for i in idx:
        train = idx != i
        slope, intercept = _linear_fit(x[train], y[train])
        preds[i] = slope * x[i] + intercept
    return float(np.mean(np.abs(preds - y)))


def compute_calibration(df: pd.DataFrame, cameras=None, min_points: int = 4,
                         min_distinct: int = 3) -> tuple:
    """Fit per-camera linear calibrations with pooled fallback.

    Returns (result, meta) where `result` maps transect_cam -> the calibration.json entry dict
    (without "_meta"), and `meta` is the dict to store under "_meta".
    """
    per_cam_xy = {cam: (g["depth_mask_mean"].to_numpy(float), g["board_distance_m"].to_numpy(float))
                  for cam, g in df.groupby("transect_cam")}

    fitted = {}
    for cam, (x, y) in per_cam_xy.items():
        n_points = len(x)
        n_distinct = int(np.unique(y).size)
        eligible = n_points >= min_points and n_distinct >= min_distinct
        if eligible:
            slope, intercept = _linear_fit(x, y)
            eligible = slope > 0
        if eligible:
            fitted[cam] = {"slope": slope, "intercept": intercept,
                            "n_points": n_points, "n_distinct_dist": n_distinct}

    if fitted:
        pooled_slope = float(np.median([v["slope"] for v in fitted.values()]))
        pooled_intercept = float(np.median([v["intercept"] for v in fitted.values()]))
        pooled_method = "pooled"
    else:
        all_x = df["depth_mask_mean"].to_numpy(float)
        all_y = df["board_distance_m"].to_numpy(float)
        if len(all_x) == 0:
            raise ValueError("no calibration points available to fit even a pooled_global line")
        pooled_slope, pooled_intercept = _linear_fit(all_x, all_y)
        pooled_method = "pooled_global"

    all_cams = set(per_cam_xy.keys())
    if cameras is not None:
        all_cams |= set(cameras)

    result = {}
    for cam in sorted(all_cams, key=str):
        if cam in fitted:
            x, y = per_cam_xy[cam]
            f = fitted[cam]
            pred = f["slope"] * x + f["intercept"]
            insample_mae = float(np.mean(np.abs(pred - y)))
            resid_sd = float(np.std(pred - y, ddof=1)) if len(x) > 1 else None
            loocv_mae = _loocv_mae(x, y)
            result[cam] = {
                "slope": f["slope"], "intercept": f["intercept"],
                "n_points": f["n_points"], "n_distinct_dist": f["n_distinct_dist"],
                "loocv_mae": loocv_mae, "insample_mae": insample_mae, "resid_sd": resid_sd,
                "method": "per_camera",
            }
        else:
            if cam in per_cam_xy:
                x, y = per_cam_xy[cam]
                n_points = len(x)
                n_distinct = int(np.unique(y).size)
                pred = pooled_slope * x + pooled_intercept
                insample_mae = float(np.mean(np.abs(pred - y)))
                resid_sd = float(np.std(pred - y, ddof=1)) if n_points > 1 else None
            else:
                n_points, n_distinct, insample_mae, resid_sd = 0, 0, None, None
            result[cam] = {
                "slope": pooled_slope, "intercept": pooled_intercept,
                "n_points": n_points, "n_distinct_dist": n_distinct,
                "loocv_mae": None, "insample_mae": insample_mae, "resid_sd": resid_sd,
                "method": pooled_method,
            }

    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "min_points": min_points,
        "min_distinct": min_distinct,
        "pooled_slope": pooled_slope,
        "pooled_intercept": pooled_intercept,
        "n_cameras_per_camera": len(fitted),
        "n_cameras_pooled": len(result) - len(fitted),
    }
    return result, meta


# --------------------------------------------------------------------------------------
# apply / (de)serialize
# --------------------------------------------------------------------------------------

def apply_calibration(depth: float, transect_cam, calib: dict) -> tuple:
    """Map a raw DA3 mask-mean depth to a calibrated distance for `transect_cam`.

    Falls back to the pooled line (stored under "_meta") for cameras absent from `calib`.
    Returns (distance, method).
    """
    entry = calib.get(transect_cam)
    if entry is None:
        meta = calib.get("_meta", {})
        slope = meta["pooled_slope"]
        intercept = meta["pooled_intercept"]
        method = "pooled"
    else:
        slope = entry["slope"]
        intercept = entry["intercept"]
        method = entry["method"]
    return slope * depth + intercept, method


def load_calibration(path) -> dict:
    with open(path) as f:
        return json.load(f)


def write_calibration_json(result: dict, meta: dict, out_path) -> None:
    payload = dict(result)
    payload["_meta"] = meta
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)


# --------------------------------------------------------------------------------------
# CLI outputs: points-used CSV, summary CSV, plot
# --------------------------------------------------------------------------------------

def build_points_used(df: pd.DataFrame, result: dict) -> pd.DataFrame:
    df = df.copy()
    fitted_pred = []
    loocv_pred = []
    for _, row in df.iterrows():
        entry = result.get(row["transect_cam"])
        depth = row["depth_mask_mean"]
        if entry is None:
            fitted_pred.append(np.nan)
            loocv_pred.append(np.nan)
            continue
        fitted_pred.append(entry["slope"] * depth + entry["intercept"])
        if entry["method"] == "per_camera":
            cam_df = df[df["transect_cam"] == row["transect_cam"]]
            x = cam_df["depth_mask_mean"].to_numpy(float)
            y = cam_df["board_distance_m"].to_numpy(float)
            pos = cam_df.index.get_loc(row.name)
            mask = np.ones(len(x), dtype=bool)
            mask[pos] = False
            slope, intercept = _linear_fit(x[mask], y[mask])
            loocv_pred.append(slope * depth + intercept)
        else:
            loocv_pred.append(np.nan)
    df["fitted_pred_m"] = fitted_pred
    df["loocv_pred_m"] = loocv_pred
    return df


def build_summary(result: dict) -> pd.DataFrame:
    rows = []
    for cam, entry in sorted(result.items(), key=lambda kv: str(kv[0])):
        rows.append({"transect_cam": cam, **entry})
    return pd.DataFrame(rows)


def plot_calibration_fits(df: pd.DataFrame, result: dict, out_path, panels_per_page: int = 60) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cams = sorted(result.keys(), key=str)
    if not cams:
        return
    pages = [cams[i:i + panels_per_page] for i in range(0, len(cams), panels_per_page)]
    out_path = Path(out_path)

    for page_idx, page_cams in enumerate(pages):
        ncols = min(6, len(page_cams))
        nrows = int(np.ceil(len(page_cams) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.8 * nrows), squeeze=False)
        for ax, cam in zip(axes.flat, page_cams):
            g = df[df["transect_cam"] == cam]
            entry = result[cam]
            if len(g):
                x = g["depth_mask_mean"].to_numpy(float)
                y = g["board_distance_m"].to_numpy(float)
                ax.scatter(x, y, s=12, color="black")
                xs = np.linspace(x.min(), x.max(), 20) if len(x) > 1 else x
            else:
                xs = np.array([0.0, 1.0])
            ax.plot(xs, entry["slope"] * xs + entry["intercept"], color="C0", lw=1)
            ax.set_title(f"{cam}\n{entry['method']} n={entry['n_points']}", fontsize=7)
            ax.tick_params(labelsize=6)
        for ax in axes.flat[len(page_cams):]:
            ax.axis("off")
        fig.tight_layout()
        page_path = out_path if len(pages) == 1 else out_path.with_stem(f"{out_path.stem}_p{page_idx + 1}")
        fig.savefig(page_path, dpi=120)
        plt.close(fig)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--points", required=True, type=Path, help="calib_points.csv")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--cameras", type=Path, default=None,
                   help="CSV with a transect_cam column listing all cameras needing calibration")
    p.add_argument("--override-csv", type=Path, default=None)
    p.add_argument("--min-points", type=int, default=4)
    p.add_argument("--min-distinct", type=int, default=3)
    p.add_argument("--min-vlm-conf", type=float, default=0.0)
    p.add_argument("--da3-model-id", type=str, default=None)
    p.add_argument("--da3-mode", type=str, default=None)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_points(args.points, override_csv=args.override_csv, min_vlm_conf=args.min_vlm_conf)
    cameras = load_cameras_list(args.cameras) if args.cameras is not None else None

    result, meta = compute_calibration(df, cameras=cameras, min_points=args.min_points,
                                        min_distinct=args.min_distinct)
    meta["source_points_csv"] = str(args.points)
    for cam, entry in result.items():
        entry["da3_model_id"] = args.da3_model_id
        entry["da3_mode"] = args.da3_mode

    write_calibration_json(result, meta, args.out_dir / "calibration.json")

    points_used = build_points_used(df, result)
    points_used.to_csv(args.out_dir / "calib_points_used.csv", index=False)

    summary = build_summary(result)
    summary.to_csv(args.out_dir / "calibration_summary.csv", index=False)

    plot_calibration_fits(df, result, args.out_dir / "calibration_fits.png")

    print(f"wrote calibration for {len(result)} cameras "
          f"({meta['n_cameras_per_camera']} per-camera, {meta['n_cameras_pooled']} pooled) "
          f"to {args.out_dir}")


if __name__ == "__main__":
    main()
