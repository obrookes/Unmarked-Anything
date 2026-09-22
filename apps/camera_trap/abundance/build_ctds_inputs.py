#!/usr/bin/env python3
"""Build CTDS (camera-trap distance sampling) inputs for the `Distance` R
package from an exported per-detection distances CSV and a per-camera
metadata CSV.

Inputs
------
--distances : exporter output CSV with (at least) columns
    transect_cam, distance, detection_datetime
  (also video_name, starting_date, year, month, day, hour, minute, second,
  ind_no, confidence, raw_distance, calib_method -- these are passed through
  or ignored as noted below).
--metadata  : configs/pss_p3/camera_metadata.csv
    (transect_cam, transect_id, ct_model, fov_deg, ct_days)
--config    : configs/pss_p3/ctds_config.yaml

Outputs (written to --out-dir)
-------------------------------
ctds_flatfile.csv   : Distance-package flatfile, one row per detection plus
                       one effort-only row (distance/object = NA) for every
                       metadata camera with zero detections. Columns:
                       Sample.Label, distance, object, Area, Region.Label,
                       Effort, transect_id, ct_model, fov_deg, ct_days,
                       calib_method.
activity_times.csv  : independent detection events per transect, used by
                       ctds_abundance.R to fit the activity model.
inputs_summary.json : counts / diagnostics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distances", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def build_flatfile(
    distances: pd.DataFrame,
    metadata: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, dict]:
    exclude_transects = set(config["exclude_transects"])
    area = config["area_km2"]
    region_label = config["region_label"]
    snapshot_interval_s = config["snapshot_interval_s"]

    meta = metadata.copy()
    meta["transect_cam"] = meta["transect_cam"].astype(str).str.lower()
    meta = meta[~meta["transect_id"].isin(exclude_transects)].reset_index(drop=True)

    meta["Effort"] = meta["ct_days"] * 86400 / snapshot_interval_s * meta["fov_deg"] / 360

    known_cams = set(meta["transect_cam"])

    det = distances.copy()
    det["transect_cam"] = det["transect_cam"].astype(str).str.lower()

    n_before = len(det)
    unknown_mask = ~det["transect_cam"].isin(known_cams)
    n_unknown = int(unknown_mask.sum())
    det = det[~unknown_mask].reset_index(drop=True)

    # detections in excluded transects: their transect_cam won't be in
    # known_cams (since excluded transects were dropped from meta), so
    # they're already captured by n_unknown / dropped here. Report separately
    # for clarity.
    n_dropped = n_unknown

    meta_by_cam = meta.set_index("transect_cam")

    det = det.merge(
        meta_by_cam[["transect_id", "ct_model", "fov_deg", "ct_days", "Effort"]],
        left_on="transect_cam",
        right_index=True,
        how="left",
    )
    det = det.rename(columns={"transect_cam": "Sample.Label"})
    det["object"] = range(1, len(det) + 1)
    det["Area"] = area
    det["Region.Label"] = region_label
    if "calib_method" not in det.columns:
        det["calib_method"] = ""

    flat_cols = [
        "Sample.Label",
        "distance",
        "object",
        "Area",
        "Region.Label",
        "Effort",
        "transect_id",
        "ct_model",
        "fov_deg",
        "ct_days",
        "calib_method",
    ]
    det_rows = det[flat_cols]

    cams_with_detections = set(det["Sample.Label"]) if len(det) else set()
    zero_det_cams = meta[~meta["transect_cam"].isin(cams_with_detections)]

    effort_rows = pd.DataFrame(
        {
            "Sample.Label": zero_det_cams["transect_cam"].values,
            "distance": pd.NA,
            "object": pd.NA,
            "Area": area,
            "Region.Label": region_label,
            "Effort": zero_det_cams["Effort"].values,
            "transect_id": zero_det_cams["transect_id"].values,
            "ct_model": zero_det_cams["ct_model"].values,
            "fov_deg": zero_det_cams["fov_deg"].values,
            "ct_days": zero_det_cams["ct_days"].values,
            "calib_method": "",
        }
    )

    flatfile = pd.concat([det_rows, effort_rows], ignore_index=True)

    summary = {
        "n_detections_in": n_before,
        "n_detections_dropped_unknown_camera": n_dropped,
        "n_detections_kept": len(det_rows),
        "n_cameras_metadata": len(meta),
        "n_cameras_with_detections": len(cams_with_detections),
        "n_cameras_effort_only": len(zero_det_cams),
        "calib_method_counts": det_rows["calib_method"].value_counts(dropna=False).to_dict()
        if len(det_rows)
        else {},
    }
    return flatfile, summary


def build_activity_times(distances: pd.DataFrame, metadata: pd.DataFrame, config: dict) -> pd.DataFrame:
    exclude_transects = set(config["exclude_transects"])
    independence_min = config["independence_min"]

    meta = metadata.copy()
    meta["transect_cam"] = meta["transect_cam"].astype(str).str.lower()
    meta = meta[~meta["transect_id"].isin(exclude_transects)]
    cam_to_transect = meta.set_index("transect_cam")["transect_id"].to_dict()

    det = distances.copy()
    det["transect_cam"] = det["transect_cam"].astype(str).str.lower()
    det = det[det["transect_cam"].isin(cam_to_transect)].copy()

    if "detection_datetime" not in det.columns or det["detection_datetime"].astype(str).str.strip().eq("").all() or det["detection_datetime"].isna().all():
        return pd.DataFrame(columns=["transect_id", "transect_cam", "detection_datetime", "time_hm", "new_event"])

    det["detection_datetime"] = pd.to_datetime(det["detection_datetime"], errors="coerce")
    det = det.dropna(subset=["detection_datetime"])
    det["transect_id"] = det["transect_cam"].map(cam_to_transect)

    det = det.sort_values(["transect_id", "detection_datetime"]).reset_index(drop=True)

    rows = []
    limit = pd.Timedelta(minutes=independence_min)
    for transect_id, grp in det.groupby("transect_id", sort=False):
        grp = grp.sort_values("detection_datetime")
        prev_time = None
        for _, r in grp.iterrows():
            dt = r["detection_datetime"]
            new_event = 1 if (prev_time is None or (dt - prev_time) > limit) else 0
            rows.append(
                {
                    "transect_id": transect_id,
                    "transect_cam": r["transect_cam"],
                    "detection_datetime": dt.isoformat(),
                    "time_hm": dt.strftime("%H:%M"),
                    "new_event": new_event,
                }
            )
            prev_time = dt

    return pd.DataFrame(rows, columns=["transect_id", "transect_cam", "detection_datetime", "time_hm", "new_event"])


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    distances = pd.read_csv(args.distances, dtype={"transect_cam": str})
    metadata = pd.read_csv(args.metadata, dtype={"transect_cam": str})

    args.out_dir.mkdir(parents=True, exist_ok=True)

    flatfile, summary = build_flatfile(distances, metadata, config)
    flatfile.to_csv(args.out_dir / "ctds_flatfile.csv", index=False)

    activity = build_activity_times(distances, metadata, config)
    activity.to_csv(args.out_dir / "activity_times.csv", index=False)
    if activity.empty:
        print(
            "WARNING: no usable detection_datetime values found; wrote an empty "
            "activity_times.csv. ctds_abundance.R will require "
            "--activity-rate/--activity-se."
        )

    if summary["n_detections_dropped_unknown_camera"]:
        print(
            f"WARNING: dropped {summary['n_detections_dropped_unknown_camera']} "
            "detection row(s) with transect_cam not present in metadata "
            "(or belonging to an excluded transect)."
        )

    (args.out_dir / "inputs_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"Wrote flatfile ({len(flatfile)} rows), activity_times ({len(activity)} rows), inputs_summary.json to {args.out_dir}")


if __name__ == "__main__":
    main()
