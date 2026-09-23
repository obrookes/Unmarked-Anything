#!/usr/bin/env python3
"""Build a video_name,start_datetime CSV (for export_job_distances_csv.py --video-times) from the
PSS P3 human-annotation spreadsheet.

Re-encoded videos lose their ffprobe creation_time, but the annotation sheet records each
video's start time. Video names follow the flat chimp-video layout, <transect_cam>_<video>.

    python apps/camera_trap/scripts/build_video_times.py \
        --annotations PSS_P3_distances_all.xlsx --output video_times.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

TIME_COLUMNS = ["Year", "Month", "Day", "Hour", "Min"]


def build_video_times(annotations: pd.DataFrame) -> pd.DataFrame:
    df = annotations.dropna(subset=["Transect_cam", "Video_name", *TIME_COLUMNS]).copy()
    parts = {c.lower(): pd.to_numeric(df[c], errors="coerce") for c in TIME_COLUMNS}
    parts["second"] = pd.to_numeric(df["Sec"], errors="coerce").fillna(0) if "Sec" in df else 0
    df["start_datetime"] = pd.to_datetime(
        pd.DataFrame({"year": parts["year"], "month": parts["month"], "day": parts["day"],
                      "hour": parts["hour"], "minute": parts["min"], "second": parts["second"]}),
        errors="coerce",
    )
    df = df.dropna(subset=["start_datetime"])
    video = df["Video_name"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(8)
    df["video_name"] = df["Transect_cam"].astype(str).str.strip() + "_" + video
    # One annotation row per individual/event; a video's start time is its earliest row.
    out = df.groupby("video_name", as_index=False)["start_datetime"].min()
    out["start_datetime"] = out["start_datetime"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    return out.sort_values("video_name")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", type=Path, required=True, help="PSS P3 annotation .xlsx or .csv.")
    parser.add_argument("--output", type=Path, required=True, help="Destination video_name,start_datetime CSV.")
    args = parser.parse_args()
    if args.annotations.suffix.lower() in (".xlsx", ".xls"):
        annotations = pd.read_excel(args.annotations, dtype=str)
    else:
        annotations = pd.read_csv(args.annotations, dtype=str)
    out = build_video_times(annotations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"wrote {len(out)} video start times to {args.output}")


if __name__ == "__main__":
    main()
