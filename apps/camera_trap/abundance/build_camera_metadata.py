#!/usr/bin/env python3
"""Build the per-camera metadata table used by the CTDS abundance pipeline.

Reads the PSS P3 distances workbook (one row per detection/video, columns
including Transect_ID, Transect_cam, CT_modele, Ctdays) and collapses it to
one row per camera with columns:

    transect_cam, transect_id, ct_model, fov_deg, ct_days

`fov_deg` is looked up from the `fov_by_model` map in the CTDS config
(configs/pss_p3/ctds_config.yaml). Cameras whose rows disagree on
transect_id / ct_model / ct_days are reported and excluded (their metadata
is ambiguous).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distances-xlsx", type=Path, required=True)
    parser.add_argument("--sheet-name", default="Sheet 1")
    parser.add_argument("--config", type=Path, default=Path("configs/pss_p3/ctds_config.yaml"))
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def build_camera_metadata(df: pd.DataFrame, fov_by_model: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (metadata, conflicts) dataframes."""
    required = ["Transect_cam", "Transect_ID", "CT_modele", "Ctdays"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input workbook missing required columns: {missing}")

    sub = df[required].copy()
    sub["Transect_cam"] = sub["Transect_cam"].str.lower()

    grouped = sub.groupby("Transect_cam")
    nunique = grouped.nunique()
    conflict_mask = (nunique["Transect_ID"] > 1) | (nunique["CT_modele"] > 1) | (nunique["Ctdays"] > 1)
    conflicting_cams = nunique.index[conflict_mask].tolist()

    conflicts = sub[sub["Transect_cam"].isin(conflicting_cams)].drop_duplicates()

    clean = sub[~sub["Transect_cam"].isin(conflicting_cams)].drop_duplicates(subset=["Transect_cam"])

    unknown_models = sorted(set(clean["CT_modele"]) - set(fov_by_model))
    if unknown_models:
        raise ValueError(f"No fov_deg configured for CT_modele values: {unknown_models}")

    metadata = pd.DataFrame(
        {
            "transect_cam": clean["Transect_cam"].values,
            "transect_id": clean["Transect_ID"].astype(int).values,
            "ct_model": clean["CT_modele"].values,
            "fov_deg": [fov_by_model[m] for m in clean["CT_modele"]],
            "ct_days": clean["Ctdays"].astype(int).values,
        }
    ).sort_values("transect_cam").reset_index(drop=True)

    return metadata, conflicts


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text())
    fov_by_model = config["fov_by_model"]

    df = pd.read_excel(args.distances_xlsx, sheet_name=args.sheet_name)
    metadata, conflicts = build_camera_metadata(df, fov_by_model)

    if not conflicts.empty:
        conflicting_cams = sorted(conflicts["Transect_cam"].unique())
        print(
            f"WARNING: {len(conflicting_cams)} camera(s) had conflicting metadata "
            f"and were excluded: {conflicting_cams}",
            file=sys.stderr,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(args.out, index=False)
    print(f"Wrote {len(metadata)} cameras to {args.out}")


if __name__ == "__main__":
    main()
