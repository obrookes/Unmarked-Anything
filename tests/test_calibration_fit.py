from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from apps.camera_trap.calibration.fit import (
    apply_calibration,
    compute_calibration,
    load_calibration,
    load_points,
    write_calibration_json,
)


def _rows(cam, video, depths, distances, vlm_conf=None):
    n = len(depths)
    d = {
        "transect_cam": [cam] * n,
        "video": [video] * n,
        "frame_idx": list(range(n)),
        "depth_mask_mean": depths,
        "board_distance_m": distances,
    }
    if vlm_conf is not None:
        d["vlm_conf"] = vlm_conf
    return pd.DataFrame(d)


def _points_csv(tmp_path, df, name="calib_points.csv"):
    path = tmp_path / name
    df.to_csv(path, index=False)
    return path


# --------------------------------------------------------------------------------------
# synthetic recovery of known slope/intercept
# --------------------------------------------------------------------------------------

def test_recovers_known_slope_intercept_with_noise():
    rng = np.random.default_rng(0)
    true_slope, true_intercept = 1.4, 0.8
    distances = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0])
    depths = (distances - true_intercept) / true_slope
    depths_noisy = depths + rng.normal(0, 0.02, size=depths.shape)

    df = _rows("camA", "vidA", depths_noisy.tolist(), distances.tolist())
    result, meta = compute_calibration(df, min_points=4, min_distinct=3)

    entry = result["camA"]
    assert entry["method"] == "per_camera"
    assert entry["slope"] == pytest.approx(true_slope, abs=0.1)
    assert entry["intercept"] == pytest.approx(true_intercept, abs=0.3)
    assert entry["n_points"] == 6
    assert entry["n_distinct_dist"] == 6
    assert entry["loocv_mae"] >= 0
    assert entry["insample_mae"] >= 0
    assert entry["resid_sd"] >= 0


# --------------------------------------------------------------------------------------
# per-camera vs pooled assignment
# --------------------------------------------------------------------------------------

def test_too_few_points_falls_back_to_pooled():
    good = _rows("good_cam", "v", [1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0])
    few = _rows("few_cam", "v", [1.0, 2.0], [2.0, 4.0])
    df = pd.concat([good, few], ignore_index=True)

    result, meta = compute_calibration(df, min_points=4, min_distinct=3)
    assert result["good_cam"]["method"] == "per_camera"
    assert result["few_cam"]["method"] == "pooled"
    assert result["few_cam"]["n_points"] == 2
    assert result["few_cam"]["slope"] == result["good_cam"]["slope"]
    assert result["few_cam"]["loocv_mae"] is None


def test_too_few_distinct_distances_falls_back_to_pooled():
    good = _rows("good_cam", "v", [1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0])
    # 4 points but only 2 distinct true distances
    dup = _rows("dup_cam", "v", [1.0, 1.1, 2.0, 2.1], [2.0, 2.0, 4.0, 4.0])
    df = pd.concat([good, dup], ignore_index=True)

    result, meta = compute_calibration(df, min_points=4, min_distinct=3)
    assert result["dup_cam"]["method"] == "pooled"
    assert result["dup_cam"]["n_distinct_dist"] == 2


def test_negative_slope_falls_back_to_pooled():
    good = _rows("good_cam", "v", [1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0])
    # depth decreases as true distance increases -> negative slope
    bad = _rows("bad_cam", "v", [4.0, 3.0, 2.0, 1.0], [2.0, 4.0, 6.0, 8.0])
    df = pd.concat([good, bad], ignore_index=True)

    result, meta = compute_calibration(df, min_points=4, min_distinct=3)
    assert result["bad_cam"]["method"] == "pooled"


def test_camera_listed_without_points_gets_pooled_fallback():
    good = _rows("good_cam", "v", [1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0])
    result, meta = compute_calibration(good, cameras=["good_cam", "no_data_cam"],
                                        min_points=4, min_distinct=3)
    assert "no_data_cam" in result
    entry = result["no_data_cam"]
    assert entry["method"] == "pooled"
    assert entry["n_points"] == 0
    assert entry["n_distinct_dist"] == 0
    assert entry["loocv_mae"] is None
    assert entry["insample_mae"] is None
    assert entry["resid_sd"] is None


# --------------------------------------------------------------------------------------
# pooled_global path
# --------------------------------------------------------------------------------------

def test_pooled_global_when_no_camera_qualifies():
    a = _rows("camA", "v", [1.0, 2.0], [2.0, 4.0])
    b = _rows("camB", "v", [1.0, 2.0], [2.5, 4.5])
    df = pd.concat([a, b], ignore_index=True)

    result, meta = compute_calibration(df, min_points=4, min_distinct=3)
    assert meta["n_cameras_per_camera"] == 0
    for cam in ("camA", "camB"):
        assert result[cam]["method"] == "pooled_global"


def test_no_points_at_all_raises():
    empty = pd.DataFrame(columns=["transect_cam", "video", "frame_idx",
                                   "depth_mask_mean", "board_distance_m"])
    with pytest.raises(ValueError):
        compute_calibration(empty)


# --------------------------------------------------------------------------------------
# override CSV
# --------------------------------------------------------------------------------------

def test_override_replaces_and_drops(tmp_path):
    df = _rows("camA", "vidA", [1.0, 2.0, 3.0], [np.nan, 4.0, 6.0])
    points_csv = _points_csv(tmp_path, df)

    override = pd.DataFrame({
        "transect_cam": ["camA", "camA"],
        "video": ["vidA", "vidA"],
        "frame_idx": [0, 1],
        "board_distance_m": [2.0, np.nan],  # fill row 0, drop row 1
    })
    override_csv = tmp_path / "override.csv"
    override.to_csv(override_csv, index=False)

    loaded = load_points(points_csv, override_csv=override_csv)
    # row0 filled in (2.0), row1 dropped (was 4.0), row2 kept (6.0)
    assert sorted(loaded["board_distance_m"].tolist()) == [2.0, 6.0]
    assert sorted(loaded["frame_idx"].tolist()) == [0, 2]


def test_min_vlm_conf_filter(tmp_path):
    df = _rows("camA", "vidA", [1.0, 2.0, 3.0], [2.0, 4.0, 6.0], vlm_conf=[0.9, 0.2, 0.5])
    points_csv = _points_csv(tmp_path, df)
    loaded = load_points(points_csv, min_vlm_conf=0.4)
    assert sorted(loaded["frame_idx"].tolist()) == [0, 2]


# --------------------------------------------------------------------------------------
# apply_calibration
# --------------------------------------------------------------------------------------

def test_apply_calibration_known_and_unknown_camera():
    calib = {
        "camA": {"slope": 2.0, "intercept": 1.0, "method": "per_camera"},
        "_meta": {"pooled_slope": 1.5, "pooled_intercept": 0.5},
    }
    dist, method = apply_calibration(3.0, "camA", calib)
    assert dist == pytest.approx(7.0)
    assert method == "per_camera"

    dist, method = apply_calibration(3.0, "unknown_cam", calib)
    assert dist == pytest.approx(5.0)
    assert method == "pooled"


# --------------------------------------------------------------------------------------
# LOOCV correctness on a tiny exact example
# --------------------------------------------------------------------------------------

def test_loocv_exact_line_zero_mae():
    # points lie exactly on distance = 2*depth + 1 -> any n-1 subset refits the same line
    depths = [1.0, 2.0, 3.0, 4.0]
    distances = [3.0, 5.0, 7.0, 9.0]
    df = _rows("camA", "v", depths, distances)
    result, meta = compute_calibration(df, min_points=4, min_distinct=3)
    entry = result["camA"]
    assert entry["slope"] == pytest.approx(2.0)
    assert entry["intercept"] == pytest.approx(1.0)
    assert entry["loocv_mae"] == pytest.approx(0.0, abs=1e-9)
    assert entry["insample_mae"] == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------------------
# json schema
# --------------------------------------------------------------------------------------

def test_calibration_json_schema(tmp_path):
    df = _rows("camA", "v", [1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0])
    result, meta = compute_calibration(df, min_points=4, min_distinct=3)
    meta["source_points_csv"] = "dummy.csv"
    out_path = tmp_path / "calibration.json"
    write_calibration_json(result, meta, out_path)

    loaded = load_calibration(out_path)
    assert "_meta" in loaded
    meta_keys = {"created_utc", "min_points", "min_distinct", "pooled_slope", "pooled_intercept",
                 "n_cameras_per_camera", "n_cameras_pooled", "source_points_csv"}
    assert meta_keys <= loaded["_meta"].keys()

    entry_keys = {"slope", "intercept", "n_points", "n_distinct_dist", "loocv_mae",
                  "insample_mae", "resid_sd", "method"}
    assert entry_keys <= loaded["camA"].keys()
    assert loaded["camA"]["method"] == "per_camera"


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def test_cli_help_and_end_to_end(tmp_path):
    fit_py = Path(__file__).resolve().parents[1] / "apps" / "camera_trap" / "calibration" / "fit.py"

    help_proc = subprocess.run([sys.executable, str(fit_py), "--help"],
                                capture_output=True, text=True)
    assert help_proc.returncode == 0
    assert "--points" in help_proc.stdout

    good = _rows("camA", "v", [1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0])
    points_csv = _points_csv(tmp_path, good)
    out_dir = tmp_path / "out"

    proc = subprocess.run(
        [sys.executable, str(fit_py), "--points", str(points_csv), "--out-dir", str(out_dir),
         "--da3-model-id", "depth-anything/DA3NESTED-GIANT-LARGE-1.1", "--da3-mode", "unconditioned"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr

    assert (out_dir / "calibration.json").exists()
    assert (out_dir / "calib_points_used.csv").exists()
    assert (out_dir / "calibration_summary.csv").exists()
    assert (out_dir / "calibration_fits.png").exists()

    calib = json.loads((out_dir / "calibration.json").read_text())
    assert calib["camA"]["da3_model_id"] == "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
    assert calib["camA"]["da3_mode"] == "unconditioned"
