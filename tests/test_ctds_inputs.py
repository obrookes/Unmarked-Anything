from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apps.camera_trap.abundance import build_ctds_inputs as bci


def _config():
    return {
        "area_km2": 3156.06,
        "region_label": "PSS_new_limits",
        "snapshot_interval_s": 2,
        "exclude_transects": [13, 29],
        "independence_min": 15,
    }


def _metadata():
    return pd.DataFrame(
        {
            "transect_cam": ["1_cam001", "2_cam002", "13_cam099"],
            "transect_id": [1, 2, 13],
            "ct_model": ["DS-30MP", "DS-32MP", "DS-30MP"],
            "fov_deg": [38, 50, 38],
            "ct_days": [10, 20, 5],
        }
    )


def test_effort_maths():
    metadata = _metadata()
    config = _config()
    distances = pd.DataFrame(
        {
            "transect_cam": ["1_cam001"],
            "distance": [5.0],
            "detection_datetime": [""],
            "calib_method": ["stereo"],
        }
    )
    flat, _ = bci.build_flatfile(distances, metadata, config)
    row = flat[flat["Sample.Label"] == "1_cam001"].iloc[0]
    expected_effort = 10 * 86400 / 2 * 38 / 360
    assert row["Effort"] == expected_effort


def test_zero_detection_cameras_present_as_effort_only_rows():
    metadata = _metadata()
    config = _config()
    distances = pd.DataFrame(
        {
            "transect_cam": ["1_cam001"],
            "distance": [5.0],
            "detection_datetime": [""],
            "calib_method": ["stereo"],
        }
    )
    flat, _ = bci.build_flatfile(distances, metadata, config)
    # transect 13 is excluded, so only camera 2_cam002 should appear as
    # effort-only (zero detections).
    cam2_rows = flat[flat["Sample.Label"] == "2_cam002"]
    assert len(cam2_rows) == 1
    assert pd.isna(cam2_rows.iloc[0]["distance"])
    assert pd.isna(cam2_rows.iloc[0]["object"])


def test_exclude_transects_applied():
    metadata = _metadata()
    config = _config()
    distances = pd.DataFrame(
        {
            "transect_cam": ["13_cam099"],
            "distance": [5.0],
            "detection_datetime": [""],
            "calib_method": ["stereo"],
        }
    )
    flat, summary = bci.build_flatfile(distances, metadata, config)
    assert "13_cam099" not in set(flat["Sample.Label"])
    # detection on an excluded transect's camera is dropped as "unknown"
    # since that camera is removed from metadata before matching.
    assert summary["n_detections_dropped_unknown_camera"] == 1


def test_unknown_cameras_dropped():
    metadata = _metadata()
    config = _config()
    distances = pd.DataFrame(
        {
            "transect_cam": ["1_cam001", "99_cam999"],
            "distance": [5.0, 3.0],
            "detection_datetime": ["", ""],
            "calib_method": ["stereo", "stereo"],
        }
    )
    flat, summary = bci.build_flatfile(distances, metadata, config)
    assert "99_cam999" not in set(flat["Sample.Label"])
    assert summary["n_detections_dropped_unknown_camera"] == 1
    assert summary["n_detections_kept"] == 1


def test_object_numbering():
    metadata = _metadata()
    config = _config()
    distances = pd.DataFrame(
        {
            "transect_cam": ["1_cam001", "1_cam001", "2_cam002"],
            "distance": [5.0, 6.0, 7.0],
            "detection_datetime": ["", "", ""],
            "calib_method": ["stereo", "stereo", "stereo"],
        }
    )
    flat, _ = bci.build_flatfile(distances, metadata, config)
    detection_rows = flat[flat["distance"].notna()]
    assert list(detection_rows["object"]) == [1, 2, 3]


def test_activity_independence_rule():
    metadata = _metadata()
    config = _config()
    distances = pd.DataFrame(
        {
            "transect_cam": ["1_cam001", "1_cam001", "1_cam001"],
            "distance": [5.0, 6.0, 7.0],
            "detection_datetime": [
                "2023-01-01T06:00:00",
                "2023-01-01T06:10:00",  # 10 min gap -> same event
                "2023-01-01T07:00:00",  # 50 min gap -> new event
            ],
        }
    )
    activity = bci.build_activity_times(distances, metadata, config)
    activity = activity.sort_values("detection_datetime").reset_index(drop=True)
    assert list(activity["new_event"]) == [1, 0, 1]


def test_activity_missing_datetime_writes_empty():
    metadata = _metadata()
    config = _config()
    distances = pd.DataFrame(
        {
            "transect_cam": ["1_cam001"],
            "distance": [5.0],
            "detection_datetime": [""],
        }
    )
    activity = bci.build_activity_times(distances, metadata, config)
    assert activity.empty
