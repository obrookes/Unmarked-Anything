from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from apps.camera_trap.scripts import export_job_distances_csv as export_mod


def _entry(depth: float, confidence: float = 0.9, center=(50.0, 50.0), bbox=(10, 10, 30, 30)):
    return {
        "depth_mask_mean": depth,
        "confidence": confidence,
        "center_xy": list(center),
        "bbox_xyxy": list(bbox),
        "track_id": 1,
    }


# --- (a) fps unit bug: native frame_index with video_fps != target_fps ---------------


def test_fps_uses_video_fps_not_target_fps():
    # Native video at 30 fps, inference subsampled to target_fps=6 (every 5th frame),
    # but frame_index stored in the JSON is the *native* frame index.
    frames = []
    for native_idx in range(0, 61, 5):  # 0, 5, 10, ..., 60  (2 seconds of native video)
        frames.append(
            {
                "frame_index": native_idx,
                "objects": [_entry(depth=2.0)],
            }
        )
    video_json = {
        "video_name": "clip",
        "video_fps": 30.0,
        "target_fps": 6.0,
        "frames": frames,
    }

    rows = export_mod.build_rows_for_video(
        video_json=video_json,
        json_path=Path("clip.json"),
        interval_seconds=2.0,
        window_seconds=1.0,
        creation_dt=None,
        apply_filters=False,
    )

    # Grid points aligned on the *native* fps: step_frames = 2s * 30fps = 60 frames.
    # first_idx=0 -> sampled_idx=0, next would be 60 (present, last_idx=60).
    seconds = sorted(row["second"] for row in rows)
    assert seconds == [0.0, 2.0]


def test_fps_identical_behaviour_when_fps_equal():
    frames = [{"frame_index": 0, "objects": [_entry(depth=3.0)]}]
    video_json = {"video_name": "clip", "video_fps": 1.0, "target_fps": 1.0, "frames": frames}
    rows = export_mod.build_rows_for_video(
        video_json=video_json,
        json_path=Path("clip.json"),
        interval_seconds=1.0,
        window_seconds=1.0,
        creation_dt=None,
        apply_filters=False,
    )
    assert len(rows) == 1
    assert rows[0]["second"] == 0.0
    assert rows[0]["distance"] == pytest.approx(3.0)


# --- (e) no calibration -> distance == raw_distance -----------------------------------


def test_no_calibration_distance_equals_raw_distance():
    video_json = {
        "video_name": "clip",
        "video_fps": 1.0,
        "frames": [{"frame_index": 0, "objects": [_entry(depth=5.0)]}],
    }
    rows = export_mod.build_rows_for_video(
        video_json=video_json,
        json_path=Path("clip.json"),
        interval_seconds=1.0,
        window_seconds=1.0,
        creation_dt=None,
        apply_filters=False,
    )
    assert len(rows) == 1
    assert rows[0]["distance"] == pytest.approx(rows[0]["raw_distance"])
    assert rows[0]["calib_method"] == ""
    assert rows[0]["raw_depth_at_point"] == ""


# --- (b) calibrated-objects lookup: applied, failed (dropped), missing (dropped) ------


def test_calibrated_objects_applied_when_present_and_not_failed():
    video_json = {
        "video_name": "clip",
        "video_fps": 1.0,
        "frames": [{"frame_index": 0, "objects": [_entry(depth=10.0)]}],
    }
    calibrated_objects = {
        ("clip", 0, "1"): {
            "distance_m": "12.5",
            "calib_method": "per_camera",
            "raw_depth_at_point": "9.9",
        }
    }
    rows = export_mod.build_rows_for_video(
        video_json=video_json,
        json_path=Path("clip.json"),
        interval_seconds=1.0,
        window_seconds=1.0,
        creation_dt=None,
        apply_filters=False,
        calibrated_objects=calibrated_objects,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["raw_distance"] == pytest.approx(10.0)
    assert row["distance"] == pytest.approx(12.5)
    assert row["calib_method"] == "per_camera"
    assert row["raw_depth_at_point"] == "9.9"


def test_calibrated_objects_failed_method_drops_row():
    video_json = {
        "video_name": "clip",
        "video_fps": 1.0,
        "frames": [{"frame_index": 0, "objects": [_entry(depth=10.0)]}],
    }
    calibrated_objects = {
        ("clip", 0, "1"): {
            "distance_m": "12.5",
            "calib_method": "failed",
            "raw_depth_at_point": "9.9",
        }
    }
    dropped_log: list = []
    rows = export_mod.build_rows_for_video(
        video_json=video_json,
        json_path=Path("clip.json"),
        interval_seconds=1.0,
        window_seconds=1.0,
        creation_dt=None,
        apply_filters=False,
        calibrated_objects=calibrated_objects,
        dropped_log=dropped_log,
    )
    assert rows == []
    assert dropped_log == [{"video_name": "clip", "track_id": 1, "reason": "calib_failed"}]


def test_calibrated_objects_no_match_drops_row():
    video_json = {
        "video_name": "clip",
        "video_fps": 1.0,
        "frames": [{"frame_index": 0, "objects": [_entry(depth=10.0)]}],
    }
    calibrated_objects: dict = {}  # no matching row for (clip, 0, "1")
    dropped_log: list = []
    rows = export_mod.build_rows_for_video(
        video_json=video_json,
        json_path=Path("clip.json"),
        interval_seconds=1.0,
        window_seconds=1.0,
        creation_dt=None,
        apply_filters=False,
        calibrated_objects=calibrated_objects,
        dropped_log=dropped_log,
    )
    assert rows == []
    assert dropped_log == [{"video_name": "clip", "track_id": 1, "reason": "calib_missing"}]


def test_load_calibrated_objects_end_to_end(tmp_path: Path):
    csv_path = tmp_path / "calibrated_objects.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "video_name",
                "frame_index",
                "track_id",
                "transect_cam",
                "distance_m",
                "raw_depth_at_point",
                "calib_method",
                "align_inlier_frac",
                "homography_used",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "video_name": "Clip.MP4",
                "frame_index": "0",
                "track_id": "1",
                "transect_cam": "49_cam003",
                "distance_m": "12.5",
                "raw_depth_at_point": "9.9",
                "calib_method": "per_camera",
                "align_inlier_frac": "0.9",
                "homography_used": "True",
            }
        )
    lookup = export_mod.load_calibrated_objects(csv_path)
    assert ("clip", 0, "1") in lookup
    assert lookup[("clip", 0, "1")]["distance_m"] == "12.5"

    video_json = {
        "video_name": "clip",
        "video_fps": 1.0,
        "frames": [{"frame_index": 0, "objects": [_entry(depth=10.0)]}],
    }
    rows = export_mod.build_rows_for_video(
        video_json=video_json,
        json_path=Path("clip.json"),
        interval_seconds=1.0,
        window_seconds=1.0,
        creation_dt=None,
        apply_filters=False,
        calibrated_objects=lookup,
    )
    assert len(rows) == 1
    assert rows[0]["distance"] == pytest.approx(12.5)
    assert rows[0]["calib_method"] == "per_camera"
    assert rows[0]["raw_depth_at_point"] == "9.9"


# --- (c) track_qc drops keep=False tracks ----------------------------------------------


def test_track_qc_drops_rejected_tracks():
    video_json = {
        "video_name": "clip",
        "video_fps": 1.0,
        "frames": [
            {
                "frame_index": 0,
                "objects": [
                    {"track_id": 1, "depth_mask_mean": 2.0, "confidence": 0.9, "center_xy": [0, 0], "bbox_xyxy": [1, 1, 5, 5]},
                    {"track_id": 2, "depth_mask_mean": 4.0, "confidence": 0.9, "center_xy": [0, 0], "bbox_xyxy": [1, 1, 5, 5]},
                ],
            }
        ],
    }
    track_qc = {("clip", "1"): False, ("clip", "2"): True}
    dropped_log: list = []
    rows = export_mod.build_rows_for_video(
        video_json=video_json,
        json_path=Path("clip.json"),
        interval_seconds=1.0,
        window_seconds=1.0,
        creation_dt=None,
        apply_filters=False,
        track_qc=track_qc,
        dropped_log=dropped_log,
    )
    assert [row["ind_no"] for row in rows] == [2]
    assert dropped_log == [{"video_name": "clip", "track_id": 1, "reason": "track_qc_reject"}]


def test_load_track_qc_matches_by_stem_case_insensitive(tmp_path: Path):
    csv_path = tmp_path / "track_qc.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["video_name", "track_id", "n_checked", "n_reject", "reject_frac", "issues", "keep"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "video_name": "Clip.MP4",
                "track_id": "1",
                "n_checked": 5,
                "n_reject": 4,
                "reject_frac": 0.8,
                "issues": "ghost",
                "keep": "False",
            }
        )
    qc = export_mod.load_track_qc(csv_path)
    assert qc[("clip", "1")] is False


# --- (d) extract_transect_cam on representative real names -----------------------------


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("171_Cam0122", "171_cam122"),
        ("T44-Cam_077", "44_cam077"),
        ("T17-17cam_82", "17_cam082"),
        (
            "vids-mission_1_phase_3_pss-E_I-T_49-49_cam03-03080020",
            "49_cam003",
        ),
    ],
)
def test_extract_transect_cam(filename, expected):
    assert export_mod.extract_transect_cam(filename) == expected


def test_extract_transect_cam_no_match_returns_none():
    assert export_mod.extract_transect_cam("no_pattern_here") is None


def test_load_video_times_skips_unparseable_rows(tmp_path: Path):
    path = tmp_path / "video_times.csv"
    path.write_text("video_name,start_datetime\n3_cam006_03240068,2024-03-24T07:15:02\nbad,not-a-date\n")
    times = export_mod.load_video_times(path)
    assert list(times) == ["3_cam006_03240068"]
    assert times["3_cam006_03240068"].hour == 7


def test_build_video_times_uses_earliest_row_per_video():
    import pandas as pd

    from apps.camera_trap.scripts.build_video_times import build_video_times

    sheet = pd.DataFrame(
        {
            "Transect_cam": ["3_cam006", "3_cam006", "15_cam126", "15_cam126"],
            "Video_name": ["03240068", "03240068", "3220171", None],
            "Year": ["2024", "2024", "2024", "2024"],
            "Month": ["3", "3", "3", "3"],
            "Day": ["24", "24", "22", "22"],
            "Hour": ["7", "7", "9", "9"],
            "Min": ["15", "14", "5", "5"],
            "Sec": ["2", "59", None, None],
        }
    )
    out = build_video_times(sheet).set_index("video_name")["start_datetime"].to_dict()
    assert out == {"15_cam126_03220171": "2024-03-22T09:05:00", "3_cam006_03240068": "2024-03-24T07:14:59"}
