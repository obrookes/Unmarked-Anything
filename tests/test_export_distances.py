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


# --- (b) calibration applied (per_camera and unknown -> pooled), raw preserved --------


def _fake_calibration():
    return {
        "49_cam003": {"slope": 2.0, "intercept": 1.0, "method": "per_camera"},
        "_meta": {"pooled_slope": 1.5, "pooled_intercept": 0.5},
    }


def test_calibration_per_camera_applied(monkeypatch):
    import apps.camera_trap.calibration.fit as fit_mod

    def fake_apply_calibration(depth, transect_cam, calib):
        entry = calib.get(transect_cam)
        if entry is None:
            meta = calib["_meta"]
            return meta["pooled_slope"] * depth + meta["pooled_intercept"], "pooled"
        return entry["slope"] * depth + entry["intercept"], entry["method"]

    monkeypatch.setattr(fit_mod, "apply_calibration", fake_apply_calibration)

    video_json = {
        "video_name": "vids-mission_1_phase_3_pss-E_I-T_49-49_cam03-03080020",
        "video_fps": 1.0,
        "frames": [{"frame_index": 0, "objects": [_entry(depth=10.0)]}],
    }
    calibration = _fake_calibration()
    rows = export_mod.build_rows_for_video(
        video_json=video_json,
        json_path=Path("clip.json"),
        interval_seconds=1.0,
        window_seconds=1.0,
        creation_dt=None,
        apply_filters=False,
        calibration=calibration,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["raw_distance"] == pytest.approx(10.0)
    assert row["transect_cam"] == "49_cam003"
    assert row["calib_method"] == "per_camera"
    assert row["distance"] == pytest.approx(2.0 * 10.0 + 1.0)


def test_calibration_unknown_camera_falls_back_to_pooled(monkeypatch):
    import apps.camera_trap.calibration.fit as fit_mod

    def fake_apply_calibration(depth, transect_cam, calib):
        entry = calib.get(transect_cam)
        if entry is None:
            meta = calib["_meta"]
            return meta["pooled_slope"] * depth + meta["pooled_intercept"], "pooled"
        return entry["slope"] * depth + entry["intercept"], entry["method"]

    monkeypatch.setattr(fit_mod, "apply_calibration", fake_apply_calibration)

    video_json = {
        "video_name": "unknown_camera_video",
        "video_fps": 1.0,
        "frames": [{"frame_index": 0, "objects": [_entry(depth=10.0)]}],
    }
    calibration = _fake_calibration()
    rows = export_mod.build_rows_for_video(
        video_json=video_json,
        json_path=Path("clip.json"),
        interval_seconds=1.0,
        window_seconds=1.0,
        creation_dt=None,
        apply_filters=False,
        calibration=calibration,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["raw_distance"] == pytest.approx(10.0)
    assert row["calib_method"] == "pooled"
    assert row["distance"] == pytest.approx(1.5 * 10.0 + 0.5)


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
