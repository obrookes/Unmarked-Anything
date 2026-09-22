from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apps.camera_trap.calibration import fit as calib_fit
from apps.camera_trap.calibration import read_boards
from apps.camera_trap.cli.dap3_cli import build_mask_storage_entry
from apps.camera_trap.vlm.engine import FakeEngine

FRAME_W, FRAME_H = 64, 48
N_FRAMES = 10


def _write_video(path: Path) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(path), fourcc, 5.0, (FRAME_W, FRAME_H))
    for i in range(N_FRAMES):
        frame = np.full((FRAME_H, FRAME_W, 3), (i * 20) % 256, dtype=np.uint8)
        vw.write(frame)
    vw.release()


def _make_video_job(job_dir: Path, video_name: str, *, depths=None, n_frames: int = N_FRAMES) -> Path:
    """Synthetic dap3 output (JSON + NPZ + tiny video) with a single track ('1'), one processed
    frame per index, using the repo's own mask-storage helpers so the fixture matches what
    dap3_cli.py writes. `depths` gives depth_mask_mean per frame (defaults to a constant)."""
    if depths is None:
        depths = [1.5] * n_frames
    video_dir = job_dir / video_name
    video_dir.mkdir(parents=True)
    video_path = video_dir / f"{video_name}.mp4"
    _write_video(video_path)

    npz_arrays: dict[str, np.ndarray] = {}
    frames = []
    for frame_idx in range(n_frames):
        mask = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
        mask[5:15, 5:15] = 1
        entry = build_mask_storage_entry(
            npz_arrays=npz_arrays,
            key_prefix=f"f{frame_idx}_obj_1_mask",
            mask=mask,
            base_entry={
                "object_index": 1,
                "track_id": 1,
                "label": "person",
                "confidence": 0.9,
                "bbox_xyxy": [5, 5, 15, 15],
                "center_xy": [10, 10],
                "mask_nonzero_pixels": 100,
                "depth_mask_mean": depths[frame_idx],
            },
            storage_format="rle",
        )
        frames.append({"frame_index": frame_idx, "status": "processed", "objects": [entry]})

    video_json = {
        "video_name": video_name,
        "video_path": str(video_path.resolve()),
        "video_fps": 5.0,
        "frame_width": FRAME_W,
        "frame_height": FRAME_H,
        "frames": frames,
    }
    (video_dir / f"{video_name}.json").write_text(json.dumps(video_json))
    np.savez(video_dir / f"{video_name}_arrays.npz", **npz_arrays)
    return video_dir / f"{video_name}.json"


def _read_csv(path: Path):
    with path.open() as f:
        return list(csv.DictReader(f))


def _write_reference_map(path: Path, rows: list[dict]) -> None:
    columns = ["transect", "cam", "transect_cam", "mission", "n_distance_annotations", "reference_dir", "n_reference_clips", "status", "note"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({**{c: "" for c in columns}, **row})


# --------------------------------------------------------------------------------------
# camera mapping
# --------------------------------------------------------------------------------------

def test_match_transect_cam_prefix_relabelled_and_missing(tmp_path: Path) -> None:
    root = tmp_path / "wcf-pps-p3"
    (root / "videos/reference/T2/Cam9").mkdir(parents=True)
    (root / "videos/reference/T75/Cam93_relabelled").mkdir(parents=True)

    rows = [
        {"transect_cam": "2_cam009", "reference_dir": "videos/reference/T2/Cam9", "status": "found"},
        {"transect_cam": "75_cam093", "reference_dir": "videos/reference/T75/Cam93_relabelled", "status": "relabelled",
         "note": "folder transect label swapped"},
        {"transect_cam": "9_cam999", "reference_dir": "videos/reference/T9/CamMissing", "status": "missing"},
    ]
    map_csv = tmp_path / "reference_video_map.csv"
    _write_reference_map(map_csv, rows)

    ref_rows = read_boards.load_reference_map(map_csv)
    assert {r["transect_cam"] for r in ref_rows} == {"2_cam009", "75_cam093"}  # missing row dropped
    ref_index = read_boards.build_reference_index(ref_rows, root)

    video_path = root / "videos/reference/T2/Cam9/clip1.mp4"
    cam, method = read_boards.match_transect_cam(video_path, ref_index, ref_rows)
    assert (cam, method) == ("2_cam009", "prefix")

    # relabelled row: transect_cam is the corrected id, not derivable from the folder path.
    video_path2 = root / "videos/reference/T75/Cam93_relabelled/clip1.mp4"
    cam2, method2 = read_boards.match_transect_cam(video_path2, ref_index, ref_rows)
    assert (cam2, method2) == ("75_cam093", "prefix")

    # a video under the "missing" camera's folder is simply unmapped.
    video_path3 = root / "videos/reference/T9/CamMissing/clip1.mp4"
    cam3, method3 = read_boards.match_transect_cam(video_path3, ref_index, ref_rows)
    assert cam3 is None


def test_match_transect_cam_basename_fallback(tmp_path: Path) -> None:
    root = tmp_path / "wcf-pps-p3"
    rows = [{"transect_cam": "4_cam146", "reference_dir": "videos/reference/somewhere_else/cam146", "status": "found"}]
    ref_rows = rows
    ref_index = read_boards.build_reference_index(ref_rows, root)

    # video lives under a differently-rooted path, but the leaf dir name still matches.
    video_path = tmp_path / "moved" / "cam146" / "clip1.mp4"
    cam, method = read_boards.match_transect_cam(video_path, ref_index, ref_rows)
    assert (cam, method) == ("4_cam146", "basename")

    video_path_unmapped = tmp_path / "moved" / "cam_unknown" / "clip1.mp4"
    cam2, method2 = read_boards.match_transect_cam(video_path_unmapped, ref_index, ref_rows)
    assert cam2 is None


# --------------------------------------------------------------------------------------
# calib row / outlier logic
# --------------------------------------------------------------------------------------

def test_build_calib_row_confidence_and_null_mapping() -> None:
    rec_high = {"transect_cam": "1_cam1", "video_name": "v", "track_id": 1, "frame_index": 0,
                "depth_mask_mean": 2.0, "verdict": {"board_visible": True, "distance_m": 8.0, "legible": True,
                                                      "confidence": "high", "raw_text": "8"}}
    row = read_boards.build_calib_row(rec_high)
    assert row["board_distance_m"] == 8.0
    assert row["vlm_conf"] == 1.0

    rec_medium = {**rec_high, "verdict": {**rec_high["verdict"], "confidence": "medium"}}
    assert read_boards.build_calib_row(rec_medium)["vlm_conf"] == 0.6

    rec_low = {**rec_high, "verdict": {**rec_high["verdict"], "confidence": "low"}}
    assert read_boards.build_calib_row(rec_low)["vlm_conf"] == 0.3

    rec_illegible = {**rec_high, "verdict": {"board_visible": True, "distance_m": None, "legible": False,
                                              "confidence": "low", "raw_text": ""}}
    row_illegible = read_boards.build_calib_row(rec_illegible)
    assert math.isnan(row_illegible["board_distance_m"])
    assert math.isnan(row_illegible["vlm_distance_raw"])

    rec_error = {"transect_cam": "1_cam1", "video_name": "v", "track_id": 1, "frame_index": 0,
                 "depth_mask_mean": 2.0, "error": "vlm call failed"}
    row_error = read_boards.build_calib_row(rec_error)
    assert math.isnan(row_error["board_distance_m"])
    assert row_error["legible"] is False


def test_flag_outliers() -> None:
    rows = [
        {"board_distance_m": 5.0},
        {"board_distance_m": 5.2},
        {"board_distance_m": 20.0},  # differs from both neighbours by > 3m
        {"board_distance_m": 5.4},
        {"board_distance_m": 5.6},
    ]
    for r in rows:
        r["outlier"] = False
    read_boards.flag_outliers(rows)
    assert rows[2]["outlier"] is True
    assert math.isnan(rows[2]["board_distance_m"])
    assert rows[0]["outlier"] is False  # first reading is never flagged (no left neighbour)
    assert rows[4]["outlier"] is False  # last reading is never flagged (no right neighbour)
    assert rows[1]["board_distance_m"] == 5.2


# --------------------------------------------------------------------------------------
# end-to-end
# --------------------------------------------------------------------------------------

def _reference_setup(tmp_path: Path, job_dir: Path, video_name: str = "video1") -> tuple[Path, Path]:
    """Reference map + root such that job_dir's video resolves to transect_cam '1_cam001'."""
    root = tmp_path / "wcf-pps-p3"
    ref_dir = root / "videos/reference/T1/Cam1"
    ref_dir.mkdir(parents=True)
    rows = [{"transect_cam": "1_cam001", "reference_dir": "videos/reference/T1/Cam1", "status": "found"}]
    map_csv = tmp_path / "reference_video_map.csv"
    _write_reference_map(map_csv, rows)
    return map_csv, root


def test_end_to_end_calib_points(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    job_dir = tmp_path / "job"
    out_dir = tmp_path / "out"
    json_path = _make_video_job(job_dir, "video1", depths=[1.0] * N_FRAMES)
    # Put the video under the reference dir so the prefix match succeeds.
    map_csv, root = _reference_setup(tmp_path, job_dir)
    ref_video_dir = root / "videos/reference/T1/Cam1"
    video_json = json.loads(json_path.read_text())
    real_video_path = Path(video_json["video_path"])
    moved_path = ref_video_dir / real_video_path.name
    moved_path.write_bytes(real_video_path.read_bytes())
    video_json["video_path"] = str(moved_path)
    json_path.write_text(json.dumps(video_json))

    responses = [
        {"board_visible": True, "distance_m": 5.0, "legible": True, "confidence": "high", "raw_text": "5"},
        {"board_visible": True, "distance_m": None, "legible": False, "confidence": "low", "raw_text": ""},
        {"board_visible": True, "distance_m": 10.0, "legible": True, "confidence": "medium", "raw_text": "10"},
    ]
    fake = FakeEngine(responses=responses)
    monkeypatch.setattr(read_boards, "VLMEngine", lambda **kwargs: fake)

    code = read_boards.main([
        "--job-dir", str(job_dir), "--out-dir", str(out_dir),
        "--reference-map", str(map_csv), "--reference-root", str(root),
        "--max-per-track", "3",
    ])
    assert code == 0
    assert len(fake.calls) == 3
    for req in fake.calls:
        assert len(req.images) == 2  # full frame + crop

    rows = _read_csv(out_dir / "calib_points.csv")
    assert len(rows) == 3
    assert all(r["transect_cam"] == "1_cam001" for r in rows)
    assert rows[0]["board_distance_m"] == "5.0"
    assert rows[0]["vlm_conf"] == "1.0"
    assert rows[1]["board_distance_m"] == ""  # illegible -> NaN -> empty
    assert rows[2]["board_distance_m"] == "10.0"
    assert rows[2]["vlm_conf"] == "0.6"

    # fit.py's loader must accept this file as-is.
    df = calib_fit.load_points(out_dir / "calib_points.csv")
    assert len(df) == 2  # the illegible NaN row is dropped by load_points


def test_dry_run_makes_no_vlm_calls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    job_dir = tmp_path / "job"
    out_dir = tmp_path / "out"
    json_path = _make_video_job(job_dir, "video1")
    map_csv, root = _reference_setup(tmp_path, job_dir)
    ref_video_dir = root / "videos/reference/T1/Cam1"
    video_json = json.loads(json_path.read_text())
    real_video_path = Path(video_json["video_path"])
    moved_path = ref_video_dir / real_video_path.name
    moved_path.write_bytes(real_video_path.read_bytes())
    video_json["video_path"] = str(moved_path)
    json_path.write_text(json.dumps(video_json))

    def _fail_construct(**kwargs):
        raise AssertionError("VLMEngine should not be constructed in --dry-run")

    monkeypatch.setattr(read_boards, "VLMEngine", _fail_construct)

    code = read_boards.main([
        "--job-dir", str(job_dir), "--out-dir", str(out_dir),
        "--reference-map", str(map_csv), "--reference-root", str(root),
        "--max-per-track", "3", "--dry-run",
    ])
    assert code == 0
    assert not (out_dir / "read_boards.jsonl").exists()
    assert not (out_dir / "calib_points.csv").exists()
    images = list((out_dir / "dry_run" / "video1").glob("*.jpg"))
    assert len(images) == 6  # 3 frames x (full + crop)


def test_unmapped_video_logged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    job_dir = tmp_path / "job"
    out_dir = tmp_path / "out"
    _make_video_job(job_dir, "video1")  # video stays at its default (unrelated) path
    map_csv, root = _reference_setup(tmp_path, job_dir)

    fake = FakeEngine(responses=[])
    monkeypatch.setattr(read_boards, "VLMEngine", lambda **kwargs: fake)

    code = read_boards.main([
        "--job-dir", str(job_dir), "--out-dir", str(out_dir),
        "--reference-map", str(map_csv), "--reference-root", str(root),
        "--max-per-track", "3",
    ])
    assert code == 0
    assert len(fake.calls) == 0
    unmapped = _read_csv(out_dir / "unmapped_videos.csv")
    assert len(unmapped) == 1
    assert unmapped[0]["video_name"] == "video1"
    assert not (out_dir / "calib_points.csv").exists()


def test_video_cam_csv_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    job_dir = tmp_path / "job"
    out_dir = tmp_path / "out"
    _make_video_job(job_dir, "video1")  # not under any reference_dir
    map_csv, root = _reference_setup(tmp_path, job_dir)

    video_cam_csv = tmp_path / "video_cam.csv"
    with video_cam_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video", "transect_cam"])
        writer.writeheader()
        writer.writerow({"video": "video1", "transect_cam": "override_cam"})

    responses = [{"board_visible": True, "distance_m": 3.0, "legible": True, "confidence": "high", "raw_text": "3"}]
    fake = FakeEngine(responses=responses)
    monkeypatch.setattr(read_boards, "VLMEngine", lambda **kwargs: fake)

    code = read_boards.main([
        "--job-dir", str(job_dir), "--out-dir", str(out_dir),
        "--reference-map", str(map_csv), "--reference-root", str(root),
        "--video-cam-csv", str(video_cam_csv), "--max-per-track", "3",
    ])
    assert code == 0
    rows = _read_csv(out_dir / "calib_points.csv")
    assert all(r["transect_cam"] == "override_cam" for r in rows)


def test_resume_skips_already_done_video(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    job_dir = tmp_path / "job"
    out_dir = tmp_path / "out"
    json_path = _make_video_job(job_dir, "video1")
    map_csv, root = _reference_setup(tmp_path, job_dir)
    ref_video_dir = root / "videos/reference/T1/Cam1"
    video_json = json.loads(json_path.read_text())
    real_video_path = Path(video_json["video_path"])
    moved_path = ref_video_dir / real_video_path.name
    moved_path.write_bytes(real_video_path.read_bytes())
    video_json["video_path"] = str(moved_path)
    json_path.write_text(json.dumps(video_json))

    ok_response = [{"board_visible": True, "distance_m": 5.0, "legible": True, "confidence": "high", "raw_text": "5"}]
    fake = FakeEngine(responses=ok_response)
    monkeypatch.setattr(read_boards, "VLMEngine", lambda **kwargs: fake)

    args = [
        "--job-dir", str(job_dir), "--out-dir", str(out_dir),
        "--reference-map", str(map_csv), "--reference-root", str(root),
        "--max-per-track", "3",
    ]
    read_boards.main(args)
    n_calls_first = len(fake.calls)
    assert n_calls_first == 3

    read_boards.main(args)  # no --overwrite: video1 already in read_boards.jsonl
    assert len(fake.calls) == n_calls_first

    read_boards.main(args + ["--overwrite"])
    assert len(fake.calls) == n_calls_first * 2
