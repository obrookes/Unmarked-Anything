from __future__ import annotations

import csv
import json
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

from apps.camera_trap.calibration import find_sign_frames as fsf
from apps.camera_trap.calibration import refmap
from apps.camera_trap.vlm.engine import FakeEngine

FRAME_W, FRAME_H = 64, 48


def _write_video(path: Path, n_frames: int, fps: float = 5.0) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(path), fourcc, fps, (FRAME_W, FRAME_H))
    for i in range(n_frames):
        frame = np.full((FRAME_H, FRAME_W, 3), (i * 17) % 256, dtype=np.uint8)
        vw.write(frame)
    vw.release()


def _write_reference_map(path: Path, rows: list[dict]) -> None:
    columns = ["transect", "cam", "transect_cam", "mission", "n_distance_annotations", "reference_dir", "n_reference_clips", "status", "note"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({**{c: "" for c in columns}, **row})


def _read_csv(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------------------------------
# refmap
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

    ref_rows = refmap.load_reference_map(map_csv)
    assert {r["transect_cam"] for r in ref_rows} == {"2_cam009", "75_cam093"}  # missing row dropped
    ref_index = refmap.build_reference_index(ref_rows, root)

    video_path = root / "videos/reference/T2/Cam9/clip1.mp4"
    cam, method = refmap.match_transect_cam(video_path, ref_index, ref_rows)
    assert (cam, method) == ("2_cam009", "prefix")

    # relabelled row: transect_cam is the corrected id, not derivable from the folder path.
    video_path2 = root / "videos/reference/T75/Cam93_relabelled/clip1.mp4"
    cam2, method2 = refmap.match_transect_cam(video_path2, ref_index, ref_rows)
    assert (cam2, method2) == ("75_cam093", "prefix")

    # a video under the "missing" camera's folder is simply unmapped.
    video_path3 = root / "videos/reference/T9/CamMissing/clip1.mp4"
    cam3, method3 = refmap.match_transect_cam(video_path3, ref_index, ref_rows)
    assert cam3 is None


def test_match_transect_cam_basename_fallback(tmp_path: Path) -> None:
    root = tmp_path / "wcf-pps-p3"
    rows = [{"transect_cam": "4_cam146", "reference_dir": "videos/reference/somewhere_else/cam146", "status": "found"}]
    ref_rows = rows
    ref_index = refmap.build_reference_index(ref_rows, root)

    video_path = tmp_path / "moved" / "cam146" / "clip1.mp4"
    cam, method = refmap.match_transect_cam(video_path, ref_index, ref_rows)
    assert (cam, method) == ("4_cam146", "basename")

    video_path_unmapped = tmp_path / "moved" / "cam_unknown" / "clip1.mp4"
    cam2, method2 = refmap.match_transect_cam(video_path_unmapped, ref_index, ref_rows)
    assert cam2 is None


def test_discover_reference_videos_walks_map_rows(tmp_path: Path) -> None:
    root = tmp_path / "wcf-pps-p3"
    cam1_dir = root / "videos/reference/T1/Cam1"
    cam2_dir = root / "videos/reference/T2/Cam2_relabelled"
    missing_dir = root / "videos/reference/T3/Cam3"
    for d in (cam1_dir, cam2_dir):
        d.mkdir(parents=True)
    (cam1_dir / "clip1.MP4").write_bytes(b"x")  # case-insensitive extension match
    (cam1_dir / "clip2.mov").write_bytes(b"x")
    (cam1_dir / "notes.txt").write_bytes(b"x")  # non-video, ignored
    (cam2_dir / "clipA.mp4").write_bytes(b"x")

    rows = [
        {"transect_cam": "1_cam001", "reference_dir": "videos/reference/T1/Cam1", "status": "found"},
        {"transect_cam": "2_cam002", "reference_dir": "videos/reference/T2/Cam2_relabelled", "status": "relabelled"},
        {"transect_cam": "3_cam003", "reference_dir": "videos/reference/T3/Cam3", "status": "missing"},
    ]
    map_csv = tmp_path / "reference_video_map.csv"
    _write_reference_map(map_csv, rows)

    ref_rows = refmap.load_reference_map(map_csv)
    videos = refmap.discover_reference_videos(ref_rows, root)
    assert not missing_dir.exists()  # sanity: never created, so it couldn't leak in anyway
    assert {(p.name, cam) for p, cam in videos} == {
        ("clip1.MP4", "1_cam001"),
        ("clip2.mov", "1_cam001"),
        ("clipA.mp4", "2_cam002"),
    }


# --------------------------------------------------------------------------------------
# sampling
# --------------------------------------------------------------------------------------

def test_sample_frame_indices() -> None:
    assert fsf.sample_frame_indices(fps=10.0, frame_count=10, scan_fps=10.0) == list(range(10))
    assert fsf.sample_frame_indices(fps=10.0, frame_count=10, scan_fps=5.0) == [0, 2, 4, 6, 8]
    assert fsf.sample_frame_indices(fps=0.0, frame_count=10, scan_fps=5.0) == []
    assert fsf.sample_frame_indices(fps=10.0, frame_count=0, scan_fps=5.0) == []


# --------------------------------------------------------------------------------------
# event grouping
# --------------------------------------------------------------------------------------

def _rec(frame_idx, distance_m, legible=True, confidence="high", sign_box=None):
    return {"frame_idx": frame_idx, "distance_m": distance_m, "legible": legible, "confidence": confidence, "sign_box": sign_box}


def test_group_events_gap_tolerance_and_min_agree() -> None:
    # 5.0 held for frames 0,1 with an illegible gap at frame 2 (within max_gap_s), then 3.
    records = [
        _rec(0, 5.0),
        _rec(1, 5.0),
        _rec(2, None, legible=False),
        _rec(3, 5.0),
    ]
    events = fsf.group_events(records, fps=10.0, max_gap_s=1.0, min_agree=2)
    assert len(events) == 1
    assert events[0]["distance"] == 5.0
    assert [f["frame_idx"] for f in events[0]["frames"]] == [0, 1, 3]


def test_group_events_gap_too_long_splits_run() -> None:
    # Gap of 20 frames at 10 fps = 2.0s, exceeding max_gap_s=1.0s: splits into two runs, each
    # too short to reach min_agree on its own.
    records = [_rec(0, 5.0), _rec(1, 5.0), _rec(21, 5.0), _rec(22, 5.0)]
    events = fsf.group_events(records, fps=10.0, max_gap_s=1.0, min_agree=2)
    assert len(events) == 2
    assert [f["frame_idx"] for f in events[0]["frames"]] == [0, 1]
    assert [f["frame_idx"] for f in events[1]["frames"]] == [21, 22]


def test_group_events_below_min_agree_dropped() -> None:
    records = [_rec(0, 5.0)]
    assert fsf.group_events(records, fps=10.0, max_gap_s=1.0, min_agree=2) == []


def test_group_events_distance_change_and_repeated_hold() -> None:
    # 5.0 x2, then 8.0 x2 (distance change ends the 5.0 run), then a *second* 5.0 hold later
    # (separated from the first by the 8.0 run) -- a separate event, not merged with the first.
    records = [
        _rec(0, 5.0), _rec(1, 5.0),
        _rec(2, 8.0), _rec(3, 8.0),
        _rec(10, 5.0), _rec(11, 5.0),
    ]
    events = fsf.group_events(records, fps=10.0, max_gap_s=1.0, min_agree=2)
    assert [e["distance"] for e in events] == [5.0, 8.0, 5.0]
    assert [f["frame_idx"] for f in events[0]["frames"]] == [0, 1]
    assert [f["frame_idx"] for f in events[2]["frames"]] == [10, 11]


def test_events_to_rows_representative_frame_and_conf() -> None:
    records = [
        _rec(0, 5.0, confidence="high", sign_box=[0.1, 0.1, 0.2, 0.2]),
        _rec(1, 5.0, confidence="medium", sign_box=[0.3, 0.3, 0.4, 0.4]),
        _rec(2, 5.0, confidence="low", sign_box=[0.5, 0.5, 0.6, 0.6]),
    ]
    events = fsf.group_events(records, fps=10.0, max_gap_s=1.0, min_agree=2)
    rows = fsf.events_to_rows("video1.mp4", "1_cam001", events)
    assert len(rows) == 1
    row = rows[0]
    assert row["event_id"] == "video1_e1"
    assert row["frame_idx"] == 1  # middle of [0, 1, 2]
    assert row["sign_box_norm"] == "0.300000;0.300000;0.400000;0.400000"
    assert row["n_agree"] == 3
    assert row["vlm_conf"] == pytest.approx((1.0 + 0.6 + 0.3) / 3, abs=1e-4)


# --------------------------------------------------------------------------------------
# tile fallback box mapping
# --------------------------------------------------------------------------------------

def test_map_tile_box() -> None:
    tile_box = (0.0, 0.4, 0.6, 1.0)
    mapped = fsf.map_tile_box([0.1, 0.2, 0.5, 0.6], tile_box)
    assert mapped == pytest.approx([0.06, 0.52, 0.30, 0.76])


def test_pick_best_tile_prefers_legible_then_confidence() -> None:
    candidates = [
        ((0, 0, 1, 1), {"person_with_sign": True, "legible": False, "distance_m": None, "confidence": "high"}),
        ((0, 0, 1, 1), {"person_with_sign": True, "legible": True, "distance_m": 4.0, "confidence": "low"}),
        ((0, 0, 1, 1), {"person_with_sign": True, "legible": True, "distance_m": 4.0, "confidence": "high"}),
    ]
    best = fsf.pick_best_tile(candidates)
    assert best[1]["confidence"] == "high"
    assert best[1]["legible"] is True

    assert fsf.pick_best_tile([]) is None


# --------------------------------------------------------------------------------------
# override add/drop/fix
# --------------------------------------------------------------------------------------

def _base_row(frame_idx, distance_m, event_id):
    return {
        "transect_cam": "1_cam001", "video_path": "v1.mp4", "event_id": event_id,
        "frame_idx": frame_idx, "distance_m": distance_m, "sign_box_norm": "", "n_agree": 3,
        "vlm_conf": 0.9, "source": "vlm",
    }


def test_apply_overrides_add_drop_fix() -> None:
    rows = [_base_row(5, 5.0, "v1_e1"), _base_row(20, 10.0, "v1_e2")]
    overrides = [
        {"transect_cam": "1_cam001", "video_path": "v1.mp4", "frame_idx": "20", "distance_m": "0", "action": "drop"},
        {"transect_cam": "1_cam001", "video_path": "v1.mp4", "frame_idx": "5", "distance_m": "5.5", "action": "fix", "sign_box_norm": "0.1;0.1;0.2;0.2"},
        {"transect_cam": "1_cam001", "video_path": "v1.mp4", "frame_idx": "40", "distance_m": "15.0", "action": "add", "sign_box_norm": ""},
    ]
    out = fsf.apply_overrides(rows, overrides)
    by_frame = {r["frame_idx"]: r for r in out}
    assert 20 not in by_frame  # dropped
    assert by_frame[5]["distance_m"] == 5.5
    assert by_frame[5]["source"] == "override"
    assert by_frame[5]["sign_box_norm"] == "0.1;0.1;0.2;0.2"
    assert by_frame[40]["distance_m"] == 15.0
    assert by_frame[40]["source"] == "override"
    assert by_frame[40]["event_id"] == "v1_manual1"


# --------------------------------------------------------------------------------------
# end-to-end helpers
# --------------------------------------------------------------------------------------

def _setup_single_camera(tmp_path: Path, n_frames: int, fps: float, video_name: str = "clip1") -> tuple[Path, Path, Path]:
    """Reference map + root with one camera whose reference_dir holds one synthetic video.
    Returns (map_csv, root, video_path)."""
    root = tmp_path / "wcf-pps-p3"
    ref_dir = root / "videos/reference/T1/Cam1"
    ref_dir.mkdir(parents=True)
    video_path = ref_dir / f"{video_name}.mp4"
    _write_video(video_path, n_frames, fps=fps)

    rows = [{"transect_cam": "1_cam001", "reference_dir": "videos/reference/T1/Cam1", "status": "found"}]
    map_csv = tmp_path / "reference_video_map.csv"
    _write_reference_map(map_csv, rows)
    return map_csv, root, video_path


def _ok_response(distance_m: float, confidence: str = "high"):
    return {"person_with_sign": True, "distance_m": distance_m, "legible": True, "confidence": confidence, "sign_box": [0.1, 0.1, 0.2, 0.2]}


def test_dry_run_makes_no_vlm_calls_and_writes_frames(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    map_csv, root, _video_path = _setup_single_camera(tmp_path, n_frames=6, fps=5.0)
    out_dir = tmp_path / "out"

    def _fail_construct(**kwargs):
        raise AssertionError("VLMEngine should not be constructed in --dry-run")

    monkeypatch.setattr(fsf, "VLMEngine", _fail_construct)

    code = fsf.main([
        "--reference-map", str(map_csv), "--reference-root", str(root), "--out-dir", str(out_dir),
        "--scan-fps", "5", "--dry-run",
    ])
    assert code == 0
    assert not (out_dir / "scan.jsonl").exists()
    assert not (out_dir / "calibration_frames.csv").exists()
    images = list((out_dir / "dry_run" / "clip1").glob("*.jpg"))
    assert len(images) == 6


def test_resume_skips_already_done_video(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    map_csv, root, _video_path = _setup_single_camera(tmp_path, n_frames=6, fps=5.0)
    out_dir = tmp_path / "out"

    fake = FakeEngine(responses=[_ok_response(5.0)])
    monkeypatch.setattr(fsf, "VLMEngine", lambda **kwargs: fake)

    args = ["--reference-map", str(map_csv), "--reference-root", str(root), "--out-dir", str(out_dir), "--scan-fps", "5"]
    fsf.main(args)
    n_calls_first = len(fake.calls)
    assert n_calls_first == 6

    fsf.main(args)  # no --overwrite: video already in scan.jsonl
    assert len(fake.calls) == n_calls_first

    fsf.main(args + ["--overwrite"])
    assert len(fake.calls) == n_calls_first * 2


def test_end_to_end_calibration_frames_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    map_csv, root, video_path = _setup_single_camera(tmp_path, n_frames=6, fps=5.0)
    out_dir = tmp_path / "out"

    fake = FakeEngine(responses=[_ok_response(5.0)])
    monkeypatch.setattr(fsf, "VLMEngine", lambda **kwargs: fake)

    code = fsf.main([
        "--reference-map", str(map_csv), "--reference-root", str(root), "--out-dir", str(out_dir),
        "--scan-fps", "5", "--min-agree", "2",
    ])
    assert code == 0

    rows = _read_csv(out_dir / "calibration_frames.csv")
    assert len(rows) == 1
    assert rows[0]["transect_cam"] == "1_cam001"
    assert rows[0]["video_path"] == str(video_path)
    assert rows[0]["distance_m"] == "5.0"
    assert rows[0]["n_agree"] == "6"
    assert rows[0]["source"] == "vlm"
    assert (out_dir / "events_summary.csv").exists()
    assert (out_dir / "contact_sheet.html").exists()
    assert list((out_dir / "contact_sheet_imgs").glob("*.jpg"))


def test_tile_fallback_maps_box_to_full_frame(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    map_csv, root, video_path = _setup_single_camera(tmp_path, n_frames=4, fps=2.0)
    out_dir = tmp_path / "out"

    counter = {"i": -1}

    def responder(req):
        counter["i"] += 1
        i = counter["i"]
        if i < 4:  # whole-frame batch: nobody visible, forcing tile fallback for all 4 frames
            return {"person_with_sign": False, "distance_m": None, "legible": False, "confidence": "low", "sign_box": None}
        j = i - 4
        frame_pos, tile_pos = divmod(j, 4)
        if frame_pos == 0 and tile_pos == 2:  # tile (0.0, 0.4, 0.6, 1.0) of frame 0
            return {"person_with_sign": True, "distance_m": 7.0, "legible": True, "confidence": "high", "sign_box": [0.1, 0.2, 0.5, 0.6]}
        return {"person_with_sign": False, "distance_m": None, "legible": False, "confidence": "low", "sign_box": None}

    fake = FakeEngine(responses=responder)
    monkeypatch.setattr(fsf, "VLMEngine", lambda **kwargs: fake)

    code = fsf.main([
        "--reference-map", str(map_csv), "--reference-root", str(root), "--out-dir", str(out_dir),
        "--scan-fps", "2", "--tile-fallback", "--min-agree", "1",
    ])
    assert code == 0
    assert len(fake.calls) == 4 + 4 * 4  # 4 whole-frame + 4 tiles per frame

    scan_records = [json.loads(l) for l in (out_dir / "scan.jsonl").read_text().splitlines()]
    rec0 = next(r for r in scan_records if r["frame_idx"] == 0)
    assert rec0["source"] == "vlm_tile"
    assert rec0["distance_m"] == 7.0
    assert rec0["sign_box"] == pytest.approx([0.06, 0.52, 0.30, 0.76])

    rows = _read_csv(out_dir / "calibration_frames.csv")
    assert len(rows) == 1
    assert rows[0]["sign_box_norm"] == "0.060000;0.520000;0.300000;0.760000"


def test_sharding_and_build_only_merges_shards(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "wcf-pps-p3"
    dir1 = root / "videos/reference/T1/Cam1"
    dir2 = root / "videos/reference/T2/Cam2"
    dir1.mkdir(parents=True)
    dir2.mkdir(parents=True)
    _write_video(dir1 / "clip1.mp4", n_frames=4, fps=2.0)
    _write_video(dir2 / "clip2.mp4", n_frames=4, fps=2.0)
    rows = [
        {"transect_cam": "1_cam001", "reference_dir": "videos/reference/T1/Cam1", "status": "found"},
        {"transect_cam": "2_cam002", "reference_dir": "videos/reference/T2/Cam2", "status": "found"},
    ]
    map_csv = tmp_path / "reference_video_map.csv"
    _write_reference_map(map_csv, rows)
    out_dir = tmp_path / "out"

    fake = FakeEngine(responses=[_ok_response(6.0)])
    monkeypatch.setattr(fsf, "VLMEngine", lambda **kwargs: fake)

    common = ["--reference-map", str(map_csv), "--reference-root", str(root), "--out-dir", str(out_dir),
              "--scan-fps", "2", "--min-agree", "2", "--num-shards", "2"]
    fsf.main(common + ["--shard-index", "0"])
    fsf.main(common + ["--shard-index", "1"])

    assert (out_dir / "scan.shard0.jsonl").exists()
    assert (out_dir / "scan.shard1.jsonl").exists()

    code = fsf.main([
        "--reference-map", str(map_csv), "--reference-root", str(root), "--out-dir", str(out_dir),
        "--build-only",
    ])
    assert code == 0
    rows_out = _read_csv(out_dir / "calibration_frames.csv")
    assert {r["transect_cam"] for r in rows_out} == {"1_cam001", "2_cam002"}
    assert len(rows_out) == 2


def test_override_csv_applied_at_build(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True)
    scan_records = [
        {"video_path": "nope.mp4", "transect_cam": "1_cam001", "fps": 5.0, "frame_idx": 1,
         "person_with_sign": True, "distance_m": 5.0, "legible": True, "confidence": "high", "sign_box": None, "source": "vlm_whole"},
        {"video_path": "nope.mp4", "transect_cam": "1_cam001", "fps": 5.0, "frame_idx": 2,
         "person_with_sign": True, "distance_m": 5.0, "legible": True, "confidence": "high", "sign_box": None, "source": "vlm_whole"},
    ]
    with (out_dir / "scan.jsonl").open("w") as f:
        for rec in scan_records:
            f.write(json.dumps(rec) + "\n")

    override_csv = tmp_path / "overrides.csv"
    with override_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["transect_cam", "video_path", "frame_idx", "distance_m", "action", "sign_box_norm"])
        writer.writeheader()
        # representative frame of the [1, 2] run is frames[len//2] = frame_idx 2 (the middle
        # agreeing frame for a 2-frame run, by this module's convention).
        writer.writerow({"transect_cam": "1_cam001", "video_path": "nope.mp4", "frame_idx": "2", "distance_m": "5.5", "action": "fix", "sign_box_norm": ""})

    # empty (but valid) map/root: --build-only never touches them.
    map_csv = tmp_path / "empty_map.csv"
    _write_reference_map(map_csv, [])
    root = tmp_path / "root"
    root.mkdir()

    code = fsf.main([
        "--reference-map", str(map_csv), "--reference-root", str(root), "--out-dir", str(out_dir),
        "--build-only", "--override-csv", str(override_csv), "--min-agree", "2",
    ])
    assert code == 0
    rows = _read_csv(out_dir / "calibration_frames.csv")
    assert len(rows) == 1
    assert rows[0]["distance_m"] == "5.5"
    assert rows[0]["source"] == "override"
