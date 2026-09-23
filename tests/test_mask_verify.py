from __future__ import annotations

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

from apps.camera_trap.cli.dap3_cli import build_mask_storage_entry
from apps.camera_trap.qc import mask_verify
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


def _make_video_job(
    job_dir: Path,
    video_name: str,
    *,
    track_ids: list[int] = (1,),
    n_frames: int = N_FRAMES,
) -> Path:
    """Build a synthetic dap3 output dir (JSON + NPZ + tiny video) with one processed frame per
    index, one object per track_id, using the repo's own mask-storage helpers so the fixture
    matches exactly what dap3_cli.py writes."""
    video_dir = job_dir / video_name
    video_dir.mkdir(parents=True)
    video_path = video_dir / f"{video_name}.mp4"
    _write_video(video_path)

    npz_arrays: dict[str, np.ndarray] = {}
    frames = []
    for frame_idx in range(n_frames):
        objects = []
        for tid in track_ids:
            mask = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
            x0, y0 = 5 + tid, 5 + tid
            mask[y0 : y0 + 10, x0 : x0 + 10] = 1
            entry = build_mask_storage_entry(
                npz_arrays=npz_arrays,
                key_prefix=f"f{frame_idx}_obj_{tid}_mask",
                mask=mask,
                base_entry={
                    "object_index": tid,
                    "track_id": tid,
                    "label": "animal",
                    "confidence": 0.9,
                    "bbox_xyxy": [x0, y0, x0 + 10, y0 + 10],
                    "center_xy": [x0 + 5, y0 + 5],
                    "mask_nonzero_pixels": 100,
                    "depth_mask_mean": 1.5,
                },
                storage_format="rle",
            )
            objects.append(entry)
        frames.append({"frame_index": frame_idx, "status": "processed", "objects": objects})

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


def test_sample_track_frames_even_spacing() -> None:
    assert mask_verify.sample_track_frames(list(range(10)), 3) == [0, 4, 9]
    assert mask_verify.sample_track_frames([2, 5, 8], 5) == [2, 5, 8]  # fewer than k: keep all
    assert mask_verify.sample_track_frames([0, 1, 2, 3, 4], 1) == [2]


def test_group_tracks_skips_unprocessed_frames() -> None:
    video_json = {
        "frames": [
            {"frame_index": 0, "status": "processed", "objects": [{"track_id": 1, "bbox_xyxy": [0, 0, 1, 1]}]},
            {"frame_index": 1, "status": "sam_error", "objects": [{"track_id": 1, "bbox_xyxy": [0, 0, 1, 1]}]},
            {"frame_index": 2, "status": "tracked", "objects": [{"track_id": 1, "bbox_xyxy": [0, 0, 1, 1]}]},
        ]
    }
    tracks = mask_verify.group_tracks(video_json)
    assert list(tracks.keys()) == [1]
    assert list(tracks[1].keys()) == [0, 2]


def test_end_to_end_keep_and_reject(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    job_dir = tmp_path / "job"
    out_dir = tmp_path / "qc"
    _make_video_job(job_dir, "video1", track_ids=[1])

    # 3 sampled frames for the track; script it 2 ok, 1 reject -> reject_frac 1/3 <= 0.5 -> keep.
    responses = [
        {"ok": True, "issue": "none", "animal_visible": True, "note": "fine"},
        {"ok": False, "issue": "loose", "animal_visible": True, "note": "too loose"},
        {"ok": True, "issue": "none", "animal_visible": True, "note": "fine"},
    ]
    fake = FakeEngine(responses=responses)
    monkeypatch.setattr(mask_verify, "VLMEngine", lambda **kwargs: fake)

    code = mask_verify.main(["--job-dir", str(job_dir), "--out-dir", str(out_dir), "--frames-per-track", "3"])
    assert code == 0

    jsonl_path = out_dir / "mask_qc.jsonl"
    lines = [json.loads(l) for l in jsonl_path.read_text().splitlines()]
    assert len(lines) == 3
    assert all(l["video_name"] == "video1" and l["track_id"] == 1 for l in lines)

    csv_rows = list(_read_csv(out_dir / "track_qc.csv"))
    assert len(csv_rows) == 1
    row = csv_rows[0]
    assert row["video_name"] == "video1"
    assert row["track_id"] == "1"
    assert row["n_checked"] == "3"
    assert row["n_reject"] == "1"
    assert float(row["reject_frac"]) == pytest.approx(1 / 3, abs=1e-3)
    assert row["issues"] == "loose:1"
    assert row["keep"] == "True"


def test_end_to_end_drops_track_above_reject_threshold(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    job_dir = tmp_path / "job"
    out_dir = tmp_path / "qc"
    _make_video_job(job_dir, "video1", track_ids=[1])

    responses = [
        {"ok": False, "issue": "fragment", "animal_visible": True, "note": "bad"},
        {"ok": False, "issue": "merged", "animal_visible": True, "note": "bad"},
        {"ok": True, "issue": "none", "animal_visible": True, "note": "ok"},
    ]
    fake = FakeEngine(responses=responses)
    monkeypatch.setattr(mask_verify, "VLMEngine", lambda **kwargs: fake)

    mask_verify.main(["--job-dir", str(job_dir), "--out-dir", str(out_dir), "--frames-per-track", "3", "--reject-frac-max", "0.5"])

    csv_rows = list(_read_csv(out_dir / "track_qc.csv"))
    row = csv_rows[0]
    assert row["n_reject"] == "2"
    assert row["keep"] == "False"


def test_all_failures_keeps_track(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    job_dir = tmp_path / "job"
    out_dir = tmp_path / "qc"
    _make_video_job(job_dir, "video1", track_ids=[1])

    fake = FakeEngine(responses=None)  # every call returns None -> "error" recorded
    monkeypatch.setattr(mask_verify, "VLMEngine", lambda **kwargs: fake)

    mask_verify.main(["--job-dir", str(job_dir), "--out-dir", str(out_dir), "--frames-per-track", "3"])

    jsonl_path = out_dir / "mask_qc.jsonl"
    lines = [json.loads(l) for l in jsonl_path.read_text().splitlines()]
    assert all("error" in l for l in lines)

    csv_rows = list(_read_csv(out_dir / "track_qc.csv"))
    row = csv_rows[0]
    assert row["n_checked"] == "0"
    assert row["keep"] == "True"


def test_resume_skips_already_done_video(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    job_dir = tmp_path / "job"
    out_dir = tmp_path / "qc"
    _make_video_job(job_dir, "video1", track_ids=[1])

    ok_response = [{"ok": True, "issue": "none", "animal_visible": True, "note": "fine"}]
    fake = FakeEngine(responses=ok_response)
    monkeypatch.setattr(mask_verify, "VLMEngine", lambda **kwargs: fake)

    mask_verify.main(["--job-dir", str(job_dir), "--out-dir", str(out_dir), "--frames-per-track", "3"])
    n_calls_first_run = len(fake.calls)
    assert n_calls_first_run == 3

    # Second run without --overwrite: video1 already in mask_qc.jsonl, should be skipped entirely.
    mask_verify.main(["--job-dir", str(job_dir), "--out-dir", str(out_dir), "--frames-per-track", "3"])
    assert len(fake.calls) == n_calls_first_run  # no new calls

    # With --overwrite, it is reprocessed.
    mask_verify.main(["--job-dir", str(job_dir), "--out-dir", str(out_dir), "--frames-per-track", "3", "--overwrite"])
    assert len(fake.calls) == n_calls_first_run * 2


def test_dry_run_writes_images_and_no_jsonl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    job_dir = tmp_path / "job"
    out_dir = tmp_path / "qc"
    _make_video_job(job_dir, "video1", track_ids=[1])

    def _fail_construct(**kwargs):
        raise AssertionError("VLMEngine should not be constructed in --dry-run")

    monkeypatch.setattr(mask_verify, "VLMEngine", _fail_construct)

    code = mask_verify.main(["--job-dir", str(job_dir), "--out-dir", str(out_dir), "--frames-per-track", "3", "--dry-run"])
    assert code == 0
    assert not (out_dir / "mask_qc.jsonl").exists()
    dry_run_dir = out_dir / "dry_run" / "video1"
    images = list(dry_run_dir.glob("*.jpg"))
    assert len(images) == 6  # 3 frames x (raw + overlay)


def _read_csv(path: Path):
    import csv

    with path.open() as f:
        yield from csv.DictReader(f)
