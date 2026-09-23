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

from apps.camera_trap.calibration import apply as applymod
from apps.camera_trap.calibration import extrinsic
from apps.camera_trap.cli.dap3_cli import build_mask_storage_entry

H, W = 64, 64
MIN_DEPTH = 1.0
MAX_DEPTH = 50.0


# --------------------------------------------------------------------------------------
# fixture builders
# --------------------------------------------------------------------------------------

def _identity_knots():
    # piecewise_from_knots(x, y) with y == x is the identity function over/around this range.
    return np.array([0.0, 100.0]), np.array([0.0, 100.0])


def _background_depth() -> np.ndarray:
    """Smooth synthetic scene depth (metres), well inside [MIN_DEPTH, MAX_DEPTH)."""
    cols = np.linspace(5.0, 15.0, W)
    return np.tile(cols, (H, 1)).astype(np.float64)


def _write_camera_calib(calib_dir: Path, cam_name: str, *, all_far: bool = False) -> None:
    calib_dir.mkdir(parents=True, exist_ok=True)
    depth_bg = np.full((H, W), MAX_DEPTH, dtype=np.float64) if all_far else _background_depth()
    anchor_disp_raw = (1.0 / depth_bg).astype(np.float32)
    anchor_img = np.random.default_rng(0).integers(0, 255, size=(H, W, 3), dtype=np.uint8)
    anchor_person_mask = np.zeros((H, W), dtype=bool)
    knots_x, knots_y = _identity_knots()
    np.savez(
        calib_dir / f"{cam_name}.npz",
        anchor_disp_raw=anchor_disp_raw,
        anchor_img=anchor_img,
        anchor_person_mask=anchor_person_mask,
        knots_x=knots_x,
        knots_y=knots_y,
        max_depth=np.float64(MAX_DEPTH),
        min_depth=np.float64(MIN_DEPTH),
        anchor_distance_m=np.float64(10.0),
        da3_model_id=np.str_("fake-model"),
    )


def _write_pooled_calib(calib_dir: Path) -> None:
    calib_dir.mkdir(parents=True, exist_ok=True)
    knots_x, knots_y = _identity_knots()
    np.savez(
        calib_dir / "_pooled.npz",
        knots_x=knots_x,
        knots_y=knots_y,
        max_depth=np.float64(MAX_DEPTH),
        min_depth=np.float64(MIN_DEPTH),
    )


def _blob_mask(cy: int, cx: int, half: int = 4) -> np.ndarray:
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[cy - half : cy + half, cx - half : cx + half] = 1
    return mask


def _write_video(path: Path, n_frames: int = 1) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(path), fourcc, 5.0, (W, H))
    for i in range(n_frames):
        vw.write(np.full((H, W, 3), (i * 20) % 256, dtype=np.uint8))
    vw.release()


def _make_job(
    job_dir: Path,
    video_name: str,
    *,
    depth_raw: np.ndarray,
    mask: np.ndarray,
    track_id: int = 1,
    depth_dtype=np.float32,
    with_video: bool = False,
) -> Path:
    video_dir = job_dir / video_name
    video_dir.mkdir(parents=True)

    npz_arrays: dict[str, np.ndarray] = {"f0_depth": depth_raw.astype(depth_dtype)}
    entry = build_mask_storage_entry(
        npz_arrays=npz_arrays,
        key_prefix="f0_obj_0_mask",
        mask=mask,
        base_entry={
            "object_index": 0,
            "track_id": track_id,
            "label": "animal",
            "confidence": 0.9,
            "bbox_xyxy": [0, 0, W, H],
            "center_xy": [W // 2, H // 2],
        },
        storage_format="raw",
    )
    frame = {
        "frame_index": 0,
        "status": "processed",
        "objects": [entry],
        "npz_keys": {"depth": "f0_depth"},
    }
    video_path = video_dir / f"{video_name}.mp4"
    if with_video:
        _write_video(video_path)

    video_json = {
        "video_name": video_name,
        "video_path": str(video_path.resolve()) if with_video else None,
        "video_fps": 5.0,
        "frames": [frame],
    }
    (video_dir / f"{video_name}.json").write_text(json.dumps(video_json))
    np.savez(video_dir / f"{video_name}_arrays.npz", **npz_arrays)
    return video_dir / f"{video_name}.json"


def _read_csv_rows(csv_path: Path) -> list[dict]:
    import csv

    with csv_path.open(newline="") as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------------------------------
# per-camera recovery
# --------------------------------------------------------------------------------------

def test_per_camera_recovers_known_distance(tmp_path: Path):
    calib_dir = tmp_path / "calib"
    _write_camera_calib(calib_dir, "17_cam082")
    _write_pooled_calib(calib_dir)

    depth_bg = _background_depth()
    true_disp = (1.0 / depth_bg).astype(np.float64)
    animal_distance = 8.0
    true_disp[H // 2 - 4 : H // 2 + 4, W // 2 - 4 : W // 2 + 4] = 1.0 / animal_distance

    m0, c0 = 1.3, 0.05  # uniform affine miscalibration applied to the whole frame's disparity
    frame_disp = m0 * true_disp + c0
    depth_raw = 1.0 / frame_disp

    mask = _blob_mask(H // 2, W // 2)
    job_dir = tmp_path / "job"
    _make_job(job_dir, "17_Cam082_vid", depth_raw=depth_raw, mask=mask)

    out_dir = tmp_path / "out"
    code = applymod.main(["--job-dir", str(job_dir), "--calib-dir", str(calib_dir), "--out-dir", str(out_dir)])
    assert code == 0

    rows = _read_csv_rows(out_dir / "calibrated_objects.csv")
    assert len(rows) == 1
    row = rows[0]
    assert row["calib_method"] == "per_camera"
    assert row["transect_cam"] == "17_cam082"
    assert float(row["distance_m"]) == pytest.approx(animal_distance, abs=0.5)
    assert float(row["align_inlier_frac"]) > 0.9

    summary = json.loads((out_dir / "apply_summary.json").read_text())
    assert summary["by_method"]["per_camera"] == 1


def test_failure_path_all_excluded(tmp_path: Path):
    calib_dir = tmp_path / "calib"
    _write_camera_calib(calib_dir, "17_cam082", all_far=True)  # anchor depth == max_depth everywhere
    _write_pooled_calib(calib_dir)

    depth_raw = np.full((H, W), 8.0, dtype=np.float64)
    mask = _blob_mask(H // 2, W // 2)
    job_dir = tmp_path / "job"
    _make_job(job_dir, "17_Cam082_vid", depth_raw=depth_raw, mask=mask)

    out_dir = tmp_path / "out"
    code = applymod.main(["--job-dir", str(job_dir), "--calib-dir", str(calib_dir), "--out-dir", str(out_dir)])
    assert code == 0

    rows = _read_csv_rows(out_dir / "calibrated_objects.csv")
    assert len(rows) == 1
    assert rows[0]["calib_method"] == "failed"
    assert rows[0]["distance_m"] == "nan" or rows[0]["distance_m"] == ""

    summary = json.loads((out_dir / "apply_summary.json").read_text())
    assert summary["failures_by_reason"].get("too_few_pixels") == 1


# --------------------------------------------------------------------------------------
# pooled path
# --------------------------------------------------------------------------------------

def test_pooled_path_for_unknown_camera(tmp_path: Path):
    calib_dir = tmp_path / "calib"
    _write_pooled_calib(calib_dir)  # no per-camera npz at all

    known_distance = 6.0
    depth_raw = np.full((H, W), 100.0, dtype=np.float64)  # far background, clipped
    mask = _blob_mask(H // 2, W // 2)
    depth_raw[H // 2 - 4 : H // 2 + 4, W // 2 - 4 : W // 2 + 4] = known_distance

    job_dir = tmp_path / "job"
    _make_job(job_dir, "no_cam_pattern_video", depth_raw=depth_raw, mask=mask)

    out_dir = tmp_path / "out"
    code = applymod.main(["--job-dir", str(job_dir), "--calib-dir", str(calib_dir), "--out-dir", str(out_dir)])
    assert code == 0

    rows = _read_csv_rows(out_dir / "calibrated_objects.csv")
    assert len(rows) == 1
    assert rows[0]["calib_method"] == "pooled"
    assert float(rows[0]["distance_m"]) == pytest.approx(known_distance, abs=1e-3)


def test_force_pooled_overrides_known_camera(tmp_path: Path):
    calib_dir = tmp_path / "calib"
    _write_camera_calib(calib_dir, "17_cam082")
    _write_pooled_calib(calib_dir)

    known_distance = 6.0
    depth_raw = np.full((H, W), 100.0, dtype=np.float64)
    mask = _blob_mask(H // 2, W // 2)
    depth_raw[H // 2 - 4 : H // 2 + 4, W // 2 - 4 : W // 2 + 4] = known_distance

    job_dir = tmp_path / "job"
    _make_job(job_dir, "17_Cam082_vid", depth_raw=depth_raw, mask=mask)

    out_dir = tmp_path / "out"
    code = applymod.main(
        ["--job-dir", str(job_dir), "--calib-dir", str(calib_dir), "--out-dir", str(out_dir), "--force-pooled"]
    )
    assert code == 0
    rows = _read_csv_rows(out_dir / "calibrated_objects.csv")
    assert rows[0]["calib_method"] == "pooled"


# --------------------------------------------------------------------------------------
# depth array variations
# --------------------------------------------------------------------------------------

def test_float16_depth(tmp_path: Path):
    calib_dir = tmp_path / "calib"
    _write_camera_calib(calib_dir, "17_cam082")
    _write_pooled_calib(calib_dir)

    depth_bg = _background_depth()
    true_disp = (1.0 / depth_bg).astype(np.float64)
    animal_distance = 8.0
    true_disp[H // 2 - 4 : H // 2 + 4, W // 2 - 4 : W // 2 + 4] = 1.0 / animal_distance
    depth_raw = 1.0 / true_disp  # no affine miscalibration -> looser tolerance covers fp16 noise

    mask = _blob_mask(H // 2, W // 2)
    job_dir = tmp_path / "job"
    _make_job(job_dir, "17_Cam082_vid", depth_raw=depth_raw, mask=mask, depth_dtype=np.float16)

    out_dir = tmp_path / "out"
    code = applymod.main(["--job-dir", str(job_dir), "--calib-dir", str(calib_dir), "--out-dir", str(out_dir)])
    assert code == 0
    rows = _read_csv_rows(out_dir / "calibrated_objects.csv")
    assert rows[0]["calib_method"] == "per_camera"
    assert float(rows[0]["distance_m"]) == pytest.approx(animal_distance, abs=0.5)


def test_different_depth_resolution(tmp_path: Path):
    calib_dir = tmp_path / "calib"
    _write_camera_calib(calib_dir, "17_cam082")
    _write_pooled_calib(calib_dir)

    depth_bg = _background_depth()
    true_disp = (1.0 / depth_bg).astype(np.float64)
    animal_distance = 8.0
    true_disp[H // 2 - 4 : H // 2 + 4, W // 2 - 4 : W // 2 + 4] = 1.0 / animal_distance
    depth_raw_full = 1.0 / true_disp

    small_h, small_w = H // 2, W // 2
    depth_raw_small = cv2.resize(depth_raw_full.astype(np.float32), (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    mask_small = cv2.resize(_blob_mask(H // 2, W // 2), (small_w, small_h), interpolation=cv2.INTER_NEAREST)

    job_dir = tmp_path / "job"
    video_dir = job_dir / "17_Cam082_vid"
    video_dir.mkdir(parents=True)
    npz_arrays = {"f0_depth": depth_raw_small.astype(np.float32)}
    entry = build_mask_storage_entry(
        npz_arrays=npz_arrays, key_prefix="f0_obj_0_mask", mask=mask_small,
        base_entry={"object_index": 0, "track_id": 1, "bbox_xyxy": [0, 0, small_w, small_h]},
        storage_format="raw",
    )
    frame = {"frame_index": 0, "status": "processed", "objects": [entry], "npz_keys": {"depth": "f0_depth"}}
    video_json = {"video_name": "17_Cam082_vid", "video_path": None, "frames": [frame]}
    (video_dir / "17_Cam082_vid.json").write_text(json.dumps(video_json))
    np.savez(video_dir / "17_Cam082_vid_arrays.npz", **npz_arrays)

    out_dir = tmp_path / "out"
    code = applymod.main(["--job-dir", str(job_dir), "--calib-dir", str(calib_dir), "--out-dir", str(out_dir)])
    assert code == 0
    rows = _read_csv_rows(out_dir / "calibrated_objects.csv")
    assert rows[0]["calib_method"] == "per_camera"
    assert float(rows[0]["distance_m"]) == pytest.approx(animal_distance, abs=1.0)


# --------------------------------------------------------------------------------------
# extrinsic recalibration (fake matcher)
# --------------------------------------------------------------------------------------

def test_extrinsic_recalibration_identity_homography(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    calib_dir = tmp_path / "calib"
    _write_camera_calib(calib_dir, "17_cam082")
    _write_pooled_calib(calib_dir)

    depth_bg = _background_depth()
    true_disp = (1.0 / depth_bg).astype(np.float64)
    animal_distance = 8.0
    true_disp[H // 2 - 4 : H // 2 + 4, W // 2 - 4 : W // 2 + 4] = 1.0 / animal_distance
    m0, c0 = 1.3, 0.05
    frame_disp = m0 * true_disp + c0
    depth_raw = 1.0 / frame_disp
    mask = _blob_mask(H // 2, W // 2)

    job_dir = tmp_path / "job"
    _make_job(job_dir, "17_Cam082_vid", depth_raw=depth_raw, mask=mask, with_video=True)

    lightglue_weights = tmp_path / "lightglue.onnx"
    lightglue_weights.write_bytes(b"fake weights")  # never loaded: ExtrinsicRecalibrator is faked below

    class FakeRecalibrator:
        def __init__(self, lightglue_weights):
            pass

        def estimate(self, baseline_img, img):
            return extrinsic.HomographyEstimate(
                homography=np.eye(3), num_matches=50, num_inliers=40, inlier_ratio=0.9, reprojection_error=0.3,
            )

    monkeypatch.setattr(applymod.extrinsic, "ExtrinsicRecalibrator", FakeRecalibrator)

    out_dir = tmp_path / "out"
    code = applymod.main(
        ["--job-dir", str(job_dir), "--calib-dir", str(calib_dir), "--out-dir", str(out_dir),
         "--extrinsic-recalibration", "--lightglue-weights", str(lightglue_weights)]
    )
    assert code == 0
    rows = _read_csv_rows(out_dir / "calibrated_objects.csv")
    assert rows[0]["homography_used"] == "True"
    assert float(rows[0]["distance_m"]) == pytest.approx(animal_distance, abs=0.5)


def test_lightglue_weights_required_without_flag_errors(tmp_path: Path):
    calib_dir = tmp_path / "calib"
    _write_pooled_calib(calib_dir)
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    out_dir = tmp_path / "out"
    with pytest.raises(SystemExit):
        applymod.main(
            ["--job-dir", str(job_dir), "--calib-dir", str(calib_dir), "--out-dir", str(out_dir), "--extrinsic-recalibration"]
        )


def test_lightglue_weights_missing_file_errors(tmp_path: Path):
    calib_dir = tmp_path / "calib"
    _write_pooled_calib(calib_dir)
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    out_dir = tmp_path / "out"
    with pytest.raises(FileNotFoundError):
        applymod.main(
            [
                "--job-dir", str(job_dir), "--calib-dir", str(calib_dir), "--out-dir", str(out_dir),
                "--extrinsic-recalibration", "--lightglue-weights", str(tmp_path / "missing.onnx"),
            ]
        )


# --------------------------------------------------------------------------------------
# sharding + merge
# --------------------------------------------------------------------------------------

def test_sharding_and_merge(tmp_path: Path):
    calib_dir = tmp_path / "calib"
    _write_camera_calib(calib_dir, "17_cam082")
    _write_pooled_calib(calib_dir)

    depth_bg = _background_depth()
    animal_distance = 8.0

    def make_depth():
        true_disp = (1.0 / depth_bg).astype(np.float64)
        true_disp[H // 2 - 4 : H // 2 + 4, W // 2 - 4 : W // 2 + 4] = 1.0 / animal_distance
        return 1.0 / true_disp

    job_dir = tmp_path / "job"
    mask = _blob_mask(H // 2, W // 2)
    video_names = [f"17_Cam082_vid{i}" for i in range(4)]
    for name in video_names:
        _make_job(job_dir, name, depth_raw=make_depth(), mask=mask)

    out_dir = tmp_path / "out"
    for shard in (0, 1):
        code = applymod.main(
            ["--job-dir", str(job_dir), "--calib-dir", str(calib_dir), "--out-dir", str(out_dir),
             "--shard-index", str(shard), "--num-shards", "2"]
        )
        assert code == 0

    assert (out_dir / "calibrated_objects.shard0.csv").exists()
    assert (out_dir / "calibrated_objects.shard1.csv").exists()

    code = applymod.main(["--job-dir", str(job_dir), "--calib-dir", str(calib_dir), "--out-dir", str(out_dir), "--merge"])
    assert code == 0

    rows = _read_csv_rows(out_dir / "calibrated_objects.csv")
    assert len(rows) == 4
    assert {r["video_name"] for r in rows} == set(video_names)

    summary = json.loads((out_dir / "apply_summary.json").read_text())
    assert summary["n_objects"] == 4
    assert summary["by_method"]["per_camera"] == 4


# --------------------------------------------------------------------------------------
# resume
# --------------------------------------------------------------------------------------

def test_resume_skips_already_done_video(tmp_path: Path):
    calib_dir = tmp_path / "calib"
    _write_camera_calib(calib_dir, "17_cam082")
    _write_pooled_calib(calib_dir)

    depth_bg = _background_depth()
    true_disp = (1.0 / depth_bg).astype(np.float64)
    true_disp[H // 2 - 4 : H // 2 + 4, W // 2 - 4 : W // 2 + 4] = 1.0 / 8.0
    depth_raw = 1.0 / true_disp
    mask = _blob_mask(H // 2, W // 2)

    job_dir = tmp_path / "job"
    _make_job(job_dir, "17_Cam082_vid", depth_raw=depth_raw, mask=mask)

    out_dir = tmp_path / "out"
    applymod.main(["--job-dir", str(job_dir), "--calib-dir", str(calib_dir), "--out-dir", str(out_dir)])
    rows_first = _read_csv_rows(out_dir / "calibrated_objects.csv")
    assert len(rows_first) == 1

    code = applymod.main(["--job-dir", str(job_dir), "--calib-dir", str(calib_dir), "--out-dir", str(out_dir)])
    assert code == 0
    rows_second = _read_csv_rows(out_dir / "calibrated_objects.csv")
    assert len(rows_second) == 1  # not duplicated

    code = applymod.main(["--job-dir", str(job_dir), "--calib-dir", str(calib_dir), "--out-dir", str(out_dir), "--overwrite"])
    assert code == 0
    rows_overwrite = _read_csv_rows(out_dir / "calibrated_objects.csv")
    assert len(rows_overwrite) == 1
