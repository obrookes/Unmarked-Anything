from __future__ import annotations

import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apps.camera_trap.calibration import build_reference as br
from apps.camera_trap.calibration import timmh

RNG = np.random.default_rng(0)
SHAPE = (60, 80)  # H, W


# --------------------------------------------------------------------------------------
# synthetic instance builder: shared background "true" disparity field + a person blob at
# 1/distance, each instance run through its own (drifting) affine to mimic a fresh DA3 call.
# --------------------------------------------------------------------------------------

def _bg_field(shape=SHAPE) -> np.ndarray:
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    return 0.01 + 0.04 * (yy.astype(np.float64) / h)


BG = _bg_field()


def _mask_box(box, shape=SHAPE) -> np.ndarray:
    x0, y0, x1, y1 = box
    mask = np.zeros(shape, dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def _make_instance(
    distance_m: float,
    box=(10, 10, 25, 25),
    *,
    m: float = 1.0,
    c: float = 0.0,
    vlm_conf: float = 1.0,
    event_id: str = "e0",
    frame_idx: int = 0,
    video_path: str = "v.mp4",
    marker: int = 0,
) -> dict:
    true_disp = BG.copy()
    mask = _mask_box(box)
    true_disp[mask] = 1.0 / distance_m
    disp_raw = m * true_disp + c
    frame_bgr = np.full((*SHAPE, 3), marker, dtype=np.uint8)
    return {
        "event_id": event_id,
        "video_path": video_path,
        "frame_idx": frame_idx,
        "distance_m": distance_m,
        "vlm_conf": vlm_conf,
        "mask_score": 0.9,
        "mask_area_px": int(mask.sum()),
        "disp": disp_raw,
        "person_mask": mask,
        "frame_bgr": frame_bgr,
    }


# --------------------------------------------------------------------------------------
# build_camera_calibration
# --------------------------------------------------------------------------------------

def test_recovers_distances_with_per_instance_affine_drift():
    distances = [3.0, 6.0, 10.0]
    instances = []
    for d in distances:
        for rep in range(2):
            m = RNG.uniform(0.5, 2.0)
            c = RNG.uniform(-0.01, 0.01)
            box = (10 + rep * 30, 10, 25 + rep * 30, 25)
            instances.append(_make_instance(d, box=box, m=m, c=c, event_id=f"e{d}_{rep}"))

    # This fixture uses a fixed-size box regardless of distance_m (it's only testing recovery
    # from per-instance affine drift), so it doesn't model mask_area ~ 1/distance^2, and it
    # deliberately gives every instance a wildly different raw-disparity affine (m, c) despite
    # tagging them all with the same "video" -- neither the size- nor alignment-consistency
    # filter's assumptions hold here, so loosen both.
    npz_dict, rows, summary = br.build_camera_calibration(
        instances, transect_cam="camA", size_tolerance=1e6, align_tolerance=1e6
    )

    assert summary["method"] == "per_camera"
    assert all(r["status"] == "ok" for r in rows)
    assert summary["loo_mae_m"] < 0.5
    assert summary["insample_mae_m"] < 0.5


def test_anchor_is_smallest_area_tie_broken_by_vlm_conf():
    small_low_conf = _make_instance(10.0, box=(10, 10, 20, 20), vlm_conf=0.2, marker=1)
    small_high_conf = _make_instance(10.0, box=(40, 10, 50, 20), vlm_conf=0.9, marker=2)
    large = _make_instance(5.0, box=(10, 30, 40, 60), vlm_conf=1.0, marker=3)

    npz_dict, rows, summary = br.build_camera_calibration(
        [small_low_conf, large, small_high_conf], transect_cam="camB"
    )

    assert summary["anchor_distance_m"] == 10.0
    assert npz_dict["anchor_img"][0, 0, 0] == 2  # small_high_conf's marker, not small_low_conf's


def test_align_failure_when_mask_covers_almost_whole_frame():
    anchor = _make_instance(10.0, box=(10, 10, 25, 25))
    # Mask covering nearly the whole frame leaves < min_pixels(500) background pixels once
    # unioned with the anchor's mask, so align_disparity must fail for this instance.
    huge_box = (0, 0, SHAPE[1] - 1, SHAPE[0] - 1)
    bad = _make_instance(5.0, box=huge_box, event_id="bad")

    npz_dict, rows, summary = br.build_camera_calibration([anchor, bad], transect_cam="camC")

    bad_row = next(r for r in rows if r["event_id"] == "bad")
    assert bad_row["status"] == "align_failed"
    assert np.isnan(bad_row["x_aligned"])
    assert not np.isnan(bad_row["x_raw"])  # raw median is unaffected by alignment


def test_single_distance_camera_falls_back_to_pooled():
    instances = [
        _make_instance(8.0, box=(10, 10, 25, 25), event_id="a"),
        _make_instance(8.0, box=(40, 10, 55, 25), event_id="b"),
    ]
    npz_dict, rows, summary = br.build_camera_calibration(instances, transect_cam="camD")

    assert summary["method"] == "pooled"
    assert summary["loo_mae_m"] is None
    assert npz_dict["knots_x"] is None
    assert npz_dict["knots_y"] is None
    assert npz_dict["anchor_distance_m"] == 8.0
    assert npz_dict["calib_method"] == "pooled_anchor"


def test_no_instances_returns_none_npz():
    npz_dict, rows, summary = br.build_camera_calibration([], transect_cam="camE")
    assert npz_dict is None
    assert rows == []
    assert summary["n_instances"] == 0
    assert summary["method"] == "pooled"


def test_npz_keys_exact_for_per_camera_method():
    # per_camera now requires >=3 distinct distances (a validated, cross-checked curve).
    distances = [4.0, 9.0, 14.0]
    instances = [_make_instance(d, box=(10 + i * 20, 10, 20 + i * 20, 20)) for i, d in enumerate(distances)]
    npz_dict, rows, summary = br.build_camera_calibration(
        instances, transect_cam="camF", da3_model_id="foo", size_tolerance=1e6
    )

    expected_keys = {
        "anchor_disp_raw", "anchor_img", "anchor_person_mask", "knots_x", "knots_y",
        "max_depth", "min_depth", "anchor_distance_m", "da3_model_id", "calib_method",
    }
    assert set(npz_dict.keys()) == expected_keys
    assert npz_dict["anchor_disp_raw"].dtype == np.float32
    assert npz_dict["anchor_person_mask"].dtype == bool
    assert npz_dict["da3_model_id"] == "foo"
    assert npz_dict["calib_method"] == "per_camera"


# --------------------------------------------------------------------------------------
# robust-calibration filters (size/alignment/disparity consistency, anchor, fallback)
# --------------------------------------------------------------------------------------

def test_size_filter_rejects_vlm_misreads_and_keeps_good_fit():
    # Genuine instances: mask_area ~ 1/distance^2 (k = distance*sqrt(area) ~ constant ~40).
    genuine = [
        (2.0, (2, 5, 22, 25)),    # area 400
        (4.0, (25, 5, 35, 15)),   # area 100
        (6.0, (38, 5, 45, 12)),   # area 49
        (8.0, (48, 5, 53, 10)),   # area 25
        (12.0, (2, 32, 5, 35)),   # area 9
    ]
    # VLM misreads: a real (small) mask paired with a "2.0 m" label, and a mid-size mask (like a
    # real near-distance person) mislabeled "24.0 m", mirroring the reported 182_cam068 bug.
    misreads = [
        (2.0, (10, 32, 12, 34)),    # area 4 -- far too small for 2 m
        (24.0, (16, 32, 26, 42)),   # area 100 -- far too big for 24 m
    ]
    instances = [_make_instance(d, box=box, event_id=f"good_{d}") for d, box in genuine]
    instances += [_make_instance(d, box=box, event_id=f"misread_{i}") for i, (d, box) in enumerate(misreads)]

    npz_dict, rows, summary = br.build_camera_calibration(instances, transect_cam="camMix")

    misread_rows = [r for r in rows if r["event_id"].startswith("misread")]
    good_rows = [r for r in rows if r["event_id"].startswith("good")]
    assert all(r["status"] == "size_inconsistent" for r in misread_rows)
    assert all(r["status"] == "ok" for r in good_rows)
    assert summary["method"] == "per_camera"
    assert summary["n_size_rejected"] == 2
    assert summary["loo_mae_m"] < 1.0


def test_fewer_than_3_instances_skips_size_filter():
    # Wildly inconsistent k (tiny mask far outside what its distance implies) would normally be
    # rejected, but with only 2 instances the size filter must not run at all.
    tiny = _make_instance(2.0, box=(10, 10, 12, 12), event_id="tiny")  # area 4
    normal = _make_instance(6.0, box=(10, 30, 40, 60), event_id="normal")  # area 900

    npz_dict, rows, summary = br.build_camera_calibration([tiny, normal], transect_cam="camTiny")

    assert all(r["status"] != "size_inconsistent" for r in rows)
    assert summary["n_size_rejected"] == 0


def test_loo_mae_over_threshold_falls_back_to_pooled():
    rows = [
        {"distance_m": 3.0, "mask_area_px": 1000.0, "x_aligned": 0.55, "x_raw": 0.55, "video": "v"},
        {"distance_m": 3.0, "mask_area_px": 1000.0, "x_aligned": 0.20, "x_raw": 0.20, "video": "v"},
        {"distance_m": 6.0, "mask_area_px": 250.0, "x_aligned": 0.25, "x_raw": 0.25, "video": "v"},
        {"distance_m": 10.0, "mask_area_px": 90.0, "x_aligned": 0.15, "x_raw": 0.15, "video": "v"},
    ]
    rows_out, summary = br.calibrate_camera_from_rows(
        rows, size_tolerance=1e6, align_tolerance=1e6, disp_tolerance=1e6
    )
    assert summary["method"] == "pooled"
    assert "loo_mae>3.0" in summary["fallback_reason"]
    assert summary["knots_x"] is None and summary["knots_y"] is None


def test_isotonic_pava_recovers_from_noisy_far_reversal():
    # A clean x ~= 1/distance relationship except a small reversal between two nearby far
    # distances (10 m, 11 m) -- the kind of measurement noise expected when disparity is nearly
    # flat out there (x should fall monotonically with distance; here it briefly rises). The
    # isotonic PAVA step must pool that one violating pair (n_isotonic_pooled == 1) into a single
    # knot rather than rejecting the whole camera as non-monotone.
    x_by_dist = {
        1.0: 1.000, 2.0: 0.500, 3.0: 0.333, 4.0: 0.250, 5.0: 0.200,
        10.0: 0.095, 11.0: 0.100, 12.0: 0.083,
    }
    rows = [
        {"distance_m": d, "mask_area_px": 1000.0, "x_aligned": x, "x_raw": x, "video": "v"}
        for d, x in x_by_dist.items()
    ]
    rows_out, summary = br.calibrate_camera_from_rows(
        rows, size_tolerance=1e6, align_tolerance=1e6, disp_tolerance=1e6
    )
    assert summary["method"] == "per_camera"
    assert summary["n_isotonic_pooled"] == 1
    assert summary["loo_mae_m"] is not None and summary["loo_mae_m"] < 3.0


def test_isotonic_collapse_falls_back_too_few_distances():
    # 3 distinct distances is the bare minimum; if isotonic pooling merges them all the way down
    # to <3 knots the camera must still fall back (sanity net for a degenerate case), with the new
    # reason string (non_monotone is no longer used at all).
    rows = [
        {"distance_m": 3.0, "mask_area_px": 1000.0, "x_aligned": 0.15, "x_raw": 0.15, "video": "v"},
        {"distance_m": 6.0, "mask_area_px": 250.0, "x_aligned": 0.35, "x_raw": 0.35, "video": "v"},
        {"distance_m": 10.0, "mask_area_px": 90.0, "x_aligned": 0.10, "x_raw": 0.10, "video": "v"},
    ]
    rows_out, summary = br.calibrate_camera_from_rows(
        rows, size_tolerance=1e6, align_tolerance=1e6, disp_tolerance=1e6
    )
    assert summary["method"] == "pooled"
    assert "too_few_distances_after_isotonic" in summary["fallback_reason"]
    assert "non_monotone" not in summary["fallback_reason"]


def test_loo_truncated_to_near_distances_by_default():
    # Near points (<=12 m, the CTDS-truncation default) exactly follow x = 1/distance; one 20 m
    # point is deliberately inconsistent with that relationship (its own held-out LOO prediction,
    # extrapolated from the near curve, is wildly off) but must not blow up the gated loo_mae_m
    # since --loo-max-distance defaults to 12.0 m.
    near = {2.0: 0.5, 4.0: 0.25, 6.0: 1 / 6, 8.0: 0.125, 10.0: 0.1, 12.0: 1 / 12}
    rows = [
        {"distance_m": d, "mask_area_px": 1000.0, "x_aligned": x, "x_raw": x, "video": "v"}
        for d, x in near.items()
    ]
    rows.append({"distance_m": 20.0, "mask_area_px": 1000.0, "x_aligned": 0.02, "x_raw": 0.02, "video": "v"})

    rows_out, summary = br.calibrate_camera_from_rows(
        rows, size_tolerance=1e6, align_tolerance=1e6, disp_tolerance=1e6
    )
    assert summary["method"] == "per_camera"
    assert summary["loo_mae_m"] is not None and summary["loo_mae_m"] < 3.0
    assert summary["loo_mae_all_m"] is not None and summary["loo_mae_all_m"] > summary["loo_mae_m"]


def test_align_tolerance_default_keeps_near_frames_with_proximity_scale_drift():
    # DA3's per-frame alignment scale genuinely shifts when a near person fills the frame: model
    # that as x_aligned/x_raw = 1.7 for near frames (2-3 m) vs. 1.0 for far frames (6-14 m), all
    # sharing one video/camera group. The underlying x_aligned itself still cleanly follows
    # x = 1/distance (a good, fittable curve) -- only the alignment-consistency ratio differs.
    ratio_by_dist = {2.0: 1.7, 3.0: 1.7, 6.0: 1.0, 8.0: 1.0, 10.0: 1.0, 12.0: 1.0, 14.0: 1.0}
    rows = [
        {
            "distance_m": d, "mask_area_px": 1000.0,
            "x_aligned": 1.0 / d, "x_raw": (1.0 / d) / ratio, "video": "v",
        }
        for d, ratio in ratio_by_dist.items()
    ]

    # Default --align-tolerance (3.0): near frames' r = log(1.7) deviates from the far-dominated
    # group median (log(1.0) = 0) by log(1.7) ~= 0.53, well under log(3.0) ~= 1.10 -- kept.
    rows_default, summary_default = br.calibrate_camera_from_rows(
        [dict(r) for r in rows], size_tolerance=1e6, disp_tolerance=1e6,
    )
    near_default = [r for r in rows_default if r["distance_m"] in (2.0, 3.0)]
    assert all(r["status"] == "ok" for r in near_default)
    assert summary_default["method"] == "per_camera"

    # The OLD default (1.5): log(1.7) ~= 0.53 > log(1.5) ~= 0.41 -- near frames get rejected.
    rows_old, _ = br.calibrate_camera_from_rows(
        [dict(r) for r in rows], size_tolerance=1e6, disp_tolerance=1e6, align_tolerance=1.5,
    )
    near_old = [r for r in rows_old if r["distance_m"] in (2.0, 3.0)]
    assert all(r["status"] == "align_inconsistent" for r in near_old)


def test_video_scale_normalization_recovers_per_video_factors():
    # 3 videos on one camera, all sharing the same true x = 1/distance relationship, but each
    # video's x_aligned carries a different residual multiplicative scale (a_v): the reference
    # video (most distinct distances -> video "ref") at 1.0, "lo" at 0.7, "hi" at 1.3.
    scale_by_video = {"ref": 1.0, "lo": 0.7, "hi": 1.3}
    dists_by_video = {"ref": [2.0, 4.0, 6.0, 8.0, 10.0], "lo": [2.0, 4.0, 6.0], "hi": [4.0, 6.0, 8.0]}
    rows = []
    for video, dists in dists_by_video.items():
        a_v = scale_by_video[video]
        for d in dists:
            x = a_v / d
            rows.append({
                "distance_m": d, "mask_area_px": 1000.0, "x_aligned": x, "x_raw": x, "video": video,
            })

    rows_out, summary = br.calibrate_camera_from_rows(
        rows, size_tolerance=1e6, align_tolerance=1e6, disp_tolerance=1e6,
    )

    assert all(r["status"] == "ok" for r in rows_out)
    assert summary["method"] == "per_camera"
    assert summary["n_videos_scaled"] == 2

    recovered = {r["video"]: r["video_scale"] for r in rows_out}
    for video, true_scale in scale_by_video.items():
        assert recovered[video] == pytest.approx(true_scale, rel=0.05)


def test_pooled_excludes_size_and_disp_inconsistent_rows():
    rows = [
        {"transect_cam": "c", "status": "ok", "distance_m": 4.0, "x_raw": 0.25},
        {"transect_cam": "c", "status": "align_failed", "distance_m": 4.0, "x_raw": float("nan")},
        {"transect_cam": "c", "status": "size_inconsistent", "distance_m": 4.0, "x_raw": 999.0},
        {"transect_cam": "c", "status": "disp_inconsistent", "distance_m": 4.0, "x_raw": -999.0},
        {"transect_cam": "c", "status": "align_inconsistent", "distance_m": 4.0, "x_raw": -999.0},
    ]
    pooled = br.build_pooled(rows)
    assert pooled is not None
    assert pooled["n_instances"] == 1  # only the "ok" row


# --------------------------------------------------------------------------------------
# build_pooled
# --------------------------------------------------------------------------------------

def test_build_pooled_recovers_distance_relationship_with_no_drift():
    rows = []
    for cam, dists in {"cam1": [4.0, 8.0], "cam2": [4.0, 8.0, 12.0]}.items():
        for d in dists:
            rows.append({
                "transect_cam": cam, "status": "ok", "distance_m": d, "x_raw": 1.0 / d + 0.001,
            })

    pooled = br.build_pooled(rows)
    assert pooled is not None
    assert set(pooled.keys()) == {"knots_x", "knots_y", "max_depth", "min_depth", "n_instances", "n_cameras"}
    assert pooled["n_instances"] == 5
    assert pooled["n_cameras"] == 2

    curve = timmh.piecewise_from_knots(pooled["knots_x"], pooled["knots_y"])
    for d in (4.0, 8.0, 12.0):
        pred_depth = 1.0 / curve(np.array([1.0 / d]))[0]
        assert pred_depth == pytest.approx(d, abs=0.2)


def test_build_pooled_empty_returns_none():
    assert br.build_pooled([]) is None
    assert br.build_pooled([{"transect_cam": "c", "status": "no_mask", "distance_m": 1.0, "x_raw": float("nan")}]) is None


# --------------------------------------------------------------------------------------
# mask selection by sign box overlap
# --------------------------------------------------------------------------------------

def test_select_person_mask_picks_best_sign_box_overlap():
    shape = SHAPE
    mask_a = _mask_box((0, 0, 10, 10), shape)
    mask_b = _mask_box((30, 20, 50, 40), shape)
    masks = [
        (mask_a, 0.95, [0, 0, 10, 10]),
        (mask_b, 0.5, [30, 20, 50, 40]),
    ]
    # sign box centered on mask_b's region.
    sign_box_norm = (32 / shape[1], 22 / shape[0], 48 / shape[1], 38 / shape[0])

    chosen = br.select_person_mask(masks, sign_box_norm, shape)
    assert chosen is not None
    assert np.array_equal(chosen[0], mask_b)


def test_select_person_mask_falls_back_to_highest_score_when_no_overlap():
    shape = SHAPE
    mask_a = _mask_box((0, 0, 10, 10), shape)
    mask_b = _mask_box((30, 20, 50, 40), shape)
    masks = [
        (mask_a, 0.3, [0, 0, 10, 10]),
        (mask_b, 0.8, [30, 20, 50, 40]),
    ]
    # sign box far from both masks.
    sign_box_norm = (0.9, 0.9, 0.99, 0.99)

    chosen = br.select_person_mask(masks, sign_box_norm, shape)
    assert np.array_equal(chosen[0], mask_b)  # higher score wins the fallback


def test_select_person_mask_empty_returns_none():
    assert br.select_person_mask([], (0.0, 0.0, 1.0, 1.0), SHAPE) is None


# --------------------------------------------------------------------------------------
# CLI end-to-end with fake segmenter + fake DA3, on tiny cv2 videos
# --------------------------------------------------------------------------------------

FRAME_W, FRAME_H = 80, 60
PERSON_BOX = (30, 20, 50, 40)  # x0, y0, x1, y1


def _write_video(path: Path, distance_by_frame: dict[int, float], n_frames: int) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(path), fourcc, 5.0, (FRAME_W, FRAME_H))
    for i in range(n_frames):
        d = distance_by_frame.get(i, 0.0)
        val = int(round(d * 10))
        frame = np.full((FRAME_H, FRAME_W, 3), val, dtype=np.uint8)
        vw.write(frame)
    vw.release()


class _FakeDA3:
    def to(self, device):
        return self

    def inference(self, frames_rgb, use_ray_pose=False, infer_gs=False, export_dir=None):
        depths = []
        yy, _ = np.mgrid[0:FRAME_H, 0:FRAME_W]
        bg_disp = 0.01 + 0.04 * (yy.astype(np.float64) / FRAME_H)  # non-constant so alignment can fit
        bg_depth = 1.0 / bg_disp
        for f in frames_rgb:
            d = float(f[0, 0, 0]) / 10.0
            depth = bg_depth.astype(np.float32).copy()
            x0, y0, x1, y1 = PERSON_BOX
            depth[y0:y1, x0:x1] = d
            depths.append(depth)
        return SimpleNamespace(depth=depths)


def _fake_segmenter(frame_bgr, prompt):
    x0, y0, x1, y1 = PERSON_BOX
    mask = np.zeros(frame_bgr.shape[:2], dtype=bool)
    mask[y0:y1, x0:x1] = True
    return [(mask, 0.9, [x0, y0, x1, y1])]


def _sign_box_norm_str() -> str:
    x0, y0, x1, y1 = PERSON_BOX
    return f"{x0 / FRAME_W};{y0 / FRAME_H};{x1 / FRAME_W};{y1 / FRAME_H}"


def test_cli_end_to_end_with_fakes(tmp_path, monkeypatch):
    ref_root = tmp_path / "ref"
    ref_root.mkdir()
    video_a = ref_root / "camA_vid.mp4"
    video_b = ref_root / "camB_vid.mp4"
    # per_camera now requires >=3 distinct distances (a validated, cross-checked curve), so camA
    # needs a 3rd frame/distance.
    _write_video(video_a, {0: 5.0, 1: 9.0, 2: 14.0}, n_frames=3)
    _write_video(video_b, {0: 6.0}, n_frames=1)

    calib_csv = tmp_path / "calibration_frames.csv"
    with calib_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "transect_cam", "video_path", "event_id", "frame_idx", "distance_m",
            "sign_box_norm", "n_agree", "vlm_conf", "source",
        ])
        writer.writerow(["camA", "camA_vid.mp4", "ev1", 0, 5.0, _sign_box_norm_str(), 2, 0.9, "vlm"])
        writer.writerow(["camA", "camA_vid.mp4", "ev2", 1, 9.0, _sign_box_norm_str(), 2, 0.9, "vlm"])
        writer.writerow(["camA", "camA_vid.mp4", "ev4", 2, 14.0, _sign_box_norm_str(), 2, 0.9, "vlm"])
        writer.writerow(["camB", "camB_vid.mp4", "ev3", 0, 6.0, _sign_box_norm_str(), 2, 0.9, "vlm"])

    cameras_csv = tmp_path / "cameras.csv"
    with cameras_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["transect_cam"])
        writer.writerow(["camA"])
        writer.writerow(["camB"])
        writer.writerow(["camC"])  # no calibration instances at all

    out_dir = tmp_path / "out"

    monkeypatch.setattr(br, "_make_segmenter", lambda args: _fake_segmenter)
    monkeypatch.setattr(
        "depth_anything_3.api.DepthAnything3.from_pretrained",
        staticmethod(lambda *a, **k: _FakeDA3()),
    )

    br.main([
        "--calibration-frames", str(calib_csv),
        "--reference-root", str(ref_root),
        "--out-dir", str(out_dir),
        "--sam3-model-path", "unused.pt",
        "--da3-batch-size", "4",
        "--device", "cpu",
        "--cameras", str(cameras_csv),
    ])

    assert (out_dir / "calib" / "camA.npz").exists()
    assert (out_dir / "calib" / "_pooled.npz").exists()
    assert (out_dir / "calibration_summary.csv").exists()
    assert (out_dir / "calibration_instances.csv").exists()
    assert (out_dir / "calibration_plots" / "camA.png").exists()
    assert list((out_dir / "instances" / "camA").glob("*.jpg"))

    npz = np.load(out_dir / "calib" / "camA.npz")
    for key in ("anchor_disp_raw", "anchor_img", "anchor_person_mask", "knots_x", "knots_y",
                "max_depth", "min_depth", "anchor_distance_m", "da3_model_id", "calib_method"):
        assert key in npz.files
    assert str(npz["calib_method"]) == "per_camera"

    with (out_dir / "calibration_summary.csv").open() as f:
        summary_rows = list(csv.DictReader(f))
    cams_in_summary = {r["transect_cam"] for r in summary_rows}
    assert cams_in_summary == {"camA", "camB", "camC"}
    cam_a_row = next(r for r in summary_rows if r["transect_cam"] == "camA")
    assert cam_a_row["method"] == "per_camera"
    cam_c_row = next(r for r in summary_rows if r["transect_cam"] == "camC")
    assert cam_c_row["n_instances"] == "0"


def test_cli_resume_skips_existing_camera_npz(tmp_path, monkeypatch):
    ref_root = tmp_path / "ref"
    ref_root.mkdir()
    video_a = ref_root / "camA_vid.mp4"
    _write_video(video_a, {0: 5.0, 1: 10.0}, n_frames=2)

    calib_csv = tmp_path / "calibration_frames.csv"
    with calib_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "transect_cam", "video_path", "event_id", "frame_idx", "distance_m",
            "sign_box_norm", "n_agree", "vlm_conf", "source",
        ])
        writer.writerow(["camA", "camA_vid.mp4", "ev1", 0, 5.0, _sign_box_norm_str(), 2, 0.9, "vlm"])
        writer.writerow(["camA", "camA_vid.mp4", "ev2", 1, 10.0, _sign_box_norm_str(), 2, 0.9, "vlm"])

    out_dir = tmp_path / "out"

    monkeypatch.setattr(br, "_make_segmenter", lambda args: _fake_segmenter)
    monkeypatch.setattr(
        "depth_anything_3.api.DepthAnything3.from_pretrained",
        staticmethod(lambda *a, **k: _FakeDA3()),
    )

    argv = [
        "--calibration-frames", str(calib_csv),
        "--reference-root", str(ref_root),
        "--out-dir", str(out_dir),
        "--sam3-model-path", "unused.pt",
        "--da3-batch-size", "4",
        "--device", "cpu",
    ]
    br.main(argv)
    npz_path = out_dir / "calib" / "camA.npz"
    mtime_before = npz_path.stat().st_mtime_ns

    # Second run without --overwrite must not touch camA's npz, even though _make_segmenter/DA3
    # would raise if actually invoked (proves the camera's GPU work was skipped).
    def _boom(args):
        raise AssertionError("segmenter should not be constructed on a resumed run")

    monkeypatch.setattr(br, "_make_segmenter", _boom)
    br.main(argv)
    assert npz_path.stat().st_mtime_ns == mtime_before


# --------------------------------------------------------------------------------------
# regression tests against real Isambard-run instances CSVs (no disparity maps recorded, so this
# exercises `calibrate_camera_from_rows`, the pure filter+fit layer, directly on each fixture's
# already-computed distance_m/mask_area_px/x_aligned/x_raw columns; 'video' is derived from
# event_id's "<video>_<event>" prefix). Parametrised over fixture files so another CSV of
# real-world cameras can be dropped in later by adding one FIXTURE_EXPECTATIONS entry.
# --------------------------------------------------------------------------------------

def _load_fixture_rows(fixture_name: str) -> dict[str, list[dict]]:
    fixture_path = REPO_ROOT / "tests" / "fixtures" / fixture_name
    by_cam: dict[str, list[dict]] = {}
    with fixture_path.open(newline="") as f:
        for r in csv.DictReader(f):
            # A missing/NaN x_aligned means the instance was never aligned (e.g. dropped by the
            # old run's size filter before alignment even ran) -- unusable here regardless of the
            # old run's recorded status.
            raw = r.get("x_aligned", "")
            if raw in ("", "nan"):
                continue
            x_aligned = float(raw)
            if not np.isfinite(x_aligned):
                continue
            x_raw_raw = r.get("x_raw", "")
            x_raw = float(x_raw_raw) if x_raw_raw not in ("", "nan") else float("nan")
            by_cam.setdefault(r["transect_cam"], []).append({
                "distance_m": float(r["distance_m"]),
                "mask_area_px": float(r["mask_area_px"]) if r.get("mask_area_px") not in ("", None) else None,
                "x_aligned": x_aligned,
                "x_raw": x_raw,
                "video": r["event_id"].split("_")[0],
            })
    return by_cam


# Per fixture, per camera: {} means "no specific expectation beyond the global invariants below
# (still per_camera => loo_mae_m < 3.0)"; {"method": ...} pins the exact outcome.
FIXTURE_EXPECTATIONS: dict[str, dict[str, dict]] = {
    "calibration_instances_smoke.csv": {
        "124_cam106": {"method": "per_camera"},
        "15_cam126": {"method": "per_camera"},
        "25_cam076": {"method": "per_camera"},
        "182_cam068": {},  # formerly broken; per_camera (validated) or pooled both acceptable
        "3_cam006": {},
    },
    "calibration_instances_fallback.csv": {
        # Clean 1->12 m and 14->1 m walks; must now recover as per_camera.
        "86_cam145": {"method": "per_camera"},
        "102_cam014": {},
        "103_cam008": {},
        "135_cam025": {},
        "137_cam123": {},
        "59_cam129": {},
        "75_cam093": {},
    },
}


@pytest.mark.parametrize("fixture_name", sorted(FIXTURE_EXPECTATIONS))
def test_fixture_recovers_expected_calibration(fixture_name):
    by_cam = _load_fixture_rows(fixture_name)
    expected = FIXTURE_EXPECTATIONS[fixture_name]
    assert set(by_cam) == set(expected)

    results = {}
    for cam, rows in by_cam.items():
        _, summary = br.calibrate_camera_from_rows(rows)
        results[cam] = summary
        print(
            f"{fixture_name} {cam}: method={summary['method']} loo_mae_m={summary['loo_mae_m']} "
            f"fallback_reason={summary['fallback_reason']!r}"
        )

    for cam, s in results.items():
        # Global invariant: nothing may end up per_camera with a blown-up (or unmeasured) LOO MAE.
        if s["method"] == "per_camera":
            assert s["loo_mae_m"] is not None and np.isfinite(s["loo_mae_m"]) and s["loo_mae_m"] < 3.0
        else:
            assert s["fallback_reason"] != ""

        want = expected[cam].get("method")
        if want is not None:
            assert s["method"] == want, f"{cam}: expected method={want!r}, got {s['method']!r}"
