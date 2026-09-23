from __future__ import annotations

import numpy as np
import pytest

from apps.camera_trap.calibration.timmh import (
    align_disparity,
    calibrate,
    depth_from_disparity,
    disparity_from_depth,
    interior_point,
    piecewise_from_knots,
    piecewise_linear_calibration,
    resize_to,
)


# --- calibrate ---------------------------------------------------------


def test_calibrate_ransac_recovers_line_with_outliers():
    rng = np.random.RandomState(0)
    x = np.linspace(0.1, 10, 100)
    y = 2 * x + 0.1
    y_noisy = y + rng.normal(0, 0.02, size=x.shape)

    n_outliers = 20
    outlier_idx = rng.choice(len(x), size=n_outliers, replace=False)
    y_noisy[outlier_idx] += rng.choice([-1, 1], size=n_outliers) * rng.uniform(5, 10, size=n_outliers)

    fn = calibrate(x, y_noisy, method="ransac")
    assert fn.m == pytest.approx(2.0, abs=0.1)
    assert fn.c == pytest.approx(0.1, abs=0.2)
    assert fn.inlier_frac > 0.7
    np.testing.assert_allclose(fn(x), fn.m * x + fn.c)


def test_calibrate_leastsquares_clean_data():
    x = np.linspace(0.1, 10, 50)
    y = 2 * x + 0.1
    fn = calibrate(x, y, method="leastsquares")
    assert fn.m == pytest.approx(2.0, abs=1e-8)
    assert fn.c == pytest.approx(0.1, abs=1e-8)
    assert fn.inlier_frac == 1.0


def test_calibrate_respects_masked_arrays():
    x = np.ma.array(np.linspace(0.1, 10, 50), mask=False)
    y = np.ma.array(2 * x.data + 0.1, mask=False)
    # corrupt some entries and mask them out
    x.mask[:10] = True
    y.data[:10] = 1000.0

    fn = calibrate(x, y, method="leastsquares")
    assert fn.m == pytest.approx(2.0, abs=1e-6)
    assert fn.c == pytest.approx(0.1, abs=1e-6)


# --- piecewise_linear_calibration --------------------------------------


def test_piecewise_interpolation():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([10.0, 20.0, 40.0])
    fn = piecewise_linear_calibration(x, y)
    assert fn(1.5) == pytest.approx(15.0)
    assert fn(2.5) == pytest.approx(30.0)


def test_piecewise_linear_extrapolation():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([10.0, 20.0, 30.0])
    fn = piecewise_linear_calibration(x, y)
    # slope 10 throughout -> extrapolate linearly (clipped to eps at/below 0)
    assert fn(0.0) == pytest.approx(1e-6, abs=1e-9)
    assert fn(4.0) == pytest.approx(40.0)


def test_piecewise_eps_clipping():
    x = np.array([1.0, 2.0])
    y = np.array([1.0, 2.0])
    fn = piecewise_linear_calibration(x, y, eps=1e-6)
    assert fn(-100.0) >= 1e-6
    assert fn(-100.0) == pytest.approx(1e-6)


def test_piecewise_duplicate_x_merged():
    x = np.array([1.0, 1.0, 2.0])
    y = np.array([10.0, 20.0, 40.0])
    fn = piecewise_linear_calibration(x, y)
    np.testing.assert_allclose(fn.knots_x, [1.0, 2.0])
    np.testing.assert_allclose(fn.knots_y, [15.0, 40.0])


def test_piecewise_single_point_constant():
    fn = piecewise_linear_calibration(np.array([5.0]), np.array([3.0]))
    assert fn(0.0) == pytest.approx(3.0)
    assert fn(100.0) == pytest.approx(3.0)


def test_piecewise_from_knots_reproduces():
    x = np.array([1.0, 2.0, 3.0, 5.0])
    y = np.array([1.0, 3.0, 4.0, 10.0])
    fn = piecewise_linear_calibration(x, y)
    fn2 = piecewise_from_knots(fn.knots_x, fn.knots_y)
    query = np.linspace(-5, 10, 50)
    np.testing.assert_allclose(fn(query), fn2(query))


# --- align_disparity -----------------------------------------------------


def test_align_disparity_recovers_anchor_on_background():
    rng = np.random.RandomState(1)
    h, w = 60, 80
    anchor = rng.uniform(1.0, 5.0, size=(h, w))

    true_m, true_c = 1.7, 0.05
    disp = (anchor - true_c) / true_m
    disp += rng.normal(0, 0.001, size=disp.shape)

    exclude = np.zeros((h, w), dtype=bool)
    exclude[20:35, 30:50] = True
    disp_wrong = disp.copy()
    disp_wrong[exclude] = 999.0

    aligned, info = align_disparity(disp_wrong, anchor, exclude, method="ransac", min_pixels=100)

    assert aligned is not None
    assert info["status"] == "ok"
    background = ~exclude
    np.testing.assert_allclose(aligned[background], anchor[background], atol=0.05)


def test_align_disparity_too_few_pixels():
    h, w = 20, 20
    disp = np.ones((h, w))
    anchor = np.ones((h, w))
    exclude = np.ones((h, w), dtype=bool)
    exclude[0, 0] = False  # only 1 valid pixel

    aligned, info = align_disparity(disp, anchor, exclude, min_pixels=500)
    assert aligned is None
    assert info["status"] == "too_few_pixels"


def test_resize_to_bool_uses_nearest():
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 2:5] = True
    resized = resize_to(mask, (20, 20))
    assert resized.dtype == bool
    assert resized.any()


def test_disparity_depth_roundtrip():
    depth = np.array([1.0, 2.0, 5.0])
    disp = disparity_from_depth(depth)
    np.testing.assert_allclose(disp, 1.0 / depth)
    back = depth_from_disparity(disp, min_depth=0.5, max_depth=10.0)
    np.testing.assert_allclose(back, depth)


# --- interior_point --------------------------------------------------------


def test_interior_point_rectangle_centre():
    mask = np.zeros((20, 30), dtype=bool)
    mask[5:15, 10:20] = True
    row, col = interior_point(mask)
    assert 9 <= row <= 10
    assert 14 <= col <= 15


def test_interior_point_l_shape_inside_mask():
    mask = np.zeros((20, 20), dtype=bool)
    mask[0:20, 0:5] = True
    mask[15:20, 0:20] = True
    row, col = interior_point(mask)
    assert mask[row, col]


def test_interior_point_empty_mask_returns_none():
    mask = np.zeros((10, 10), dtype=bool)
    assert interior_point(mask) is None
