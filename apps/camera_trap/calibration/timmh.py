"""Calibration primitives ported from Haucke et al. 2022, distance-estimation.

Ported (disparity-space only, i.e. timmh's ``calibrate_metric=False`` / ``exp=1``) from:
  https://github.com/timmh/distance-estimation (utils.py, run.py)

MIT License

Copyright (c) 2019 Timm Haucke and contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Optional

import cv2
import numpy as np
import scipy.ndimage as ndimage
from sklearn import linear_model

RANDOM_SEED = 42


@contextmanager
def random_seed_manager(seed: int = RANDOM_SEED):
    """Deterministically seed numpy's global RNG for the duration of a block, then restore it.

    Port of timmh utils.random_seed_manager (utils.py:25).
    """
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def _split_mask(arr):
    """Return the boolean mask of a plain or masked array (all-False if unmasked)."""
    mask = arr.mask if hasattr(arr, "mask") else np.zeros_like(arr, dtype=bool)
    if mask.shape == ():
        mask = np.full(np.asarray(arr).shape, bool(mask), dtype=bool)
    return mask


def calibrate(x, y, method: str = "ransac", n: int = 2) -> Callable[[np.ndarray], np.ndarray]:
    """Fit y = m * x + c and return the calibration function.

    Port of timmh utils.calibrate (utils.py:101-183), restricted to the "ransac" and
    "leastsquares" methods (POLY / RANSAC_POLY are out of scope here).
    """
    assert n in (1, 2)
    assert method in ("ransac", "leastsquares")
    assert len(x) >= 2 and len(y) >= 2 and len(x) == len(y), (
        f"inconsistent sample length in calibration: len(x)={len(x)}, len(y)={len(y)}"
    )

    with random_seed_manager():
        x_mask = _split_mask(x)
        y_mask = _split_mask(y)
        mask = x_mask | y_mask

        x = np.asarray(x)[~mask]
        y = np.asarray(y)[~mask]
        x, y = x.reshape(-1), y.reshape(-1)

        if method == "ransac":
            def is_model_valid(model, X_, y_):
                return (model.coef_ > 0).all()

            estimator = linear_model.LinearRegression(positive=True)
            ransac = linear_model.RANSACRegressor(
                estimator=estimator, is_model_valid=is_model_valid, random_state=RANDOM_SEED
            )
            ransac.fit(np.array(x).reshape(-1, 1), np.array(y).reshape(-1, 1))
            c = ransac.predict(np.array([0]).reshape(-1, 1)) if n == 2 else 0
            m = ransac.predict(np.array([1]).reshape(-1, 1)) - c
            m, c = (m.item() if hasattr(m, "item") else float(m)), (c.item() if hasattr(c, "item") else float(c))

            fn = lambda data: m * np.asarray(data) + c  # noqa: E731
            fn.m = m
            fn.c = c
            fn.inlier_frac = float(np.mean(ransac.inlier_mask_))
            return fn

        # method == "leastsquares"
        try:
            if n == 2:
                A = np.vstack([x, np.ones(len(x))]).T
                m, c = np.linalg.lstsq(A, y, rcond=None)[0]
            else:
                c = 0
                A = x.T
                m = np.linalg.lstsq(A, y, rcond=None)[0]
            m, c = float(np.asarray(m).item()), float(np.asarray(c).item())

            fn = lambda data: m * np.asarray(data) + c  # noqa: E731
            fn.m = m
            fn.c = c
            fn.inlier_frac = 1.0
            return fn
        except np.linalg.LinAlgError:
            return calibrate(x, y, method="ransac", n=n)


def piecewise_linear_calibration(x, y, eps: float = 1e-6) -> Callable[[np.ndarray], np.ndarray]:
    """Create a monotonic piecewise-linear calibration function y = f(x).

    Exact port of timmh utils.piecewise_linear_calibration (utils.py:184-244).

    - Interpolates linearly between calibration points.
    - Extrapolates linearly using the first/last segment.
    - Clips outputs to be >= eps (avoids negative/zero inverse depths).
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if len(x) == 0 or len(y) == 0 or len(x) != len(y):
        raise ValueError(f"Invalid calibration points: len(x)={len(x)} len(y)={len(y)}")

    if len(x) == 1:
        y0 = float(y[0])

        def f(data):
            data = np.asarray(data, dtype=np.float64)
            return np.full_like(data, max(eps, y0), dtype=np.float64)

        f.knots_x = x.copy()
        f.knots_y = np.array([max(eps, y0)])
        return f

    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]

    # Merge duplicate x values by averaging y (rare, but avoids division by zero in slopes).
    unique_x, inverse = np.unique(x_sorted, return_inverse=True)
    if len(unique_x) != len(x_sorted):
        summed_y = np.zeros(len(unique_x), dtype=np.float64)
        counts = np.zeros(len(unique_x), dtype=np.int64)
        for i, group in enumerate(inverse):
            summed_y[group] += y_sorted[i]
            counts[group] += 1
        x_sorted = unique_x
        y_sorted = summed_y / np.maximum(counts, 1)

    if len(x_sorted) == 1:
        y0 = float(y_sorted[0])

        def f(data):
            data = np.asarray(data, dtype=np.float64)
            return np.full_like(data, max(eps, y0), dtype=np.float64)

        f.knots_x = x_sorted.copy()
        f.knots_y = np.array([max(eps, y0)])
        return f

    return _piecewise_from_sorted_knots(x_sorted, y_sorted, eps)


def _piecewise_from_sorted_knots(x_sorted, y_sorted, eps):
    left_dx = float(x_sorted[1] - x_sorted[0])
    right_dx = float(x_sorted[-1] - x_sorted[-2])
    left_slope = float((y_sorted[1] - y_sorted[0]) / left_dx) if left_dx != 0 else 0.0
    right_slope = float((y_sorted[-1] - y_sorted[-2]) / right_dx) if right_dx != 0 else 0.0

    def f(data):
        data = np.asarray(data, dtype=np.float64)
        out = np.interp(data, x_sorted, y_sorted)
        out = np.where(data < x_sorted[0], y_sorted[0] + (data - x_sorted[0]) * left_slope, out)
        out = np.where(data > x_sorted[-1], y_sorted[-1] + (data - x_sorted[-1]) * right_slope, out)
        return np.clip(out, eps, np.inf)

    f.knots_x = x_sorted.copy()
    f.knots_y = y_sorted.copy()
    return f


def piecewise_from_knots(knots_x, knots_y, eps: float = 1e-6) -> Callable[[np.ndarray], np.ndarray]:
    """Rebuild a piecewise_linear_calibration function from previously saved knots."""
    return piecewise_linear_calibration(knots_x, knots_y, eps=eps)


def resize_to(arr, shape_hw, interpolation=cv2.INTER_LINEAR):
    """Resize arr to shape_hw=(H, W) if not already at that shape.

    Port of timmh utils.resize (utils.py:68), generalized to accept an explicit target
    shape/interpolation. Boolean arrays are resized with nearest-neighbor interpolation
    regardless of the requested interpolation, to keep them boolean.
    """
    target_h, target_w = shape_hw[0], shape_hw[1]
    if arr.shape[0] == target_h and arr.shape[1] == target_w:
        return arr

    if arr.dtype == bool:
        resized = cv2.resize(
            arr.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST
        )
        return resized.astype(bool)

    resized = cv2.resize(arr.astype(np.float64), (target_w, target_h), interpolation=interpolation)
    return resized


def blur_and_downsample(img, calibration_downsampling_factor: float = 1 / 8, calibration_blur_sigma: float = 41):
    """Port of timmh utils.blur_and_downsample (utils.py:442)."""
    mask = img.mask if (hasattr(img, "mask") and img.mask.shape != ()) else None
    img = img if mask is None else img.data

    img = ndimage.gaussian_filter(img, sigma=calibration_blur_sigma)
    img = cv2.resize(
        img,
        None,
        fx=calibration_downsampling_factor,
        fy=calibration_downsampling_factor,
        interpolation=cv2.INTER_LINEAR,
    )

    if mask is not None:
        mask = (mask * 255).astype(np.uint8)
        mask = ndimage.gaussian_filter(mask, sigma=calibration_blur_sigma)
        mask = cv2.resize(
            mask,
            None,
            fx=calibration_downsampling_factor,
            fy=calibration_downsampling_factor,
            interpolation=cv2.INTER_LINEAR,
        )
        img = np.ma.masked_where(mask > 127, img)

    return img


def condition_disparity(disp, eps: float = 1e-6):
    """Port of timmh utils.condition_disparity (utils.py:471)."""
    disp = ndimage.median_filter(disp, size=3)
    disp = disp - np.min(disp)
    disp = disp / np.std(disp)
    disp = np.clip(disp, eps, np.inf)
    return disp


_MAX_ALIGN_PIXELS = 200_000


def align_disparity(
    disp,
    anchor_disp_raw,
    exclude_mask,
    method: str = "ransac",
    min_pixels: int = 500,
    blur: bool = False,
):
    """Align disp onto the scale/offset of anchor_disp_raw, excluding exclude_mask pixels.

    Port of the detection-frame alignment block in timmh run.py (run.py:315-342): disp is
    resized to the anchor's shape, then ``calibrate(disp[~exclude], anchor[~exclude])`` is
    fit (optionally on blurred/downsampled fields) and applied to the full disp array.

    Returns (aligned_disp_full, info) where aligned_disp_full is None and info["status"] is
    "too_few_pixels" or "invalid_fit" on failure.
    """
    disp = resize_to(disp, anchor_disp_raw.shape[:2])
    exclude_mask = resize_to(np.asarray(exclude_mask, dtype=bool), anchor_disp_raw.shape[:2])

    valid = ~exclude_mask
    n_pixels = int(np.count_nonzero(valid))
    info = {"m": None, "c": None, "inlier_frac": None, "n_pixels": n_pixels, "status": "ok"}

    if n_pixels < min_pixels:
        info["status"] = "too_few_pixels"
        return None, info

    x_fit = np.ma.masked_where(exclude_mask, disp)
    y_fit = np.ma.masked_where(exclude_mask, anchor_disp_raw)

    if blur:
        x_fit = blur_and_downsample(x_fit)
        y_fit = blur_and_downsample(y_fit)

    # Deterministic subsample for speed, applied to unmasked entries only.
    x_valid_mask = _split_mask(x_fit)
    y_valid_mask = _split_mask(y_fit)
    fit_valid = ~(x_valid_mask | y_valid_mask)
    valid_idx = np.flatnonzero(fit_valid.reshape(-1))
    x_flat = np.asarray(x_fit).reshape(-1)
    y_flat = np.asarray(y_fit).reshape(-1)

    if valid_idx.size < min_pixels:
        info["status"] = "too_few_pixels"
        return None, info

    if valid_idx.size > _MAX_ALIGN_PIXELS:
        rng = np.random.RandomState(RANDOM_SEED)
        valid_idx = rng.choice(valid_idx, size=_MAX_ALIGN_PIXELS, replace=False)

    x_sub = x_flat[valid_idx]
    y_sub = y_flat[valid_idx]

    try:
        fn = calibrate(x_sub, y_sub, method=method)
    except Exception:
        info["status"] = "invalid_fit"
        return None, info

    info["m"], info["c"], info["inlier_frac"] = fn.m, fn.c, fn.inlier_frac
    aligned_disp_full = fn(disp)
    return aligned_disp_full, info


def disparity_from_depth(depth, eps: float = 1e-6):
    """disp = 1 / clip(depth, eps, inf)"""
    return 1.0 / np.clip(depth, eps, np.inf)


def depth_from_disparity(disp, min_depth, max_depth):
    """Port of timmh run.py:342: depth = clip(disp, 1/max_depth, 1/min_depth) ** -1"""
    return np.clip(disp, 1.0 / max_depth, 1.0 / min_depth) ** -1


def interior_point(mask) -> Optional[tuple]:
    """Return the (row, col) of the mask pixel farthest from its boundary.

    Port of the SAM interior-point sampling in timmh run.py (run.py:402-419): the mask is
    zero-padded by 1px, a Euclidean distance transform is computed, and the argmax location
    (mapped back to full-image coordinates) is returned. Returns None for an empty mask.
    """
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return None

    mask_padded = np.pad(mask, ((1, 1), (1, 1)))
    dist = cv2.distanceTransform((mask_padded * 255).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_3)
    sample_location = np.unravel_index(np.argmax(dist, axis=None), dist.shape)
    row = max(0, min(mask.shape[0] - 1, sample_location[0] - 1))
    col = max(0, min(mask.shape[1] - 1, sample_location[1] - 1))
    return int(row), int(col)
