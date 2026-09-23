"""Extrinsic (camera-shift) recalibration: align a detection frame onto the anchor frame.

Ported (LightGlue-ONNX feature matching, RANSAC homography, quality gates, reuse of the
previous accepted homography) from:
  https://github.com/timmh/distance-estimation (extrinsic_recalibration.py)

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

Unlike the upstream module, ``LightGlueONNX`` here never downloads weights: a local
``.onnx`` path must be supplied explicitly (``--lightglue-weights`` on the apply.py CLI, which is
required whenever ``--extrinsic-recalibration`` is passed), and ``onnxruntime`` is imported
lazily so the rest of this module (homography estimation, warping) works without it installed.
Unlike the upstream module, there is no SIFT fallback matcher: LightGlue-ONNX is the only
matcher used.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


@dataclass
class HomographyEstimate:
    homography: np.ndarray
    num_matches: int
    num_inliers: int
    inlier_ratio: float
    reprojection_error: float
    reused_previous: bool = False


class LightGlueONNX:
    """SuperPoint+LightGlue ONNX matcher. Requires a local weights file; never downloads."""

    def __init__(self, weights_path: str | Path, max_image_size: int = 1024):
        self.weights_path = str(weights_path)
        self.max_image_size = max_image_size
        self._session = None

    def _load_model(self):
        if self._session is not None:
            return
        import onnxruntime  # lazy import: optional dependency, offline compute nodes

        self._session = onnxruntime.InferenceSession(
            self.weights_path, providers=["CPUExecutionProvider"]
        )

    def __call__(self, img0, img1):
        self._load_model()
        input_tensors, scales0, scales1 = self._prepare_inputs(img0, img1)
        output_names = [output.name for output in self._session.get_outputs()]
        outputs = self._session.run(output_names, input_tensors)
        output_map = dict(zip(output_names, outputs))
        pts0, pts1, scores = self._parse_outputs(output_map)
        pts0 = pts0.copy()
        pts1 = pts1.copy()
        pts0[:, 0] /= scales0[0]
        pts0[:, 1] /= scales0[1]
        pts1[:, 0] /= scales1[0]
        pts1[:, 1] /= scales1[1]
        return pts0, pts1, scores

    def _prepare_inputs(self, img0, img1):
        inputs = self._session.get_inputs()
        tensor0, scales0 = self._preprocess(img0, inputs[0].shape)
        tensor1, scales1 = self._preprocess(img1, inputs[-1].shape)
        if len(inputs) == 1:
            return {inputs[0].name: np.concatenate([tensor0, tensor1], axis=0)}, scales0, scales1
        if len(inputs) == 2:
            return {inputs[0].name: tensor0, inputs[1].name: tensor1}, scales0, scales1
        raise RuntimeError(f"Unsupported LightGlue ONNX input count: {len(inputs)}")

    def _preprocess(self, img, input_shape):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        gray = gray.astype(np.float32) / 255.0
        h, w = gray.shape
        fixed_h, fixed_w = self._fixed_hw(input_shape)
        if fixed_h is not None and fixed_w is not None:
            target_h, target_w = fixed_h, fixed_w
        else:
            scale = min(1.0, self.max_image_size / max(h, w))
            target_h = max(8, int(round(h * scale / 8) * 8))
            target_w = max(8, int(round(w * scale / 8) * 8))
        if (h, w) != (target_h, target_w):
            gray = cv2.resize(gray, (target_w, target_h), interpolation=cv2.INTER_AREA)
        tensor = gray[None, None, :, :].astype(np.float32)
        return tensor, (target_w / w, target_h / h)

    def _fixed_hw(self, input_shape):
        if input_shape is None or len(input_shape) < 4:
            return None, None
        h, w = input_shape[-2], input_shape[-1]
        if isinstance(h, int) and isinstance(w, int) and h > 0 and w > 0:
            return h, w
        return None, None

    def _parse_outputs(self, output_map):
        outputs = {name.lower(): value for name, value in output_map.items()}
        keypoints0 = self._get_output(outputs, ["keypoints0", "kpts0", "keypoints_0", "kpts_0"])
        keypoints1 = self._get_output(outputs, ["keypoints1", "kpts1", "keypoints_1", "kpts_1"])
        matched0 = self._get_output(outputs, ["matched_keypoints0", "mkpts0", "matches_keypoints0"], required=False)
        matched1 = self._get_output(outputs, ["matched_keypoints1", "mkpts1", "matches_keypoints1"], required=False)

        if matched0 is not None and matched1 is not None:
            scores = self._get_output(outputs, ["scores", "mscores", "matching_scores", "mscores0"], required=False)
            matched0 = self._points(matched0)
            matched1 = self._points(matched1)
            return matched0, matched1, self._scores(scores, len(matched0))

        matches = self._get_output(outputs, ["matches", "matches0", "matches_0"])
        scores = self._get_output(outputs, ["scores", "mscores", "matching_scores", "mscores0"], required=False)
        keypoints0 = self._points(keypoints0)
        keypoints1 = self._points(keypoints1)

        matches = np.asarray(matches)
        if matches.ndim == 3:
            matches = matches[0]
        if matches.ndim == 2 and matches.shape[0] == 1:
            matches = matches[0]
        if matches.ndim == 1:
            valid = matches >= 0
            idx0 = np.nonzero(valid)[0]
            idx1 = matches[valid].astype(np.int64)
            score_values = self._scores(scores, len(matches))[valid]
        else:
            valid = np.all(matches >= 0, axis=1)
            idx0 = matches[valid, 0].astype(np.int64)
            idx1 = matches[valid, 1].astype(np.int64)
            score_values = self._scores(scores, len(matches))[valid]

        valid_idx = (idx0 < len(keypoints0)) & (idx1 < len(keypoints1))
        return keypoints0[idx0[valid_idx]], keypoints1[idx1[valid_idx]], score_values[valid_idx]

    def _get_output(self, outputs, names, required=True):
        for name in names:
            if name in outputs:
                return outputs[name]
        if required:
            raise RuntimeError(f"LightGlue ONNX output not found. Expected one of {names}, got {list(outputs.keys())}")
        return None

    def _points(self, points):
        points = np.asarray(points, dtype=np.float32)
        if points.ndim == 3:
            points = points[0]
        return points.reshape(-1, 2)

    def _scores(self, scores, length):
        if scores is None:
            return np.ones(length, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        if len(scores) == length:
            return scores
        return np.ones(length, dtype=np.float32)


class ExtrinsicRecalibrator:
    """Estimates (and caches) a homography aligning a detection frame onto the anchor frame.

    LightGlue-ONNX is the only matcher used (``lightglue_weights`` is required). Quality gates
    and "reuse previous accepted homography" behaviour follow the upstream
    ``ExtrinsicRecalibrator`` (extrinsic_recalibration.py).
    """

    def __init__(self, lightglue_weights: str | Path):
        self.lightglue = LightGlueONNX(lightglue_weights)
        self.previous_homography: Optional[np.ndarray] = None

    def reset(self):
        self.previous_homography = None

    def estimate(self, baseline_img: np.ndarray, img: np.ndarray) -> Optional[HomographyEstimate]:
        pts0, pts1, scores = self.lightglue(baseline_img, img)

        estimate = self._estimate_homography(pts0, pts1, scores)
        if estimate is not None and self._is_quality_acceptable(estimate, img.shape[:2]):
            self.previous_homography = estimate.homography
            return estimate

        if self.previous_homography is not None:
            return HomographyEstimate(
                homography=self.previous_homography,
                num_matches=0 if estimate is None else estimate.num_matches,
                num_inliers=0 if estimate is None else estimate.num_inliers,
                inlier_ratio=0.0 if estimate is None else estimate.inlier_ratio,
                reprojection_error=np.inf if estimate is None else estimate.reprojection_error,
                reused_previous=True,
            )
        return None

    def _estimate_homography(self, pts0, pts1, scores) -> Optional[HomographyEstimate]:
        if len(pts0) < 8 or len(pts1) < 8:
            return None
        order = np.argsort(scores)[::-1] if len(scores) == len(pts0) else np.arange(len(pts0))
        pts0, pts1 = pts0[order], pts1[order]
        homography, inlier_mask = cv2.findHomography(pts1, pts0, cv2.RANSAC, 4.0)
        if homography is None or inlier_mask is None:
            return None
        inliers = inlier_mask.ravel().astype(bool)
        if not np.any(inliers):
            return None
        projected = cv2.perspectiveTransform(pts1[inliers, None, :], homography)[:, 0, :]
        errors = np.linalg.norm(projected - pts0[inliers], axis=1)
        return HomographyEstimate(
            homography=homography,
            num_matches=len(pts0),
            num_inliers=int(np.sum(inliers)),
            inlier_ratio=float(np.mean(inliers)),
            reprojection_error=float(np.median(errors)),
        )

    def _is_quality_acceptable(self, estimate: HomographyEstimate, image_shape) -> bool:
        if estimate.num_inliers < 16 or estimate.inlier_ratio < 0.25 or estimate.reprojection_error > 6.0:
            return False
        return is_homography_sane(estimate.homography, image_shape)


def is_homography_sane(homography: np.ndarray, image_shape) -> bool:
    h, w = image_shape
    if not np.all(np.isfinite(homography)):
        return False
    corners = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    warped = cv2.perspectiveTransform(corners[None, :, :], homography)[0]
    if not np.all(np.isfinite(warped)):
        return False
    source_area = cv2.contourArea(corners)
    warped_area = abs(cv2.contourArea(warped.astype(np.float32)))
    area_ratio = warped_area / max(source_area, 1.0)
    if area_ratio < 0.2 or area_ratio > 5.0:
        return False
    max_displacement = np.max(np.linalg.norm(warped - corners, axis=1))
    return max_displacement <= max(h, w) * 0.75


def rescale_homography(homography: np.ndarray, from_shape_hw, to_shape_hw) -> np.ndarray:
    """Conjugate a homography estimated at ``from_shape_hw`` resolution to ``to_shape_hw``.

    Both the domain and codomain of ``homography`` are assumed to represent the same physical
    frame, just resampled together (i.e. this is not for a homography between two genuinely
    different resolutions/crops). Used to reuse a homography estimated on downscaled matcher
    inputs at the resolution of the disparity/mask arrays being warped.
    """
    fh, fw = from_shape_hw[0], from_shape_hw[1]
    th, tw = to_shape_hw[0], to_shape_hw[1]
    if (fh, fw) == (th, tw):
        return homography
    sx, sy = tw / fw, th / fh
    S = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float64)
    return S @ homography.astype(np.float64) @ np.linalg.inv(S)


def warp_image(img: np.ndarray, homography: np.ndarray, target_shape_hw) -> np.ndarray:
    return cv2.warpPerspective(
        img,
        homography,
        (target_shape_hw[1], target_shape_hw[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def warp_depth(depth: np.ndarray, homography: np.ndarray, target_shape_hw) -> np.ndarray:
    return cv2.warpPerspective(
        depth.astype(np.float64),
        homography,
        (target_shape_hw[1], target_shape_hw[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def warp_mask(mask: np.ndarray, homography: np.ndarray, target_shape_hw) -> np.ndarray:
    warped = cv2.warpPerspective(
        mask.astype(np.uint8),
        homography,
        (target_shape_hw[1], target_shape_hw[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped.astype(bool)
