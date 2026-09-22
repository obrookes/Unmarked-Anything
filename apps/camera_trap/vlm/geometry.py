"""Pixel-space box geometry helpers, ported from vision-llm-ann-corrector/geometry.py.

Only the functions this package uses are ported (mask_bbox_xyxy, dilate_xyxy). numpy-only, no
torch/sam3 imports.
"""

from __future__ import annotations

import numpy as np


def mask_bbox_xyxy(mask: np.ndarray):
    """Bounding box of the True pixels of a bool (H, W) mask, or None if empty."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def dilate_xyxy(box_xyxy, frac: float, width: int, height: int):
    """Grow box_xyxy by `frac` of its own width/height on each side, clipped to the image."""
    x0, y0, x1, y1 = box_xyxy
    bw, bh = x1 - x0, y1 - y0
    dx, dy = bw * frac, bh * frac
    return [
        max(0.0, x0 - dx),
        max(0.0, y0 - dy),
        min(float(width), x1 + dx),
        min(float(height), y1 + dy),
    ]
