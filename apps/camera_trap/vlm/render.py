"""Mask decode + overlay rendering for VLM QC images.

`rle_decode` is copied from vision-llm-ann-verifier/images.py:rle_decode (same as
vision-llm-ann-corrector/rle.py:rle_decode) -- a numpy-only COCO RLE decoder, no pycocotools
dependency required. It's the fallback path for environments (e.g. the vLLM container) that may
lack pycocotools; prefer `depth_anything_3.utils.camera_trap_masks.load_mask_array` when
pycocotools is available.
"""

from __future__ import annotations

import cv2
import numpy as np

from .geometry import dilate_xyxy

# Copied from vision-llm-ann-verifier/images.py's PALETTE.
PALETTE = [
    (66, 133, 244), (52, 168, 83), (234, 67, 53), (251, 188, 5),
    (154, 92, 232), (0, 172, 193), (255, 112, 67), (156, 204, 101),
    (240, 98, 146), (3, 169, 244), (139, 195, 74), (255, 160, 0),
]


def rle_decode(rle: dict) -> np.ndarray:
    """Decode a COCO RLE dict ({"size": [h, w], "counts": str | list}) to a bool (H, W) mask."""
    h, w = rle["size"]
    s = rle["counts"]
    if isinstance(s, str):
        counts = []
        i = 0
        m = 0
        while i < len(s):
            x = 0
            k = 0
            while True:
                c = ord(s[i]) - 48
                x |= (c & 0x1F) << (5 * k)
                more = c & 0x20
                i += 1
                k += 1
                if not more:
                    if c & 0x10:
                        x |= -1 << (5 * k)
                    break
            if m > 2:
                x += counts[m - 2]
            counts.append(x)
            m += 1
    else:
        counts = list(s)
    counts = np.asarray(counts, dtype=np.int64)
    ends = np.cumsum(counts)
    starts = ends - counts
    mask = np.zeros(h * w, dtype=bool)
    for st, en in zip(starts[1::2], ends[1::2]):
        mask[st:en] = True
    return mask.reshape((w, h)).T  # COCO RLE is column-major


def render_overlay(frame: np.ndarray, masks: list[np.ndarray], labels: list[str] | None = None,
                    alpha: float = 0.5) -> np.ndarray:
    """Blend each mask's fill + 2px contour + numeric/text label onto a copy of `frame`.

    `frame` and the output share the same channel convention (e.g. both BGR) -- this function
    only blends colour, it doesn't care which.
    """
    out = frame.copy()
    for k, mask in enumerate(masks):
        color = np.array(PALETTE[k % len(PALETTE)], dtype=np.float64)
        out[mask] = (alpha * out[mask] + (1 - alpha) * color).astype(np.uint8)

        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, tuple(int(c) for c in color), 2)

        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            continue
        x0, y0 = int(xs.min()), int(ys.min())
        label = str(labels[k]) if labels is not None else str(k)
        pos = (max(x0, 2), y0 - 6 if y0 > 24 else min(y0 + 22, out.shape[0] - 4))
        cv2.putText(out, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def crop_around(frame: np.ndarray, bbox_xyxy, pad_frac: float = 0.25) -> np.ndarray:
    """Crop `frame` to bbox_xyxy dilated by pad_frac of its own size, clipped to the frame."""
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = dilate_xyxy(bbox_xyxy, pad_frac, w, h)
    x0, y0 = int(x0), int(y0)
    x1, y1 = max(int(round(x1)), x0 + 1), max(int(round(y1)), y0 + 1)
    return frame[y0:y1, x0:x1]


def downscale_max_side(frame: np.ndarray, max_side: int = 1024) -> np.ndarray:
    """Resize `frame` so its longer side is at most max_side; no-op if already smaller."""
    h, w = frame.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return frame
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
