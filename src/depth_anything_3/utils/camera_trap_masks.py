from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

try:
    from pycocotools import mask as coco_mask
except ImportError:  # pragma: no cover - dependency may be absent in local test env
    coco_mask = None


MASK_ENCODING_RAW = "raw"
MASK_ENCODING_COCO_RLE = "coco_rle"
MASK_STORAGE_FORMAT_CHOICES = ("raw", "rle", "both")


def _require_coco_mask() -> Any:
    if coco_mask is None:
        raise RuntimeError(
            "pycocotools is required for COCO RLE mask support. Install project dependencies first."
        )
    return coco_mask


def _array_lookup(npz_data: Mapping[str, Any], key: str) -> np.ndarray:
    if key not in npz_data:
        raise KeyError(key)
    return np.asarray(npz_data[key])


def normalize_binary_mask(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D mask array, got shape {arr.shape}")
    return (arr > 0).astype(np.uint8)


def encode_mask_to_coco_rle(mask: np.ndarray) -> tuple[str, list[int]]:
    encoder = _require_coco_mask()
    mask_u8 = np.asfortranarray(normalize_binary_mask(mask))
    encoded = encoder.encode(mask_u8)
    counts = encoded["counts"]
    if isinstance(counts, bytes):
        counts_text = counts.decode("utf-8")
    else:
        counts_text = str(counts)
    size = [int(v) for v in encoded["size"]]
    return counts_text, size


def decode_coco_rle_to_mask(*, counts: str, size: list[int] | tuple[int, int]) -> np.ndarray:
    decoder = _require_coco_mask()
    if len(size) != 2:
        raise ValueError(f"Expected size [height, width], got {size!r}")
    decoded = decoder.decode({"counts": counts.encode("utf-8"), "size": [int(size[0]), int(size[1])]})
    decoded = np.asarray(decoded)
    if decoded.ndim == 3:
        decoded = decoded[..., 0]
    return normalize_binary_mask(decoded)


def encode_mask_to_rle_npz_payload(mask: np.ndarray) -> tuple[np.ndarray, list[int]]:
    counts, size = encode_mask_to_coco_rle(mask)
    return np.asarray(counts, dtype=np.str_), size


def load_mask_array(npz_data: Mapping[str, Any], entry: dict[str, Any]) -> np.ndarray:
    key = entry.get("key")
    if not isinstance(key, str) or not key:
        raise KeyError(f"Missing mask key in entry: {entry!r}")

    encoding = str(entry.get("encoding") or MASK_ENCODING_RAW)
    if encoding == MASK_ENCODING_RAW:
        return normalize_binary_mask(_array_lookup(npz_data, key))
    if encoding != MASK_ENCODING_COCO_RLE:
        raise ValueError(f"Unsupported mask encoding: {encoding!r}")

    payload = _array_lookup(npz_data, key)
    if payload.shape == ():
        counts = str(payload.item())
    else:
        counts = str(payload.reshape(-1)[0])
    size = entry.get("size")
    if not isinstance(size, (list, tuple)):
        raise ValueError(f"Missing RLE size metadata for key {key!r}")
    return decode_coco_rle_to_mask(counts=counts, size=size)


def load_mask_bool(npz_data: Mapping[str, Any], entry: dict[str, Any]) -> np.ndarray:
    return load_mask_array(npz_data=npz_data, entry=entry).astype(bool)
