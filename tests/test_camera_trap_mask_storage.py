from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from apps.camera_trap.cli import dap3_cli
from apps.camera_trap.scripts import export_job_distances_csv, merge_old_depth_into_stream_npz
from depth_anything_3.utils import camera_trap_viz
from depth_anything_3.utils.camera_trap_masks import (
    MASK_ENCODING_COCO_RLE,
    MASK_ENCODING_RAW,
    decode_coco_rle_to_mask,
    encode_mask_to_coco_rle,
    load_mask_bool,
)


def _write_npz(tmp_path: Path, **arrays: np.ndarray) -> Path:
    npz_path = tmp_path / "arrays.npz"
    np.savez(npz_path, **arrays)
    return npz_path


def test_build_mask_storage_entry_raw() -> None:
    arrays: dict[str, np.ndarray] = {}
    entry = dap3_cli.build_mask_storage_entry(
        npz_arrays=arrays,
        key_prefix="f1_mask_ape",
        mask=np.array([[0, 1], [2, 0]], dtype=np.uint8),
        base_entry={"prompt": "ape"},
        storage_format="raw",
    )

    assert entry["encoding"] == MASK_ENCODING_RAW
    assert entry["key"] == "f1_mask_ape_raw"
    assert entry["raw_key"] == "f1_mask_ape_raw"
    assert entry["size"] == [2, 2]
    np.testing.assert_array_equal(arrays["f1_mask_ape_raw"], np.array([[0, 1], [1, 0]], dtype=np.uint8))


def test_build_mask_storage_entry_both(monkeypatch: pytest.MonkeyPatch) -> None:
    arrays: dict[str, np.ndarray] = {}
    monkeypatch.setattr(
        dap3_cli,
        "encode_mask_to_rle_npz_payload",
        lambda mask: (np.asarray("stub-counts", dtype=np.str_), [2, 2]),
    )

    entry = dap3_cli.build_mask_storage_entry(
        npz_arrays=arrays,
        key_prefix="f1_obj_0_mask",
        mask=np.array([[1, 0], [0, 1]], dtype=np.uint8),
        base_entry={"object_index": 0},
        storage_format="both",
    )

    assert entry["encoding"] == MASK_ENCODING_RAW
    assert entry["key"] == "f1_obj_0_mask_raw"
    assert entry["raw_key"] == "f1_obj_0_mask_raw"
    assert entry["rle_key"] == "f1_obj_0_mask_rle"
    assert entry["size"] == [2, 2]
    assert arrays["f1_obj_0_mask_rle"].shape == ()


def test_load_mask_bool_legacy_raw_entry(tmp_path: Path) -> None:
    npz_path = _write_npz(tmp_path, legacy_mask=np.array([[0, 1], [0, 2]], dtype=np.uint8))
    with np.load(npz_path, allow_pickle=False) as npz_data:
        mask = load_mask_bool(npz_data=npz_data, entry={"key": "legacy_mask"})
    np.testing.assert_array_equal(mask, np.array([[False, True], [False, True]]))


def test_encode_decode_coco_rle_round_trip() -> None:
    pytest.importorskip("pycocotools.mask")
    mask = np.array(
        [
            [0, 1, 1, 0],
            [0, 0, 1, 0],
            [1, 1, 0, 0],
        ],
        dtype=np.uint8,
    )
    counts, size = encode_mask_to_coco_rle(mask)
    decoded = decode_coco_rle_to_mask(counts=counts, size=size)
    np.testing.assert_array_equal(decoded, mask)


def test_camera_trap_viz_build_union_mask_rle(tmp_path: Path) -> None:
    pytest.importorskip("pycocotools.mask")
    mask_a = np.array([[0, 1], [0, 0]], dtype=np.uint8)
    mask_b = np.array([[0, 0], [1, 0]], dtype=np.uint8)
    counts_a, size_a = encode_mask_to_coco_rle(mask_a)
    counts_b, size_b = encode_mask_to_coco_rle(mask_b)
    npz_path = _write_npz(
        tmp_path,
        mask_a=np.asarray(counts_a, dtype=np.str_),
        mask_b=np.asarray(counts_b, dtype=np.str_),
    )

    frame_row = {
        "frame_index": 3,
        "npz_keys": {
            "masks": [
                {"key": "mask_a", "encoding": MASK_ENCODING_COCO_RLE, "size": size_a},
                {"key": "mask_b", "encoding": MASK_ENCODING_COCO_RLE, "size": size_b},
            ]
        },
    }

    with np.load(npz_path, allow_pickle=False) as npz_data:
        union = camera_trap_viz.build_union_mask(npz_data=npz_data, frame_row=frame_row)
    np.testing.assert_array_equal(union, np.array([[False, True], [True, False]]))


def test_export_build_rows_for_video_rle(tmp_path: Path) -> None:
    pytest.importorskip("pycocotools.mask")
    mask = np.array([[1, 0], [1, 0]], dtype=np.uint8)
    counts, size = encode_mask_to_coco_rle(mask)
    npz_path = _write_npz(
        tmp_path,
        f0_depth=np.array([[2.0, 7.0], [4.0, 9.0]], dtype=np.float32),
        f0_obj_0_mask_rle=np.asarray(counts, dtype=np.str_),
    )
    video_json = {
        "video_name": "clip",
        "video_fps": 1.0,
        "frames": [
            {
                "frame_index": 0,
                "objects": [
                    {
                        "track_id": 7,
                        "key": "f0_obj_0_mask_rle",
                        "encoding": MASK_ENCODING_COCO_RLE,
                        "size": size,
                    }
                ],
            }
        ],
    }

    rows = export_job_distances_csv.build_rows_for_video(
        video_json=video_json,
        json_path=tmp_path / "clip.json",
        npz_path=npz_path,
        interval_seconds=1.0,
        fps_override=None,
        creation_dt=None,
    )

    assert len(rows) == 1
    assert rows[0]["ind_no"] == 7
    assert rows[0]["distance"] == pytest.approx(3.0)


def test_merge_build_union_mask_supports_rle() -> None:
    pytest.importorskip("pycocotools.mask")
    mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    counts, size = encode_mask_to_coco_rle(mask)
    frame_row = {
        "frame_index": 5,
        "npz_keys": {
            "masks": [
                {"key": "mask_rle", "encoding": MASK_ENCODING_COCO_RLE, "size": size},
            ]
        },
    }
    union, missing = merge_old_depth_into_stream_npz.build_union_mask(
        frame_row=frame_row,
        arrays={"mask_rle": np.asarray(counts, dtype=np.str_)},
    )

    assert missing == 0
    assert union is not None
    np.testing.assert_array_equal(union, np.array([[False, True], [True, False]]))
