from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from apps.camera_trap.scripts import validate_mask_rle_roundtrip as validator
from depth_anything_3.utils.camera_trap_masks import encode_mask_to_coco_rle


def _write_video_fixture(
    tmp_path: Path,
    *,
    raw_mask: np.ndarray,
    rle_mask: np.ndarray | None = None,
    include_pair_metadata: bool = True,
) -> validator.ResolvedVideo:
    video_dir = tmp_path / "demo"
    video_dir.mkdir()
    json_path = video_dir / "demo.json"
    npz_path = video_dir / "demo_arrays.npz"

    if rle_mask is None:
        rle_mask = raw_mask

    counts, size = encode_mask_to_coco_rle(rle_mask)
    arrays: dict[str, np.ndarray] = {
        "f0_mask_ape_raw": raw_mask.astype(np.uint8),
        "f0_mask_ape_rle": np.asarray(counts, dtype=np.str_),
        "f0_obj_0_mask_raw": raw_mask.astype(np.uint8),
        "f0_obj_0_mask_rle": np.asarray(counts, dtype=np.str_),
    }
    np.savez(npz_path, **arrays)

    prompt_entry = {
        "prompt_index": 0,
        "prompt": "ape",
        "slug": "ape",
        "size": size,
        "encoding": "raw",
        "key": "f0_mask_ape_raw",
    }
    object_entry = {
        "object_index": 0,
        "track_id": 5,
        "label": "ape",
        "size": size,
        "encoding": "raw",
        "key": "f0_obj_0_mask_raw",
    }
    if include_pair_metadata:
        prompt_entry.update({"raw_key": "f0_mask_ape_raw", "rle_key": "f0_mask_ape_rle"})
        object_entry.update({"raw_key": "f0_obj_0_mask_raw", "rle_key": "f0_obj_0_mask_rle"})

    payload = {
        "video_name": "demo",
        "video_fps": 5.0,
        "frames": [
            {
                "frame_index": 0,
                "status": "processed",
                "npz_keys": {
                    "masks": [prompt_entry],
                    "objects": [object_entry],
                },
            }
        ],
    }
    json_path.write_text(json.dumps(payload), encoding="utf-8")
    return validator.ResolvedVideo(video_name="demo", video_dir=video_dir, json_path=json_path, npz_path=npz_path)


def test_validate_mask_pair_exact_match(tmp_path: Path) -> None:
    pytest.importorskip("pycocotools.mask")
    mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    resolved = _write_video_fixture(tmp_path, raw_mask=mask)
    member_sizes = validator._member_size_map(resolved.npz_path)

    with np.load(resolved.npz_path, allow_pickle=False) as npz_data:
        entry = json.loads(resolved.json_path.read_text())["frames"][0]["npz_keys"]["masks"][0]
        result = validator.validate_mask_pair(
            video_name="demo",
            frame_index=0,
            npz_data=npz_data,
            entry=entry,
            mask_kind="prompt",
            member_sizes=member_sizes,
            strict_pairs=True,
        )

    assert result is not None
    assert result.exact_match is True
    assert result.diff_pixels == 0
    assert result.error is None


def test_validate_mask_pair_detects_mismatch(tmp_path: Path) -> None:
    pytest.importorskip("pycocotools.mask")
    raw_mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    mismatched_rle_mask = np.array([[0, 1], [0, 0]], dtype=np.uint8)
    resolved = _write_video_fixture(tmp_path, raw_mask=raw_mask, rle_mask=mismatched_rle_mask)
    member_sizes = validator._member_size_map(resolved.npz_path)

    with np.load(resolved.npz_path, allow_pickle=False) as npz_data:
        entry = json.loads(resolved.json_path.read_text())["frames"][0]["npz_keys"]["masks"][0]
        result = validator.validate_mask_pair(
            video_name="demo",
            frame_index=0,
            npz_data=npz_data,
            entry=entry,
            mask_kind="prompt",
            member_sizes=member_sizes,
            strict_pairs=True,
        )

    assert result is not None
    assert result.exact_match is False
    assert result.diff_pixels == 1
    assert result.error is None


def test_validate_mask_pair_missing_pair_metadata_strict(tmp_path: Path) -> None:
    pytest.importorskip("pycocotools.mask")
    mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    resolved = _write_video_fixture(tmp_path, raw_mask=mask, include_pair_metadata=False)

    with np.load(resolved.npz_path, allow_pickle=False) as npz_data:
        entry = json.loads(resolved.json_path.read_text())["frames"][0]["npz_keys"]["masks"][0]
        result = validator.validate_mask_pair(
            video_name="demo",
            frame_index=0,
            npz_data=npz_data,
            entry=entry,
            mask_kind="prompt",
            member_sizes={},
            strict_pairs=True,
        )

    assert result is not None
    assert result.exact_match is False
    assert "missing raw_key or rle_key" in str(result.error)


def test_validate_video_summary_and_csv(tmp_path: Path) -> None:
    pytest.importorskip("pycocotools.mask")
    mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    resolved = _write_video_fixture(tmp_path, raw_mask=mask)
    summary = validator.validate_video(
        resolved=resolved,
        output_dir=tmp_path / "reports",
        max_frames=None,
        skip_visuals=True,
        allow_missing_video=False,
        visual_fps=2.0,
        overlay_alpha=0.45,
        strict_pairs=True,
    )

    assert summary["entries_checked"] == 2
    assert summary["mismatched_entries"] == 0
    assert Path(summary["detail_csv"]).is_file()


def test_main_returns_nonzero_on_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("pycocotools.mask")
    raw_mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    mismatched_rle_mask = np.array([[0, 1], [0, 0]], dtype=np.uint8)
    resolved = _write_video_fixture(tmp_path, raw_mask=raw_mask, rle_mask=mismatched_rle_mask)
    run_root = tmp_path / "run"
    run_root.mkdir()
    target_dir = run_root / resolved.video_dir.name
    resolved.video_dir.rename(target_dir)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_mask_rle_roundtrip.py",
            "--run-root",
            str(run_root),
            "--output-dir",
            str(tmp_path / "out"),
            "--skip-visuals",
        ],
    )

    code = validator.main()
    assert code == 1
    summary_path = tmp_path / "out" / "mask_rle_validation_summary.json"
    assert summary_path.is_file()
