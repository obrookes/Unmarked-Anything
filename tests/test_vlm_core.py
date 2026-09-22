from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apps.camera_trap.vlm import models, render
from apps.camera_trap.vlm.engine import FakeEngine, VLMEngine, VLMRequest


def test_rle_decode_matches_pycocotools_round_trip() -> None:
    pytest.importorskip("pycocotools.mask")
    from depth_anything_3.utils.camera_trap_masks import encode_mask_to_coco_rle

    rng = np.random.default_rng(0)
    for _ in range(5):
        mask = (rng.random((37, 53)) > 0.6).astype(np.uint8)
        counts, size = encode_mask_to_coco_rle(mask)
        decoded = render.rle_decode({"size": size, "counts": counts})
        np.testing.assert_array_equal(decoded.astype(np.uint8), mask)


def test_render_overlay_shape_and_changes_pixels() -> None:
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    mask = np.zeros((20, 30), dtype=bool)
    mask[5:10, 5:10] = True
    out = render.render_overlay(frame, [mask], labels=["0"])
    assert out.shape == frame.shape
    assert out.dtype == frame.dtype
    assert not np.array_equal(out, frame)
    # Outside the mask, far from any contour/label pixels, should be untouched.
    assert tuple(out[0, 0]) == (0, 0, 0)


def test_crop_around_shape() -> None:
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    bbox = [50, 40, 70, 60]  # 20x20 box
    crop = render.crop_around(frame, bbox, pad_frac=0.25)
    # Dilated box grows by 25% each side -> 30x30, clipped to frame bounds.
    assert crop.shape[0] == 30
    assert crop.shape[1] == 30


def test_crop_around_clips_to_frame() -> None:
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    bbox = [0, 0, 10, 10]
    crop = render.crop_around(frame, bbox, pad_frac=1.0)
    assert crop.shape[0] <= 50
    assert crop.shape[1] <= 50


def test_downscale_max_side() -> None:
    frame = np.zeros((2000, 1000, 3), dtype=np.uint8)
    out = render.downscale_max_side(frame, max_side=1024)
    assert max(out.shape[0], out.shape[1]) == 1024
    small = np.zeros((10, 10, 3), dtype=np.uint8)
    assert render.downscale_max_side(small, max_side=1024) is small


def test_models_resolve_known_and_unknown() -> None:
    spec = models.resolve("qwen")
    assert spec.model_id == "Qwen/Qwen3.8-27B"
    other = models.resolve("some-org/some-model")
    assert other.model_id == "some-org/some-model"
    assert other.tensor_parallel == spec.tensor_parallel


def test_vlm_engine_construction_does_not_import_vllm() -> None:
    # vllm is not installed in this env; constructing the engine must not try to import it
    # (only run_json's _ensure_engine should).
    engine = VLMEngine(model_id="qwen")
    assert engine.spec.model_id == "Qwen/Qwen3.8-27B"


def test_fake_engine_run_json_fixed_list() -> None:
    engine = FakeEngine(responses=[{"ok": True, "issue": "none"}, None])
    reqs = [
        VLMRequest(images=[np.zeros((4, 4, 3), dtype=np.uint8)], prompt="p1"),
        VLMRequest(images=[np.zeros((4, 4, 3), dtype=np.uint8)], prompt="p2"),
    ]
    results = engine.run_json(reqs, schema={"type": "object"})
    assert results == [{"ok": True, "issue": "none"}, None]
    assert engine.last_errors == [None, "fake: no scripted response"]
    assert engine.calls == reqs


def test_fake_engine_run_json_callable() -> None:
    engine = FakeEngine(responses=lambda req: {"prompt": req.prompt})
    reqs = [VLMRequest(images=[], prompt="hello")]
    results = engine.run_json(reqs)
    assert results == [{"prompt": "hello"}]
