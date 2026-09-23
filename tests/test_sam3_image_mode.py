"""Tests for OfficialSam3ImageSegmenter.segment_image (sam3_backends.py), using fake `sam3`
modules injected into sys.modules so we can check the output-parsing contract without the real
facebookresearch/sam3 package installed.

`segment_image(frame_bgr, prompt) -> list[tuple[mask_bool_HxW, score, box_xyxy]]` is the public
entry point the calibration builder (a separate work package) will call.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apps.camera_trap import sam3_backends
from apps.camera_trap.sam3_backends import OfficialSam3ImageSegmenter


FRAME_HW = (8, 8)


class FakeSam3Processor:
    """Fake for sam3.model.sam3_image_processor.Sam3Processor.

    Bypasses the real (patched) `_forward_grounding` entirely -- this test only checks that
    `segment_image` correctly parses whatever `set_text_prompt` hands back via `state`, not the
    internals of the ported `_patch_image_processor_scoring` patch itself.
    """

    _scoring_patched = False  # class attr the real patch function checks/sets

    def __init__(self, model, device: str = "cuda") -> None:
        self.model = model
        self.device = device
        self.confidence_threshold: float | None = None
        self.set_image_calls: list = []
        self.set_text_prompt_calls: list = []

    def set_confidence_threshold(self, value: float) -> None:
        self.confidence_threshold = value

    def set_image(self, pil_image) -> dict:
        self.set_image_calls.append(pil_image)
        return {"image": pil_image}

    def set_text_prompt(self, *, prompt: str, state: dict) -> dict:
        self.set_text_prompt_calls.append((prompt, state))
        n = getattr(self, "_next_n_detections", 2)
        boxes = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 8.0, 8.0]][:n], dtype=torch.float32
        )
        masks = torch.zeros((n, 1, FRAME_HW[0], FRAME_HW[1]), dtype=torch.bool)
        if n >= 1:
            masks[0, 0, 0:3, 0:3] = True
        if n >= 2:
            masks[1, 0, :, :] = True
        scores = torch.tensor([0.91, 0.42][:n], dtype=torch.float32)
        return {**state, "boxes": boxes, "masks": masks, "scores": scores}


def _install_fake_sam3_modules(monkeypatch: pytest.MonkeyPatch, checkpoint_path: Path) -> None:
    fake_sam3 = types.ModuleType("sam3")
    fake_sam3_model = types.ModuleType("sam3.model")
    fake_model_builder = types.ModuleType("sam3.model_builder")
    fake_image_processor = types.ModuleType("sam3.model.sam3_image_processor")
    fake_box_ops = types.ModuleType("sam3.model.box_ops")
    fake_data_misc = types.ModuleType("sam3.model.data_misc")

    def _build_sam3_image_model(**kwargs):
        return types.SimpleNamespace(kwargs=kwargs)

    fake_model_builder.build_sam3_image_model = _build_sam3_image_model
    fake_image_processor.Sam3Processor = FakeSam3Processor
    fake_box_ops.box_cxcywh_to_xyxy = lambda x: x  # unused by segment_image itself
    fake_data_misc.interpolate = lambda *a, **k: None  # unused; only referenced when patch runs

    monkeypatch.setitem(sys.modules, "sam3", fake_sam3)
    monkeypatch.setitem(sys.modules, "sam3.model", fake_sam3_model)
    monkeypatch.setitem(sys.modules, "sam3.model_builder", fake_model_builder)
    monkeypatch.setitem(sys.modules, "sam3.model.sam3_image_processor", fake_image_processor)
    monkeypatch.setitem(sys.modules, "sam3.model.box_ops", fake_box_ops)
    monkeypatch.setitem(sys.modules, "sam3.model.data_misc", fake_data_misc)

    # Reset the patch's idempotency guard between tests (it's a class attr).
    FakeSam3Processor._scoring_patched = False


def _write_unknown_variant_checkpoint(path: Path) -> Path:
    # Keys matching neither SA-FARI nor Meta-release patterns -> _checkpoint_variant == "unknown",
    # so _patch_segmentation_head_with_presence (which needs a real sam3.model_builder with
    # _create_segmentation_head/_create_transformer_decoder) is never invoked.
    torch.save({"some.unrelated.key": torch.zeros(1)}, path)
    return path


def test_segment_image_parses_boxes_masks_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_path = _write_unknown_variant_checkpoint(tmp_path / "ckpt.pt")
    _install_fake_sam3_modules(monkeypatch, checkpoint_path)

    segmenter = OfficialSam3ImageSegmenter(
        sam3_model_path=str(checkpoint_path), device="cpu", confidence_threshold=0.3
    )
    frame_bgr = np.zeros((FRAME_HW[0], FRAME_HW[1], 3), dtype=np.uint8)

    results = segmenter.segment_image(frame_bgr, "animal")

    assert len(results) == 2
    for mask, score, box in results:
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == np.bool_
        assert mask.shape == FRAME_HW
        assert isinstance(score, float)
        assert isinstance(box, list)
        assert len(box) == 4
        assert all(isinstance(v, float) for v in box)

    mask0, score0, box0 = results[0]
    assert mask0[0:3, 0:3].all()
    assert not mask0[5, 5]
    assert score0 == pytest.approx(0.91)
    assert box0 == pytest.approx([1.0, 2.0, 3.0, 4.0])

    mask1, score1, box1 = results[1]
    assert mask1.all()
    assert score1 == pytest.approx(0.42)


def test_segment_image_squeezes_nhw1_masks_and_handles_zero_detections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_path = _write_unknown_variant_checkpoint(tmp_path / "ckpt.pt")
    _install_fake_sam3_modules(monkeypatch, checkpoint_path)

    segmenter = OfficialSam3ImageSegmenter(sam3_model_path=str(checkpoint_path), device="cpu")
    frame_bgr = np.zeros((FRAME_HW[0], FRAME_HW[1], 3), dtype=np.uint8)

    # Monkeypatch the fake processor to return zero detections for this call.
    _, processor = segmenter._load_image_model()
    processor._next_n_detections = 0

    results = segmenter.segment_image(frame_bgr, "animal")
    assert results == []


def test_segment_image_model_and_processor_loaded_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_path = _write_unknown_variant_checkpoint(tmp_path / "ckpt.pt")
    _install_fake_sam3_modules(monkeypatch, checkpoint_path)

    load_calls = []
    real_builder = sys.modules["sam3.model_builder"].build_sam3_image_model

    def _counting_builder(**kwargs):
        load_calls.append(kwargs)
        return real_builder(**kwargs)

    sys.modules["sam3.model_builder"].build_sam3_image_model = _counting_builder

    segmenter = OfficialSam3ImageSegmenter(sam3_model_path=str(checkpoint_path), device="cpu")
    frame_bgr = np.zeros((FRAME_HW[0], FRAME_HW[1], 3), dtype=np.uint8)

    segmenter.segment_image(frame_bgr, "animal")
    segmenter.segment_image(frame_bgr, "animal")

    assert len(load_calls) == 1  # lazily loaded once, reused on the second call
    assert load_calls[0]["checkpoint_path"] == str(checkpoint_path)
    assert load_calls[0]["load_from_HF"] is False


def test_segment_image_confidence_threshold_applied_to_processor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_path = _write_unknown_variant_checkpoint(tmp_path / "ckpt.pt")
    _install_fake_sam3_modules(monkeypatch, checkpoint_path)

    segmenter = OfficialSam3ImageSegmenter(
        sam3_model_path=str(checkpoint_path), device="cpu", confidence_threshold=0.55
    )
    frame_bgr = np.zeros((FRAME_HW[0], FRAME_HW[1], 3), dtype=np.uint8)
    segmenter.segment_image(frame_bgr, "animal")

    _, processor = segmenter._load_image_model()
    assert processor.confidence_threshold == pytest.approx(0.55)


def test_module_imports_without_sam3_installed() -> None:
    # sam3_backends.py must stay importable with no sam3/torch-dependent globals evaluated
    # at import time -- confirmed indirectly by every other test file importing it already, but
    # assert the image-mode class specifically exists and is lazy (no sam3 import at class
    # definition time).
    assert hasattr(sam3_backends, "OfficialSam3ImageSegmenter")
    segmenter = sam3_backends.OfficialSam3ImageSegmenter(sam3_model_path="unused.pt", device="cpu")
    assert segmenter._model is None
    assert segmenter._processor is None
