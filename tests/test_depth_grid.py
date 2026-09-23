"""Tests for --depth-interval-seconds (the DA3 depth grid) in dap3_cli.py/sam3_backends.py.

Covers: `compute_depth_grid_step`, `compute_missed_grid_frames`, the official-backend
stride-union-grid sampling, the ultralytics intersection+warning path, and the
process_video-level frame classification (grid frames go to DA3 and stay "processed"/
"empty_mask"; non-grid frames become "tracked" with null depth and are never sent to DA3),
including float16 native-resolution NPZ depth storage and the `--depth-interval-seconds 0`
old-behaviour escape hatch.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import cv2
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
from apps.camera_trap.cli import dap3_cli
from apps.camera_trap.sam3_backends import OfficialSam3Backend, SamFrameResult


FRAME_SIZE = (8, 8)  # (width, height)
NATIVE_DEPTH_SHAPE = (4, 4)  # simulates DA3's native output resolution != frame resolution


def _write_synthetic_video(path: Path, num_frames: int, fps: float) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, FRAME_SIZE)
    assert writer.isOpened()
    for i in range(num_frames):
        frame = np.full((FRAME_SIZE[1], FRAME_SIZE[0], 3), (i * 5) % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()


# ---------------------------------------------------------------------------
# compute_depth_grid_step / compute_missed_grid_frames (pure math)
# ---------------------------------------------------------------------------


def test_compute_depth_grid_step_matches_exporter_formula() -> None:
    # export_job_distances_csv.py: step_frames = max(1, round(interval_seconds * fps_value))
    for fps, interval in [(30.0, 2.0), (25.0, 2.0), (6.0, 1.5), (23.976, 3.0)]:
        expected = max(1, int(round(interval * fps)))
        assert dap3_cli.compute_depth_grid_step(fps, interval) == expected


def test_compute_depth_grid_step_disabled_and_zero_fps() -> None:
    assert dap3_cli.compute_depth_grid_step(30.0, 0.0) is None
    assert dap3_cli.compute_depth_grid_step(30.0, -1.0) is None
    assert dap3_cli.compute_depth_grid_step(0.0, 2.0) == 1


def test_compute_missed_grid_frames_divisible_step_has_no_misses() -> None:
    # sample_interval=5, depth_grid_step=60: 60 % 5 == 0, so every grid frame is reachable.
    missed, total = sam3_backends.compute_missed_grid_frames(
        sample_interval=5, depth_grid_step=60, frame_count_est=130
    )
    assert total == 3  # 0, 60, 120
    assert missed == 0


def test_compute_missed_grid_frames_non_divisible_step_has_misses() -> None:
    # sample_interval=4, depth_grid_step=50: 50 % 4 == 2, no grid frame is a stride multiple.
    missed, total = sam3_backends.compute_missed_grid_frames(
        sample_interval=4, depth_grid_step=50, frame_count_est=110
    )
    assert total == 3  # 0, 50, 100
    assert missed == 1  # only 50 is not a multiple of 4 (0 and 100 are)


def test_compute_missed_grid_frames_disabled_grid() -> None:
    assert sam3_backends.compute_missed_grid_frames(
        sample_interval=4, depth_grid_step=None, frame_count_est=100
    ) == (0, 0)


# ---------------------------------------------------------------------------
# OfficialSam3Backend: yielded frame set = stride frames UNION grid frames
# ---------------------------------------------------------------------------


class FakeOfficialPredictor:
    """Minimal fake for sam3's video predictor.handle_request/handle_stream_request."""

    def __init__(self) -> None:
        self.sessions: dict[str, dict] = {}
        self.closed_sessions: list[str] = []
        self._next_session_id = 0

    def handle_request(self, request: dict) -> dict:
        rtype = request["type"]
        if rtype == "start_session":
            sid = f"sess-{self._next_session_id}"
            self._next_session_id += 1
            self.sessions[sid] = {"resource_path": request["resource_path"]}
            return {"session_id": sid}
        if rtype == "add_prompt":
            return {}
        if rtype == "close_session":
            self.closed_sessions.append(request["session_id"])
            return {}
        raise ValueError(f"unexpected request type: {rtype}")

    def handle_stream_request(self, request: dict):
        # No detections needed for this test: we only care about which frame_idx values the
        # backend samples, not about per-frame objects.
        return iter(())


def test_official_backend_yields_stride_union_grid_frames(tmp_path: Path) -> None:
    video_path = tmp_path / "vid.mp4"
    video_fps = 25.0
    num_frames = 110
    _write_synthetic_video(video_path, num_frames=num_frames, fps=video_fps)

    target_fps = 6.0
    sample_interval = dap3_cli.compute_sample_interval(video_fps, target_fps)
    assert sample_interval == 4  # round(25/6)
    depth_grid_step = dap3_cli.compute_depth_grid_step(video_fps, 2.0)
    assert depth_grid_step == 50  # round(25*2), NOT a multiple of 4

    backend = OfficialSam3Backend(sam3_model_path="unused.pt", device="cpu")
    backend._predictor = FakeOfficialPredictor()

    warnings: list[str] = []
    results = list(
        backend.iter_video(
            video_path=video_path,
            mode="track",
            prompts=["animal"],
            prompt_slugs=["animal"],
            sample_interval=sample_interval,
            video_fps=video_fps,
            frame_count_est=num_frames,
            estimated_sampled_frames=None,
            warnings=warnings,
            depth_grid_step=depth_grid_step,
        )
    )
    got_indices = sorted({r.frame_idx for r in results})

    stride_indices = set(range(0, num_frames, sample_interval))
    grid_indices = set(range(0, num_frames, depth_grid_step))
    expected = sorted(stride_indices | grid_indices)

    assert got_indices == expected
    # The grid step is deliberately not a stride multiple, so this union must be a strict
    # superset of the plain stride sampling (proving the official backend really added grid
    # frames the stride alone would have missed).
    assert grid_indices - stride_indices  # sanity: the fixture actually exercises the union
    assert (grid_indices - stride_indices).issubset(got_indices)


def test_official_backend_no_grid_step_samples_stride_only(tmp_path: Path) -> None:
    video_path = tmp_path / "vid.mp4"
    video_fps = 25.0
    num_frames = 40
    _write_synthetic_video(video_path, num_frames=num_frames, fps=video_fps)

    backend = OfficialSam3Backend(sam3_model_path="unused.pt", device="cpu")
    backend._predictor = FakeOfficialPredictor()

    results = list(
        backend.iter_video(
            video_path=video_path,
            mode="track",
            prompts=["animal"],
            prompt_slugs=["animal"],
            sample_interval=4,
            video_fps=video_fps,
            frame_count_est=num_frames,
            estimated_sampled_frames=None,
            warnings=[],
            depth_grid_step=None,
        )
    )
    assert sorted({r.frame_idx for r in results}) == list(range(0, num_frames, 4))


# ---------------------------------------------------------------------------
# UltralyticsBackend: stride can't be widened -> intersection only + one warning
# ---------------------------------------------------------------------------


class _FakeSam3FrameResult:
    """Duck-types just enough of an ultralytics Results object for extract_prompt_masks_and_objects."""

    masks = None
    boxes = None
    names: dict = {}


def test_ultralytics_frame_mode_intersects_grid_with_stride_and_warns_once(tmp_path: Path) -> None:
    video_path = tmp_path / "vid.mp4"
    video_fps = 25.0
    num_frames = 110
    _write_synthetic_video(video_path, num_frames=num_frames, fps=video_fps)

    sample_interval = 4
    depth_grid_step = 50  # not a multiple of 4 -> some grid frames unreachable by the stride

    backend = sam3_backends.UltralyticsBackend(
        sam3_model_path="unused.pt", conf=0.25, half=False
    )
    backend._sam3_frame = lambda source, text: [_FakeSam3FrameResult()]

    warnings: list[str] = []
    results = list(
        backend._iter_frame_mode(
            video_path=video_path,
            prompts=["animal"],
            prompt_slugs=["animal"],
            sample_interval=sample_interval,
            frame_count_est=num_frames,
            warnings=warnings,
            depth_grid_step=depth_grid_step,
        )
    )
    got_indices = sorted({r.frame_idx for r in results})
    assert got_indices == list(range(0, num_frames, sample_interval))  # stride only, no union

    assert len(warnings) == 1
    assert "sample_interval=4" in warnings[0]
    assert "depth_grid_step=50" in warnings[0]
    assert "1 of 3" in warnings[0]  # grid frames 0,50,100 in [0,110); only 50 is missed


def test_ultralytics_frame_mode_no_warning_when_grid_divides_stride(tmp_path: Path) -> None:
    video_path = tmp_path / "vid.mp4"
    video_fps = 30.0
    num_frames = 60
    _write_synthetic_video(video_path, num_frames=num_frames, fps=video_fps)

    backend = sam3_backends.UltralyticsBackend(
        sam3_model_path="unused.pt", conf=0.25, half=False
    )
    backend._sam3_frame = lambda source, text: [_FakeSam3FrameResult()]

    warnings: list[str] = []
    list(
        backend._iter_frame_mode(
            video_path=video_path,
            prompts=["animal"],
            prompt_slugs=["animal"],
            sample_interval=5,
            frame_count_est=num_frames,
            warnings=warnings,
            depth_grid_step=60,  # 60 % 5 == 0: fully covered
        )
    )
    assert warnings == []


# ---------------------------------------------------------------------------
# process_video-level frame classification (grid -> DA3/"processed"/"empty_mask",
# non-grid -> "tracked" with null depth, never sent to DA3)
# ---------------------------------------------------------------------------


class _GridFakeBackend:
    """Yields one SamFrameResult per stride-sampled frame_idx, all with a non-empty mask."""

    def __init__(self, prompts: list[str], prompt_slugs: list[str], frame_indices: list[int]) -> None:
        self.prompts = prompts
        self.prompt_slugs = prompt_slugs
        self.frame_indices = frame_indices

    def prepare_for_video(self, *, mode: str) -> list[str]:
        return []

    def iter_video(self, **kwargs):
        for frame_idx in self.frame_indices:
            frame_bgr = np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8)
            mask = np.zeros((FRAME_SIZE[1], FRAME_SIZE[0]), dtype=np.uint8)
            mask[2:5, 2:5] = 1
            object_rows = [
                {
                    "object_index": 0,
                    "track_id": frame_idx,
                    "prompt_index": 0,
                    "prompt": self.prompts[0],
                    "slug": self.prompt_slugs[0],
                    "label": self.prompts[0],
                    "confidence": 0.9,
                    "bbox_xyxy": [2, 2, 4, 4],
                    "center_xy": [3, 3],
                    "mask_nonzero_pixels": int(mask.sum()),
                    "mask": mask,
                }
            ]
            yield SamFrameResult(
                frame_idx=frame_idx,
                frame_bgr=frame_bgr,
                prompt_masks=[mask],
                object_rows=object_rows,
                track_summary={"active_track_count": 1, "tracks": []},
                timing_ms_sam=1.0,
                status=None,
                error=None,
            )


def _fake_da3_batch_calls(calls: list[int]):
    def _run(da3, frames_bgr):
        calls.append(len(frames_bgr))
        return [
            np.full(NATIVE_DEPTH_SHAPE, 3.5, dtype=np.float32) for _ in frames_bgr
        ]

    return _run


def test_process_video_depth_grid_classification_and_npz(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video_path = tmp_path / "vid.mp4"
    video_fps = 30.0
    _write_synthetic_video(video_path, num_frames=121, fps=video_fps)

    prompts = ["animal"]
    prompt_slugs = dap3_cli.build_prompt_slugs(prompts)

    target_fps = 6.0
    sample_interval = dap3_cli.compute_sample_interval(video_fps, target_fps)
    assert sample_interval == 5
    depth_grid_step = dap3_cli.compute_depth_grid_step(video_fps, 2.0)
    assert depth_grid_step == 60

    stride_indices = list(range(0, 121, sample_interval))  # 0,5,...,120
    grid_indices = {i for i in stride_indices if i % depth_grid_step == 0}
    assert grid_indices == {0, 60, 120}

    da3_calls: list[int] = []
    monkeypatch.setattr(dap3_cli, "run_da3_inference_batch", _fake_da3_batch_calls(da3_calls))

    args = dap3_cli.parse_args(
        [
            "--input-video-dir", str(tmp_path),
            "--output-dir", str(tmp_path / "out"),
            "--sam3-model-path", "weights/sam3/model.pt",
            "--sam3-text-prompts", *prompts,
            "--da3-batch-size", "4",
            "--target-fps", str(target_fps),
            "--depth-interval-seconds", "2.0",
            "--npz",
        ]
    )

    backend = _GridFakeBackend(prompts, prompt_slugs, stride_indices)
    entry = dap3_cli.process_video(
        video_path=video_path,
        output_root=tmp_path / "out",
        args=args,
        backend=backend,
        da3=object(),
        da3_stream_config=Path(args.da3_stream_config),
        prompt_slugs=prompt_slugs,
        device=torch.device("cpu"),
        da3_batch_size=4,
        track_setup_actions=[],
    )

    assert entry["status"] == "success"
    assert entry["depth_grid_step_frames"] == 60
    assert sum(da3_calls) == len(grid_indices)  # DA3 only called for grid frames

    import json

    data = json.loads(Path(entry["json_path"]).read_text())
    assert data["depth_interval_seconds"] == pytest.approx(2.0)
    assert data["depth_grid_step_frames"] == 60

    frames_by_idx = {f["frame_index"]: f for f in data["frames"]}
    assert set(frames_by_idx) == set(stride_indices)

    for idx in grid_indices:
        frame = frames_by_idx[idx]
        assert frame["status"] == "processed"
        assert frame["depth_mask_mean"] == pytest.approx(3.5)
        assert frame["depth_shape"] == list(NATIVE_DEPTH_SHAPE)
        for obj in frame["objects"]:
            assert obj["depth_mask_mean"] == pytest.approx(3.5)

    non_grid_indices = set(stride_indices) - grid_indices
    assert non_grid_indices  # sanity: fixture actually exercises non-grid frames
    for idx in non_grid_indices:
        frame = frames_by_idx[idx]
        assert frame["status"] == "tracked"
        assert frame["depth_mask_mean"] is None
        assert frame["depth_center_value"] is None
        assert frame["depth_shape"] is None
        assert frame["objects"], "tracked frames must still carry SAM objects"
        for obj in frame["objects"]:
            assert obj["depth_mask_mean"] is None
            assert obj["depth_center_value"] is None

    npz_path = Path(entry["npz_path"])
    assert npz_path.exists()
    with np.load(npz_path, allow_pickle=True) as npz_data:
        for idx in grid_indices:
            depth_key = f"f{idx}_depth"
            assert depth_key in npz_data
            assert npz_data[depth_key].dtype == np.float16
            assert npz_data[depth_key].shape == NATIVE_DEPTH_SHAPE
        for idx in non_grid_indices:
            assert f"f{idx}_depth" not in npz_data
            # Masks/objects are still present for tracked (non-grid) frames.
            mask_keys = [k for k in npz_data.files if k.startswith(f"f{idx}_mask_")]
            assert mask_keys

    assert entry["counts"]["tracked_frames"] == len(non_grid_indices)
    assert entry["counts"]["processed_frames"] == len(grid_indices)


def test_process_video_depth_interval_zero_keeps_old_behaviour(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video_path = tmp_path / "vid.mp4"
    video_fps = 6.0
    _write_synthetic_video(video_path, num_frames=6, fps=video_fps)

    prompts = ["animal"]
    prompt_slugs = dap3_cli.build_prompt_slugs(prompts)

    da3_calls: list[int] = []
    monkeypatch.setattr(dap3_cli, "run_da3_inference_batch", _fake_da3_batch_calls(da3_calls))

    args = dap3_cli.parse_args(
        [
            "--input-video-dir", str(tmp_path),
            "--output-dir", str(tmp_path / "out"),
            "--sam3-model-path", "weights/sam3/model.pt",
            "--sam3-text-prompts", *prompts,
            "--da3-batch-size", "4",
            "--depth-interval-seconds", "0",
        ]
    )
    assert dap3_cli.compute_depth_grid_step(video_fps, args.depth_interval_seconds) is None

    stride_indices = [0, 1, 2, 3, 4, 5]
    backend = _GridFakeBackend(prompts, prompt_slugs, stride_indices)
    entry = dap3_cli.process_video(
        video_path=video_path,
        output_root=tmp_path / "out",
        args=args,
        backend=backend,
        da3=object(),
        da3_stream_config=Path(args.da3_stream_config),
        prompt_slugs=prompt_slugs,
        device=torch.device("cpu"),
        da3_batch_size=4,
        track_setup_actions=[],
    )

    assert entry["status"] == "success"
    assert entry["depth_grid_step_frames"] is None
    assert sum(da3_calls) == len(stride_indices)  # every non-empty-mask frame goes to DA3

    import json

    data = json.loads(Path(entry["json_path"]).read_text())
    assert data["depth_grid_step_frames"] is None
    statuses = {f["status"] for f in data["frames"]}
    assert statuses == {"processed"}
    assert "tracked" not in statuses
