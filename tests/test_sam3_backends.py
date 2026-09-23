from __future__ import annotations

import json
import sys
from pathlib import Path

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
from apps.camera_trap.sam3_backends import PROMPT_TRACK_ID_OFFSET, OfficialSam3Backend, SamFrameResult


FRAME_SIZE = (8, 8)  # (width, height)


def _write_synthetic_video(path: Path, num_frames: int, fps: float = 6.0) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, FRAME_SIZE)
    assert writer.isOpened()
    for i in range(num_frames):
        frame = np.full((FRAME_SIZE[1], FRAME_SIZE[0], 3), (i * 20) % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _mask(row: int) -> np.ndarray:
    mask = np.zeros((FRAME_SIZE[1], FRAME_SIZE[0]), dtype=bool)
    mask[row, row] = True
    return mask


class FakeOfficialPredictor:
    """Fake for sam3's video predictor.handle_request/handle_stream_request.

    `per_prompt_frames[prompt]` is a list of per-frame dicts (one per sampled frame, in
    propagation order): {"ids": [...], "probs": [...], "masks": [np.ndarray bool HxW, ...]}.
    """

    def __init__(self, per_prompt_frames: dict[str, list[dict]]) -> None:
        self.per_prompt_frames = per_prompt_frames
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
            self.sessions[request["session_id"]]["prompt"] = request["text"]
            return {}
        if rtype == "close_session":
            self.closed_sessions.append(request["session_id"])
            return {}
        raise ValueError(f"unexpected request type: {rtype}")

    def handle_stream_request(self, request: dict):
        sid = request["session_id"]
        prompt = self.sessions[sid]["prompt"]
        frames = self.per_prompt_frames[prompt]
        for i, frame in enumerate(frames):
            masks = frame["masks"]
            out_binary_masks = (
                np.stack(masks) if masks else np.zeros((0, FRAME_SIZE[1], FRAME_SIZE[0]), dtype=bool)
            )
            yield {
                "frame_index": i,
                "outputs": {
                    "out_obj_ids": list(frame["ids"]),
                    "out_probs": list(frame["probs"]),
                    "out_boxes_xywh": [[0.0, 0.0, 0.0, 0.0] for _ in frame["ids"]],
                    "out_binary_masks": out_binary_masks,
                },
            }


def _build_backend(tmp_path: Path, per_prompt_frames: dict[str, list[dict]], det_threshold: float = 0.5) -> OfficialSam3Backend:
    backend = OfficialSam3Backend(sam3_model_path="unused.pt", det_threshold=det_threshold, device="cpu")
    backend._predictor = FakeOfficialPredictor(per_prompt_frames)  # skip lazy sam3 import entirely
    return backend


def test_official_backend_sampled_frame_indices(tmp_path: Path) -> None:
    video_path = tmp_path / "vid.mp4"
    _write_synthetic_video(video_path, num_frames=6)

    per_prompt_frames = {
        "animal": [
            {"ids": [1], "probs": [0.9], "masks": [_mask(1)]},
            {"ids": [1], "probs": [0.9], "masks": [_mask(2)]},
            {"ids": [1], "probs": [0.9], "masks": [_mask(3)]},
        ]
    }
    backend = _build_backend(tmp_path, per_prompt_frames)

    results = list(
        backend.iter_video(
            video_path=video_path,
            mode="track",
            prompts=["animal"],
            prompt_slugs=["animal"],
            sample_interval=2,
            video_fps=6.0,
            frame_count_est=6,
            estimated_sampled_frames=3,
            warnings=[],
        )
    )
    assert [r.frame_idx for r in results] == [0, 2, 4]
    assert backend._predictor.closed_sessions == ["sess-0"]


def test_official_backend_contract_keys_and_types(tmp_path: Path) -> None:
    video_path = tmp_path / "vid.mp4"
    _write_synthetic_video(video_path, num_frames=3)

    per_prompt_frames = {"animal": [{"ids": [7], "probs": [0.8], "masks": [_mask(1)]}]}
    backend = _build_backend(tmp_path, per_prompt_frames)

    results = list(
        backend.iter_video(
            video_path=video_path,
            mode="track",
            prompts=["animal"],
            prompt_slugs=["animal"],
            sample_interval=1,
            video_fps=3.0,
            frame_count_est=3,
            estimated_sampled_frames=3,
            warnings=[],
        )
    )
    assert len(results) == 3
    result = results[0]
    assert isinstance(result, SamFrameResult)
    assert result.status is None
    assert isinstance(result.frame_bgr, np.ndarray)
    assert len(result.prompt_masks) == 1
    assert result.prompt_masks[0].dtype == np.uint8
    assert len(result.object_rows) == 1
    row = result.object_rows[0]
    for key in (
        "object_index",
        "track_id",
        "prompt_index",
        "prompt",
        "slug",
        "label",
        "confidence",
        "bbox_xyxy",
        "center_xy",
        "mask_nonzero_pixels",
        "mask",
    ):
        assert key in row
    assert row["mask"].dtype == np.uint8
    assert isinstance(result.track_summary, dict)
    assert "active_track_count" in result.track_summary and "tracks" in result.track_summary


def test_official_backend_multi_prompt_mapping_and_track_id_offset(tmp_path: Path) -> None:
    video_path = tmp_path / "vid.mp4"
    _write_synthetic_video(video_path, num_frames=1)

    per_prompt_frames = {
        "animal": [{"ids": [1], "probs": [0.9], "masks": [_mask(1)]}],
        "bird": [{"ids": [1], "probs": [0.9], "masks": [_mask(2)]}],
    }
    backend = _build_backend(tmp_path, per_prompt_frames)

    results = list(
        backend.iter_video(
            video_path=video_path,
            mode="track",
            prompts=["animal", "bird"],
            prompt_slugs=["animal", "bird"],
            sample_interval=1,
            video_fps=2.0,
            frame_count_est=2,
            estimated_sampled_frames=1,
            warnings=[],
        )
    )
    assert len(results) == 1
    rows = {row["prompt"]: row for row in results[0].object_rows}
    assert set(rows.keys()) == {"animal", "bird"}
    assert rows["animal"]["prompt_index"] == 0
    assert rows["animal"]["slug"] == "animal"
    assert rows["animal"]["track_id"] == 1
    assert rows["bird"]["prompt_index"] == 1
    assert rows["bird"]["slug"] == "bird"
    # Same native obj id (1) in both prompts' sessions must stay unique across prompts.
    assert rows["bird"]["track_id"] == PROMPT_TRACK_ID_OFFSET + 1
    assert rows["animal"]["track_id"] != rows["bird"]["track_id"]


def test_official_backend_track_id_persists_across_frames(tmp_path: Path) -> None:
    video_path = tmp_path / "vid.mp4"
    _write_synthetic_video(video_path, num_frames=3)

    per_prompt_frames = {
        "animal": [
            {"ids": [5], "probs": [0.9], "masks": [_mask(1)]},
            {"ids": [5], "probs": [0.9], "masks": [_mask(2)]},
            {"ids": [5], "probs": [0.9], "masks": [_mask(3)]},
        ]
    }
    backend = _build_backend(tmp_path, per_prompt_frames)

    results = list(
        backend.iter_video(
            video_path=video_path,
            mode="track",
            prompts=["animal"],
            prompt_slugs=["animal"],
            sample_interval=1,
            video_fps=3.0,
            frame_count_est=3,
            estimated_sampled_frames=3,
            warnings=[],
        )
    )
    track_ids = [r.object_rows[0]["track_id"] for r in results]
    assert track_ids == [5, 5, 5]


def test_official_backend_threshold_filtering(tmp_path: Path) -> None:
    video_path = tmp_path / "vid.mp4"
    _write_synthetic_video(video_path, num_frames=1)

    per_prompt_frames = {
        "animal": [
            {"ids": [1, 2], "probs": [0.9, 0.1], "masks": [_mask(1), _mask(2)]},
        ]
    }
    backend = _build_backend(tmp_path, per_prompt_frames, det_threshold=0.5)

    results = list(
        backend.iter_video(
            video_path=video_path,
            mode="track",
            prompts=["animal"],
            prompt_slugs=["animal"],
            sample_interval=1,
            video_fps=1.0,
            frame_count_est=1,
            estimated_sampled_frames=1,
            warnings=[],
        )
    )
    assert len(results) == 1
    ids = [row["track_id"] for row in results[0].object_rows]
    assert ids == [1]  # obj id 2 (prob 0.1) dropped by det_threshold


def test_official_backend_frame_mode_raises_clear_error(tmp_path: Path) -> None:
    backend = OfficialSam3Backend(sam3_model_path="unused.pt")
    with pytest.raises(NotImplementedError, match="track only"):
        backend.prepare_for_video(mode="frame")


def test_parse_args_rejects_official_backend_with_frame_mode() -> None:
    with pytest.raises(SystemExit):
        dap3_cli.parse_args(
            [
                "--input-video-dir", "in",
                "--output-dir", "out",
                "--sam3-model-path", "weights/sam3/model.pt",
                "--sam3-text-prompts", "ape",
                "--da3-batch-size", "4",
                "--sam3-backend", "official",
                "--sam3-mode", "frame",
            ]
        )


def test_parse_args_sam3_backend_defaults() -> None:
    args = dap3_cli.parse_args(
        [
            "--input-video-dir", "in",
            "--output-dir", "out",
            "--sam3-model-path", "weights/sam3/model.pt",
            "--sam3-text-prompts", "ape",
            "--da3-batch-size", "4",
        ]
    )
    assert args.sam3_backend == "official"
    assert args.sam3_det_threshold == pytest.approx(0.5)


class _FakeBackend:
    """Backend-agnostic fake: yields pre-built SamFrameResult objects for process_video."""

    def __init__(self, prompts: list[str], prompt_slugs: list[str]) -> None:
        self.prompts = prompts
        self.prompt_slugs = prompt_slugs

    def prepare_for_video(self, *, mode: str) -> list[str]:
        return []

    def iter_video(self, **kwargs):
        for i in range(3):
            frame_idx = i * kwargs["sample_interval"]
            frame_bgr = np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8)
            mask = np.zeros((FRAME_SIZE[1], FRAME_SIZE[0]), dtype=np.uint8)
            mask[2:5, 2:5] = 1
            object_rows = [
                {
                    "object_index": 0,
                    "track_id": i,
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


def test_process_video_end_to_end_with_fake_backend_and_fake_da3(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video_path = tmp_path / "vid.mp4"
    _write_synthetic_video(video_path, num_frames=6)

    prompts = ["animal"]
    prompt_slugs = dap3_cli.build_prompt_slugs(prompts)

    monkeypatch.setattr(
        dap3_cli,
        "run_da3_inference_batch",
        lambda da3, frames_bgr: [np.full((FRAME_SIZE[1], FRAME_SIZE[0]), 2.5, dtype=np.float32) for _ in frames_bgr],
    )

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

    entry = dap3_cli.process_video(
        video_path=video_path,
        output_root=tmp_path / "out",
        args=args,
        backend=_FakeBackend(prompts, prompt_slugs),
        da3=object(),
        da3_stream_config=Path(args.da3_stream_config),
        prompt_slugs=prompt_slugs,
        device=torch.device("cpu"),
        da3_batch_size=4,
        track_setup_actions=[],
    )

    assert entry["status"] == "success"
    json_path = Path(entry["json_path"])
    assert json_path.exists()
    data = json.loads(json_path.read_text())
    processed = [f for f in data["frames"] if f["status"] == "processed"]
    assert len(processed) == 3
    for frame in processed:
        assert frame["objects"], "expected at least one object per processed frame"
        for obj in frame["objects"]:
            assert obj["depth_mask_mean"] == pytest.approx(2.5)
