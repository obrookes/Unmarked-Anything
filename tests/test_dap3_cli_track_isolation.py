import types

import pytest

from apps.camera_trap.cli import dap3_cli


def test_parse_args_track_defaults() -> None:
    args = dap3_cli.parse_args(
        [
            "--input-video-dir",
            "in",
            "--output-dir",
            "out",
            "--sam3-model-path",
            "weights/sam3/model.pt",
            "--sam3-text-prompts",
            "ape",
            "--da3-batch-size",
            "4",
        ]
    )
    assert args.sam3_track_isolation == "recreate"
    assert args.sam3_track_tail_policy == "warn_and_finalize"
    assert args.da3_mode == "batch"
    assert args.mask_storage_format == "both"


def test_parse_args_track_overrides() -> None:
    args = dap3_cli.parse_args(
        [
            "--input-video-dir",
            "in",
            "--output-dir",
            "out",
            "--sam3-model-path",
            "weights/sam3/model.pt",
            "--sam3-text-prompts",
            "ape",
            "--da3-batch-size",
            "4",
            "--sam3-track-isolation",
            "both",
            "--sam3-track-tail-policy",
            "fail_fast",
        ]
    )
    assert args.sam3_track_isolation == "both"
    assert args.sam3_track_tail_policy == "fail_fast"


def test_parse_args_da3_stream_mode() -> None:
    args = dap3_cli.parse_args(
        [
            "--input-video-dir",
            "in",
            "--output-dir",
            "out",
            "--sam3-model-path",
            "weights/sam3/model.pt",
            "--sam3-text-prompts",
            "ape",
            "--da3-batch-size",
            "4",
            "--da3-mode",
            "stream",
            "--da3-stream-config",
            "da3_streaming/configs/base_config.yaml",
        ]
    )
    assert args.da3_mode == "stream"
    assert args.da3_stream_config == "da3_streaming/configs/base_config.yaml"


def test_parse_args_mask_storage_override() -> None:
    args = dap3_cli.parse_args(
        [
            "--input-video-dir",
            "in",
            "--output-dir",
            "out",
            "--sam3-model-path",
            "weights/sam3/model.pt",
            "--sam3-text-prompts",
            "ape",
            "--da3-batch-size",
            "4",
            "--mask-storage-format",
            "rle",
        ]
    )
    assert args.mask_storage_format == "rle"


class _FakeTracker:
    def __init__(self) -> None:
        self.inference_state = {"tracker": 1}
        self.reset_image_calls = 0

    def reset_image(self) -> None:
        self.reset_image_calls += 1


class _FakePredictor:
    def __init__(self) -> None:
        self.inference_state = {"num_frames": 123, "x": 1}
        self.model = object()
        self.dataset = object()
        self.batch = object()
        self.results = object()
        self.seen = 12
        self.reset_prompts_calls = 0
        self.reset_image_calls = 0
        self.tracker = _FakeTracker()

    def reset_prompts(self) -> None:
        self.reset_prompts_calls += 1

    def reset_image(self) -> None:
        self.reset_image_calls += 1


def test_reset_sam3_track_predictor_state_clears_fields() -> None:
    predictor = _FakePredictor()
    actions = dap3_cli.reset_sam3_track_predictor_state(predictor)
    assert "reset_prompts_skipped" in actions
    assert "reset_image" in actions
    assert predictor.reset_prompts_calls == 0
    assert predictor.reset_image_calls == 1
    assert predictor.inference_state == {}
    assert predictor.dataset is None
    assert predictor.batch is None
    assert predictor.results is None
    assert predictor.seen == 0
    assert predictor.tracker.inference_state == {}
    assert predictor.tracker.reset_image_calls == 1


def test_prepare_track_predictor_recreate(monkeypatch: pytest.MonkeyPatch) -> None:
    created = []

    def _factory(overrides):
        obj = types.SimpleNamespace(created_with=overrides)
        created.append(obj)
        return obj

    monkeypatch.setattr(dap3_cli, "create_sam3_track_predictor", _factory)
    predictor, actions = dap3_cli.prepare_sam3_track_predictor_for_video(
        sam3_track=object(),
        isolation_mode="recreate",
        sam3_overrides={"a": 1},
    )
    assert predictor is created[0]
    assert actions == ["recreate"]


def test_prepare_track_predictor_reset_reuses_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    predictor = object()
    reset_calls = []

    def _reset(obj):
        reset_calls.append(obj)
        return ["cleared"]

    monkeypatch.setattr(dap3_cli, "reset_sam3_track_predictor_state", _reset)
    monkeypatch.setattr(
        dap3_cli,
        "create_sam3_track_predictor",
        lambda overrides: (_ for _ in ()).throw(AssertionError("create should not be called")),
    )
    out, actions = dap3_cli.prepare_sam3_track_predictor_for_video(
        sam3_track=predictor,
        isolation_mode="reset",
        sam3_overrides={},
    )
    assert out is predictor
    assert reset_calls == [predictor]
    assert actions == ["reset", "reset:cleared"]


def test_prepare_track_predictor_both_resets_then_recreates(monkeypatch: pytest.MonkeyPatch) -> None:
    order = []

    def _reset(obj):
        order.append("reset")
        return ["old"]

    def _factory(overrides):
        order.append("create")
        return types.SimpleNamespace()

    monkeypatch.setattr(dap3_cli, "reset_sam3_track_predictor_state", _reset)
    monkeypatch.setattr(dap3_cli, "create_sam3_track_predictor", _factory)
    _, actions = dap3_cli.prepare_sam3_track_predictor_for_video(
        sam3_track=object(),
        isolation_mode="both",
        sam3_overrides={},
    )
    assert order == ["reset", "create"]
    assert actions == ["reset_previous", "reset:old", "recreate"]


def test_validate_track_state_num_frames_mismatch_raises() -> None:
    predictor = types.SimpleNamespace(inference_state={"num_frames": 14})
    with pytest.raises(RuntimeError, match="state mismatch"):
        dap3_cli.validate_track_state_num_frames(
            sam3_track=predictor,
            video_stem="video_a",
            expected_total_frames=10,
            expected_sampled_frames=10,
            sample_interval=24,
        )


def test_validate_track_state_num_frames_matches_total_frames() -> None:
    predictor = types.SimpleNamespace(inference_state={"num_frames": 1454})
    out = dap3_cli.validate_track_state_num_frames(
        sam3_track=predictor,
        video_stem="video_a",
        expected_total_frames=1454,
        expected_sampled_frames=61,
        sample_interval=24,
    )
    assert out == 1454


def test_handle_track_stream_index_error_warn_and_finalize() -> None:
    warnings = []
    should_finalize = dap3_cli.handle_track_stream_index_error(
        exc=IndexError("tail"),
        video_stem="video_a",
        sampled_idx=3,
        dataset_frame=72,
        sample_interval=24,
        isolation_mode="recreate",
        tail_policy="warn_and_finalize",
        warnings=warnings,
    )
    assert should_finalize is True
    assert len(warnings) == 1
    assert "video=video_a" in warnings[0]


def test_handle_track_stream_index_error_fail_fast() -> None:
    with pytest.raises(RuntimeError, match="fail-fast"):
        dap3_cli.handle_track_stream_index_error(
            exc=IndexError("tail"),
            video_stem="video_a",
            sampled_idx=3,
            dataset_frame=72,
            sample_interval=24,
            isolation_mode="recreate",
            tail_policy="fail_fast",
            warnings=[],
        )
