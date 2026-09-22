"""SAM3 segmentation/tracking backends for the camera-trap CLI (dap3_cli.py).

Two backends implement the same seam: iterate sampled frames of a video and yield a
`SamFrameResult` per sampled frame, carrying already-converted `(prompt_masks, object_rows,
track_summary)` in the output contract dap3_cli.py needs. dap3_cli.py's per-video loop then
becomes backend-agnostic: mask-union/empty-mask bookkeeping, DA3 dispatch, and JSON/NPZ writing
all operate purely on `SamFrameResult` objects.

- `UltralyticsBackend`: the pre-existing ultralytics SAM3 port. All predictor construction,
  track-mode isolation/reset, tail-policy handling, and result extraction logic previously in
  dap3_cli.py lives here unchanged; `ultralytics` is imported lazily so importing this module
  (or dap3_cli.py) does not require it to be installed.
- `OfficialSam3Backend`: the official facebookresearch/sam3 package (SA-FARI fine-tuned
  checkpoints), ported from vision-llm-ann-generator/sam3_runner.py. `sam3`/`torch` are imported
  lazily. Track mode only (video predictor + propagate_in_video); frame mode raises a clear
  NotImplementedError (the image-model path is not ported here).

Multiple text prompts on the official backend: unlike the ultralytics predictor (one call with
all prompts at once), the confirmed official API (sam3_runner.py) only demonstrates a single text
prompt per session. Rather than guess at undocumented multi-prompt-per-session semantics, this
backend opens one sam3 session per prompt per video (add_prompt + propagate_in_video each), then
merges the per-prompt results by frame index. Track ids are namespaced per prompt
(`prompt_index * PROMPT_TRACK_ID_OFFSET + native_id`) to stay unique across prompts within a
video, matching the contract's "track ids unique per video" requirement.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np

TRACK_ISOLATION_CHOICES = ("recreate", "reset", "both")
TRACK_TAIL_POLICY_CHOICES = ("warn_and_finalize", "fail_fast")
PROMPT_TRACK_ID_OFFSET = 100_000


@dataclass
class SamFrameResult:
    """One sampled frame's SAM3 output, already converted to dap3_cli's contract.

    `status`/`error` mirror dap3_cli's existing per-frame status semantics:
      - status=None: success; `prompt_masks`/`object_rows` are populated (frame_bgr always set).
      - status="sam_error": SAM3 call/parsing failed; `error` set, frame_bgr may or may not be
        available (best effort), prompt_masks/object_rows/track_summary are None.
      - status="frame_decode_error": video decode ended early (frame-mode tail case only);
        frame_bgr/prompt_masks/object_rows/track_summary are all None.
    """

    frame_idx: int
    frame_bgr: np.ndarray | None
    prompt_masks: list[np.ndarray] | None
    object_rows: list[dict[str, Any]] | None
    track_summary: dict[str, Any] | None
    timing_ms_sam: float | None
    status: str | None = None
    error: str | None = None


def mask_extent_bbox_center(mask_bool: np.ndarray) -> tuple[list[int] | None, list[int] | None]:
    """bbox_xyxy/center_xy from a boolean mask's nonzero extent, or (None, None) if empty."""
    if not mask_bool.any():
        return None, None
    ys, xs = np.where(mask_bool)
    xmin, xmax = int(xs.min()), int(xs.max())
    ymin, ymax = int(ys.min()), int(ys.max())
    bbox_xyxy = [xmin, ymin, xmax, ymax]
    center_xy = [int((xmin + xmax) // 2), int((ymin + ymax) // 2)]
    return bbox_xyxy, center_xy


def _find_prompt_idx_by_label(label: str, prompts: list[str]) -> int | None:
    if not label:
        return None
    norm_label = label.strip().casefold()
    for idx, prompt in enumerate(prompts):
        if prompt.strip().casefold() == norm_label:
            return idx
    return None


def extract_prompt_masks_and_objects(
    sam_result: Any,
    prompts: list[str],
    prompt_slugs: list[str],
    frame_shape_hw: tuple[int, int],
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    """Convert an ultralytics SAM3 `Results` object into (prompt_masks, object_rows)."""
    height, width = frame_shape_hw
    prompt_masks = [np.zeros((height, width), dtype=bool) for _ in prompts]
    object_rows: list[dict[str, Any]] = []

    if sam_result.masks is None or sam_result.masks.data is None:
        return [mask.astype(np.uint8) for mask in prompt_masks], object_rows

    masks_data = sam_result.masks.data.detach().cpu().numpy()
    if masks_data.size == 0:
        return [mask.astype(np.uint8) for mask in prompt_masks], object_rows
    det_masks = masks_data > 0
    num_dets = det_masks.shape[0]

    boxes = sam_result.boxes
    xyxy = (
        boxes.xyxy.detach().cpu().numpy()
        if boxes is not None and boxes.xyxy is not None
        else np.zeros((num_dets, 4), dtype=np.float32)
    )
    confs = (
        boxes.conf.detach().cpu().numpy()
        if boxes is not None and boxes.conf is not None
        else np.zeros((num_dets,), dtype=np.float32)
    )

    cls_ids = None
    if boxes is not None and boxes.cls is not None:
        cls_ids = boxes.cls.detach().cpu().numpy().astype(np.int64)
    track_ids = (
        boxes.id.detach().cpu().numpy().astype(np.int64)
        if boxes is not None and getattr(boxes, "is_track", False) and boxes.id is not None
        else np.full((num_dets,), -1, dtype=np.int64)
    )
    names = sam_result.names if hasattr(sam_result, "names") else None

    for det_idx in range(num_dets):
        det_mask = det_masks[det_idx]
        if det_mask.shape != (height, width):
            det_mask = cv2.resize(
                det_mask.astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        cls_id = int(cls_ids[det_idx]) if cls_ids is not None and det_idx < len(cls_ids) else -1
        prompt_idx = None
        if len(prompts) > 0:
            if 0 <= cls_id < len(prompts):
                prompt_idx = cls_id
            elif isinstance(names, dict) and cls_id in names:
                prompt_idx = _find_prompt_idx_by_label(str(names[cls_id]), prompts)

            # Fallback for unknown class mapping: assign to first prompt.
            if prompt_idx is None:
                prompt_idx = 0
            prompt_masks[prompt_idx] |= det_mask

        mask_pixels = int(det_mask.sum())
        bbox_xyxy, center_xy = mask_extent_bbox_center(det_mask)
        if bbox_xyxy is None and det_idx < len(xyxy):
            x1, y1, x2, y2 = [int(round(v)) for v in xyxy[det_idx].tolist()]
            x1 = max(0, min(x1, width - 1))
            x2 = max(0, min(x2, width - 1))
            y1 = max(0, min(y1, height - 1))
            y2 = max(0, min(y2, height - 1))
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1
            bbox_xyxy = [x1, y1, x2, y2]
            center_xy = [int((x1 + x2) // 2), int((y1 + y2) // 2)]

        label = str(cls_id)
        if isinstance(names, dict) and cls_id in names:
            label = str(names[cls_id])

        object_rows.append(
            {
                "object_index": int(det_idx),
                "track_id": int(track_ids[det_idx]) if det_idx < len(track_ids) and int(track_ids[det_idx]) >= 0 else None,
                "prompt_index": int(prompt_idx) if prompt_idx is not None else None,
                "prompt": prompts[prompt_idx] if prompt_idx is not None and prompt_idx < len(prompts) else None,
                "slug": prompt_slugs[prompt_idx] if prompt_idx is not None and prompt_idx < len(prompt_slugs) else None,
                "label": label,
                "confidence": float(confs[det_idx]) if det_idx < len(confs) else None,
                "bbox_xyxy": bbox_xyxy,
                "center_xy": center_xy,
                "mask_nonzero_pixels": mask_pixels,
                "mask": det_mask.astype(np.uint8),
            }
        )

    return [mask.astype(np.uint8) for mask in prompt_masks], object_rows


def extract_track_summary(sam_result: Any) -> dict[str, Any]:
    """Per-frame track summary from an ultralytics SAM3 track-mode `Results` object."""
    summary = {"active_track_count": 0, "tracks": []}
    if sam_result is None or sam_result.boxes is None or len(sam_result.boxes) == 0:
        return summary

    boxes = sam_result.boxes
    xyxy = boxes.xyxy.detach().cpu().numpy() if boxes.xyxy is not None else np.zeros((0, 4))
    confs = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else np.zeros((len(boxes),))
    classes = boxes.cls.detach().cpu().numpy().astype(np.int64) if boxes.cls is not None else np.zeros((len(boxes),), dtype=np.int64)
    ids = (
        boxes.id.detach().cpu().numpy().astype(np.int64)
        if getattr(boxes, "is_track", False) and boxes.id is not None
        else np.full((len(boxes),), -1, dtype=np.int64)
    )

    names = sam_result.names if hasattr(sam_result, "names") else {}
    det_mask_pixels = np.zeros((len(boxes),), dtype=np.int64)
    if sam_result.masks is not None and sam_result.masks.data is not None:
        det_masks = sam_result.masks.data.detach().cpu().numpy() > 0
        if det_masks.shape[0] == len(boxes):
            det_mask_pixels = det_masks.reshape(det_masks.shape[0], -1).sum(axis=1).astype(np.int64)

    track_ids_non_null: set[int] = set()
    tracks: list[dict[str, Any]] = []
    for i in range(len(boxes)):
        cls_id = int(classes[i]) if i < len(classes) else -1
        track_id = int(ids[i]) if i < len(ids) and int(ids[i]) >= 0 else None
        if track_id is not None:
            track_ids_non_null.add(track_id)

        label = str(cls_id)
        if isinstance(names, dict) and cls_id in names:
            label = str(names[cls_id])

        bbox = [int(round(v)) for v in xyxy[i].tolist()] if i < len(xyxy) else None
        tracks.append(
            {
                "track_id": track_id,
                "label": label,
                "confidence": float(confs[i]) if i < len(confs) else None,
                "bbox_xyxy": bbox,
                "mask_pixels": int(det_mask_pixels[i]) if i < len(det_mask_pixels) else 0,
            }
        )

    summary["tracks"] = tracks
    summary["active_track_count"] = len(track_ids_non_null) if track_ids_non_null else len(tracks)
    return summary


def track_summary_from_object_rows(object_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-frame track summary built from already-extracted object_rows (official backend)."""
    track_ids_non_null: set[int] = set()
    tracks: list[dict[str, Any]] = []
    for row in object_rows:
        track_id = row.get("track_id")
        if track_id is not None:
            track_ids_non_null.add(track_id)
        tracks.append(
            {
                "track_id": track_id,
                "label": row.get("label"),
                "confidence": row.get("confidence"),
                "bbox_xyxy": row.get("bbox_xyxy"),
                "mask_pixels": int(row.get("mask_nonzero_pixels") or 0),
            }
        )
    return {
        "active_track_count": len(track_ids_non_null) if track_ids_non_null else len(tracks),
        "tracks": tracks,
    }


def create_sam3_track_predictor(sam3_overrides: dict[str, Any]) -> Any:
    try:
        from ultralytics.models.sam import SAM3VideoSemanticPredictor
    except ImportError:
        SAM3VideoSemanticPredictor = None
    if SAM3VideoSemanticPredictor is None:
        raise ImportError(
            "SAM3VideoSemanticPredictor is not available in this ultralytics build. "
            "Upgrade ultralytics or run with --sam3-mode frame."
        )
    return SAM3VideoSemanticPredictor(overrides=sam3_overrides)


def reset_sam3_track_predictor_state(sam3_track: Any) -> list[str]:
    actions: list[str] = []
    if sam3_track is None:
        return actions

    # Do not call reset_prompts() in reset mode.
    # Some ultralytics builds expect prompt-side model internals (for example language feature caches)
    # to persist across calls after model setup; clearing them here can cause KeyError on next video.
    if callable(getattr(sam3_track, "reset_prompts", None)):
        actions.append("reset_prompts_skipped")

    reset_image = getattr(sam3_track, "reset_image", None)
    if callable(reset_image):
        reset_image()
        actions.append("reset_image")

    if hasattr(sam3_track, "inference_state"):
        inference_state = getattr(sam3_track, "inference_state")
        if isinstance(inference_state, dict):
            inference_state.clear()
            actions.append("inference_state.clear")
        else:
            setattr(sam3_track, "inference_state", {})
            actions.append("inference_state={}")
    else:
        setattr(sam3_track, "inference_state", {})
        actions.append("inference_state={}")

    for attr_name in ("dataset", "batch", "results"):
        if hasattr(sam3_track, attr_name):
            setattr(sam3_track, attr_name, None)
            actions.append(f"{attr_name}=None")
    if hasattr(sam3_track, "seen"):
        setattr(sam3_track, "seen", 0)
        actions.append("seen=0")

    tracker = getattr(sam3_track, "tracker", None)
    if tracker is None:
        return actions

    tracker_reset_image = getattr(tracker, "reset_image", None)
    if callable(tracker_reset_image):
        tracker_reset_image()
        actions.append("tracker.reset_image")

    if hasattr(tracker, "inference_state"):
        tracker_inference_state = getattr(tracker, "inference_state")
        if isinstance(tracker_inference_state, dict):
            tracker_inference_state.clear()
            actions.append("tracker.inference_state.clear")
        else:
            setattr(tracker, "inference_state", {})
            actions.append("tracker.inference_state={}")

    return actions


def prepare_sam3_track_predictor_for_video(
    *,
    sam3_track: Any,
    isolation_mode: str,
    sam3_overrides: dict[str, Any],
) -> tuple[Any, list[str]]:
    actions: list[str] = []
    if isolation_mode not in TRACK_ISOLATION_CHOICES:
        raise ValueError(
            f"Unknown --sam3-track-isolation '{isolation_mode}'. "
            f"Expected one of: {', '.join(TRACK_ISOLATION_CHOICES)}."
        )

    if isolation_mode == "recreate":
        return create_sam3_track_predictor(sam3_overrides), ["recreate"]

    if isolation_mode == "reset":
        if sam3_track is None:
            sam3_track = create_sam3_track_predictor(sam3_overrides)
            actions.append("create")
        reset_actions = reset_sam3_track_predictor_state(sam3_track)
        actions.append("reset")
        actions.extend([f"reset:{name}" for name in reset_actions])
        return sam3_track, actions

    # isolation_mode == "both"
    if sam3_track is not None:
        reset_actions = reset_sam3_track_predictor_state(sam3_track)
        actions.append("reset_previous")
        actions.extend([f"reset:{name}" for name in reset_actions])
    sam3_track = create_sam3_track_predictor(sam3_overrides)
    actions.append("recreate")
    return sam3_track, actions


def validate_track_state_num_frames(
    *,
    sam3_track: Any,
    video_stem: str,
    expected_total_frames: int | None,
    expected_sampled_frames: int | None,
    sample_interval: int,
) -> int | None:
    inference_state = getattr(sam3_track, "inference_state", None)
    num_frames = (
        int(inference_state.get("num_frames"))
        if isinstance(inference_state, dict) and inference_state.get("num_frames") is not None
        else None
    )
    # SAM3VideoSemanticPredictor tracks full video length (`dataset.frames`), not sampled frame count.
    # Validate against total input frames and keep sampled-frame info for diagnostics only.
    if (
        expected_total_frames is not None
        and num_frames is not None
        and abs(num_frames - expected_total_frames) > 1
    ):
        raise RuntimeError(
            "SAM3 track predictor state mismatch for video "
            f"'{video_stem}': inference_state.num_frames={num_frames}, "
            f"expected_total_frames={expected_total_frames}, "
            f"expected_sampled_frames={expected_sampled_frames}, sample_interval={sample_interval}. "
            "This usually indicates cross-video state leakage."
        )
    return num_frames


def handle_track_stream_index_error(
    *,
    exc: Exception,
    video_stem: str,
    sampled_idx: int,
    dataset_frame: Any,
    sample_interval: int,
    isolation_mode: str,
    tail_policy: str,
    warnings: list[str],
) -> bool:
    context = (
        f"video={video_stem} sampled_idx={sampled_idx} dataset_frame={dataset_frame} "
        f"sample_interval={sample_interval} isolation={isolation_mode}"
    )
    if tail_policy == "warn_and_finalize":
        warning = (
            "SAM3 track stream ended with IndexError; finalizing partial results "
            f"({context}): {type(exc).__name__}: {exc}"
        )
        warnings.append(warning)
        print(f"Warning [{video_stem}]: {warning}")
        return True

    if tail_policy == "fail_fast":
        raise RuntimeError(
            "SAM3 track stream raised IndexError and fail-fast policy is enabled "
            f"({context}): {type(exc).__name__}: {exc}"
        ) from exc

    raise ValueError(
        f"Unknown --sam3-track-tail-policy '{tail_policy}'. "
        f"Expected one of: {', '.join(TRACK_TAIL_POLICY_CHOICES)}."
    )


class UltralyticsBackend:
    """The pre-existing ultralytics SAM3 port, unchanged behaviour, behind the backend seam."""

    name = "ultralytics"

    def __init__(
        self,
        *,
        sam3_model_path: str,
        conf: float,
        half: bool,
        track_isolation: str = "recreate",
        track_tail_policy: str = "warn_and_finalize",
    ) -> None:
        self.sam3_model_path = sam3_model_path
        self.conf = conf
        self.half = half
        self.track_isolation = track_isolation
        self.track_tail_policy = track_tail_policy
        self._sam3_frame: Any = None
        self._sam3_track: Any = None
        self.last_tracker_num_frames: int | None = None

    def _sam3_overrides(self) -> dict[str, Any]:
        return dict(
            conf=self.conf,
            task="segment",
            mode="predict",
            model=self.sam3_model_path,
            half=self.half,
            save=False,
            verbose=False,
        )

    def prepare_for_video(self, *, mode: str) -> list[str]:
        """Called once per video before `iter_video`. Returns predictor setup actions taken."""
        if mode == "frame":
            if self._sam3_frame is None:
                from ultralytics.models.sam import SAM3SemanticPredictor

                self._sam3_frame = SAM3SemanticPredictor(overrides=self._sam3_overrides())
            return []

        self._sam3_track, actions = prepare_sam3_track_predictor_for_video(
            sam3_track=self._sam3_track,
            isolation_mode=self.track_isolation,
            sam3_overrides=self._sam3_overrides(),
        )
        return actions

    def iter_video(
        self,
        *,
        video_path: Path,
        mode: str,
        prompts: list[str],
        prompt_slugs: list[str],
        sample_interval: int,
        video_fps: float,
        frame_count_est: int,
        estimated_sampled_frames: int | None,
        warnings: list[str],
    ) -> Iterator[SamFrameResult]:
        if mode == "frame":
            yield from self._iter_frame_mode(
                video_path=video_path,
                prompts=prompts,
                prompt_slugs=prompt_slugs,
                sample_interval=sample_interval,
                frame_count_est=frame_count_est,
            )
        else:
            yield from self._iter_track_mode(
                video_path=video_path,
                prompts=prompts,
                prompt_slugs=prompt_slugs,
                sample_interval=sample_interval,
                frame_count_est=frame_count_est,
                estimated_sampled_frames=estimated_sampled_frames,
                warnings=warnings,
            )

    def _iter_frame_mode(
        self,
        *,
        video_path: Path,
        prompts: list[str],
        prompt_slugs: list[str],
        sample_interval: int,
        frame_count_est: int,
    ) -> Iterator[SamFrameResult]:
        if self._sam3_frame is None:
            raise RuntimeError("SAM3 frame predictor is not initialized.")
        cap = cv2.VideoCapture(str(video_path))
        try:
            frame_idx = 0
            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    if frame_count_est > 0 and frame_idx < frame_count_est - 1:
                        yield SamFrameResult(
                            frame_idx=frame_idx,
                            frame_bgr=None,
                            prompt_masks=None,
                            object_rows=None,
                            track_summary=None,
                            timing_ms_sam=None,
                            status="frame_decode_error",
                            error=None,
                        )
                    break

                if frame_idx % sample_interval != 0:
                    frame_idx += 1
                    continue

                try:
                    sam_start = time.perf_counter()
                    sam_results = self._sam3_frame(source=frame_bgr, text=prompts)
                    sam_result = sam_results[0]
                    prompt_masks, object_entries = extract_prompt_masks_and_objects(
                        sam_result,
                        prompts,
                        prompt_slugs,
                        frame_bgr.shape[:2],
                    )
                    sam_ms = (time.perf_counter() - sam_start) * 1000.0
                except Exception as exc:
                    yield SamFrameResult(
                        frame_idx=frame_idx,
                        frame_bgr=frame_bgr,
                        prompt_masks=None,
                        object_rows=None,
                        track_summary=None,
                        timing_ms_sam=None,
                        status="sam_error",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    frame_idx += 1
                    continue

                yield SamFrameResult(
                    frame_idx=frame_idx,
                    frame_bgr=frame_bgr,
                    prompt_masks=prompt_masks,
                    object_rows=object_entries,
                    track_summary=None,
                    timing_ms_sam=sam_ms,
                    status=None,
                    error=None,
                )
                frame_idx += 1
        finally:
            cap.release()

    def _iter_track_mode(
        self,
        *,
        video_path: Path,
        prompts: list[str],
        prompt_slugs: list[str],
        sample_interval: int,
        frame_count_est: int,
        estimated_sampled_frames: int | None,
        warnings: list[str],
    ) -> Iterator[SamFrameResult]:
        if self._sam3_track is None:
            raise RuntimeError("SAM3 track predictor is not initialized.")

        sam3_track = self._sam3_track
        video_stem = Path(video_path).stem
        sampled_idx = 0
        validated_track_state = False
        track_stream = sam3_track(
            source=str(video_path),
            text=prompts,
            stream=True,
            vid_stride=sample_interval,
        )
        track_iter = iter(track_stream)
        while True:
            try:
                sam_result = next(track_iter)
            except StopIteration:
                break
            except IndexError as exc:
                dataset_frame = getattr(getattr(sam3_track, "dataset", None), "frame", None)
                should_finalize = handle_track_stream_index_error(
                    exc=exc,
                    video_stem=video_stem,
                    sampled_idx=sampled_idx,
                    dataset_frame=dataset_frame,
                    sample_interval=sample_interval,
                    isolation_mode=self.track_isolation,
                    tail_policy=self.track_tail_policy,
                    warnings=warnings,
                )
                if should_finalize:
                    break

            default_frame_idx = sampled_idx * sample_interval
            dataset_frame = getattr(getattr(sam3_track, "dataset", None), "frame", None)
            frame_idx = (
                int(dataset_frame) - 1
                if isinstance(dataset_frame, int) and dataset_frame > 0
                else default_frame_idx
            )
            sampled_idx += 1
            if not validated_track_state:
                self.last_tracker_num_frames = validate_track_state_num_frames(
                    sam3_track=sam3_track,
                    video_stem=video_stem,
                    expected_total_frames=frame_count_est if frame_count_est > 0 else None,
                    expected_sampled_frames=estimated_sampled_frames,
                    sample_interval=sample_interval,
                )
                validated_track_state = True

            track_summary = extract_track_summary(sam_result)
            sam_ms = None
            speed = getattr(sam_result, "speed", None)
            if isinstance(speed, dict):
                inference_ms = speed.get("inference")
                sam_ms = float(inference_ms) if inference_ms is not None else None

            try:
                frame_bgr = getattr(sam_result, "orig_img", None)
                if frame_bgr is None:
                    raise RuntimeError("SAM3 track result did not include orig_img.")
                prompt_masks, object_entries = extract_prompt_masks_and_objects(
                    sam_result,
                    prompts,
                    prompt_slugs,
                    frame_bgr.shape[:2],
                )
            except Exception as exc:
                yield SamFrameResult(
                    frame_idx=frame_idx,
                    frame_bgr=None,
                    prompt_masks=None,
                    object_rows=None,
                    track_summary=track_summary,
                    timing_ms_sam=sam_ms,
                    status="sam_error",
                    error=f"{type(exc).__name__}: {exc}",
                )
                continue

            yield SamFrameResult(
                frame_idx=frame_idx,
                frame_bgr=frame_bgr,
                prompt_masks=prompt_masks,
                object_rows=object_entries,
                track_summary=track_summary,
                timing_ms_sam=sam_ms,
                status=None,
                error=None,
            )


def _checkpoint_variant(checkpoint: str) -> str:
    """Peek at the state-dict keys (mmap, no GPU) and classify the presence mechanism.

    "decoder_presence_token": Meta's release sam3.pt (detector.transformer.decoder.presence_token.*)
    "seg_head_presence":      SA-FARI fine-tunes (detector.segmentation_head.presence_head.*)
    "unknown":                neither (loaded with the builder defaults)
    """
    import torch

    sd = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    keys = list(sd.keys())
    if any(k.startswith("detector.segmentation_head.presence_head.") for k in keys):
        return "seg_head_presence"
    if any(k.startswith("detector.transformer.decoder.presence_token") for k in keys):
        return "decoder_presence_token"
    return "unknown"


def _patch_segmentation_head_with_presence() -> None:
    """Make the model match the SA-FARI checkpoints (segmentation-head presence, no decoder
    presence token). Ported from vision-llm-ann-generator/sam3_runner.py; idempotent."""
    from sam3 import model_builder as mb

    if getattr(mb, "_presence_head_patched", False):
        return
    orig = mb._create_segmentation_head

    def patched(*args, **kwargs):
        head = orig(*args, **kwargs)
        head.presence_head = mb._create_dot_product_scoring()
        return head

    mb._create_segmentation_head = patched

    orig_dec = mb._create_transformer_decoder

    def patched_dec(*args, **kwargs):
        dec = orig_dec(*args, **kwargs)
        dec.presence_token = None
        dec.presence_token_head = None
        dec.presence_token_out_norm = None
        return dec

    mb._create_transformer_decoder = patched_dec
    mb._presence_head_patched = True


def _to_numpy(x):
    if hasattr(x, "cpu"):  # torch tensor (may be bf16 under autocast; numpy has no bf16)
        if hasattr(x, "is_floating_point") and x.is_floating_point():
            x = x.float()
        x = x.cpu().numpy()
    return np.asarray(x)


def _to_list(x):
    if hasattr(x, "cpu"):
        if hasattr(x, "is_floating_point") and x.is_floating_point():
            x = x.float()
        x = x.cpu().numpy()
    if hasattr(x, "tolist"):
        return x.tolist()
    return list(x)


def _iter_propagation(prop_resp):
    """Normalise propagate_in_video's response into an iterable of (frame_idx, out).

    Ported from sam3_runner.py: the installed sam3 yields {"frame_index", "outputs"} dicts;
    the other shapes are kept for robustness against future sam3 versions.
    """
    if hasattr(prop_resp, "__iter__") and not isinstance(prop_resp, dict):
        for item in prop_resp:
            if isinstance(item, (tuple, list)) and len(item) == 2:
                yield item[0], item[1]
            elif isinstance(item, dict) and "frame_index" in item:
                yield item["frame_index"], item.get("outputs", item)
            else:
                raise TypeError(f"unrecognised propagate_in_video item shape: {type(item)}")
        return
    if isinstance(prop_resp, dict):
        if "results" in prop_resp:
            prop_resp = prop_resp["results"]
        if isinstance(prop_resp, dict):
            for k, v in prop_resp.items():
                yield int(k), v
            return
        if isinstance(prop_resp, list):
            for i, v in enumerate(prop_resp):
                yield v.get("frame_index", i) if isinstance(v, dict) else i, v
            return
    raise TypeError(f"unrecognised propagate_in_video response shape: {type(prop_resp)}")


def _parse_propagation_output(out: Any, det_threshold: float | None) -> tuple[list[int], list[float], list[np.ndarray]]:
    """(obj_ids, probs, masks[bool HxW]) from one propagate_in_video frame output."""
    obj_ids = _to_list(out["out_obj_ids"])
    probs = _to_list(out["out_probs"])
    masks_raw = out.get("out_binary_masks")
    masks: list[np.ndarray] = []
    if masks_raw is not None:
        masks_np = _to_numpy(masks_raw)
        masks = [masks_np[i].astype(bool) for i in range(masks_np.shape[0])]

    if det_threshold is not None and probs:
        keep = [i for i, p in enumerate(probs) if p >= det_threshold]
        obj_ids = [obj_ids[i] for i in keep]
        probs = [probs[i] for i in keep]
        masks = [masks[i] for i in keep] if masks else masks

    return [int(i) for i in obj_ids], [float(p) for p in probs], masks


class OfficialSam3Backend:
    """Official facebookresearch/sam3 (SA-FARI checkpoints), ported from sam3_runner.py.

    Track mode only. Frame mode is not implemented (the image-model/Sam3Processor path from
    sam3_runner.py was not ported) and raises NotImplementedError with a clear message.
    """

    name = "official"

    def __init__(self, *, sam3_model_path: str, det_threshold: float = 0.5, device: str = "cuda") -> None:
        self.sam3_model_path = sam3_model_path
        self.det_threshold = det_threshold
        self.device = device
        self._predictor: Any = None

    def prepare_for_video(self, *, mode: str) -> list[str]:
        if mode != "track":
            raise NotImplementedError(
                "--sam3-backend official currently supports --sam3-mode track only "
                "(the sam3 image-model + Sam3Processor path is not implemented here). "
                "Use --sam3-backend ultralytics for --sam3-mode frame."
            )
        return []

    def _load_predictor(self) -> Any:
        if self._predictor is not None:
            return self._predictor
        from sam3.model_builder import build_sam3_video_predictor

        variant = _checkpoint_variant(self.sam3_model_path)
        kwargs: dict[str, Any] = dict(checkpoint_path=self.sam3_model_path)
        if variant == "seg_head_presence":
            kwargs["has_presence_token"] = False
            _patch_segmentation_head_with_presence()
        print(f"OfficialSam3Backend: checkpoint variant={variant} kwargs={kwargs}", flush=True)
        self._predictor = build_sam3_video_predictor(**kwargs)
        return self._predictor

    def iter_video(
        self,
        *,
        video_path: Path,
        mode: str,
        prompts: list[str],
        prompt_slugs: list[str],
        sample_interval: int,
        video_fps: float,
        frame_count_est: int,
        estimated_sampled_frames: int | None,
        warnings: list[str],
    ) -> Iterator[SamFrameResult]:
        if mode != "track":
            raise NotImplementedError(
                "--sam3-backend official currently supports --sam3-mode track only."
            )
        from PIL import Image

        predictor = self._load_predictor()

        cap = cv2.VideoCapture(str(video_path))
        frame_indices: list[int] = []
        frames_bgr: list[np.ndarray] = []
        try:
            frame_idx = 0
            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    break
                if frame_idx % sample_interval == 0:
                    frame_indices.append(frame_idx)
                    frames_bgr.append(frame_bgr)
                frame_idx += 1
        finally:
            cap.release()

        if not frame_indices:
            return

        pil_frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames_bgr]
        # sam3_video_inference.py treats a list of length 1 as a still image (disabling
        # propagation), so fall back to the source video path in that edge case, matching
        # sam3_runner.py.
        use_pil_list = len(pil_frames) > 1
        resource = pil_frames if use_pil_list else str(video_path)

        per_frame_objects: dict[int, list[dict[str, Any]]] = {idx: [] for idx in frame_indices}
        video_stem = Path(video_path).stem
        sam_start = time.perf_counter()

        for prompt_index, prompt in enumerate(prompts):
            session_id = None
            try:
                start_resp = predictor.handle_request(dict(type="start_session", resource_path=resource))
                session_id = start_resp["session_id"] if isinstance(start_resp, dict) else start_resp

                predictor.handle_request(
                    dict(
                        type="add_prompt",
                        session_id=session_id,
                        frame_index=0,
                        text=prompt,
                        output_prob_thresh=self.det_threshold,
                    )
                )

                prop_resp = predictor.handle_stream_request(
                    dict(
                        type="propagate_in_video",
                        session_id=session_id,
                        propagation_direction="forward",
                        start_frame_index=0,
                        output_prob_thresh=self.det_threshold,
                    )
                )

                for i, (pos_idx, out) in enumerate(_iter_propagation(prop_resp)):
                    if i >= len(frame_indices):
                        break
                    source_idx = frame_indices[pos_idx] if use_pil_list else frame_indices[i]
                    obj_ids, probs, masks = _parse_propagation_output(out, self.det_threshold)
                    for local_i, obj_id in enumerate(obj_ids):
                        mask_bool = masks[local_i] if local_i < len(masks) else None
                        if mask_bool is None or not mask_bool.any():
                            continue
                        prob = probs[local_i] if local_i < len(probs) else None
                        bbox_xyxy, center_xy = mask_extent_bbox_center(mask_bool)
                        track_id = prompt_index * PROMPT_TRACK_ID_OFFSET + int(obj_id)
                        per_frame_objects.setdefault(source_idx, []).append(
                            {
                                "object_index": None,
                                "track_id": track_id,
                                "prompt_index": prompt_index,
                                "prompt": prompt,
                                "slug": prompt_slugs[prompt_index],
                                "label": prompt,
                                "confidence": prob,
                                "bbox_xyxy": bbox_xyxy,
                                "center_xy": center_xy,
                                "mask_nonzero_pixels": int(mask_bool.sum()),
                                "mask": mask_bool.astype(np.uint8),
                            }
                        )
            except Exception as exc:
                warnings.append(
                    f"OfficialSam3Backend: prompt '{prompt}' failed for video "
                    f"'{video_stem}': {type(exc).__name__}: {exc}"
                )
            finally:
                if session_id is not None:
                    try:
                        predictor.handle_request(dict(type="close_session", session_id=session_id))
                    except Exception:
                        pass

        total_sam_ms = (time.perf_counter() - sam_start) * 1000.0
        per_frame_sam_ms = total_sam_ms / max(1, len(frame_indices))

        for seq_idx, fidx in enumerate(frame_indices):
            rows = per_frame_objects.get(fidx, [])
            for obj_idx, row in enumerate(rows):
                row["object_index"] = obj_idx
            frame_bgr = frames_bgr[seq_idx]
            prompt_masks = [np.zeros(frame_bgr.shape[:2], dtype=np.uint8) for _ in prompts]
            for row in rows:
                pm_idx = row["prompt_index"]
                if 0 <= pm_idx < len(prompt_masks):
                    prompt_masks[pm_idx] |= row["mask"]

            yield SamFrameResult(
                frame_idx=fidx,
                frame_bgr=frame_bgr,
                prompt_masks=prompt_masks,
                object_rows=rows,
                track_summary=track_summary_from_object_rows(rows),
                timing_ms_sam=per_frame_sam_ms,
                status=None,
                error=None,
            )
