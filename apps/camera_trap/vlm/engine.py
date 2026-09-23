"""Generic local-VLM engine: batched (images + prompt) requests -> parsed JSON dicts.

This is the reusable core: given a list of `VLMRequest` (each a list of images plus prompt
text, optionally its own JSON schema), `VLMEngine.run_json` returns one parsed dict per request
(or None on a generation/parse failure). It is used by mask QC
(apps/camera_trap/qc/mask_verify.py) and is intended for reuse by the distance-board-reading
package too.

vllm and its transformers/torch dependencies are only imported lazily, inside
`VLMEngine._ensure_engine`, so importing this module (and everything that merely imports it,
e.g. for the CLI's --help) works in environments without vllm installed -- such as the local dev
conda env used for tests, which use `FakeEngine` instead.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .models import ModelSpec, resolve


@dataclass
class VLMRequest:
    """One request: a list of images (numpy array or PIL.Image) + prompt text.

    `color` says how to interpret numpy arrays ("rgb" or "bgr"; ignored for PIL images and for
    single-channel arrays). `schema` overrides the JSON schema passed to `run_json` for this
    request only (leave None to use the call-level schema).
    """

    images: list
    prompt: str
    schema: dict | None = None
    color: str = "rgb"


def _to_pil(image, color: str = "rgb"):
    from PIL import Image

    if isinstance(image, Image.Image):
        return image
    arr = np.asarray(image)
    if arr.ndim == 2:
        return Image.fromarray(arr).convert("RGB")
    if color == "bgr":
        arr = arr[:, :, ::-1]
    return Image.fromarray(arr)


def _structured_output_kwarg(schema: dict) -> dict:
    """StructuredOutputsParams (current vLLM) with a fallback to GuidedDecodingParams (older)."""
    try:
        from vllm.sampling_params import StructuredOutputsParams

        return {"structured_outputs": StructuredOutputsParams(json=schema)}
    except ImportError:
        from vllm.sampling_params import GuidedDecodingParams

        return {"guided_decoding": GuidedDecodingParams(json=schema)}


class VLMEngine:
    """Offline vLLM engine wrapper. Weights must already be cached locally (HF_HUB_OFFLINE=1)."""

    def __init__(
        self,
        model_id: str = "qwen",
        tensor_parallel_size: int | None = None,
        max_model_len: int | None = None,
        gpu_mem_util: float | None = None,
        thinking: bool = False,
    ):
        self.spec: ModelSpec = resolve(model_id)
        self.thinking = thinking
        self.tensor_parallel_size = tensor_parallel_size or self.spec.tensor_parallel
        self.max_model_len = max_model_len or self.spec.max_model_len
        self.gpu_mem_util = gpu_mem_util or self.spec.gpu_mem
        self._llm = None
        self._tokenizer = None
        self.last_errors: list[str | None] = []

    def _ensure_engine(self) -> None:
        if self._llm is not None:
            return
        from transformers import AutoTokenizer
        from vllm import LLM

        self._tokenizer = AutoTokenizer.from_pretrained(self.spec.model_id)
        self._llm = LLM(
            model=self.spec.model_id,
            tensor_parallel_size=self.tensor_parallel_size,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_mem_util,
            enable_prefix_caching=True,
            seed=0,
        )

    def run_json(
        self,
        requests: list[VLMRequest],
        schema: dict | None = None,
        max_tokens: int = 768,
        chunk: int = 32,
    ) -> list[dict | None]:
        """Run all requests through the engine in chunks, returning one parsed dict per request.

        A failure (generation error or invalid JSON) yields None at that position; the reason is
        recorded in `self.last_errors` (same length/order as `requests`), so callers never need a
        try/except around this call.
        """
        from vllm import SamplingParams

        self._ensure_engine()
        results: list[dict | None] = [None] * len(requests)
        errors: list[str | None] = [None] * len(requests)

        for start in range(0, len(requests), chunk):
            batch = requests[start : start + chunk]
            prompts = []
            sampling = []
            for req in batch:
                req_schema = req.schema or schema
                prompt = self.spec.render_prompt(
                    self._tokenizer, req.prompt, think=self.thinking, media=len(req.images)
                )
                images = [_to_pil(im, req.color) for im in req.images]
                prompts.append({"prompt": prompt, "multi_modal_data": {"image": images}})

                sp_kwargs: dict[str, Any] = dict(max_tokens=max_tokens, seed=0)
                qwen_thinking = self.thinking and self.spec.family != "glm5_next"
                sp_kwargs.update({"temperature": 1.0, "top_p": 0.95} if qwen_thinking else self.spec.sampling)
                if req_schema is not None:
                    sp_kwargs.update(_structured_output_kwarg(req_schema))
                sampling.append(SamplingParams(**sp_kwargs))

            try:
                outputs = self._llm.generate(prompts, sampling, use_tqdm=False)
            except Exception as e:  # engine-level failure: whole chunk fails, keep going
                for i in range(len(batch)):
                    errors[start + i] = f"generation error: {e!r}"
                continue

            for i, out in enumerate(outputs):
                idx = start + i
                text = out.outputs[0].text
                _, answer = ModelSpec.split_reasoning(text)
                try:
                    results[idx] = json.loads(answer)
                except json.JSONDecodeError as e:
                    errors[idx] = f"json parse error: {e}"

        self.last_errors = errors
        return results


class FakeEngine:
    """Test double with VLMEngine's `run_json` interface; no vllm/torch import.

    `responses` is either a callable `(VLMRequest) -> dict | None` or a fixed list of
    dicts/None consumed in request order (cycling if shorter than the request list). Every call
    is recorded in `self.calls` for assertions.
    """

    def __init__(self, responses=None):
        self.responses = responses
        self.calls: list[VLMRequest] = []
        self.last_errors: list[str | None] = []

    def run_json(
        self,
        requests: list[VLMRequest],
        schema: dict | None = None,
        max_tokens: int = 768,
        chunk: int = 32,
    ) -> list[dict | None]:
        self.calls.extend(requests)
        results: list[dict | None] = []
        errors: list[str | None] = []
        for i, req in enumerate(requests):
            if callable(self.responses):
                val = self.responses(req)
            elif self.responses:
                val = self.responses[i % len(self.responses)]
            else:
                val = None
            results.append(val)
            errors.append(None if val is not None else "fake: no scripted response")
        self.last_errors = errors
        return results
