"""Minimal local-VLM model registry.

Trimmed from vision-llm-ann-verifier/models.py to what this package's callers need: a default
model (Qwen3.8-27B, matching that repo's "qwen" alias), a chat-template prompt builder for N
images + one text part, and best-effort resolution of arbitrary HF ids. Stdlib + no heavy
imports, so it is safe to import from the local dev env (no vllm/torch installed there).
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Qwen model-card recommendation for non-thinking mode.
QWEN_SAMPLING = dict(temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5)


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    model_id: str
    max_model_len: int
    gpu_mem: float
    tensor_parallel: int
    sampling: dict = field(default_factory=dict)

    def render_prompt(self, tokenizer, prompt_text: str, think: bool = False, media: int = 1) -> str:
        """Chat-template `media` image parts followed by one text part into a prompt string.

        Only the tokenizer is needed (tokenize=False), so this works regardless of whether the
        environment's transformers ships the model's processor class.
        """
        media_parts = [{"type": "image"}] * media
        messages = [{"role": "user", "content": media_parts + [{"type": "text", "text": prompt_text}]}]
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=think
            )
        except TypeError:  # template doesn't accept enable_thinking
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    @staticmethod
    def split_reasoning(text: str):
        """Split generated text into (reasoning, answer). reasoning is None when no </think> is present."""
        head, sep, tail = text.partition("</think>")
        if not sep:
            return None, text.strip()
        return head.replace("<think>", "", 1).strip(), tail.strip()


SPECS = {
    "qwen": ModelSpec(
        alias="qwen",
        model_id="Qwen/Qwen3.8-27B",
        max_model_len=28672,
        gpu_mem=0.92,
        tensor_parallel=1,
        sampling=QWEN_SAMPLING,
    ),
}


def resolve(name: str) -> ModelSpec:
    """Alias ('qwen') or an HF model id. Unknown ids get a generic spec based on 'qwen'."""
    if name in SPECS:
        return SPECS[name]
    for spec in SPECS.values():
        if spec.model_id.lower() == name.lower():
            return spec
    base = SPECS["qwen"]
    return ModelSpec(
        alias=name,
        model_id=name,
        max_model_len=base.max_model_len,
        gpu_mem=base.gpu_mem,
        tensor_parallel=base.tensor_parallel,
        sampling=base.sampling,
    )
