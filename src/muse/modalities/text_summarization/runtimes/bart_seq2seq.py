"""BartSeq2SeqRuntime: generic runtime over any HF seq2seq summarizer.

One class wraps `transformers.AutoModelForSeq2SeqLM` for any repo on
HuggingFace that ships a BART/PEGASUS/T5-shape summarizer. Pulled via
the HF resolver: `muse pull hf://facebook/bart-large-cnn` synthesizes
a manifest pointing at this class.

Deferred imports follow the muse pattern: torch + transformers stay as
module-top sentinels (None) until _ensure_deps() lazy-imports them.
Tests patch the sentinels directly; _ensure_deps short-circuits on
non-None.

Length-to-max_new_tokens mapping is owned here, not in the codec or
the route layer. Single source of truth.
"""
from __future__ import annotations

import logging
from typing import Any

from muse.modalities.text_summarization.protocol import SummarizationResult


logger = logging.getLogger(__name__)


torch: Any = None
AutoModelForSeq2SeqLM: Any = None
AutoTokenizer: Any = None


# Frozen mapping. Length values match the Cohere /v1/summarize request
# spec exactly. `max_new_tokens` choices balance "useful summary" against
# "fits a typical news paragraph".
LENGTH_TO_MAX_TOKENS: dict[str, int] = {
    "short": 80,
    "medium": 180,
    "long": 400,
}


VALID_LENGTHS = frozenset(LENGTH_TO_MAX_TOKENS.keys())
VALID_FORMATS = frozenset({"paragraph", "bullets"})


def _ensure_deps() -> None:
    global torch, AutoModelForSeq2SeqLM, AutoTokenizer
    if torch is None:
        try:
            import torch as _t
            torch = _t
        except Exception as e:  # noqa: BLE001
            logger.debug("BartSeq2SeqRuntime: torch unavailable: %s", e)
    if AutoModelForSeq2SeqLM is None:
        try:
            from transformers import (
                AutoModelForSeq2SeqLM as _amfsl,
                AutoTokenizer as _atok,
            )
            AutoModelForSeq2SeqLM = _amfsl
            AutoTokenizer = _atok
        except Exception as e:  # noqa: BLE001
            logger.debug("BartSeq2SeqRuntime: transformers unavailable: %s", e)


def _resolve_dtype(dtype: str) -> Any:
    """Map a string dtype to the torch dtype object, or None when torch
    isn't available (caller will skip dtype kwargs)."""
    if torch is None:
        return None
    return {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }.get(dtype, torch.float32)


def _select_device(device: str) -> str:
    if device != "auto":
        return device
    if torch is None:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _set_inference_mode(model: Any) -> None:
    """Switch model to inference (no-grad) mode if the method exists.

    Wrapped in a helper so the runtime body stays readable and tests
    can patch this without intercepting the model object's attribute.
    """
    fn = getattr(model, "eval", None)
    if callable(fn):
        fn()


class BartSeq2SeqRuntime:
    """Generic seq2seq summarizer runtime."""

    model_id: str

    def __init__(
        self,
        *,
        model_id: str,
        hf_repo: str,
        local_dir: str | None = None,
        device: str = "auto",
        dtype: str = "float32",
        default_length: str = "medium",
        default_format: str = "paragraph",
        max_input_tokens: int = 1024,
        **_: Any,
    ) -> None:
        _ensure_deps()
        if AutoModelForSeq2SeqLM is None or AutoTokenizer is None:
            raise RuntimeError(
                "transformers is not installed; run `muse pull` "
                "or install `transformers` into this venv"
            )
        self.model_id = model_id
        self._device = _select_device(device)
        self._dtype = dtype
        self._default_length = default_length
        self._default_format = default_format
        self._max_input_tokens = max_input_tokens

        src = local_dir or hf_repo
        logger.info(
            "loading seq2seq summarizer from %s (device=%s, dtype=%s, max_input_tokens=%d)",
            src, self._device, dtype, max_input_tokens,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(src)
        torch_dtype = _resolve_dtype(dtype)
        kwargs = {}
        if torch_dtype is not None:
            kwargs["torch_dtype"] = torch_dtype
        self._model = AutoModelForSeq2SeqLM.from_pretrained(src, **kwargs)
        self._model = self._model.to(self._device)
        _set_inference_mode(self._model)

    def summarize(
        self,
        text: str,
        length: str | None = None,
        format: str | None = None,
    ) -> SummarizationResult:
        """Produce a summary; honors `length` -> max_new_tokens mapping.

        Truncates inputs longer than `max_input_tokens` and records
        a `truncation_warning` in the result metadata so the caller can
        surface it to the end user. `format` is recorded but does not
        affect generation for non-instruction models like BART-CNN.
        """
        eff_length = length or self._default_length
        eff_format = format or self._default_format
        if eff_length not in VALID_LENGTHS:
            raise ValueError(
                f"length must be one of {sorted(VALID_LENGTHS)}; got {eff_length!r}"
            )
        if eff_format not in VALID_FORMATS:
            raise ValueError(
                f"format must be one of {sorted(VALID_FORMATS)}; got {eff_format!r}"
            )
        max_new_tokens = LENGTH_TO_MAX_TOKENS[eff_length]

        # Tokenize with truncation. The tokenizer returns a BatchEncoding;
        # we pull input_ids out for `.generate()` and the un-truncated
        # length for the truncation_warning.
        full_encoded = self._tokenizer(text, return_tensors="pt")
        full_len = full_encoded["input_ids"].shape[-1]
        truncated_encoded = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self._max_input_tokens,
        )
        input_ids = truncated_encoded["input_ids"].to(self._device)
        prompt_tokens = int(input_ids.shape[-1])

        # generate() returns shape (batch, seq_len) including any
        # decoder start tokens; length excludes the input.
        output_ids = self._model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
        )
        # Single batch element. Decode and count completion tokens
        # post-decoding (more honest than the raw token-id sequence
        # because generate may pad / start with a bos/decoder-start).
        summary = self._tokenizer.decode(
            output_ids[0], skip_special_tokens=True,
        )
        completion_tokens = int(output_ids.shape[-1])

        metadata: dict[str, Any] = {}
        if full_len > self._max_input_tokens:
            metadata["truncation_warning"] = True
            metadata["truncated_from_tokens"] = full_len
            metadata["truncated_to_tokens"] = self._max_input_tokens

        return SummarizationResult(
            summary=summary,
            length=eff_length,
            format=eff_format,
            model_id=self.model_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            metadata=metadata,
        )
