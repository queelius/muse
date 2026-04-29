"""facebook/bart-large-cnn (BART large fine-tuned on CNN/DailyMail).

Curated default for `text/summarization`. ~400MB on disk; CPU-friendly.
1024-token max_input_tokens (BART's hard limit). News summarization.

License: Apache 2.0.

Wraps `transformers.AutoModelForSeq2SeqLM`; lazy imports so muse pull
+ muse --help work without ML deps installed.
"""
from __future__ import annotations

import logging
from typing import Any

from muse.modalities.text_summarization.protocol import SummarizationResult


logger = logging.getLogger(__name__)


# Sentinels (lazy-import pattern matches sd_turbo, soprano_80m, etc.).
torch: Any = None
AutoModelForSeq2SeqLM: Any = None
AutoTokenizer: Any = None


# Length to max_new_tokens mapping (matches the runtime). Frozen.
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
            logger.debug("bart_large_cnn: torch unavailable: %s", e)
    if AutoModelForSeq2SeqLM is None:
        try:
            from transformers import (
                AutoModelForSeq2SeqLM as _amfsl,
                AutoTokenizer as _atok,
            )
            AutoModelForSeq2SeqLM = _amfsl
            AutoTokenizer = _atok
        except Exception as e:  # noqa: BLE001
            logger.debug("bart_large_cnn: transformers unavailable: %s", e)


MANIFEST = {
    "model_id": "bart-large-cnn",
    "modality": "text/summarization",
    "hf_repo": "facebook/bart-large-cnn",
    "description": (
        "BART large CNN: news summarization, ~400MB, CPU-friendly, Apache 2.0"
    ),
    "license": "Apache 2.0",
    "pip_extras": (
        "torch>=2.1.0",
        "transformers>=4.36.0",
    ),
    "system_packages": (),
    "capabilities": {
        # CPU-friendly is the muse story for this model. Honest annotation.
        "device": "cpu",
        "default_length": "medium",
        "default_format": "paragraph",
        "supports_dialog_summarization": False,
        # Measured peak inference, BART-large at fp32 + 1024 ctx.
        "memory_gb": 1.5,
        "max_input_tokens": 1024,
    },
    # BART tokenizer requires merges.txt + vocab.json (BPE pair); without
    # both, the tokenizer load fails with a non-obvious error. Listed
    # explicitly so the snapshot_download allow_patterns can't drop them.
    "allow_patterns": [
        "*.safetensors", "*.json", "*.txt",
        "merges.txt", "vocab.json", "tokenizer*",
    ],
}


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
    """Switch the model to no-grad inference mode if the method exists."""
    fn = getattr(model, "eval", None)
    if callable(fn):
        fn()


class Model:
    """bart-large-cnn backend (seq2seq summarizer)."""

    model_id = MANIFEST["model_id"]

    def __init__(
        self,
        *,
        hf_repo: str = MANIFEST["hf_repo"],
        local_dir: str | None = None,
        device: str = "cpu",
        default_length: str = "medium",
        default_format: str = "paragraph",
        max_input_tokens: int = 1024,
        **_: Any,
    ) -> None:
        _ensure_deps()
        if AutoModelForSeq2SeqLM is None or AutoTokenizer is None:
            raise RuntimeError(
                "transformers is not installed; run `muse pull bart-large-cnn` "
                "or install `transformers` into this venv"
            )
        self._device = _select_device(device)
        self._default_length = default_length
        self._default_format = default_format
        self._max_input_tokens = max_input_tokens

        src = local_dir or hf_repo
        logger.info(
            "loading bart-large-cnn from %s (device=%s, max_input_tokens=%d)",
            src, self._device, max_input_tokens,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(src)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(src)
        self._model = self._model.to(self._device)
        _set_inference_mode(self._model)

    def summarize(
        self,
        text: str,
        length: str | None = None,
        format: str | None = None,
    ) -> SummarizationResult:
        """Produce a summary for `text`.

        BART-CNN is not instruction-tuned; `format` is recorded in the
        result metadata but does not affect generation. Use the
        `bart-cnn-samsum` or future instruction-tuned summarizers when
        format-respecting bullets are needed.
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

        output_ids = self._model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
        )
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
