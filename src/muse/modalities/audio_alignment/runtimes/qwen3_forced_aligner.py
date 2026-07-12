"""Transformers runtime for Qwen3 ForcedAligner 0.6B."""
from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from muse.core.runtime_helpers import LoadTimer, select_device, set_inference_mode
from muse.modalities.audio_alignment.decoding import decode_audio
from muse.modalities.audio_alignment.protocol import (
    AlignmentWord,
    AudioAlignmentResult,
    UnalignableTextError,
    UnsupportedAlignmentLanguageError,
)


logger = logging.getLogger(__name__)
torch: Any = None
AutoProcessor: Any = None
AutoModelForTokenClassification: Any = None

SUPPORTED_LANGUAGES = (
    "English",
    "Chinese",
    "Cantonese",
    "French",
    "German",
    "Italian",
    "Japanese",
    "Korean",
    "Portuguese",
    "Russian",
    "Spanish",
)
_LANGUAGE_ALIASES = {
    "en": "English",
    "english": "English",
    "zh": "Chinese",
    "zh-cn": "Chinese",
    "cmn": "Chinese",
    "mandarin": "Chinese",
    "chinese": "Chinese",
    "yue": "Cantonese",
    "zh-hk": "Cantonese",
    "cantonese": "Cantonese",
    "fr": "French",
    "french": "French",
    "de": "German",
    "german": "German",
    "it": "Italian",
    "italian": "Italian",
    "ja": "Japanese",
    "jp": "Japanese",
    "japanese": "Japanese",
    "ko": "Korean",
    "kr": "Korean",
    "korean": "Korean",
    "pt": "Portuguese",
    "pt-br": "Portuguese",
    "portuguese": "Portuguese",
    "ru": "Russian",
    "russian": "Russian",
    "es": "Spanish",
    "spanish": "Spanish",
}
_REQUIRED_FILES = (
    "config.json",
    "model.safetensors",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


def _ensure_deps() -> None:
    global torch, AutoProcessor, AutoModelForTokenClassification
    if (
        torch is not None
        and AutoProcessor is not None
        and AutoModelForTokenClassification is not None
    ):
        return
    try:
        import torch as _torch
        from transformers import (
            AutoModelForTokenClassification as _AutoModel,
            AutoProcessor as _AutoProcessor,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Qwen3 ForcedAligner requires torch and a recent transformers; "
            "run `muse pull` to install the model dependencies"
        ) from exc
    torch = _torch
    AutoProcessor = _AutoProcessor
    AutoModelForTokenClassification = _AutoModel


def normalize_language(language: str | None) -> str | None:
    """Return the canonical Qwen language name or reject the value."""
    if language is None or not language.strip():
        return None
    normalized = language.strip().lower().replace("_", "-")
    if normalized == "auto":
        return None
    canonical = _LANGUAGE_ALIASES.get(normalized)
    if canonical is None:
        raise UnsupportedAlignmentLanguageError(
            language, supported=SUPPORTED_LANGUAGES,
        )
    return canonical


def _model_source(local_dir: str | None, hf_repo: str) -> str:
    source = local_dir or hf_repo
    path = Path(source)
    if local_dir is not None and not path.is_dir():
        raise FileNotFoundError(
            f"Qwen3 ForcedAligner snapshot not found at {path}; "
            "pull the model first"
        )
    if path.is_dir():
        missing = [name for name in _REQUIRED_FILES if not (path / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Qwen3 ForcedAligner snapshot at {path} is missing: "
                f"{', '.join(missing)}"
            )
    return str(path) if path.is_dir() else source


def _dtype_for_device(device: str) -> Any:
    if device == "cpu":
        return torch.float32
    if device.startswith("cuda"):
        supports_bf16 = getattr(torch.cuda, "is_bf16_supported", None)
        if callable(supports_bf16) and supports_bf16():
            return torch.bfloat16
    return torch.float16


def _item_value(item: Any, name: str) -> Any:
    if isinstance(item, Mapping):
        return item[name]
    return getattr(item, name)


def _timestamp_marker_confidences(
    logits: Any,
    input_ids: Any,
    timestamp_token_id: int,
) -> list[float]:
    """Return the top timestamp-bin probability at each marker position."""
    marker_logits = logits[input_ids == timestamp_token_id]
    if int(marker_logits.shape[0]) == 0:
        return []
    probabilities = torch.softmax(marker_logits.float(), dim=-1).amax(dim=-1)
    return [float(value) for value in probabilities.detach().cpu().tolist()]


def _model_context_limit(model: Any) -> int | None:
    """Read the text-position limit across composite model configs."""
    config = model.config
    text_config = getattr(config, "text_config", None)
    value = getattr(text_config, "max_position_embeddings", None)
    if value is None:
        value = getattr(config, "max_position_embeddings", None)
    if value is None or int(value) <= 0:
        return None
    return int(value)


def _word_confidence(markers: list[float], index: int) -> float | None:
    offset = index * 2
    if offset + 1 >= len(markers):
        return None
    value = (markers[offset] + markers[offset + 1]) / 2.0
    return round(min(1.0, max(0.0, value)), 6)


class Qwen3ForcedAlignerRuntime:
    """Align trusted transcripts without re-transcribing the audio."""

    def __init__(
        self,
        *,
        model_id: str,
        hf_repo: str,
        local_dir: str | None = None,
        device: str = "auto",
        sample_rate: int = 16000,
        max_duration_seconds: float = 300.0,
        max_input_tokens: int = 8192,
        max_reference_words: int = 2048,
        **_: Any,
    ) -> None:
        _ensure_deps()
        self.model_id = model_id
        self._device = select_device(device, torch_module=torch)
        self._dtype = _dtype_for_device(self._device)
        self._sample_rate = int(sample_rate)
        self._max_duration_seconds = float(max_duration_seconds)
        self._max_reference_words = int(max_reference_words)
        if self._max_reference_words <= 0:
            raise ValueError("max_reference_words must be positive")
        configured_context = int(max_input_tokens)
        if configured_context <= 0:
            raise ValueError("max_input_tokens must be positive")
        source = _model_source(local_dir, hf_repo)
        with LoadTimer(f"Qwen3 ForcedAligner from {source}", logger):
            self._processor = AutoProcessor.from_pretrained(source)
            self._model = AutoModelForTokenClassification.from_pretrained(
                source,
                dtype=self._dtype,
            )
            moved = self._model.to(self._device)
            if moved is not None:
                self._model = moved
        set_inference_mode(self._model)
        model_context = _model_context_limit(self._model)
        self._max_input_tokens = (
            configured_context
            if model_context is None
            else min(configured_context, model_context)
        )

    def align(
        self,
        audio_path: str,
        transcript: str,
        *,
        language: str | None = None,
        max_duration_seconds: float | None = None,
    ) -> AudioAlignmentResult:
        reference_text = transcript.strip()
        if not reference_text:
            raise UnalignableTextError("reference text must not be empty")
        canonical_language = normalize_language(language)
        reference_words = self._processor.split_words_for_alignment(
            reference_text, canonical_language,
        )
        if not reference_words:
            raise UnalignableTextError(
                "reference text contains no alignable words"
            )
        if len(reference_words) > self._max_reference_words:
            raise UnalignableTextError(
                f"reference text contains {len(reference_words)} alignable "
                f"words; maximum is {self._max_reference_words}"
            )
        request_limit = (
            self._max_duration_seconds
            if max_duration_seconds is None
            else float(max_duration_seconds)
        )
        limit = min(self._max_duration_seconds, request_limit)
        if limit <= 0:
            raise ValueError("max_duration_seconds must be positive")

        decoded = decode_audio(
            audio_path,
            sample_rate=self._sample_rate,
            max_duration_seconds=limit,
        )
        waveform = decoded.waveform.squeeze(0).detach().cpu().float().numpy()
        aligner_inputs, word_lists = (
            self._processor.prepare_forced_aligner_inputs(
                audio=waveform,
                transcript=reference_text,
                language=canonical_language,
            )
        )
        if not word_lists or not word_lists[0]:
            raise UnalignableTextError(
                "reference text contains no alignable words"
            )
        if len(word_lists[0]) > self._max_reference_words:
            raise UnalignableTextError(
                f"reference text contains {len(word_lists[0])} alignable "
                f"words; maximum is {self._max_reference_words}"
            )
        input_ids = aligner_inputs["input_ids"]
        input_tokens = int(input_ids.shape[-1])
        if input_tokens > self._max_input_tokens:
            raise UnalignableTextError(
                f"prepared alignment contains {input_tokens} input tokens; "
                f"maximum is {self._max_input_tokens}"
            )
        aligner_inputs = aligner_inputs.to(self._device, self._dtype)
        with torch.inference_mode():
            outputs = self._model(**aligner_inputs)

        input_ids = aligner_inputs["input_ids"]
        timestamp_token_id = self._model.config.timestamp_token_id
        decoded_batches = self._processor.decode_forced_alignment(
            logits=outputs.logits,
            input_ids=input_ids,
            word_lists=word_lists,
            timestamp_token_id=timestamp_token_id,
        )
        if not decoded_batches or not decoded_batches[0]:
            raise UnalignableTextError(
                "the model could not align the reference text"
            )
        raw_words = decoded_batches[0]
        expected_count = len(word_lists[0])
        if len(raw_words) != expected_count:
            raise UnalignableTextError(
                "the model returned an incomplete word alignment"
            )
        markers = _timestamp_marker_confidences(
            outputs.logits, input_ids, timestamp_token_id,
        )

        words: list[AlignmentWord] = []
        clamped_count = 0
        for index, item in enumerate(raw_words):
            raw_start = float(_item_value(item, "start_time"))
            raw_end = float(_item_value(item, "end_time"))
            if not math.isfinite(raw_start) or not math.isfinite(raw_end):
                raise UnalignableTextError(
                    "the model returned a non-finite timestamp"
                )
            start = min(decoded.duration_seconds, max(0.0, raw_start))
            end = min(decoded.duration_seconds, max(start, raw_end))
            if start != raw_start or end != raw_end:
                clamped_count += 1
            words.append(AlignmentWord(
                word=str(_item_value(item, "text")),
                start=round(start, 6),
                end=round(end, 6),
                confidence=_word_confidence(markers, index),
            ))

        confidences = [
            item.confidence for item in words if item.confidence is not None
        ]
        metadata: dict[str, Any] = {
            "family": "qwen3-forced-aligner",
            "sample_rate": decoded.sample_rate,
            "word_count": len(words),
            "confidence_type": "mean_start_end_timestamp_probability",
        }
        if confidences:
            metadata["mean_confidence"] = round(
                sum(confidences) / len(confidences), 6,
            )
        if clamped_count:
            metadata["clamped_timestamp_count"] = clamped_count
        return AudioAlignmentResult(
            text=reference_text,
            language=canonical_language,
            duration_seconds=round(decoded.duration_seconds, 6),
            words=words,
            metadata=metadata,
        )
