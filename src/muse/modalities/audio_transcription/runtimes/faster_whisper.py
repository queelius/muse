"""FasterWhisperModel: generic runtime over any Systran CT2 Whisper repo.

One class serves any faster-whisper-format repo on HF. Pulled via the
HF resolver (muse.core.resolvers_hf): `muse pull
hf://Systran/faster-whisper-base` synthesizes a manifest pointing at
this class.

Deferred imports follow the muse pattern: torch and WhisperModel stay
as module-top sentinels (None) until _ensure_deps() lazy-imports them.
Tests patch the sentinels directly; _ensure_deps short-circuits on
non-None, so pre-populated mocks survive.
"""
from __future__ import annotations

import logging
from typing import Any, Literal

from muse.modalities.audio_transcription import (
    Segment,
    TranscriptionResult,
    Word,
)


logger = logging.getLogger(__name__)

torch: Any = None
WhisperModel: Any = None


def _ensure_deps() -> None:
    global torch, WhisperModel
    if torch is None:
        try:
            import torch as _t
            torch = _t
        except Exception as e:  # noqa: BLE001
            logger.debug("FasterWhisperModel torch unavailable: %s", e)
    if WhisperModel is None:
        try:
            from faster_whisper import WhisperModel as _w
            WhisperModel = _w
        except Exception as e:  # noqa: BLE001
            logger.debug("FasterWhisperModel faster-whisper unavailable: %s", e)


class FasterWhisperModel:
    """Generic faster-whisper runtime."""

    model_id: str

    def __init__(
        self,
        *,
        model_id: str,
        hf_repo: str,
        local_dir: str | None = None,
        device: str = "auto",
        compute_type: str | None = None,
        beam_size: int = 5,
        **_: Any,
    ) -> None:
        _ensure_deps()
        if WhisperModel is None:
            raise RuntimeError(
                "faster-whisper is not installed; run `muse pull` or "
                "install `faster-whisper` into this venv"
            )
        self.model_id = model_id
        self._device = _select_device(device)
        if compute_type is None:
            compute_type = "float16" if self._device == "cuda" else "int8"
        src = local_dir or hf_repo
        logger.info(
            "loading faster-whisper from %s (device=%s, compute_type=%s)",
            src, self._device, compute_type,
        )
        self._model = WhisperModel(
            src, device=self._device, compute_type=compute_type,
        )
        self._beam_size = beam_size

    def transcribe(
        self,
        audio_path: str,
        *,
        task: Literal["transcribe", "translate"] = "transcribe",
        language: str | None = None,
        prompt: str | None = None,
        temperature: float = 0.0,
        word_timestamps: bool = False,
        vad_filter: bool = False,
        **_: Any,
    ) -> TranscriptionResult:
        segments_iter, info = self._model.transcribe(
            audio_path,
            task=task,
            language=language,
            initial_prompt=prompt,
            temperature=temperature,
            word_timestamps=word_timestamps,
            vad_filter=vad_filter,
            beam_size=self._beam_size,
        )
        segments: list[Segment] = []
        for i, s in enumerate(segments_iter):
            words: list[Word] | None = None
            raw_words = getattr(s, "words", None) or []
            if raw_words:
                words = [Word(word=w.word, start=w.start, end=w.end) for w in raw_words]
            segments.append(Segment(
                id=i, start=s.start, end=s.end, text=s.text.strip(),
                words=words,
            ))
        return TranscriptionResult(
            text=" ".join(s.text for s in segments).strip(),
            language=info.language,
            duration=info.duration,
            segments=segments,
            task=task,
        )


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
