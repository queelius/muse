"""Protocol + dataclasses for audio/transcription."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol


@dataclass
class Word:
    """A single word with its start/end timestamps."""
    word: str
    start: float
    end: float


@dataclass
class Segment:
    """A transcript segment (Whisper's native unit of output)."""
    id: int
    start: float
    end: float
    text: str
    words: list[Word] | None


@dataclass
class TranscriptionResult:
    """Full transcription output.

    - text: concatenated transcript for the whole file
    - language: detected or user-specified ISO-639-1 code
    - duration: input audio duration in seconds
    - segments: Whisper segments in time order
    - task: 'transcribe' (source-language transcript) or 'translate'
      (source-language audio to English transcript)
    """
    text: str
    language: str
    duration: float
    segments: list[Segment]
    task: Literal["transcribe", "translate"]


class TranscriptionModel(Protocol):
    """Structural protocol any ASR backend satisfies.

    FasterWhisperModel (the generic runtime) satisfies this without
    inheriting. Tests use fakes that match the signature structurally.
    """

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
        **kwargs: Any,
    ) -> TranscriptionResult: ...
