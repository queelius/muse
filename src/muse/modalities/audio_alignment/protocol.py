"""Protocol and result types for the ``audio/alignment`` modality."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


class AudioAlignmentDurationExceededError(ValueError):
    """Decoded audio exceeds the amount of work allowed for one request."""

    def __init__(
        self,
        *,
        maximum_seconds: float,
        actual_seconds: float | None = None,
    ) -> None:
        self.maximum_seconds = float(maximum_seconds)
        self.actual_seconds = (
            None if actual_seconds is None else float(actual_seconds)
        )
        detail = (
            f" ({self.actual_seconds:.3f}s received)"
            if self.actual_seconds is not None
            else ""
        )
        super().__init__(
            f"audio duration exceeds {self.maximum_seconds:.3f}s{detail}"
        )


class AudioAlignmentDecodeError(ValueError):
    """The uploaded payload could not be decoded as supported audio."""


class UnsupportedAlignmentLanguageError(ValueError):
    """The requested language is not supported by the active aligner."""

    def __init__(self, language: str, *, supported: tuple[str, ...]) -> None:
        self.language = language
        self.supported = supported
        super().__init__(
            f"unsupported alignment language {language!r}; supported: "
            f"{', '.join(supported)}"
        )


class UnalignableTextError(ValueError):
    """The reference text cannot be tokenized or aligned."""


@dataclass(frozen=True)
class AlignmentWord:
    """One reference-text token with its forced-alignment timestamps."""

    word: str
    start: float
    end: float
    confidence: float | None = None


@dataclass
class AudioAlignmentResult:
    """Word timestamps for trusted reference text and one audio clip."""

    text: str
    language: str | None
    duration_seconds: float
    words: list[AlignmentWord]
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class AudioAlignmentModel(Protocol):
    """Structural protocol implemented by forced-alignment backends."""

    model_id: str

    def align(
        self,
        audio_path: str,
        transcript: str,
        *,
        language: str | None = None,
        max_duration_seconds: float | None = None,
    ) -> AudioAlignmentResult:
        """Align trusted reference text against one locally stored file."""
        ...
