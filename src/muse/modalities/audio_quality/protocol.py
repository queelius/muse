"""Protocol and result types for the ``audio/quality`` modality."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable


ScoreDirection = Literal[
    "higher_is_better",
    "lower_is_better",
    "descriptive",
]


class AudioDurationExceededError(ValueError):
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


@dataclass(frozen=True)
class AudioQualityScore:
    """One named quality axis and its nominal interpretation."""

    value: float
    minimum: float | None = None
    maximum: float | None = None
    direction: ScoreDirection = "higher_is_better"


@dataclass
class AudioQualityResult:
    """Quality scores for one audio clip.

    ``primary_score`` identifies the axis callers should use for simple
    ranking. Models may expose additional axes with different scales, so
    every score carries its own range and direction.
    """

    scores: dict[str, AudioQualityScore]
    primary_score: str
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class AudioQualityModel(Protocol):
    """Structural protocol implemented by audio-quality backends."""

    model_id: str

    def assess(
        self,
        audio_path: str,
        *,
        max_duration_seconds: float | None = None,
    ) -> AudioQualityResult:
        """Assess one locally stored audio file."""
        ...
