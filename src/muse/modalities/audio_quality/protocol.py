"""Protocol and result types for the ``audio/quality`` modality."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

# Re-exported: the shared decoder raises it for over-long uploads.
from muse.modalities._common.audio_decode import AudioDurationExceededError  # noqa: F401


ScoreDirection = Literal[
    "higher_is_better",
    "lower_is_better",
    "descriptive",
]


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
