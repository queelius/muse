"""Protocol + dataclasses for audio/classification."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class AudioClassificationResult:
    """One audio file's classification output.

    `scores` is the model's full label distribution for the input.
    `multi_label` is True for sigmoid-head multi-label classifiers
    (e.g., AudioSet event classifiers where multiple events may be
    present), False for softmax single-label heads (sentiment-style,
    language-ID-style).

    The codec's `top_k` parameter sorts by score desc and truncates;
    `multi_label` is metadata for downstream consumers (e.g., the
    moderation-style "any score >= threshold" rule).
    """
    scores: dict[str, float]
    multi_label: bool


@runtime_checkable
class AudioClassifierModel(Protocol):
    """Structural protocol any audio-classifier backend satisfies.

    HFAudioClassifier (the generic runtime) satisfies this without
    inheriting. Tests use fakes that match the signature structurally.
    """

    def classify(self, audio_path: str) -> list[AudioClassificationResult]:
        """Return one AudioClassificationResult.

        Returns a length-1 list for symmetry with text_classification's
        contract. Future batch support could return multiple entries.
        """
        ...
