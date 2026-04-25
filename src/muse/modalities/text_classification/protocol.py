"""Protocol + dataclasses for text/classification."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class ClassificationResult:
    """One input's classification output.

    scores: {label: confidence in [0, 1]} from the model's id2label space.
    multi_label: True if scores are independent (sigmoid head). False if
    mutually exclusive (softmax; sums to ~1.0).

    The codec uses multi_label to decide whether `flagged` derives from
    "any score >= threshold" (multi-label) or "argmax score >= threshold"
    (single-label).
    """
    scores: dict[str, float]
    multi_label: bool


@runtime_checkable
class TextClassifierModel(Protocol):
    """Structural protocol any text-classifier backend satisfies.

    HFTextClassifier (the generic runtime, in Task 5) satisfies this
    without inheriting. Tests use fakes that match the signature
    structurally.
    """

    def classify(self, input: str | list[str]) -> list[ClassificationResult]:
        """Return one ClassificationResult per input.

        Even when `input` is a scalar str, returns a list of length 1.
        Order matches input order in batch mode.
        """
        ...
