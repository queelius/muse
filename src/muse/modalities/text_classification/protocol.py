"""Protocol + dataclasses for text/classification."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class ClassificationResult:
    """One input's classification output.

    scores: {label: confidence in [0, 1]} from the model's id2label space
    (fine-tuned classifiers) or from the request's candidate_labels
    (zero-shot).
    multi_label: True if scores are independent (sigmoid head, or
    zero-shot with multi_label=True). False if mutually exclusive
    (softmax; sums to ~1.0).

    The /v1/moderations codec uses multi_label to decide whether
    `flagged` derives from "any score >= threshold" or "argmax score
    >= threshold". The /v1/text/classifications codec uses scores to
    build a sorted [{label, score}] list and ignores multi_label
    (the response shape is uniform across both modes).
    """
    scores: dict[str, float]
    multi_label: bool


@runtime_checkable
class TextClassifierModel(Protocol):
    """Structural protocol any text-classifier backend satisfies.

    HFTextClassifier (the generic runtime) satisfies this without
    inheriting. Tests use fakes that match the signature structurally.
    """

    def classify(self, input: str | list[str]) -> list[ClassificationResult]:
        """Return one ClassificationResult per input.

        Even when `input` is a scalar str, returns a list of length 1.
        Order matches input order in batch mode.
        """
        ...


@runtime_checkable
class ZeroShotClassifierModel(Protocol):
    """Structural protocol for zero-shot NLI runtimes.

    HFZeroShotPipeline (the generic runtime) satisfies this without
    inheriting. Models that satisfy ZeroShotClassifierModel must
    declare `supports_zero_shot: True` in their MANIFEST capabilities
    so the route's capability gate accepts candidate_labels.

    Wire shape is the same as TextClassifierModel.classify: returns
    list[ClassificationResult], one per input, with scores keyed by
    the candidate labels. Lets the codec layer treat both runtimes
    uniformly.
    """

    def classify_zero_shot(
        self,
        input: str | list[str],
        candidate_labels: list[str],
        *,
        multi_label: bool = False,
    ) -> list[ClassificationResult]:
        """Run zero-shot NLI on each input against the candidate labels.

        With multi_label=False, scores are softmaxed across candidate
        labels (sum to ~1.0). With multi_label=True, each label is
        scored independently via NLI entailment probability (no
        normalization). The runtime sets multi_label on each
        ClassificationResult to match the call.
        """
        ...
