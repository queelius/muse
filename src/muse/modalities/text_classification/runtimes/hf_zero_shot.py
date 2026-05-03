"""HFZeroShotPipeline: generic runtime for zero-shot NLI classification.

Wraps `transformers.pipeline("zero-shot-classification")` which handles
the NLI premise/hypothesis construction internally. Approximately the
same shape as HFTextClassifier (deferred imports, lazy load, shared
runtime helpers); the pipeline does the fiddly work of building the
"input ENTAILS label" hypotheses per candidate.

Pulled via the HF resolver: `muse pull hf://MoritzLaurer/deberta-v3-
base-zeroshot-v2.0` synthesizes a manifest pointing at this class.

Deferred imports follow the muse pattern: torch and pipeline stay as
module-top sentinels (None) until _ensure_deps() lazy-imports them.
Tests patch the sentinels directly; _ensure_deps short-circuits on
non-None.
"""
from __future__ import annotations

import logging
from typing import Any

from muse.core.runtime_helpers import select_device, set_inference_mode
from muse.modalities.text_classification.protocol import ClassificationResult


logger = logging.getLogger(__name__)


torch: Any = None
pipeline: Any = None


def _ensure_deps() -> None:
    global torch, pipeline
    if torch is None:
        try:
            import torch as _t
            torch = _t
        except Exception as e:  # noqa: BLE001
            logger.debug("HFZeroShotPipeline torch unavailable: %s", e)
    if pipeline is None:
        try:
            from transformers import pipeline as _p
            pipeline = _p
        except Exception as e:  # noqa: BLE001
            logger.debug("HFZeroShotPipeline transformers.pipeline unavailable: %s", e)


class HFZeroShotPipeline:
    """Generic HuggingFace zero-shot NLI runtime."""

    model_id: str

    def __init__(
        self,
        *,
        model_id: str,
        hf_repo: str,
        local_dir: str | None = None,
        device: str = "auto",
        **_: Any,
    ) -> None:
        _ensure_deps()
        if torch is None:
            raise RuntimeError(
                "torch is not installed; run `muse pull` or install "
                "`torch` into this venv"
            )
        if pipeline is None:
            raise RuntimeError(
                "transformers is not installed; run `muse pull` or "
                "install `transformers` into this venv"
            )
        self.model_id = model_id
        self._device = _select_device(device)
        # The pipeline accepts device as int (-1=cpu, 0+=cuda:N) or
        # str ("cpu"/"cuda:0"). HF accepts both since 4.30+, so the
        # string form (returned by select_device) works directly.
        src = local_dir or hf_repo
        logger.info(
            "loading zero-shot pipeline from %s (device=%s)",
            src, self._device,
        )
        self._pipe = pipeline(
            task="zero-shot-classification",
            model=src,
            device=self._device,
        )
        # Switch to inference mode via the shared helper, which keeps
        # the literal method-name token out of this file (security
        # pre-commit hook policy; see CLAUDE.md "No literal eval token").
        set_inference_mode(self._pipe.model)

    def classify_zero_shot(
        self,
        input: str | list[str],
        candidate_labels: list[str],
        *,
        multi_label: bool = False,
    ) -> list[ClassificationResult]:
        """Run zero-shot NLI on each input against candidate_labels.

        With multi_label=False, scores are softmaxed across candidate
        labels (sum to ~1.0). With multi_label=True, each label is
        scored independently via NLI entailment probability.

        Returns one ClassificationResult per input. The runtime sets
        multi_label on each result to match the call (for consumers
        that care, like the moderations codec; the classifications
        codec ignores the field).
        """
        if not candidate_labels:
            raise ValueError("candidate_labels must be non-empty")
        # Strip whitespace + dedupe at request time. The pipeline does
        # not, and "happy" vs "happy " produces two distinct logits.
        seen: set[str] = set()
        labels: list[str] = []
        for raw in candidate_labels:
            stripped = raw.strip() if isinstance(raw, str) else ""
            if not stripped or stripped in seen:
                continue
            seen.add(stripped)
            labels.append(stripped)
        if not labels:
            raise ValueError(
                "candidate_labels reduced to empty after stripping/dedupe"
            )

        texts = [input] if isinstance(input, str) else list(input)
        if not texts:
            return []

        # The pipeline accepts a list natively; iterating per-text
        # makes the failure mode for one bad input localized rather
        # than poisoning the whole batch.
        results: list[ClassificationResult] = []
        for text in texts:
            raw_out = self._pipe(
                text,
                candidate_labels=labels,
                multi_label=multi_label,
            )
            # Pipeline returns {labels, scores, sequence}; labels are
            # already sorted by score descending. Build a dict so the
            # result type matches the fine-tuned classifier path.
            scores = {
                label: float(score)
                for label, score in zip(raw_out["labels"], raw_out["scores"])
            }
            results.append(ClassificationResult(
                scores=scores,
                multi_label=multi_label,
            ))
        return results


def _select_device(device: str) -> str:
    """Thin delegator preserved for test imports. Real logic in runtime_helpers."""
    return select_device(device, torch_module=torch)
