"""HFTextClassifier: generic runtime over any HF text-classification model.

One class wraps `AutoModelForSequenceClassification` + `AutoTokenizer`
for any HuggingFace text-classifier. Pulled via the HF resolver:
`muse pull hf://KoalaAI/Text-Moderation` synthesizes a manifest
pointing at this class.

Deferred imports follow the muse pattern: torch, AutoTokenizer, and
AutoModelForSequenceClassification stay as module-top sentinels (None)
until _ensure_deps() lazy-imports them. Tests patch the sentinels
directly; _ensure_deps short-circuits on non-None.
"""
from __future__ import annotations

import logging
from typing import Any

from muse.core.runtime_helpers import select_device, set_inference_mode
from muse.modalities.text_classification.protocol import ClassificationResult


logger = logging.getLogger(__name__)

torch: Any = None
AutoTokenizer: Any = None
AutoModelForSequenceClassification: Any = None


def _ensure_deps() -> None:
    global torch, AutoTokenizer, AutoModelForSequenceClassification
    if torch is None:
        try:
            import torch as _t
            torch = _t
        except Exception as e:  # noqa: BLE001
            logger.debug("HFTextClassifier torch unavailable: %s", e)
    if AutoTokenizer is None:
        try:
            from transformers import AutoTokenizer as _tk
            AutoTokenizer = _tk
        except Exception as e:  # noqa: BLE001
            logger.debug("HFTextClassifier AutoTokenizer unavailable: %s", e)
    if AutoModelForSequenceClassification is None:
        try:
            from transformers import AutoModelForSequenceClassification as _m
            AutoModelForSequenceClassification = _m
        except Exception as e:  # noqa: BLE001
            logger.debug("HFTextClassifier AutoModel unavailable: %s", e)


class HFTextClassifier:
    """Generic HuggingFace text-classification runtime."""

    model_id: str

    def __init__(
        self,
        *,
        model_id: str,
        hf_repo: str,
        local_dir: str | None = None,
        device: str = "auto",
        max_length: int = 512,
        **_: Any,
    ) -> None:
        _ensure_deps()
        if AutoTokenizer is None or AutoModelForSequenceClassification is None:
            raise RuntimeError(
                "transformers is not installed; run `muse pull` or "
                "install `transformers` into this venv"
            )
        self.model_id = model_id
        self._device = _select_device(device)
        self._max_length = max_length

        src = local_dir or hf_repo
        logger.info(
            "loading text classifier from %s (device=%s)",
            src, self._device,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(src)
        self._model = AutoModelForSequenceClassification.from_pretrained(src)
        self._model = self._model.to(self._device)
        # Switch to inference mode via the shared helper, which keeps the
        # literal method-name token out of this file (security pre-commit
        # hook policy; see CLAUDE.md "No literal eval token").
        set_inference_mode(self._model)

        cfg = self._model.config
        self._id2label: dict[int, str] = dict(getattr(cfg, "id2label", {}))
        self._multi_label = (
            getattr(cfg, "problem_type", None) == "multi_label_classification"
        )

    def classify(self, input: str | list[str]) -> list[ClassificationResult]:
        texts = [input] if isinstance(input, str) else list(input)
        if not texts:
            return []

        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt",
        )
        encoded = encoded.to(self._device)

        outputs = self._model(**encoded)
        logits = outputs.logits  # shape: (batch, num_labels)

        if self._multi_label:
            probs = torch.sigmoid(logits)
        else:
            probs = torch.softmax(logits, dim=-1)

        probs_np = probs.detach().cpu().numpy()  # (batch, num_labels)

        results: list[ClassificationResult] = []
        for row in probs_np:
            scores = {
                self._id2label.get(i, str(i)): float(row[i])
                for i in range(row.shape[-1])
            }
            results.append(ClassificationResult(
                scores=scores,
                multi_label=self._multi_label,
            ))
        return results


def _select_device(device: str) -> str:
    """Thin delegator preserved for test imports. Real logic in runtime_helpers."""
    return select_device(device, torch_module=torch)
