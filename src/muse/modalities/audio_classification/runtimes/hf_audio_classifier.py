"""HFAudioClassifier: generic runtime over AutoModelForAudioClassification.

Wraps `transformers.AutoModelForAudioClassification` + `AutoFeatureExtractor`.
Audio decoding via librosa (already a dep of audio_embedding /
audio_transcription); resampling per-model via the feature extractor's
`sampling_rate` attribute.

Supports both softmax single-label heads (sentiment-style emotion,
language-ID) and sigmoid multi-label heads (AudioSet event
classifiers where multiple events may be present at once). Detection
via `model.config.problem_type == "multi_label_classification"`.

Deferred imports follow the muse pattern: torch, transformers, librosa
as module-top sentinels populated by _ensure_deps(). Tests patch the
sentinels directly.
"""
from __future__ import annotations

import logging
from typing import Any

from muse.core.runtime_helpers import (
    LoadTimer, dtype_for_name, select_device, set_inference_mode,
)
from muse.modalities.audio_classification.protocol import (
    AudioClassificationResult,
)


logger = logging.getLogger(__name__)


torch: Any = None
AutoModelForAudioClassification: Any = None
AutoFeatureExtractor: Any = None
librosa: Any = None


def _ensure_deps() -> None:
    global torch, AutoModelForAudioClassification, AutoFeatureExtractor, librosa
    if torch is None:
        try:
            import torch as _t
            torch = _t
        except Exception as e:  # noqa: BLE001
            logger.debug("HFAudioClassifier torch unavailable: %s", e)
    if AutoModelForAudioClassification is None:
        try:
            from transformers import AutoModelForAudioClassification as _m
            AutoModelForAudioClassification = _m
        except Exception as e:  # noqa: BLE001
            logger.debug(
                "HFAudioClassifier AutoModelForAudioClassification unavailable: %s",
                e,
            )
    if AutoFeatureExtractor is None:
        try:
            from transformers import AutoFeatureExtractor as _p
            AutoFeatureExtractor = _p
        except Exception as e:  # noqa: BLE001
            logger.debug(
                "HFAudioClassifier AutoFeatureExtractor unavailable: %s", e,
            )
    if librosa is None:
        try:
            import librosa as _l
            librosa = _l
        except Exception as e:  # noqa: BLE001
            logger.debug("HFAudioClassifier librosa unavailable: %s", e)


class HFAudioClassifier:
    """Generic audio classification runtime."""

    model_id: str

    def __init__(
        self,
        *,
        model_id: str,
        hf_repo: str,
        local_dir: str | None = None,
        device: str = "auto",
        dtype: str = "fp32",
        multi_label: bool | None = None,
        **_: Any,
    ) -> None:
        _ensure_deps()
        if torch is None:
            raise RuntimeError(
                "torch is not installed; run `muse pull` or install "
                "`torch` into this venv"
            )
        if AutoModelForAudioClassification is None or AutoFeatureExtractor is None:
            raise RuntimeError(
                "transformers is not installed; run `muse pull` or "
                "install `transformers` into this venv"
            )
        if librosa is None:
            raise RuntimeError(
                "librosa is not installed; run `muse pull` or install "
                "`librosa` into this venv"
            )
        self.model_id = model_id
        self._device = _select_device(device)
        self._dtype = _resolve_dtype(dtype)
        src = local_dir or hf_repo
        with LoadTimer(f"loading audio classifier from {src}", logger):
            self._extractor = AutoFeatureExtractor.from_pretrained(src)
            self._model = AutoModelForAudioClassification.from_pretrained(
                src, torch_dtype=self._dtype,
            )
            self._model = self._model.to(self._device)
        set_inference_mode(self._model)

        cfg = self._model.config
        self._id2label: dict[int, str] = {}
        for k, v in (getattr(cfg, "id2label", None) or {}).items():
            try:
                self._id2label[int(k)] = str(v)
            except (TypeError, ValueError):
                continue
        config_multi_label = (
            getattr(cfg, "problem_type", None) == "multi_label_classification"
        )
        # OR the manifest hint with the config check: hints can flip
        # False -> True (e.g., AST/AudioSet checkpoints whose config
        # lacks problem_type) but never override a config-declared
        # True. Hints are additive metadata, not a clobber.
        if multi_label is None:
            self._multi_label = config_multi_label
        else:
            self._multi_label = bool(multi_label) or config_multi_label
        self._sampling_rate = int(
            getattr(self._extractor, "sampling_rate", 16000)
        )

    def classify(self, audio_path: str) -> list[AudioClassificationResult]:
        """Classify one audio file. Returns a length-1 list."""
        # librosa.load resamples to the model's expected rate and
        # downmixes to mono.
        waveform, _sr = librosa.load(
            audio_path, sr=self._sampling_rate, mono=True,
        )
        inputs = self._extractor(
            waveform,
            sampling_rate=self._sampling_rate,
            return_tensors="pt",
        )
        inputs = {
            k: (v.to(self._device) if hasattr(v, "to") else v)
            for k, v in inputs.items()
        }
        outputs = self._model(**inputs)
        logits = outputs.logits  # (1, num_labels)

        if self._multi_label:
            probs = torch.sigmoid(logits)
        else:
            probs = torch.softmax(logits, dim=-1)
        probs_row = probs.detach().cpu().to(torch.float32).numpy()[0]

        scores = {
            self._id2label.get(i, str(i)): float(probs_row[i])
            for i in range(probs_row.shape[-1])
        }
        return [AudioClassificationResult(
            scores=scores, multi_label=self._multi_label,
        )]


def _select_device(device: str) -> str:
    return select_device(device, torch_module=torch)


def _resolve_dtype(dtype: str):
    return dtype_for_name(dtype, torch_module=torch)
