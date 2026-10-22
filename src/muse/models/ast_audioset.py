"""Bundled muse model: ast-audioset.

Audio Spectrogram Transformer (AST) finetuned on AudioSet (527 classes).
~340MB, BSD-3-Clause. Default for /v1/audio/classifications: covers
speech / music / animals / vehicles / alarms / household sounds with
broad coverage.

Multi-label (sigmoid head): a single audio can be tagged with multiple
events (e.g., "music" + "speech" for a podcast intro).

Wraps HFAudioClassifier.
"""
from __future__ import annotations

from muse.modalities.audio_classification.runtimes import HFAudioClassifier


MANIFEST = {
    "model_id": "ast-audioset",
    "modality": "audio/classification",
    "hf_repo": "MIT/ast-finetuned-audioset-10-10-0.4593",
    "revision": "f826b80d28226b62986cc218e5cec390b1096902",
    "description": (
        "AST AudioSet: 527 classes (speech, music, animals, vehicles, "
        "alarms, household, etc.), multi-label, BSD-3-Clause."
    ),
    "summary": "Tag audio clips with AudioSet sound-event labels (speech, music, instruments, animals, alarms, ...)",
    "strengths": [
        "527-class multi-label scores, so overlapping sounds can all score high",
        "CPU-friendly, ~0.7 GB",
        "BSD-3-Clause",
    ],
    "limits": [
        "AudioSet labels only; no emotion or spoken-language ID (see the curated audio/classification models)",
        "One set of labels per clip; no timestamps",
    ],
    "quality_tier": "good",
    "license": "BSD-3-Clause",
    "pip_extras": [
        "torch>=2.1.0", "transformers>=4.40.0",
        "librosa>=0.10.0",
    ],
    "system_packages": [],
    "capabilities": {
        "device": "auto",
        "memory_gb": 0.7,
        "multi_label": True,
        "num_labels": 527,
    },
}


class Model(HFAudioClassifier):
    pass
