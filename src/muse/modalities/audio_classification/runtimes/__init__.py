"""Generic runtime for audio/classification.

HFAudioClassifier wraps `AutoModelForAudioClassification` +
`AutoFeatureExtractor` with librosa decode upstream. Supports both
softmax (single-label) and sigmoid (multi-label) heads.
"""
from muse.modalities.audio_classification.runtimes.hf_audio_classifier import (
    HFAudioClassifier,
)


__all__ = ["HFAudioClassifier"]
