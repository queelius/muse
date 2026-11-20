"""WAV and Opus encoding for audio/speech responses.

The PCM16 conversion, WAV writer and owned-ffmpeg Opus encoder live in
`muse.modalities._common.audio_codec` (shared with audio/generation); this
module keeps speech's mono-only WAV entry point.
"""
from __future__ import annotations

import numpy as np

from muse.modalities._common.audio_codec import (  # noqa: F401  (re-exports)
    AudioFormatError,
    float_to_pcm16,
    pcm16_wav_bytes,
    wav_bytes_to_opus,
)


def audio_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Convert float32 [-1, 1] mono audio to a 16-bit PCM WAV bytestring."""
    if audio.ndim != 1:
        raise AudioFormatError(f"expected 1-D audio, got shape {audio.shape}")
    return pcm16_wav_bytes(audio, sample_rate, channels=1)
