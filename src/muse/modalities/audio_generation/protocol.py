"""Modality protocol for audio/generation.

Defines AudioGenerationModel (the backend contract) and
AudioGenerationResult (the synthesis return type). Backends satisfy
the protocol structurally, no base class required.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np


@dataclass
class AudioGenerationResult:
    """One generated audio clip plus provenance metadata.

    audio: numpy float32 array. Shape `(samples,)` for mono or
        `(samples, channels)` for multi-channel. The codec converts to
        int16 PCM (or compressed format) at output.
    sample_rate: per-model output sample rate (Hz). Stable Audio Open
        is 44100; MusicGen typically 32000; AudioGen 16000.
    channels: 1 for mono, 2 for stereo. Codec respects this when
        writing the WAV/FLAC header.
    duration_seconds: real duration of the synthesized clip; may
        differ slightly from the requested duration when the model
        rounds to its internal grid.
    metadata: dict of model-specific provenance (prompt, steps,
        guidance, seed, model_id).
    """
    audio: np.ndarray
    sample_rate: int
    channels: int
    duration_seconds: float
    metadata: dict = field(default_factory=dict)


@runtime_checkable
class AudioGenerationModel(Protocol):
    """Structural protocol any audio-generation backend satisfies."""

    @property
    def model_id(self) -> str: ...

    def generate(
        self,
        prompt: str,
        *,
        duration: float | None = None,
        seed: int | None = None,
        steps: int | None = None,
        guidance: float | None = None,
        negative_prompt: str | None = None,
        **kwargs: Any,
    ) -> AudioGenerationResult: ...
