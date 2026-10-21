"""Temporary probe WAV for audio-input modalities' PROBE_DEFAULTS.

Stdlib-only: modality `__init__` modules import this, and discovery
imports those without numpy / torch available.
"""
from __future__ import annotations

import math
import os
import struct
import tempfile
import wave
from contextlib import contextmanager
from typing import Iterator


_SAMPLE_RATE = 16000


@contextmanager
def probe_wav(seconds: float = 1.0, *, tone_hz: float | None = None) -> Iterator[str]:
    """Yield the path of a 16kHz mono 16-bit WAV, removed on exit.

    Silence by default; `tone_hz` writes a low-amplitude sine instead
    (for models that need some signal, e.g. alignment / quality).
    """
    n = int(_SAMPLE_RATE * seconds)
    if tone_hz is None:
        frames = b"\x00\x00" * n
    else:
        frames = b"".join(
            struct.pack("<h", int(2500 * math.sin(2 * math.pi * tone_hz * i / _SAMPLE_RATE)))
            for i in range(n)
        )
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        path = tmp.name
    try:
        with wave.open(path, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(_SAMPLE_RATE)
            wav.writeframes(frames)
        yield path
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
