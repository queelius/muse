"""Tests for the shared probe WAV helper."""
from __future__ import annotations

import os
import wave

import pytest

from muse.modalities._common.probe_audio import probe_wav


@pytest.mark.parametrize("tone", [None, 220])
def test_probe_wav_writes_16k_mono_and_removes_it(tone):
    with probe_wav(0.5, tone_hz=tone) as path:
        with wave.open(path, "rb") as w:
            assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 16000)
            frames = w.readframes(w.getnframes())
        assert len(frames) == 2 * 8000
        assert any(frames) is (tone is not None)
    assert not os.path.exists(path)


def test_probe_wav_removed_when_body_raises():
    with pytest.raises(RuntimeError):
        with probe_wav() as path:
            raise RuntimeError("probe failed")
    assert not os.path.exists(path)
