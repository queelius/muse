"""Tests for the audio_speech text normalizer's time expansion.

Regression guard for L4: the H:MM:SS fallback emitted a Python set-repr
(`{'45'}`) instead of the minutes string, degrading TTS prosody.
"""
from __future__ import annotations

import re

from muse.modalities.audio_speech.utils.text_normalizer import (
    _expand_time,
    _time_re,
)


def _expand(text: str) -> str:
    return re.sub(_time_re, _expand_time, text)


class TestExpandTime:
    def test_hmmss_minutes_not_set_repr(self):
        # 9:45:30 -> minutes "45" does not start with 0 and isn't "00", so it
        # hit the fallback branch, which was a set literal {minutes}.
        out = _expand("9:45:30")
        assert "{" not in out and "}" not in out
        assert "'" not in out
        assert out == "9 45 30"

    def test_hmmss_leading_zero_minutes(self):
        assert _expand("9:05:30") == "9 oh 05 30"

    def test_hmmss_zero_minutes(self):
        assert _expand("9:00:30") == "9 oh oh 30"

    def test_hh_mm_unaffected(self):
        assert _expand("12:45") == "12 45"
