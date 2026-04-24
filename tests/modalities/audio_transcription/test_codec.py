"""Codec: TranscriptionResult to bytes for 5 response formats."""
import json as _json

import pytest

from muse.modalities.audio_transcription import (
    Segment,
    TranscriptionResult,
    Word,
)
from muse.modalities.audio_transcription.codec import (
    encode_transcription,
    _format_srt_ts,
    _format_vtt_ts,
)


@pytest.fixture
def two_segments():
    return TranscriptionResult(
        text="Hello and welcome. This is a test.",
        language="en",
        duration=9.1,
        task="transcribe",
        segments=[
            Segment(id=0, start=0.0, end=4.52, text="Hello and welcome.",
                    words=[Word("Hello", 0.0, 0.48),
                           Word("and", 0.48, 0.72),
                           Word("welcome.", 0.72, 4.52)]),
            Segment(id=1, start=4.52, end=9.1, text="This is a test.",
                    words=[Word("This", 4.52, 4.80),
                           Word("is", 4.80, 5.00),
                           Word("a", 5.00, 5.10),
                           Word("test.", 5.10, 9.1)]),
        ],
    )


# --- Timestamp formatters ---

@pytest.mark.parametrize("secs,expected", [
    (0.0, "00:00:00,000"),
    (0.999, "00:00:00,999"),
    (3661.5, "01:01:01,500"),
    (7325.123, "02:02:05,123"),
    # Rounding-edge cases: sub-integer values must roll over cleanly
    # to the next whole second (regression: pre-fix produced "00:00:00,1000")
    (0.9999, "00:00:01,000"),
    (3599.9999, "01:00:00,000"),      # rolls minutes into hours
    (1.0 - 1e-9, "00:00:01,000"),     # float-epsilon below one second
])
def test_format_srt_ts(secs, expected):
    assert _format_srt_ts(secs) == expected


@pytest.mark.parametrize("secs,expected", [
    (0.0, "00:00:00.000"),
    (3661.5, "01:01:01.500"),
    (0.9999, "00:00:01.000"),
    (3599.9999, "01:00:00.000"),
])
def test_format_vtt_ts(secs, expected):
    assert _format_vtt_ts(secs) == expected


# --- json ---

def test_json_format_is_text_only(two_segments):
    body, ct = encode_transcription(two_segments, "json")
    assert ct == "application/json"
    parsed = _json.loads(body)
    assert parsed == {"text": "Hello and welcome. This is a test."}


# --- text ---

def test_text_format_is_raw_transcript(two_segments):
    body, ct = encode_transcription(two_segments, "text")
    assert ct == "text/plain"
    assert body.decode() == "Hello and welcome. This is a test."


# --- srt ---

def test_srt_format_has_correct_shape(two_segments):
    body, ct = encode_transcription(two_segments, "srt")
    assert ct == "application/x-subrip"
    content = body.decode()
    # 2 numbered blocks, comma-millisecond separator, blank-line delimited
    assert content.startswith("1\n00:00:00,000 --> 00:00:04,520\n")
    assert "\n\n2\n00:00:04,520 --> 00:00:09,100\n" in content
    assert "Hello and welcome." in content
    assert "This is a test." in content


# --- vtt ---

def test_vtt_format_has_header_and_periods(two_segments):
    body, ct = encode_transcription(two_segments, "vtt")
    assert ct == "text/vtt"
    content = body.decode()
    assert content.startswith("WEBVTT\n\n")
    # period separator, --> between timestamps
    assert "00:00:00.000 --> 00:00:04.520" in content
    assert "00:00:04.520 --> 00:00:09.100" in content


# --- verbose_json (segments only) ---

def test_verbose_json_segments_only(two_segments):
    body, ct = encode_transcription(two_segments, "verbose_json")
    assert ct == "application/json"
    d = _json.loads(body)
    assert d["task"] == "transcribe"
    assert d["language"] == "en"
    assert d["duration"] == 9.1
    assert d["text"] == "Hello and welcome. This is a test."
    assert len(d["segments"]) == 2
    assert d["segments"][0]["text"] == "Hello and welcome."
    # Without word granularity, `words` key must be absent (matches OpenAI)
    assert "words" not in d


# --- verbose_json (with words) ---

def test_verbose_json_with_words_flattened(two_segments):
    body, ct = encode_transcription(two_segments, "verbose_json", include_words=True)
    d = _json.loads(body)
    assert "words" in d
    # Words flattened from per-segment into top-level
    assert len(d["words"]) == 7  # 3 + 4
    assert d["words"][0] == {"word": "Hello", "start": 0.0, "end": 0.48}
    assert d["words"][-1]["word"] == "test."


# --- dispatcher errors ---

def test_unknown_format_raises(two_segments):
    with pytest.raises(ValueError, match="unknown response_format"):
        encode_transcription(two_segments, "xml")
