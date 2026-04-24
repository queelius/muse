"""Response encoding for /v1/audio/transcriptions and /v1/audio/translations.

OpenAI defines 5 formats; this codec is the pure function that turns a
TranscriptionResult into bytes + content-type for any of them. Tests cover
the formatters independently of FastAPI.
"""
from __future__ import annotations

import json
from typing import Any

from muse.modalities.audio_transcription.protocol import TranscriptionResult


def encode_transcription(
    result: TranscriptionResult,
    fmt: str,
    *,
    include_words: bool = False,
) -> tuple[bytes, str]:
    """Return (body_bytes, content_type) for the requested response_format.

    `include_words` only affects verbose_json; the word list is flattened
    across segments and placed at the top level of the response object to
    match OpenAI's shape.
    """
    if fmt == "json":
        return json.dumps(_to_json(result)).encode(), "application/json"
    if fmt == "text":
        return _to_text(result).encode(), "text/plain"
    if fmt == "srt":
        return _to_srt(result).encode(), "application/x-subrip"
    if fmt == "vtt":
        return _to_vtt(result).encode(), "text/vtt"
    if fmt == "verbose_json":
        return (
            json.dumps(_to_verbose_json(result, include_words=include_words)).encode(),
            "application/json",
        )
    raise ValueError(f"unknown response_format {fmt!r}")


def _to_json(r: TranscriptionResult) -> dict:
    return {"text": r.text}


def _to_text(r: TranscriptionResult) -> str:
    return r.text


def _to_srt(r: TranscriptionResult) -> str:
    parts = []
    for i, s in enumerate(r.segments, start=1):
        parts.append(
            f"{i}\n"
            f"{_format_srt_ts(s.start)} --> {_format_srt_ts(s.end)}\n"
            f"{s.text}\n"
        )
    return "\n".join(parts)


def _to_vtt(r: TranscriptionResult) -> str:
    parts = ["WEBVTT\n"]
    for s in r.segments:
        parts.append(
            f"{_format_vtt_ts(s.start)} --> {_format_vtt_ts(s.end)}\n"
            f"{s.text}\n"
        )
    return "\n".join(parts)


def _to_verbose_json(r: TranscriptionResult, *, include_words: bool) -> dict[str, Any]:
    out: dict[str, Any] = {
        "task": r.task,
        "language": r.language,
        "duration": r.duration,
        "text": r.text,
        "segments": [
            {"id": s.id, "start": s.start, "end": s.end, "text": s.text}
            for s in r.segments
        ],
    }
    if include_words:
        out["words"] = [
            {"word": w.word, "start": w.start, "end": w.end}
            for s in r.segments
            for w in (s.words or [])
        ]
    return out


def _format_srt_ts(seconds: float) -> str:
    """SubRip: HH:MM:SS,mmm (comma before milliseconds)."""
    hours, rem = divmod(seconds, 3600)
    mins, secs = divmod(rem, 60)
    whole = int(secs)
    ms = int(round((secs - whole) * 1000))
    return f"{int(hours):02d}:{int(mins):02d}:{whole:02d},{ms:03d}"


def _format_vtt_ts(seconds: float) -> str:
    """WebVTT: HH:MM:SS.mmm (period before milliseconds)."""
    hours, rem = divmod(seconds, 3600)
    mins, secs = divmod(rem, 60)
    whole = int(secs)
    ms = int(round((secs - whole) * 1000))
    return f"{int(hours):02d}:{int(mins):02d}:{whole:02d}.{ms:03d}"
