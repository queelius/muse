# audio/transcription (Whisper family) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship muse's 5th modality, `audio/transcription`, via a generic `FasterWhisperModel` runtime serving any Systran faster-whisper HF repo. OpenAI wire-compat for `/v1/audio/transcriptions` and `/v1/audio/translations`. HF resolver sniffs CT2 shape. Three curated entries (tiny/base/large-v3). First muse modality with multipart/form-data file upload; keep the pattern inline until a second multipart modality arrives.

**Architecture:** Modality subpackage at `src/muse/modalities/audio_transcription/` mirrors the existing four modalities: `protocol.py` with `TranscriptionModel` Protocol + result dataclasses, `routes.py` building a FastAPI router, `codec.py` with 5 pure formatters, `client.py` for HTTP, `runtimes/faster_whisper.py` wrapping the CT2 engine. No bundled `src/muse/models/*.py` scripts; every Whisper variant is resolver-pulled. HF resolver grows a `_looks_like_faster_whisper` branch.

**Tech Stack:** faster-whisper (CT2 engine), FastAPI multipart, PyAV/ffmpeg (audio decode), pytest, httpx (TranscriptionClient).

**Spec:** `docs/superpowers/specs/2026-04-24-audio-transcription-whisper-design.md`

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `src/muse/modalities/audio_transcription/__init__.py` | create | exports `MODALITY` + `build_router`; re-exports Protocol + dataclasses |
| `src/muse/modalities/audio_transcription/protocol.py` | create | `Word`, `Segment`, `TranscriptionResult` dataclasses; `TranscriptionModel` Protocol |
| `src/muse/modalities/audio_transcription/codec.py` | create | 5 pure format fns + `encode_transcription` dispatcher |
| `src/muse/modalities/audio_transcription/routes.py` | create | `build_router(registry) -> APIRouter` with 2 multipart endpoints |
| `src/muse/modalities/audio_transcription/client.py` | create | `TranscriptionClient` |
| `src/muse/modalities/audio_transcription/runtimes/__init__.py` | create | empty package marker |
| `src/muse/modalities/audio_transcription/runtimes/faster_whisper.py` | create | `FasterWhisperModel` generic runtime |
| `src/muse/core/resolvers_hf.py` | modify | add faster-whisper branch to `_sniff_repo_shape`, `resolve`, `search` |
| `src/muse/curated.yaml` | modify | +3 entries (whisper-tiny/base/large-v3) |
| `tests/modalities/audio_transcription/` | create | full unit suite mirroring the 4 existing modality test dirs |
| `tests/core/test_resolvers_hf.py` | modify | CT2 sniff + resolve tests |
| `tests/core/test_curated.py` | modify | the 3 new entries parse cleanly |
| `tests/cli_impl/test_e2e_asr.py` | create | one `@pytest.mark.slow` test: multipart + codec end-to-end |
| `tests/integration/conftest.py` | modify | `whisper_model` fixture |
| `tests/integration/test_remote_asr.py` | create | one opt-in test hitting a real server |
| `tests/fixtures/asr_sample.wav` | create | short test clip (~1s, 16kHz mono) for the integration test |
| `CLAUDE.md` | modify | add `audio/transcription` to modality list; multipart-pattern note |
| `README.md` | modify | modality list + endpoint block |
| `pyproject.toml` | modify | version 0.12.1 to 0.13.0 |

---

### Task 1: Protocol + dataclasses

**Files:**
- Create: `src/muse/modalities/audio_transcription/__init__.py`
- Create: `src/muse/modalities/audio_transcription/protocol.py`
- Create: `src/muse/modalities/audio_transcription/runtimes/__init__.py` (empty)
- Test: `tests/modalities/audio_transcription/__init__.py` (empty) + `tests/modalities/audio_transcription/test_protocol.py`

- [ ] **Step 1: Write the failing protocol tests**

Create `tests/modalities/audio_transcription/__init__.py` (empty) and `tests/modalities/audio_transcription/test_protocol.py`:

```python
"""Protocol + dataclass shape tests for audio/transcription."""
from muse.modalities.audio_transcription import (
    MODALITY,
    Word,
    Segment,
    TranscriptionResult,
    TranscriptionModel,
)


def test_modality_tag_is_mime_shaped():
    assert MODALITY == "audio/transcription"


def test_word_dataclass_roundtrip():
    w = Word(word="hello", start=0.0, end=0.5)
    assert w.word == "hello"
    assert w.end == 0.5


def test_segment_with_and_without_words():
    s1 = Segment(id=0, start=0.0, end=1.0, text="hi", words=None)
    assert s1.words is None
    s2 = Segment(
        id=1, start=1.0, end=2.0, text="world",
        words=[Word("world", 1.0, 2.0)],
    )
    assert len(s2.words) == 1
    assert s2.words[0].word == "world"


def test_transcription_result_minimal():
    r = TranscriptionResult(
        text="hi world", language="en", duration=2.0,
        segments=[
            Segment(id=0, start=0.0, end=1.0, text="hi", words=None),
            Segment(id=1, start=1.0, end=2.0, text="world", words=None),
        ],
        task="transcribe",
    )
    assert r.text == "hi world"
    assert r.language == "en"
    assert r.task == "transcribe"
    assert len(r.segments) == 2


def test_transcription_model_protocol_is_structural():
    """Any class with a matching `transcribe` signature satisfies the protocol."""
    class Fake:
        def transcribe(self, audio_path, **kwargs):
            return TranscriptionResult(
                text="", language="en", duration=0.0,
                segments=[], task="transcribe",
            )
    # This import-time check matters: if Protocol is misdefined
    # (e.g. accidentally ABC), structural subtyping breaks.
    fake: TranscriptionModel = Fake()  # type: ignore[assignment]
    assert fake.transcribe("/tmp/x").text == ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/modalities/audio_transcription/test_protocol.py -v`
Expected: ModuleNotFoundError on `muse.modalities.audio_transcription`.

- [ ] **Step 3: Create the modality package**

Create `src/muse/modalities/audio_transcription/__init__.py`:

```python
"""audio/transcription modality: automatic speech recognition.

Public surface:
  - MODALITY: str (MIME-shaped tag; used by discover_modalities)
  - build_router(registry) -> APIRouter (mounted by the worker)
  - Word, Segment, TranscriptionResult dataclasses
  - TranscriptionModel Protocol

Wire contract (OpenAI-compat):
  - POST /v1/audio/transcriptions
  - POST /v1/audio/translations

The `build_router` import is deferred (circular between routes.py and
this __init__), matching the pattern used by audio_speech.
"""
from __future__ import annotations

from muse.modalities.audio_transcription.protocol import (
    Word,
    Segment,
    TranscriptionResult,
    TranscriptionModel,
)


MODALITY = "audio/transcription"


def build_router(registry):
    """Lazy import of routes.build_router to keep __init__ deps-light.

    routes.py imports FastAPI at module top; __init__ must import cheaply
    so discovery works in the supervisor process (no ML deps installed).
    """
    from muse.modalities.audio_transcription.routes import (
        build_router as _build,
    )
    return _build(registry)


__all__ = [
    "MODALITY",
    "build_router",
    "Word",
    "Segment",
    "TranscriptionResult",
    "TranscriptionModel",
]
```

Create `src/muse/modalities/audio_transcription/runtimes/__init__.py` (empty file with a one-line module docstring):

```python
"""Generic runtimes for audio/transcription."""
```

Create `src/muse/modalities/audio_transcription/protocol.py`:

```python
"""Protocol + dataclasses for audio/transcription."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol


@dataclass
class Word:
    """A single word with its start/end timestamps."""
    word: str
    start: float
    end: float


@dataclass
class Segment:
    """A transcript segment (Whisper's native unit of output)."""
    id: int
    start: float
    end: float
    text: str
    words: list[Word] | None


@dataclass
class TranscriptionResult:
    """Full transcription output.

    - text: concatenated transcript for the whole file
    - language: detected or user-specified ISO-639-1 code
    - duration: input audio duration in seconds
    - segments: Whisper segments in time order
    - task: 'transcribe' (source-language transcript) or 'translate'
      (source-language audio to English transcript)
    """
    text: str
    language: str
    duration: float
    segments: list[Segment]
    task: Literal["transcribe", "translate"]


class TranscriptionModel(Protocol):
    """Structural protocol any ASR backend satisfies.

    FasterWhisperModel (the generic runtime) satisfies this without
    inheriting. Tests use fakes that match the signature structurally.
    """

    def transcribe(
        self,
        audio_path: str,
        *,
        task: Literal["transcribe", "translate"] = "transcribe",
        language: str | None = None,
        prompt: str | None = None,
        temperature: float = 0.0,
        word_timestamps: bool = False,
        vad_filter: bool = False,
        **kwargs: Any,
    ) -> TranscriptionResult: ...
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/modalities/audio_transcription/test_protocol.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/muse/modalities/audio_transcription/__init__.py \
        src/muse/modalities/audio_transcription/protocol.py \
        src/muse/modalities/audio_transcription/runtimes/__init__.py \
        tests/modalities/audio_transcription/__init__.py \
        tests/modalities/audio_transcription/test_protocol.py
git commit -m "$(cat <<'EOF'
feat(asr): audio/transcription modality skeleton and protocol

MODALITY tag + Word/Segment/TranscriptionResult dataclasses + the
TranscriptionModel structural protocol. routes.py and codec.py
land in follow-up commits.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Codec (5 response formats)

**Files:**
- Create: `src/muse/modalities/audio_transcription/codec.py`
- Test: `tests/modalities/audio_transcription/test_codec.py`

- [ ] **Step 1: Write the failing codec tests**

Create `tests/modalities/audio_transcription/test_codec.py`:

```python
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
])
def test_format_srt_ts(secs, expected):
    assert _format_srt_ts(secs) == expected


@pytest.mark.parametrize("secs,expected", [
    (0.0, "00:00:00.000"),
    (3661.5, "01:01:01.500"),
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/modalities/audio_transcription/test_codec.py -v`
Expected: ModuleNotFoundError on `muse.modalities.audio_transcription.codec`.

- [ ] **Step 3: Implement codec**

Create `src/muse/modalities/audio_transcription/codec.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/modalities/audio_transcription/test_codec.py -v`
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add src/muse/modalities/audio_transcription/codec.py \
        tests/modalities/audio_transcription/test_codec.py
git commit -m "$(cat <<'EOF'
feat(asr): codec for 5 response formats

Pure functions from TranscriptionResult to bytes: json, text, srt, vtt,
verbose_json. SRT uses comma-millisecond separator, VTT uses period,
both use HH:MM:SS. verbose_json flattens per-segment words into a
top-level list to match OpenAI's shape. Unknown formats raise.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Routes (multipart, 2 endpoints)

**Files:**
- Create: `src/muse/modalities/audio_transcription/routes.py`
- Test: `tests/modalities/audio_transcription/test_routes.py`

- [ ] **Step 1: Write the failing route tests**

Create `tests/modalities/audio_transcription/test_routes.py`:

```python
"""Route tests for /v1/audio/transcriptions and /v1/audio/translations."""
import io
import json
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from muse.core.registry import ModalityRegistry
from muse.modalities.audio_transcription import (
    MODALITY,
    Segment,
    TranscriptionResult,
    build_router,
)


def _make_client(backend) -> TestClient:
    reg = ModalityRegistry()
    reg.register(MODALITY, backend, manifest={"model_id": "whisper-tiny"})
    app = FastAPI()
    app.include_router(build_router(reg))
    return TestClient(app)


def _fake_result(task="transcribe", language="en"):
    return TranscriptionResult(
        text="hello world",
        language=language, duration=1.0, task=task,
        segments=[Segment(id=0, start=0.0, end=1.0, text="hello world", words=None)],
    )


def test_transcriptions_returns_json_by_default():
    backend = MagicMock()
    backend.model_id = "whisper-tiny"
    backend.transcribe.return_value = _fake_result()
    client = _make_client(backend)

    r = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("a.wav", b"FAKEWAV", "audio/wav")},
        data={"model": "whisper-tiny"},
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/json")
    assert r.json() == {"text": "hello world"}

    # Backend invoked with task=transcribe by default
    _, kwargs = backend.transcribe.call_args
    assert kwargs["task"] == "transcribe"


def test_translations_forces_task_translate_and_drops_language():
    backend = MagicMock()
    backend.transcribe.return_value = _fake_result(task="translate", language="en")
    client = _make_client(backend)

    r = client.post(
        "/v1/audio/translations",
        files={"file": ("a.wav", b"FAKEWAV", "audio/wav")},
        data={"model": "whisper-tiny", "language": "fr"},  # language is ignored
    )
    assert r.status_code == 200

    _, kwargs = backend.transcribe.call_args
    assert kwargs["task"] == "translate"
    assert kwargs["language"] is None, "language must be dropped on translations"


def test_response_format_srt():
    backend = MagicMock()
    backend.transcribe.return_value = _fake_result()
    client = _make_client(backend)

    r = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("a.wav", b"FAKEWAV", "audio/wav")},
        data={"model": "whisper-tiny", "response_format": "srt"},
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/x-subrip")
    assert "1\n00:00:00,000 --> 00:00:01,000\nhello world" in r.text


def test_response_format_verbose_json_without_words():
    backend = MagicMock()
    backend.transcribe.return_value = _fake_result()
    client = _make_client(backend)

    r = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("a.wav", b"FAKEWAV", "audio/wav")},
        data={"model": "whisper-tiny", "response_format": "verbose_json"},
    )
    d = r.json()
    assert d["language"] == "en"
    assert d["task"] == "transcribe"
    assert "words" not in d


def test_word_timestamps_flag_flows_to_backend():
    backend = MagicMock()
    backend.transcribe.return_value = _fake_result()
    client = _make_client(backend)

    r = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("a.wav", b"FAKEWAV", "audio/wav")},
        data=[
            ("model", "whisper-tiny"),
            ("timestamp_granularities[]", "word"),
        ],
    )
    assert r.status_code == 200
    _, kwargs = backend.transcribe.call_args
    assert kwargs["word_timestamps"] is True


def test_vad_filter_flows_to_backend():
    backend = MagicMock()
    backend.transcribe.return_value = _fake_result()
    client = _make_client(backend)

    r = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("a.wav", b"FAKEWAV", "audio/wav")},
        data={"model": "whisper-tiny", "vad_filter": "true"},
    )
    assert r.status_code == 200
    _, kwargs = backend.transcribe.call_args
    assert kwargs["vad_filter"] is True


def test_unknown_response_format_returns_400_envelope():
    backend = MagicMock()
    backend.transcribe.return_value = _fake_result()
    client = _make_client(backend)

    r = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("a.wav", b"FAKEWAV", "audio/wav")},
        data={"model": "whisper-tiny", "response_format": "xml"},
    )
    assert r.status_code == 400
    body = r.json()
    assert "error" in body
    assert "detail" not in body
    assert body["error"]["code"] == "invalid_parameter"


def test_unknown_model_returns_404_envelope():
    backend = MagicMock()
    backend.transcribe.return_value = _fake_result()
    client = _make_client(backend)

    r = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("a.wav", b"FAKEWAV", "audio/wav")},
        data={"model": "no-such-model"},
    )
    assert r.status_code == 404
    body = r.json()
    assert body["error"]["code"] == "model_not_found"


def test_empty_file_returns_400_envelope():
    backend = MagicMock()
    backend.transcribe.return_value = _fake_result()
    client = _make_client(backend)

    r = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("a.wav", b"", "audio/wav")},
        data={"model": "whisper-tiny"},
    )
    assert r.status_code == 400
    body = r.json()
    assert body["error"]["code"] == "invalid_parameter"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/modalities/audio_transcription/test_routes.py -v`
Expected: ImportError on `muse.modalities.audio_transcription.routes`.

- [ ] **Step 3: Implement routes**

Create `src/muse/modalities/audio_transcription/routes.py`:

```python
"""FastAPI routes for /v1/audio/transcriptions and /v1/audio/translations.

First muse modality with multipart/form-data uploads. Pattern: FastAPI
UploadFile + Form fields, saved to a tempfile, passed as path to the
backend. If a second multipart modality (images/edits, audio input) lands,
factor out into muse.modalities._common.uploads (see CLAUDE.md TODO).

Size cap: MUSE_ASR_MAX_MB env var (default 100). 4x OpenAI's 25 MB since
we're self-hosted.
"""
from __future__ import annotations

import logging
import os
import tempfile
from typing import Literal

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from muse.core.errors import ModelNotFoundError
from muse.core.registry import ModalityRegistry
from muse.modalities.audio_transcription import MODALITY
from muse.modalities.audio_transcription.codec import encode_transcription


logger = logging.getLogger(__name__)

VALID_FORMATS = {"json", "text", "srt", "vtt", "verbose_json"}


def _max_upload_bytes() -> int:
    mb = int(os.environ.get("MUSE_ASR_MAX_MB", "100"))
    return mb * 1024 * 1024


def build_router(registry: ModalityRegistry) -> APIRouter:
    router = APIRouter()

    async def _handle(
        *,
        task: Literal["transcribe", "translate"],
        file: UploadFile,
        model: str,
        language: str | None,
        prompt: str | None,
        response_format: str,
        temperature: float,
        timestamp_granularities: list[str],
        vad_filter: bool,
    ) -> Response:
        if response_format not in VALID_FORMATS:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "code": "invalid_parameter",
                        "message": (
                            f"response_format must be one of {sorted(VALID_FORMATS)}; "
                            f"got {response_format!r}"
                        ),
                        "type": "invalid_request_error",
                    }
                },
            )

        max_bytes = _max_upload_bytes()
        data = await file.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail={
                    "error": {
                        "code": "payload_too_large",
                        "message": f"file exceeds MUSE_ASR_MAX_MB={max_bytes // (1024*1024)}",
                        "type": "invalid_request_error",
                    }
                },
            )
        if not data:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "code": "invalid_parameter",
                        "message": "file is empty",
                        "type": "invalid_request_error",
                    }
                },
            )

        backend = registry.get(MODALITY, model)
        if backend is None:
            raise ModelNotFoundError(model, MODALITY)

        want_words = "word" in timestamp_granularities

        with tempfile.NamedTemporaryFile(
            suffix=_suffix_for_upload(file.filename), delete=True,
        ) as tmp:
            tmp.write(data)
            tmp.flush()
            try:
                result = backend.transcribe(
                    tmp.name,
                    task=task,
                    language=None if task == "translate" else language,
                    prompt=prompt,
                    temperature=temperature,
                    word_timestamps=want_words,
                    vad_filter=vad_filter,
                )
            except Exception as e:  # noqa: BLE001
                # PyAV/ffmpeg decode failures land here; we don't try to
                # distinguish them finely (message surface is good enough).
                msg = str(e).lower()
                if "decoder" in msg or "format" in msg or "ffmpeg" in msg:
                    raise HTTPException(
                        status_code=415,
                        detail={
                            "error": {
                                "code": "unsupported_media_type",
                                "message": f"audio decode failed: {e}",
                                "type": "invalid_request_error",
                            }
                        },
                    )
                raise

        body, content_type = encode_transcription(
            result, response_format, include_words=want_words,
        )
        return Response(content=body, media_type=content_type)

    @router.post("/v1/audio/transcriptions")
    async def transcriptions(
        file: UploadFile = File(...),
        model: str = Form(...),
        language: str | None = Form(None),
        prompt: str | None = Form(None),
        response_format: str = Form("json"),
        temperature: float = Form(0.0),
        timestamp_granularities: list[str] = Form(
            default_factory=list, alias="timestamp_granularities[]",
        ),
        vad_filter: bool = Form(False),
    ):
        return await _handle(
            task="transcribe",
            file=file, model=model, language=language, prompt=prompt,
            response_format=response_format, temperature=temperature,
            timestamp_granularities=timestamp_granularities,
            vad_filter=vad_filter,
        )

    @router.post("/v1/audio/translations")
    async def translations(
        file: UploadFile = File(...),
        model: str = Form(...),
        prompt: str | None = Form(None),
        response_format: str = Form("json"),
        temperature: float = Form(0.0),
        timestamp_granularities: list[str] = Form(
            default_factory=list, alias="timestamp_granularities[]",
        ),
        vad_filter: bool = Form(False),
    ):
        return await _handle(
            task="translate",
            file=file, model=model, language=None, prompt=prompt,
            response_format=response_format, temperature=temperature,
            timestamp_granularities=timestamp_granularities,
            vad_filter=vad_filter,
        )

    return router


def _suffix_for_upload(filename: str | None) -> str:
    """Preserve the upload's suffix so ffmpeg/PyAV can sniff the format."""
    if not filename or "." not in filename:
        return ""
    return "." + filename.rsplit(".", 1)[-1]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/modalities/audio_transcription/test_routes.py -v`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add src/muse/modalities/audio_transcription/routes.py \
        tests/modalities/audio_transcription/test_routes.py
git commit -m "$(cat <<'EOF'
feat(asr): FastAPI routes for transcriptions + translations

POST /v1/audio/transcriptions and /v1/audio/translations. Multipart
file upload + form fields. /translations forces task=translate and
drops language. Error envelopes for unknown model (404), bad
response_format (400), empty file (400), decoder failure (415),
size cap (413). MUSE_ASR_MAX_MB env var (default 100).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Client

**Files:**
- Create: `src/muse/modalities/audio_transcription/client.py`
- Test: `tests/modalities/audio_transcription/test_client.py`

- [ ] **Step 1: Write the failing client tests**

Create `tests/modalities/audio_transcription/test_client.py`:

```python
"""TranscriptionClient: HTTP wrapper."""
from unittest.mock import patch

import httpx
import pytest
import respx

from muse.modalities.audio_transcription import TranscriptionClient  # noqa: F401


@respx.mock
def test_client_transcribe_default_json_format():
    route = respx.post("http://localhost:8000/v1/audio/transcriptions").mock(
        return_value=httpx.Response(200, json={"text": "hello"})
    )
    from muse.modalities.audio_transcription import TranscriptionClient
    c = TranscriptionClient(base_url="http://localhost:8000")
    text = c.transcribe(audio=b"FAKEWAV", filename="a.wav", model="whisper-tiny")
    assert text == "hello"
    assert route.called


@respx.mock
def test_client_text_format_returns_raw_string():
    respx.post("http://localhost:8000/v1/audio/transcriptions").mock(
        return_value=httpx.Response(200, text="plain text transcript",
                                    headers={"content-type": "text/plain"})
    )
    from muse.modalities.audio_transcription import TranscriptionClient
    c = TranscriptionClient(base_url="http://localhost:8000")
    out = c.transcribe(
        audio=b"x", filename="a.wav", model="whisper-tiny", response_format="text",
    )
    assert out == "plain text transcript"


@respx.mock
def test_client_srt_format_returns_raw_string():
    srt = "1\n00:00:00,000 --> 00:00:01,000\nhello\n"
    respx.post("http://localhost:8000/v1/audio/transcriptions").mock(
        return_value=httpx.Response(200, text=srt,
                                    headers={"content-type": "application/x-subrip"})
    )
    from muse.modalities.audio_transcription import TranscriptionClient
    c = TranscriptionClient(base_url="http://localhost:8000")
    out = c.transcribe(
        audio=b"x", filename="a.wav", model="whisper-tiny", response_format="srt",
    )
    assert "hello" in out


@respx.mock
def test_client_verbose_json_returns_full_dict():
    body = {
        "task": "transcribe", "language": "en", "duration": 1.0,
        "text": "hello", "segments": [],
    }
    respx.post("http://localhost:8000/v1/audio/transcriptions").mock(
        return_value=httpx.Response(200, json=body)
    )
    from muse.modalities.audio_transcription import TranscriptionClient
    c = TranscriptionClient(base_url="http://localhost:8000")
    out = c.transcribe(
        audio=b"x", filename="a.wav", model="whisper-tiny",
        response_format="verbose_json",
    )
    assert out["language"] == "en"


@respx.mock
def test_client_translate_hits_translations_endpoint():
    route = respx.post("http://localhost:8000/v1/audio/translations").mock(
        return_value=httpx.Response(200, json={"text": "hello"})
    )
    from muse.modalities.audio_transcription import TranscriptionClient
    c = TranscriptionClient(base_url="http://localhost:8000")
    c.translate(audio=b"x", filename="a.wav", model="whisper-tiny")
    assert route.called


def test_client_env_var_base_url(monkeypatch):
    monkeypatch.setenv("MUSE_SERVER", "http://custom:9999")
    from muse.modalities.audio_transcription import TranscriptionClient
    c = TranscriptionClient()
    assert c._base_url == "http://custom:9999"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/modalities/audio_transcription/test_client.py -v`
Expected: ImportError on `TranscriptionClient`.

- [ ] **Step 3: Implement client**

Create `src/muse/modalities/audio_transcription/client.py`:

```python
"""HTTP client for /v1/audio/transcriptions and /v1/audio/translations.

Mirrors the ergonomics of the other muse clients: base_url from
MUSE_SERVER env if not passed explicitly; two methods (transcribe,
translate); response type follows response_format (str for
json/text/srt/vtt; dict for verbose_json).
"""
from __future__ import annotations

import os
from typing import Any

import httpx


class TranscriptionClient:
    """Minimal HTTP client for the audio/transcription modality."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self._base_url = (
            base_url
            or os.environ.get("MUSE_SERVER")
            or "http://localhost:8000"
        )
        self._timeout = timeout

    def transcribe(
        self,
        *,
        audio: bytes,
        filename: str,
        model: str,
        language: str | None = None,
        prompt: str | None = None,
        response_format: str = "json",
        temperature: float = 0.0,
        word_timestamps: bool = False,
        vad_filter: bool = False,
    ) -> str | dict[str, Any]:
        return self._post(
            "/v1/audio/transcriptions",
            audio=audio, filename=filename, model=model,
            language=language, prompt=prompt,
            response_format=response_format, temperature=temperature,
            word_timestamps=word_timestamps, vad_filter=vad_filter,
        )

    def translate(
        self,
        *,
        audio: bytes,
        filename: str,
        model: str,
        prompt: str | None = None,
        response_format: str = "json",
        temperature: float = 0.0,
        word_timestamps: bool = False,
        vad_filter: bool = False,
    ) -> str | dict[str, Any]:
        return self._post(
            "/v1/audio/translations",
            audio=audio, filename=filename, model=model,
            language=None, prompt=prompt,
            response_format=response_format, temperature=temperature,
            word_timestamps=word_timestamps, vad_filter=vad_filter,
        )

    def _post(
        self,
        path: str,
        *,
        audio: bytes,
        filename: str,
        model: str,
        language: str | None,
        prompt: str | None,
        response_format: str,
        temperature: float,
        word_timestamps: bool,
        vad_filter: bool,
    ):
        files = {"file": (filename, audio)}
        data: list[tuple[str, str]] = [
            ("model", model),
            ("response_format", response_format),
            ("temperature", str(temperature)),
            ("vad_filter", "true" if vad_filter else "false"),
        ]
        if language is not None:
            data.append(("language", language))
        if prompt is not None:
            data.append(("prompt", prompt))
        if word_timestamps:
            data.append(("timestamp_granularities[]", "word"))

        r = httpx.post(
            f"{self._base_url}{path}",
            files=files, data=data, timeout=self._timeout,
        )
        r.raise_for_status()

        ct = r.headers.get("content-type", "")
        if "json" in ct:
            j = r.json()
            # json format returns {"text": "..."}; verbose_json returns a full dict.
            if isinstance(j, dict) and set(j.keys()) == {"text"}:
                return j["text"]
            return j
        return r.text
```

Also update `src/muse/modalities/audio_transcription/__init__.py` to re-export `TranscriptionClient`:

```python
from muse.modalities.audio_transcription.client import TranscriptionClient
```

and add `"TranscriptionClient"` to `__all__`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/modalities/audio_transcription/test_client.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/muse/modalities/audio_transcription/client.py \
        src/muse/modalities/audio_transcription/__init__.py \
        tests/modalities/audio_transcription/test_client.py
git commit -m "$(cat <<'EOF'
feat(asr): TranscriptionClient for /v1/audio/transcriptions endpoints

Parallel API to SpeechClient / EmbeddingsClient / GenerationsClient /
ChatClient. `transcribe` and `translate` methods; response type
follows response_format. Honors MUSE_SERVER env var.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: FasterWhisperModel runtime

**Files:**
- Create: `src/muse/modalities/audio_transcription/runtimes/faster_whisper.py`
- Test: `tests/modalities/audio_transcription/runtimes/__init__.py` (empty) + `tests/modalities/audio_transcription/runtimes/test_faster_whisper.py`

- [ ] **Step 1: Write the failing runtime tests**

Create `tests/modalities/audio_transcription/runtimes/__init__.py` (empty file). Then `tests/modalities/audio_transcription/runtimes/test_faster_whisper.py`:

```python
"""FasterWhisperModel runtime: mocked-dep tests."""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _fake_info(language="en", duration=1.0):
    return SimpleNamespace(language=language, duration=duration)


def _fake_segment(id, start, end, text, words=None):
    return SimpleNamespace(id=id, start=start, end=end, text=text, words=words or [])


@pytest.fixture(autouse=True)
def _reset_module_sentinels():
    """Reset the module-top torch/WhisperModel sentinels between tests.

    Deferred-import discipline means _ensure_deps() only runs once per
    process. Each test that mocks should fully replace both sentinels.
    """
    import muse.modalities.audio_transcription.runtimes.faster_whisper as mod
    orig = (mod.torch, mod.WhisperModel)
    yield
    mod.torch, mod.WhisperModel = orig


def test_transcribe_assembles_result_from_segments():
    import muse.modalities.audio_transcription.runtimes.faster_whisper as mod
    # Stub module sentinels
    mod.torch = MagicMock(cuda=MagicMock(is_available=MagicMock(return_value=False)))
    mod.torch.backends = MagicMock(mps=None)
    fake_whisper = MagicMock()
    fake_whisper.transcribe.return_value = (
        iter([
            _fake_segment(0, 0.0, 1.0, "hello"),
            _fake_segment(1, 1.0, 2.0, "world"),
        ]),
        _fake_info("en", 2.0),
    )
    mod.WhisperModel = MagicMock(return_value=fake_whisper)

    m = mod.FasterWhisperModel(
        model_id="whisper-tiny", hf_repo="Systran/faster-whisper-tiny",
        local_dir="/fake/weights", device="cpu",
    )
    r = m.transcribe("/fake/audio.wav", task="transcribe")
    assert r.text == "hello world"
    assert r.language == "en"
    assert r.duration == 2.0
    assert r.task == "transcribe"
    assert len(r.segments) == 2
    assert r.segments[0].words is None  # word_timestamps was False


def test_word_timestamps_populates_segment_words():
    import muse.modalities.audio_transcription.runtimes.faster_whisper as mod
    mod.torch = MagicMock(cuda=MagicMock(is_available=MagicMock(return_value=False)))
    mod.torch.backends = MagicMock(mps=None)
    fake_whisper = MagicMock()
    fake_word = SimpleNamespace(word="hello", start=0.0, end=0.5)
    fake_whisper.transcribe.return_value = (
        iter([_fake_segment(0, 0.0, 1.0, "hello", words=[fake_word])]),
        _fake_info(),
    )
    mod.WhisperModel = MagicMock(return_value=fake_whisper)

    m = mod.FasterWhisperModel(
        model_id="whisper-tiny", hf_repo="x", local_dir="/fake", device="cpu",
    )
    r = m.transcribe("/fake/a.wav", word_timestamps=True)
    assert r.segments[0].words is not None
    assert r.segments[0].words[0].word == "hello"
    # Forwarded through to faster_whisper
    _, kw = fake_whisper.transcribe.call_args
    assert kw["word_timestamps"] is True


def test_task_translate_is_forwarded():
    import muse.modalities.audio_transcription.runtimes.faster_whisper as mod
    mod.torch = MagicMock(cuda=MagicMock(is_available=MagicMock(return_value=False)))
    mod.torch.backends = MagicMock(mps=None)
    fake_whisper = MagicMock()
    fake_whisper.transcribe.return_value = (iter([]), _fake_info("en", 0.0))
    mod.WhisperModel = MagicMock(return_value=fake_whisper)

    m = mod.FasterWhisperModel(
        model_id="whisper-tiny", hf_repo="x", local_dir="/fake", device="cpu",
    )
    m.transcribe("/fake/a.wav", task="translate")

    _, kw = fake_whisper.transcribe.call_args
    assert kw["task"] == "translate"


def test_device_auto_selects_cuda_when_available():
    import muse.modalities.audio_transcription.runtimes.faster_whisper as mod
    mod.torch = MagicMock(cuda=MagicMock(is_available=MagicMock(return_value=True)))
    mod.torch.backends = MagicMock(mps=None)
    captured_kwargs = {}

    def constructor(path, **kw):
        captured_kwargs.update(kw)
        return MagicMock()

    mod.WhisperModel = MagicMock(side_effect=constructor)

    mod.FasterWhisperModel(
        model_id="whisper-tiny", hf_repo="x", local_dir="/fake", device="auto",
    )
    assert captured_kwargs["device"] == "cuda"
    assert captured_kwargs["compute_type"] == "float16"


def test_device_cpu_uses_int8_compute_type():
    import muse.modalities.audio_transcription.runtimes.faster_whisper as mod
    mod.torch = MagicMock(cuda=MagicMock(is_available=MagicMock(return_value=False)))
    mod.torch.backends = MagicMock(mps=None)
    captured_kwargs = {}

    def constructor(path, **kw):
        captured_kwargs.update(kw)
        return MagicMock()

    mod.WhisperModel = MagicMock(side_effect=constructor)

    mod.FasterWhisperModel(
        model_id="whisper-tiny", hf_repo="x", local_dir="/fake", device="cpu",
    )
    assert captured_kwargs["device"] == "cpu"
    assert captured_kwargs["compute_type"] == "int8"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/modalities/audio_transcription/runtimes/ -v`
Expected: ImportError on the runtime module.

- [ ] **Step 3: Implement the runtime**

Create `src/muse/modalities/audio_transcription/runtimes/faster_whisper.py`:

```python
"""FasterWhisperModel: generic runtime over any Systran CT2 Whisper repo.

One class serves any faster-whisper-format repo on HF. Pulled via the
HF resolver (muse.core.resolvers_hf): `muse pull
hf://Systran/faster-whisper-base` synthesizes a manifest pointing at
this class.

Deferred imports follow the muse pattern: torch and WhisperModel stay
as module-top sentinels (None) until _ensure_deps() lazy-imports them.
Tests patch the sentinels directly; _ensure_deps short-circuits on
non-None, so pre-populated mocks survive.
"""
from __future__ import annotations

import logging
from typing import Any, Literal

from muse.modalities.audio_transcription import (
    Segment,
    TranscriptionResult,
    Word,
)


logger = logging.getLogger(__name__)

torch: Any = None
WhisperModel: Any = None


def _ensure_deps() -> None:
    global torch, WhisperModel
    if torch is None:
        try:
            import torch as _t
            torch = _t
        except Exception as e:  # noqa: BLE001
            logger.debug("FasterWhisperModel torch unavailable: %s", e)
    if WhisperModel is None:
        try:
            from faster_whisper import WhisperModel as _w
            WhisperModel = _w
        except Exception as e:  # noqa: BLE001
            logger.debug("FasterWhisperModel faster-whisper unavailable: %s", e)


class FasterWhisperModel:
    """Generic faster-whisper runtime."""

    model_id: str

    def __init__(
        self,
        *,
        model_id: str,
        hf_repo: str,
        local_dir: str | None = None,
        device: str = "auto",
        compute_type: str | None = None,
        beam_size: int = 5,
        **_: Any,
    ) -> None:
        _ensure_deps()
        if WhisperModel is None:
            raise RuntimeError(
                "faster-whisper is not installed; run `muse pull` or "
                "install `faster-whisper` into this venv"
            )
        self.model_id = model_id
        self._device = _select_device(device)
        if compute_type is None:
            compute_type = "float16" if self._device == "cuda" else "int8"
        src = local_dir or hf_repo
        logger.info(
            "loading faster-whisper from %s (device=%s, compute_type=%s)",
            src, self._device, compute_type,
        )
        self._model = WhisperModel(
            src, device=self._device, compute_type=compute_type,
        )
        self._beam_size = beam_size

    def transcribe(
        self,
        audio_path: str,
        *,
        task: Literal["transcribe", "translate"] = "transcribe",
        language: str | None = None,
        prompt: str | None = None,
        temperature: float = 0.0,
        word_timestamps: bool = False,
        vad_filter: bool = False,
        **_: Any,
    ) -> TranscriptionResult:
        segments_iter, info = self._model.transcribe(
            audio_path,
            task=task,
            language=language,
            initial_prompt=prompt,
            temperature=temperature,
            word_timestamps=word_timestamps,
            vad_filter=vad_filter,
            beam_size=self._beam_size,
        )
        segments: list[Segment] = []
        for i, s in enumerate(segments_iter):
            words: list[Word] | None = None
            raw_words = getattr(s, "words", None) or []
            if raw_words:
                words = [Word(word=w.word, start=w.start, end=w.end) for w in raw_words]
            segments.append(Segment(
                id=i, start=s.start, end=s.end, text=s.text.strip(),
                words=words,
            ))
        return TranscriptionResult(
            text=" ".join(s.text for s in segments).strip(),
            language=info.language,
            duration=info.duration,
            segments=segments,
            task=task,
        )


def _select_device(device: str) -> str:
    if device != "auto":
        return device
    if torch is None:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/modalities/audio_transcription/runtimes/ -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/muse/modalities/audio_transcription/runtimes/faster_whisper.py \
        tests/modalities/audio_transcription/runtimes/__init__.py \
        tests/modalities/audio_transcription/runtimes/test_faster_whisper.py
git commit -m "$(cat <<'EOF'
feat(asr): FasterWhisperModel generic runtime over Systran CT2 repos

One runtime class serves any faster-whisper-format HF repo. Deferred
torch + faster_whisper imports (module-top sentinels). device=auto
picks cuda when torch.cuda.is_available() else cpu. compute_type
defaults: float16 on cuda, int8 on cpu. transcribe() flows task,
language, prompt, temperature, word_timestamps, vad_filter through.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: HF resolver extension (CT2 sniff + resolve + search)

**Files:**
- Modify: `src/muse/core/resolvers_hf.py`
- Test: `tests/core/test_resolvers_hf.py`

- [ ] **Step 1: Write the failing resolver tests**

Append to `tests/core/test_resolvers_hf.py`:

```python
# --- faster-whisper branch ---

def _fake_ct2_whisper_siblings():
    return [
        SimpleNamespace(rfilename="model.bin"),
        SimpleNamespace(rfilename="config.json"),
        SimpleNamespace(rfilename="vocabulary.txt"),
        SimpleNamespace(rfilename="README.md"),
    ]


def test_sniff_detects_faster_whisper_shape():
    from muse.core.resolvers_hf import _sniff_repo_shape
    info = SimpleNamespace(
        siblings=_fake_ct2_whisper_siblings(),
        tags=["automatic-speech-recognition", "whisper"],
    )
    assert _sniff_repo_shape(info) == "faster-whisper"


def test_sniff_rejects_ct2_shape_without_asr_tag():
    """CT2 alone is not enough: could be an NMT repo."""
    from muse.core.resolvers_hf import _sniff_repo_shape
    info = SimpleNamespace(
        siblings=_fake_ct2_whisper_siblings(),
        tags=["machine-translation"],
    )
    assert _sniff_repo_shape(info) == "unknown"


def test_sniff_rejects_without_model_bin():
    from muse.core.resolvers_hf import _sniff_repo_shape
    info = SimpleNamespace(
        siblings=[
            SimpleNamespace(rfilename="config.json"),
            SimpleNamespace(rfilename="tokenizer.json"),
        ],
        tags=["automatic-speech-recognition"],
    )
    assert _sniff_repo_shape(info) == "unknown"


def test_resolve_faster_whisper_synthesizes_manifest():
    from muse.core.resolvers_hf import HFResolver
    resolver = HFResolver()
    info = SimpleNamespace(
        siblings=_fake_ct2_whisper_siblings(),
        tags=["automatic-speech-recognition"],
        card_data=SimpleNamespace(license="mit"),
    )
    with patch.object(resolver._api, "repo_info", return_value=info):
        resolved = resolver.resolve("hf://Systran/faster-whisper-tiny")
    assert resolved.manifest["modality"] == "audio/transcription"
    assert resolved.manifest["hf_repo"] == "Systran/faster-whisper-tiny"
    assert resolved.manifest["model_id"] == "faster-whisper-tiny"
    assert "faster-whisper>=1.0.0" in resolved.manifest["pip_extras"]
    assert "ffmpeg" in resolved.manifest["system_packages"]
    assert resolved.backend_path == (
        "muse.modalities.audio_transcription.runtimes.faster_whisper"
        ":FasterWhisperModel"
    )


def test_search_faster_whisper_yields_results():
    from muse.core.resolvers_hf import HFResolver
    resolver = HFResolver()
    fake_repos = [
        SimpleNamespace(id="Systran/faster-whisper-tiny", downloads=12345, siblings=[]),
        SimpleNamespace(id="Systran/faster-whisper-base", downloads=8000, siblings=[]),
    ]
    with patch.object(resolver._api, "list_models", return_value=fake_repos):
        results = list(resolver.search("whisper", modality="audio/transcription"))
    assert len(results) == 2
    assert all(r.modality == "audio/transcription" for r in results)
    assert all(r.uri.startswith("hf://Systran/faster-whisper-") for r in results)
```

(Ensure the file's imports include `from types import SimpleNamespace` and `from unittest.mock import patch`; add if missing.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/core/test_resolvers_hf.py -v -k "faster_whisper or ct2"`
Expected: AssertionError on the sniff tests; resolve test errors out because the branch doesn't exist.

- [ ] **Step 3: Extend `_sniff_repo_shape`**

In `src/muse/core/resolvers_hf.py`, add this helper at module level (next to `_extract_variant`):

```python
def _looks_like_faster_whisper(siblings: list[str], tags: list[str]) -> bool:
    """CT2 faster-whisper repos have model.bin + config.json +
    (vocabulary.txt or tokenizer.json), plus the ASR tag."""
    names = {Path(f).name for f in siblings}
    has_ct2_shape = (
        "model.bin" in names
        and "config.json" in names
        and ("vocabulary.txt" in names or "tokenizer.json" in names)
    )
    has_asr_tag = "automatic-speech-recognition" in tags
    return has_ct2_shape and has_asr_tag
```

Update `_sniff_repo_shape` to add the third branch:

```python
def _sniff_repo_shape(info) -> str:
    """Return one of: 'gguf' | 'sentence-transformers' | 'faster-whisper' | 'unknown'."""
    siblings = [s.rfilename for s in getattr(info, "siblings", [])]
    tags = getattr(info, "tags", None) or []
    if any(f.endswith(".gguf") for f in siblings):
        return "gguf"
    if "sentence-transformers" in tags:
        return "sentence-transformers"
    if any(Path(f).name == "sentence_transformers_config.json" for f in siblings):
        return "sentence-transformers"
    if _looks_like_faster_whisper(siblings, tags):
        return "faster-whisper"
    return "unknown"
```

- [ ] **Step 4: Add resolve + search branches**

At module top, add a new constant:

```python
FASTER_WHISPER_RUNTIME_PATH = (
    "muse.modalities.audio_transcription.runtimes.faster_whisper:FasterWhisperModel"
)
FASTER_WHISPER_PIP_EXTRAS = ("faster-whisper>=1.0.0",)
FASTER_WHISPER_SYSTEM_PACKAGES = ("ffmpeg",)
```

In `HFResolver.resolve`, add the third dispatch branch after `sentence-transformers`:

```python
if shape == "faster-whisper":
    return self._resolve_faster_whisper(repo_id, info)
```

Add the method:

```python
def _resolve_faster_whisper(self, repo_id: str, info) -> ResolvedModel:
    manifest = {
        "model_id": repo_id.split("/", 1)[-1].lower(),
        "modality": "audio/transcription",
        "hf_repo": repo_id,
        "description": f"Faster-Whisper: {repo_id}",
        "license": _repo_license(info),
        "pip_extras": list(FASTER_WHISPER_PIP_EXTRAS),
        "system_packages": list(FASTER_WHISPER_SYSTEM_PACKAGES),
        "capabilities": {},
    }

    def _download(cache_root: Path) -> Path:
        return Path(snapshot_download(
            repo_id=repo_id,
            cache_dir=str(cache_root) if cache_root else None,
        ))

    return ResolvedModel(
        manifest=manifest,
        backend_path=FASTER_WHISPER_RUNTIME_PATH,
        download=_download,
    )
```

Extend `HFResolver.search` to route `audio/transcription`:

```python
elif modality == "audio/transcription":
    yield from self._search_faster_whisper(query, sort=sort, limit=limit)
```

And add the method:

```python
def _search_faster_whisper(self, query: str, *, sort: str, limit: int):
    repos = self._api.list_models(
        search=query, filter="automatic-speech-recognition",
        sort=sort, limit=limit,
    )
    for repo in repos:
        yield SearchResult(
            uri=f"hf://{repo.id}",
            model_id=repo.id.split("/", 1)[-1].lower(),
            modality="audio/transcription",
            size_gb=None,
            downloads=getattr(repo, "downloads", None),
            license=None,
            description=repo.id,
        )
```

Also update the `search(modality=None)` case to optionally yield ASR results; do NOT add it to the "default" unmodalled search (which already covers gguf + sentence-transformers); require `--modality audio/transcription`:

```python
# In search():
else:
    raise ResolverError(
        f"HFResolver.search does not support modality {modality!r}; "
        f"supported: chat/completion, embedding/text, audio/transcription"
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/core/test_resolvers_hf.py -v`
Expected: all prior tests still green, new CT2 tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/muse/core/resolvers_hf.py tests/core/test_resolvers_hf.py
git commit -m "$(cat <<'EOF'
feat(resolver): HF resolver sniffs Systran CT2 faster-whisper repos

Adds a third _sniff_repo_shape branch after gguf + sentence-transformers:
faster-whisper. Requires model.bin + config.json + (vocabulary.txt or
tokenizer.json) siblings AND the automatic-speech-recognition tag.
resolve() synthesizes an audio/transcription manifest pointing at the
FasterWhisperModel runtime; search() routes --modality audio/transcription
to list_models(filter=automatic-speech-recognition).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Curated entries

**Files:**
- Modify: `src/muse/curated.yaml`
- Test: `tests/core/test_curated.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/core/test_curated.py`:

```python
def test_load_curated_includes_whisper_entries():
    """ASR curated shortcuts: whisper-tiny, whisper-base, whisper-large-v3."""
    entries = load_curated()
    asr_ids = {e.id for e in entries if e.modality == "audio/transcription"}
    assert {"whisper-tiny", "whisper-base", "whisper-large-v3"}.issubset(asr_ids)
    # Each points at a Systran HF URI
    for e in entries:
        if e.modality == "audio/transcription":
            assert e.uri is not None
            assert e.uri.startswith("hf://Systran/faster-whisper-")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/core/test_curated.py -v -k whisper`
Expected: AssertionError (no whisper entries yet).

- [ ] **Step 3: Add the three curated rows**

Append to `src/muse/curated.yaml`:

```yaml
# ---------- audio/transcription (ASR via faster-whisper) ----------

- id: whisper-tiny
  uri: hf://Systran/faster-whisper-tiny
  modality: audio/transcription
  size_gb: 0.08
  description: "Whisper tiny (39M, ~75MB): CPU-friendly smoke test, multilingual"

- id: whisper-base
  uri: hf://Systran/faster-whisper-base
  modality: audio/transcription
  size_gb: 0.15
  description: "Whisper base (74M, ~142MB): balanced CPU default"

- id: whisper-large-v3
  uri: hf://Systran/faster-whisper-large-v3
  modality: audio/transcription
  size_gb: 3.0
  description: "Whisper large-v3 (1550M, ~2.9GB): SotA quality, 8GB+ GPU"
```

- [ ] **Step 4: Run tests to verify**

Run: `pytest tests/core/test_curated.py -v`
Expected: all curated tests green, including the new whisper check.

- [ ] **Step 5: Commit**

```bash
git add src/muse/curated.yaml tests/core/test_curated.py
git commit -m "$(cat <<'EOF'
feat(curated): three whisper shortcuts (tiny/base/large-v3)

Newbie-friendly aliases for the most-used Systran faster-whisper
builds. Users can also pull any other CT2 repo via
`muse pull hf://...` or find via `muse search whisper`.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: E2E slow test (multipart flow end-to-end)

**Files:**
- Create: `tests/cli_impl/test_e2e_asr.py`

- [ ] **Step 1: Write the test**

Create `tests/cli_impl/test_e2e_asr.py`:

```python
"""End-to-end: multipart upload flows through FastAPI + codec correctly.

Uses a fake TranscriptionModel backend; no real weights. Skipped if
ffmpeg isn't on PATH (not that FastAPI/TestClient need it here, but
this test documents the full multipart path muse depends on).
"""
import io
import struct
import wave

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from muse.core.registry import ModalityRegistry
from muse.modalities.audio_transcription import (
    MODALITY,
    Segment,
    TranscriptionResult,
    build_router,
)


pytestmark = pytest.mark.slow


def _make_sine_wav(seconds: float = 1.0, rate: int = 16000) -> bytes:
    """Synthesize a silent/sine WAV in memory for upload (no decode needed)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        # Silence is fine; we're mocking the backend
        w.writeframes(b"\x00\x00" * int(rate * seconds))
    return buf.getvalue()


class _FakeWhisper:
    def __init__(self):
        self.called_with = None

    def transcribe(self, audio_path, **kwargs):
        self.called_with = (audio_path, kwargs)
        return TranscriptionResult(
            text="mocked transcript",
            language="en", duration=1.0, task=kwargs.get("task", "transcribe"),
            segments=[Segment(id=0, start=0.0, end=1.0, text="mocked transcript",
                              words=None)],
        )


@pytest.mark.timeout(10)
def test_multipart_flow_srt_end_to_end():
    fake = _FakeWhisper()
    reg = ModalityRegistry()
    reg.register(MODALITY, fake, manifest={"model_id": "whisper-tiny"})
    app = FastAPI()
    app.include_router(build_router(reg))

    wav = _make_sine_wav(1.0)
    client = TestClient(app)
    r = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("sample.wav", wav, "audio/wav")},
        data={"model": "whisper-tiny", "response_format": "srt"},
    )
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("application/x-subrip")
    assert "mocked transcript" in r.text
    # Backend saw a real file path
    audio_path, kwargs = fake.called_with
    assert audio_path.endswith(".wav")
    assert kwargs["task"] == "transcribe"
```

- [ ] **Step 2: Run the test**

Run: `pytest tests/cli_impl/test_e2e_asr.py -v`
Expected: 1 passed.

- [ ] **Step 3: Commit**

```bash
git add tests/cli_impl/test_e2e_asr.py
git commit -m "$(cat <<'EOF'
test(asr): e2e multipart flow through FastAPI + codec

One slow-marked test: synthesizes a WAV in-memory, uploads via
TestClient, asserts SRT-format response. Covers the full upload ->
tempfile -> backend -> codec path that no unit test individually
exercises.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: Integration test fixture + opt-in test

**Files:**
- Create: `tests/fixtures/asr_sample.wav` (short audio clip)
- Modify: `tests/integration/conftest.py`
- Create: `tests/integration/test_remote_asr.py`

- [ ] **Step 1: Bundle the audio fixture**

Generate a 2-second clip of TTS-readable speech via muse's own Kokoro (if you have it) or any other TTS. The clip must say something recognizable in English like "the quick brown fox". Save to `tests/fixtures/asr_sample.wav` (~30-100KB).

If you don't want to generate live, use a known public-domain clip or create programmatically with a `say`/`espeak`/`flite` command:

```bash
mkdir -p tests/fixtures
# Example (requires flite):
flite -t "the quick brown fox jumps over the lazy dog" -o tests/fixtures/asr_sample.wav
```

Size-check: `ls -lh tests/fixtures/asr_sample.wav` should show under 1MB.

- [ ] **Step 2: Add the fixture to conftest.py**

Append to `tests/integration/conftest.py`:

```python
# Whisper model fixture, parameterized by env var
whisper_model = require_model_fixture(
    default_id="whisper-tiny",
    env_var="MUSE_WHISPER_MODEL_ID",
)
```

(If `require_model_fixture` takes a single positional arg today, adapt to the existing signature; the pattern mirrors the existing `qwen3_embedding` and `chat_model` fixtures.)

- [ ] **Step 3: Write the integration test**

Create `tests/integration/test_remote_asr.py`:

```python
"""End-to-end ASR against a running muse server. Opt-in.

Requires: MUSE_REMOTE_SERVER set AND the target server has the
whisper_model loaded (default whisper-tiny, override via
MUSE_WHISPER_MODEL_ID).
"""
from __future__ import annotations

from pathlib import Path

import pytest


pytestmark = pytest.mark.slow


FIXTURE = Path(__file__).parent.parent / "fixtures" / "asr_sample.wav"


def test_protocol_transcribes_sample_to_text(openai_client, whisper_model):
    """Upload the bundled sample and assert the transcript is non-empty."""
    if not FIXTURE.exists():
        pytest.skip(f"missing fixture: {FIXTURE}")

    with open(FIXTURE, "rb") as f:
        r = openai_client.audio.transcriptions.create(
            model=whisper_model, file=f,
        )
    assert r.text, f"empty transcription from {whisper_model}"


def test_protocol_verbose_json_returns_segments(openai_client, whisper_model):
    if not FIXTURE.exists():
        pytest.skip(f"missing fixture: {FIXTURE}")

    with open(FIXTURE, "rb") as f:
        r = openai_client.audio.transcriptions.create(
            model=whisper_model, file=f, response_format="verbose_json",
        )
    # OpenAI SDK returns a Verbose dict-like object
    assert hasattr(r, "language")
    assert hasattr(r, "segments")
    assert hasattr(r, "duration")
    assert r.duration > 0
```

- [ ] **Step 4: Commit**

```bash
git add tests/fixtures/asr_sample.wav \
        tests/integration/conftest.py \
        tests/integration/test_remote_asr.py
git commit -m "$(cat <<'EOF'
test(integration): opt-in ASR test against a live muse server

tests/integration/test_remote_asr.py posts a bundled WAV fixture to
the server and asserts a non-empty transcript (json format) +
segment-shaped verbose_json. Auto-skipped without MUSE_REMOTE_SERVER
or when the chosen model isn't loaded. whisper_model fixture defaults
to whisper-tiny; override via MUSE_WHISPER_MODEL_ID.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: Docs (CLAUDE.md + README.md)

**Files:**
- Modify: `CLAUDE.md`
- Modify: `README.md`

- [ ] **Step 1: Update CLAUDE.md modality list**

In `CLAUDE.md`, update the "Project overview" list to include the new modality:

```
- **audio/transcription**: speech-to-text via `/v1/audio/transcriptions` and `/v1/audio/translations` (Systran faster-whisper family; any CT2 Whisper on HF)
```

And add one new bullet at the end of the "Modality conventions" section:

```
- `audio_transcription/` is muse's first modality with multipart/form-data uploads (OpenAI Whisper wire shape). `routes.py` handles UploadFile + Form fields inline. If a second multipart modality lands (images/edits, audio-conditioned audio/generation), factor out to `muse.modalities._common.uploads`.
```

- [ ] **Step 2: Update README.md**

In `README.md`, update the modality summary line and endpoints block to include `/v1/audio/transcriptions` and `/v1/audio/translations`. Match the style of the existing bullets.

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md README.md
git commit -m "$(cat <<'EOF'
docs(asr): CLAUDE.md and README.md note the new modality

Modality list gains audio/transcription; CLAUDE.md adds a note about
the first multipart-upload modality + refactor trigger for
modalities/_common/uploads.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 11: Version bump + tag + release

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Bump version**

In `pyproject.toml`, change `version = "0.12.1"` to `version = "0.13.0"`.

- [ ] **Step 2: Full suite**

Run: `pytest -m "not slow" -q 2>&1 | tail -5`
Expected: all passing. Record the new total (should be +20 to +30 over the v0.12.1 baseline).

Then: `pytest -q 2>&1 | tail -5` (includes slow) to run the new e2e ASR test.
Expected: all passing including the 1 new slow test.

- [ ] **Step 3: Commit the bump**

```bash
git add pyproject.toml
git commit -m "$(cat <<'EOF'
chore(release): v0.13.0

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 4: Tag**

```bash
git tag -a v0.13.0 -m "$(cat <<'EOF'
v0.13.0: audio/transcription modality (Whisper family)

New modality: /v1/audio/transcriptions and /v1/audio/translations,
OpenAI-wire-compat. Generic FasterWhisperModel runtime over any
Systran CT2 faster-whisper repo; HF resolver learns to sniff CT2
shape. Three curated aliases: whisper-tiny, whisper-base,
whisper-large-v3.

Muse's first modality with multipart/form-data file uploads. Pattern
lives inline in routes.py; will factor out to modalities/_common/
uploads when a second multipart modality (images/edits,
audio-conditioned generation) lands.

Pull with:

  muse pull whisper-tiny          # curated
  muse pull hf://Systran/faster-whisper-large-v3   # URI
  muse search whisper --modality audio/transcription

Transcribe via OpenAI SDK:

  from openai import OpenAI
  c = OpenAI(base_url="http://localhost:8000/v1", api_key="x")
  with open("audio.wav", "rb") as f:
      r = c.audio.transcriptions.create(model="whisper-tiny", file=f)
  print(r.text)
EOF
)"
```

- [ ] **Step 5: Push**

```bash
git push origin main
git push origin v0.13.0
```

---

## Success criteria

- `muse pull whisper-tiny` creates a venv with faster-whisper and downloads Systran/faster-whisper-tiny
- `muse serve` loads the whisper worker alongside existing models
- `curl -F file=@audio.wav -F model=whisper-tiny http://localhost:8000/v1/audio/transcriptions` returns `{"text": "..."}`
- OpenAI Python SDK `client.audio.transcriptions.create(model="whisper-tiny", file=open("a.wav","rb"))` works
- All 5 response formats return the expected content-type + body shape
- `muse search whisper --modality audio/transcription` lists CT2 HF repos
- Full fast-lane test suite green; full suite including slow green
- Tag `v0.13.0` on origin

## Out of scope (confirmed)

- Streaming
- Distil-Whisper or non-CT2 Whisper variants in curated
- Diarization
- Live microphone capture
- pywhispercpp / GGML Whisper path
