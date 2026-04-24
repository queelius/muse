"""Route tests for /v1/audio/transcriptions and /v1/audio/translations."""
import io
import json
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from muse.core.registry import ModalityRegistry
from muse.core.server import create_app
from muse.modalities.audio_transcription import (
    MODALITY,
    Segment,
    TranscriptionResult,
    build_router,
)


def _make_client(backend) -> TestClient:
    backend.model_id = "whisper-tiny"
    reg = ModalityRegistry()
    reg.register(MODALITY, backend, manifest={"model_id": "whisper-tiny"})
    app = create_app(registry=reg, routers={MODALITY: build_router(reg)})
    return TestClient(app, raise_server_exceptions=False)


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
        files=[
            ("file", ("a.wav", b"FAKEWAV", "audio/wav")),
            ("model", (None, "whisper-tiny", "text/plain")),
            ("timestamp_granularities[]", (None, "word", "text/plain")),
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


def test_payload_too_large_returns_413_envelope(monkeypatch):
    """File exceeding MUSE_ASR_MAX_MB returns the 413 envelope.

    Set the cap to 0 MB so any non-empty file overflows without
    needing to actually read 100 MB in the test.
    """
    monkeypatch.setenv("MUSE_ASR_MAX_MB", "0")
    backend = MagicMock()
    backend.transcribe.return_value = _fake_result()
    client = _make_client(backend)

    r = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("a.wav", b"anything", "audio/wav")},
        data={"model": "whisper-tiny"},
    )
    assert r.status_code == 413
    body = r.json()
    assert "error" in body
    assert "detail" not in body
    assert body["error"]["code"] == "payload_too_large"


def test_backend_decoder_error_returns_415_envelope():
    """A backend exception matching the decoder-error pattern returns 415.

    The substring gate in routes.py must catch PyAV/ffmpeg decode
    failures but re-raise unrelated RuntimeErrors cleanly.
    """
    backend = MagicMock()
    backend.model_id = "whisper-tiny"
    backend.transcribe.side_effect = RuntimeError(
        "PyAV InvalidDataError: invalid data found when processing input"
    )
    client = _make_client(backend)

    r = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("a.wav", b"NOT_REAL_AUDIO", "audio/wav")},
        data={"model": "whisper-tiny"},
    )
    assert r.status_code == 415
    body = r.json()
    assert body["error"]["code"] == "unsupported_media_type"


def test_backend_unrelated_error_re_raised_not_as_415():
    """A non-decoder RuntimeError must NOT be silently wrapped as 415."""
    backend = MagicMock()
    backend.model_id = "whisper-tiny"
    backend.transcribe.side_effect = RuntimeError("GPU out of memory")
    client = _make_client(backend)

    r = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("a.wav", b"FAKEWAV", "audio/wav")},
        data={"model": "whisper-tiny"},
    )
    # Expect a 500 (unhandled) or something NOT 415, because "GPU out of
    # memory" is not a decode failure.
    assert r.status_code != 415, (
        f"non-decoder RuntimeError must not be masked as 415; "
        f"body={r.text}"
    )
