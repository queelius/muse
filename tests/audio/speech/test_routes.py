"""Tests for /v1/audio/speech FastAPI router."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from muse.audio.speech.protocol import AudioChunk, AudioResult
from muse.audio.speech.routes import build_router
from muse.core.registry import ModalityRegistry
from muse.core.server import create_app


class FakeTTS:
    model_id = "fake-tts"
    sample_rate = 16000
    voices = ["default", "alt"]

    def synthesize(self, text, **kwargs):
        n = max(1000, len(text) * 100)
        return AudioResult(
            audio=np.zeros(n, dtype=np.float32),
            sample_rate=self.sample_rate,
            metadata={"duration": n / self.sample_rate},
        )

    def synthesize_stream(self, text, **kwargs):
        for _ in range(3):
            yield AudioChunk(audio=np.zeros(500, dtype=np.float32), sample_rate=self.sample_rate)


@pytest.fixture
def client():
    reg = ModalityRegistry()
    reg.register("audio.speech", FakeTTS())
    app = create_app(registry=reg, routers={"audio.speech": build_router(reg)})
    return TestClient(app)


def test_list_voices_endpoint(client):
    r = client.get("/v1/audio/speech/voices")
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "fake-tts"
    assert "default" in body["voices"]
    assert "alt" in body["voices"]


def test_speech_wav_response(client):
    r = client.post("/v1/audio/speech", json={
        "input": "hello world",
        "model": "fake-tts",
        "response_format": "wav",
    })
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/wav"
    assert r.content.startswith(b"RIFF")


def test_speech_default_model_when_unspecified(client):
    r = client.post("/v1/audio/speech", json={"input": "hello"})
    assert r.status_code == 200
    assert r.content.startswith(b"RIFF")


def test_unknown_model_returns_404(client):
    r = client.post("/v1/audio/speech", json={
        "input": "hello",
        "model": "does-not-exist",
    })
    assert r.status_code == 404


def test_empty_input_returns_400(client):
    r = client.post("/v1/audio/speech", json={"input": ""})
    # Pydantic v2 validation yields 422 by default; either is acceptable
    assert r.status_code in (400, 422)


def test_oversize_input_returns_400(client):
    r = client.post("/v1/audio/speech", json={"input": "x" * 60_000})
    assert r.status_code in (400, 422)


def test_streaming_response(client):
    r = client.post("/v1/audio/speech", json={
        "input": "hello world",
        "stream": True,
    })
    assert r.status_code == 200
    assert "text/event-stream" in r.headers["content-type"]


def test_404_uses_openai_error_envelope(client):
    """Unknown model must return {error:{...}} not {detail:...}."""
    r = client.post("/v1/audio/speech", json={
        "input": "hi", "model": "no-such",
    })
    assert r.status_code == 404
    body = r.json()
    # OpenAI envelope: top-level "error" with code/message/type
    assert "error" in body
    assert "detail" not in body
    err = body["error"]
    assert err["code"] == "model_not_found"
    assert err["type"] == "invalid_request_error"
    assert "no-such" in err["message"]


def test_voices_404_uses_openai_error_envelope(client):
    r = client.get("/v1/audio/speech/voices?model=no-such")
    assert r.status_code == 404
    assert "error" in r.json()


def test_streaming_yields_multiple_events_progressively(client):
    """With the producer-queue pattern, each chunk is a distinct SSE event."""
    r = client.post("/v1/audio/speech", json={
        "input": "hello world",
        "stream": True,
    })
    assert r.status_code == 200
    assert "text/event-stream" in r.headers["content-type"]
    text = r.text
    # FakeTTS yields 3 chunks then "done". Expect at least 3 data events + done.
    data_event_count = text.count("data: ")  # each SSE event starts with "data: "
    assert data_event_count >= 3
    assert "event: done" in text
