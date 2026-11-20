"""`instructions` on /v1/audio/speech and typed voice errors."""
import numpy as np
from fastapi.testclient import TestClient

from muse.core.registry import ModalityRegistry
from muse.core.server import create_app
from muse.modalities.audio_speech.protocol import (
    AudioChunk, AudioResult, UnknownVoiceError, VoiceUnreadableError,
)
from muse.modalities.audio_speech.routes import build_router


class FakeTTS:
    model_id = "fake-tts"
    sample_rate = 16000
    voices = ["narrator"]

    def __init__(self, raises=None, metadata=None, check_raises=None):
        self.calls = []
        self.raises = raises
        self.metadata = metadata or {}
        self.check_raises = check_raises
        if check_raises is not None:
            self.check_voice = self._check_voice

    def _check_voice(self, voice):
        raise self.check_raises

    def synthesize(self, text, **kwargs):
        self.calls.append(kwargs)
        if self.raises:
            raise self.raises
        return AudioResult(
            audio=np.zeros(1000, dtype=np.float32),
            sample_rate=self.sample_rate, metadata=dict(self.metadata),
        )

    def synthesize_stream(self, text, **kwargs):
        self.calls.append(kwargs)
        yield AudioChunk(audio=np.zeros(500, dtype=np.float32), sample_rate=16000)


def _client(fake, caps=None):
    reg = ModalityRegistry()
    reg.register("audio/speech", fake, manifest={
        "model_id": "fake-tts", "modality": "audio/speech",
        "capabilities": caps or {},
    })
    app = create_app(registry=reg, routers={"audio/speech": build_router(reg)})
    return TestClient(app)


def _post(c, **body):
    return c.post("/v1/audio/speech", json={"input": "hello", **body})


def test_instructions_forwarded_when_supported():
    fake = FakeTTS()
    r = _post(_client(fake, {"supports_instructions": True}), instructions="gruff dwarf")
    assert r.status_code == 200
    assert fake.calls[0]["instructions"] == "gruff dwarf"


def test_instructions_forwarded_when_streaming():
    fake = FakeTTS()
    r = _post(_client(fake, {"supports_instructions": True}),
              instructions="gruff dwarf", stream=True)
    assert r.status_code == 200
    assert fake.calls[0]["instructions"] == "gruff dwarf"


def test_instructions_absent_means_no_kwarg():
    fake = FakeTTS()
    assert _post(_client(fake, {"supports_instructions": True})).status_code == 200
    assert "instructions" not in fake.calls[0]


def test_instructions_rejected_when_unsupported():
    r = _post(_client(FakeTTS()), instructions="gruff dwarf")
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "instructions_not_supported"


def test_blank_instructions_treated_as_unset():
    fake = FakeTTS()
    r = _post(_client(fake), instructions="   ")
    assert r.status_code == 200
    assert "instructions" not in fake.calls[0]


def test_instructions_too_long():
    r = _post(_client(FakeTTS(), {"supports_instructions": True}),
              instructions="x" * 4097)
    assert r.status_code in (400, 422)


def test_unknown_voice_is_400():
    fake = FakeTTS(raises=UnknownVoiceError("alloy", ["narrator"]))
    r = _post(_client(fake), voice="alloy")
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "voice_not_found"
    assert "narrator" in r.json()["error"]["message"]


def test_unknown_voice_stream_is_400_before_sse():
    fake = FakeTTS(check_raises=UnknownVoiceError("alloy", ["narrator"]))
    r = _post(_client(fake), voice="alloy", stream=True)
    assert r.status_code == 400
    assert "text/event-stream" not in r.headers["content-type"]
    assert r.json()["error"]["code"] == "voice_not_found"


def test_unreadable_voice_is_500_without_paths():
    fake = FakeTTS(raises=VoiceUnreadableError("/secret/path"))
    r = _post(_client(fake), voice="alloy")
    assert r.status_code == 500
    assert r.json()["error"]["code"] == "voice_unreadable"
    assert "/secret" not in r.text


def test_unreadable_voice_stream_check_is_500():
    fake = FakeTTS(check_raises=VoiceUnreadableError("/secret/path"))
    r = _post(_client(fake), voice="alloy", stream=True)
    assert r.status_code == 500
    assert "/secret" not in r.text


def test_metadata_voice_none_suppresses_voice_header():
    r = _post(_client(FakeTTS(metadata={"voice": None})), voice="alloy")
    assert r.status_code == 200
    assert "x-muse-voice" not in r.headers
