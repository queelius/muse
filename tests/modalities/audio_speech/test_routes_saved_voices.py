"""POST/DELETE /v1/audio/speech/voices: save and delete reference voices."""
import io
import struct
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

from muse.core.registry import ModalityRegistry
from muse.core.server import create_app
from muse.modalities._common.audio_decode import (
    AudioDecodeError, AudioDecoderUnavailableError, AudioDurationExceededError,
    DecodedAudio,
)
from muse.modalities.audio_speech import DISCOVERY
from muse.modalities.audio_speech import routes
from muse.modalities.audio_speech.protocol import UnknownVoiceError, VoiceError
from muse.modalities.audio_speech.routes import build_router
from muse.modalities.audio_speech.voices import StoredVoice


class FakeTTS:
    model_id = "fake-tts"
    sample_rate = 24000
    voices = ["narrator"]

    def __init__(self, save_raises=None, delete_raises=None):
        self.saved = []
        self.deleted = []
        self.save_raises = save_raises
        self.delete_raises = delete_raises

    def save_voice(self, name, audio, sample_rate, transcript, *,
                   overwrite=False, source_filename=None):
        if self.save_raises:
            raise self.save_raises
        self.saved.append(dict(
            name=name, audio=audio, sample_rate=sample_rate,
            transcript=transcript, overwrite=overwrite,
            source_filename=source_filename,
        ))
        return StoredVoice(
            name=name, audio_path="/x.wav", transcript=transcript,
            sample_rate=sample_rate, duration_seconds=1.5, sha256="ab" * 32,
            read_only=False, identity=(0, 0),
        )

    def delete_voice(self, name):
        if self.delete_raises:
            raise self.delete_raises
        self.deleted.append(name)


def _client(fake, caps=None):
    reg = ModalityRegistry()
    reg.register("audio/speech", fake, manifest={
        "model_id": "fake-tts", "modality": "audio/speech",
        "capabilities": {"supports_saved_voices": True} if caps is None else caps,
    })
    app = create_app(registry=reg, routers={"audio/speech": build_router(reg)})
    return TestClient(app)


def _decoded(*_a, **_k):
    torch = pytest.importorskip("torch")
    return DecodedAudio(
        waveform=torch.zeros(1, 36000), sample_rate=24000, duration_seconds=1.5,
    )


@pytest.fixture
def decode(monkeypatch):
    monkeypatch.setattr(routes, "decode_audio", _decoded)
    return monkeypatch


def _save(c, name="grimbold", **form):
    data = {"name": name, "transcript": "Welcome, traveler.", **form}
    return c.post(
        "/v1/audio/speech/voices", data=data,
        files={"audio": ("ref.wav", b"RIFFxxxx", "audio/wav")},
    )


def test_save_voice_ok(decode):
    fake = FakeTTS()
    r = _save(_client(fake))
    assert r.status_code == 200, r.text
    assert r.json() == {
        "object": "audio.voice", "model": "fake-tts", "name": "grimbold",
        "duration_seconds": 1.5, "sha256": "ab" * 32,
    }
    (call,) = fake.saved
    assert call["name"] == "grimbold"
    assert call["transcript"] == "Welcome, traveler."
    assert call["sample_rate"] == 24000
    assert call["overwrite"] is False
    assert call["source_filename"] == "ref.wav"
    assert call["audio"].dtype == np.float32 and call["audio"].shape == (36000,)


def test_save_overwrite_flag(decode):
    fake = FakeTTS()
    assert _save(_client(fake), overwrite="true").status_code == 200
    assert fake.saved[0]["overwrite"] is True


def test_save_requires_capability(decode):
    r = _save(_client(FakeTTS(), caps={}))
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "voice_saving_not_supported"


def test_save_maps_voice_errors(decode):
    fake = FakeTTS(save_raises=VoiceError("voice_exists", "exists", 409))
    r = _save(_client(fake))
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "voice_exists"


def _raising(exc):
    def boom(*a, **k):
        raise exc
    return boom


def test_save_too_long_reference(monkeypatch):
    monkeypatch.setattr(
        routes, "decode_audio",
        _raising(AudioDurationExceededError(maximum_seconds=30.0)),
    )
    r = _save(_client(FakeTTS()))
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_reference_audio"


def test_save_undecodable(monkeypatch):
    monkeypatch.setattr(routes, "decode_audio", _raising(AudioDecodeError("bad")))
    r = _save(_client(FakeTTS()))
    assert r.status_code == 415
    assert r.json()["error"]["code"] == "unsupported_media_type"


def test_save_decoder_unavailable(monkeypatch):
    monkeypatch.setattr(
        routes, "decode_audio",
        _raising(AudioDecoderUnavailableError("no torchcodec secret-detail")),
    )
    r = _save(_client(FakeTTS()))
    assert r.status_code == 500
    assert "secret-detail" not in r.text


def test_save_oversize_upload(monkeypatch):
    monkeypatch.setenv("MUSE_VOICE_REFERENCE_MAX_BYTES", "4")
    r = _save(_client(FakeTTS()))
    assert r.status_code == 413
    assert r.json()["error"]["code"] == "payload_too_large"


def test_delete_ok_and_unknown():
    fake = FakeTTS()
    c = _client(fake)
    r = c.delete("/v1/audio/speech/voices/grimbold")
    assert r.status_code == 200
    assert r.json() == {
        "object": "audio.voice.deleted", "model": "fake-tts",
        "name": "grimbold", "deleted": True,
    }
    assert fake.deleted == ["grimbold"]
    c2 = _client(FakeTTS(delete_raises=UnknownVoiceError("nope", ["a"])))
    r2 = c2.delete("/v1/audio/speech/voices/nope")
    assert r2.status_code == 404
    assert r2.json()["error"]["code"] == "voice_not_found"


def test_delete_read_only_and_capability():
    fake = FakeTTS(delete_raises=VoiceError("voice_read_only", "ro"))
    r = _client(fake).delete("/v1/audio/speech/voices/x")
    assert r.status_code == 400 and r.json()["error"]["code"] == "voice_read_only"
    r = _client(FakeTTS(), caps={}).delete("/v1/audio/speech/voices/x")
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "voice_saving_not_supported"


def test_real_wav_through_real_decoder():
    pytest.importorskip("torchcodec")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(struct.pack("<h", 0) * 2400)
    fake = FakeTTS()
    r = _client(fake).post(
        "/v1/audio/speech/voices",
        data={"name": "a", "transcript": "hi"},
        files={"audio": ("a.wav", buf.getvalue(), "audio/wav")},
    )
    assert r.status_code == 200, r.text
    assert fake.saved[0]["audio"].ndim == 1


def test_discovery_lists_both_routes():
    routes_ = {(r["method"], r["path"]): r for r in DISCOVERY["routes"]}
    post = routes_[("POST", "/v1/audio/speech/voices")]
    assert post["mcp_tool"] == "muse_save_voice"
    assert post["request_content_type"] == "multipart/form-data"
    assert post["file_fields"] == ["audio"]
    assert routes_[("DELETE", "/v1/audio/speech/voices/{name}")]["mcp_tool"] is None
