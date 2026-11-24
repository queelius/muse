"""POST /v1/audio/speech/voices: save reference voices.

Listing, inspecting, auditioning, editing and deleting voices are light
routes; see test_light_routes.py."""
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
from muse.modalities.audio_speech.protocol import (
    UnknownVoiceError, VoiceError, VoiceUnreadableError,
)
from muse.modalities.audio_speech.routes import build_router
from muse.modalities.audio_speech.voices import StoredVoice


class FakeTTS:
    model_id = "fake-tts"
    sample_rate = 24000
    voices = ["narrator"]

    def __init__(self, save_raises=None):
        self.saved = []
        self.save_raises = save_raises

    def save_voice(self, name, audio, sample_rate, transcript, *,
                   overwrite=False, source_filename=None, metadata=None):
        if self.save_raises:
            raise self.save_raises
        self.saved.append(dict(
            name=name, audio=audio, sample_rate=sample_rate,
            transcript=transcript, overwrite=overwrite,
            source_filename=source_filename, metadata=metadata,
        ))
        return StoredVoice(
            name=name, audio_path="/x.wav", transcript=transcript,
            sample_rate=sample_rate, duration_seconds=1.5, sha256="ab" * 32,
            read_only=False, identity=(0, 0),
        )


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
        "category": None, "tags": [], "description": None, "notes": None,
        "design": None,
    }
    (call,) = fake.saved
    assert call["name"] == "grimbold"
    assert call["transcript"] == "Welcome, traveler."
    assert call["sample_rate"] == 24000
    assert call["overwrite"] is False
    assert call["source_filename"] == "ref.wav"
    assert call["audio"].dtype == np.float32 and call["audio"].shape == (36000,)


def test_save_passes_metadata_fields(decode):
    fake = FakeTTS()
    r = _save(_client(fake), category="npc/dwarf", tags="male, gravelly",
              design_instructions="gruff")
    assert r.status_code == 200, r.text
    md = fake.saved[0]["metadata"]
    assert md == {"category": "npc/dwarf", "tags": "male, gravelly",
                  "design": {"model": None, "instructions": "gruff"}}
    body = r.json()
    assert body["category"] == "npc/dwarf" and body["tags"] == ["gravelly", "male"]
    assert body["design"] == {"model": None, "instructions": "gruff"}


def test_save_without_metadata_sends_empty_metadata(decode):
    fake = FakeTTS()
    assert _save(_client(fake)).status_code == 200
    assert fake.saved[0]["metadata"] == {}


def test_save_reads_back_backend_entry_when_available(decode):
    from muse.modalities.audio_speech.voice_catalog import entry_from_meta

    fake = FakeTTS()
    fake.voice_entry = lambda name: entry_from_meta(name, {
        "transcript": "t", "sample_rate": 24000, "duration_seconds": 1.5,
        "category": "narrator/epic", "tags": ["deep"]}, read_only=False)
    body = _save(_client(fake), category="ignored/by/readback").json()
    assert body["category"] == "narrator/epic" and body["tags"] == ["deep"]


def test_save_invalid_metadata_is_400_envelope(decode):
    fake = FakeTTS(save_raises=VoiceError(
        "invalid_voice_metadata", "category: bad segment", 400))
    r = _save(_client(fake), category="Bad Cat!")
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_voice_metadata"


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
    assert not post.get("light")  # saving decodes audio in the model's venv
    delete = routes_[("DELETE", "/v1/audio/speech/voices/{name}")]
    assert delete["mcp_tool"] == "muse_delete_voice" and delete["light"] is True


class StoreBackedTTS(FakeTTS):
    def __init__(self, store):
        super().__init__()
        self.store = store

    def save_voice(self, name, audio, sample_rate, transcript, *,
                   overwrite=False, source_filename=None, metadata=None):
        return self.store.save(name, audio, sample_rate, transcript,
                               overwrite=overwrite, source_filename=source_filename,
                               metadata=metadata)


def test_save_trailing_newline_name_is_invalid(decode, tmp_path):
    from muse.modalities.audio_speech.voices import VoiceStore
    c = _client(StoreBackedTTS(VoiceStore(tmp_path)))
    r = _save(c, name="grimbold\n")
    assert r.status_code == 400, r.text
    assert r.json()["error"]["code"] == "invalid_voice_name"
    assert list(tmp_path.iterdir()) == []


def test_save_storage_unavailable_has_own_fixed_message(decode):
    fake = FakeTTS(save_raises=VoiceUnreadableError("/secret/path is bad"))
    r = _save(_client(fake))
    assert r.status_code == 500
    err = r.json()["error"]
    assert err["code"] == "voice_storage_unavailable"
    assert err["message"] == "voice storage is unavailable"
    assert "/secret" not in r.text


