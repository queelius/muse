"""Voice catalog routes served by the audio/speech light router.

Mounted the way a worker mounts them (router + light router over the
registry). For a saved-voice model every route reads or edits the voice
files on disk and never touches the backend.
"""
import json
import os
import wave
from io import BytesIO

import pytest
from fastapi.testclient import TestClient

from muse.core import config
from muse.core.discovery import light_hooks
from muse.core.light import RegistryLightCatalog
from muse.core.registry import ModalityRegistry
from muse.core.server import create_app
from muse.modalities.audio_speech.light import build_light_router
from muse.modalities.audio_speech.routes import build_router
from muse.modalities.audio_speech.voice_errors import VoiceUnreadableError

MODEL = "fake-clone"


class Sentinel:
    """A backend with no voice methods: any call into it fails the test."""

    model_id = MODEL


class PresetTTS:
    model_id = "kokoro"
    voices = ["af_heart", "bm_george"]


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSE_HOME", str(tmp_path))
    config.reset_config()
    yield tmp_path
    config.reset_config()


def _wav(seconds=2.0):
    buf = BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(b"\0\0" * int(seconds * 24000))
    return buf.getvalue()


def _voice(home, name, **meta):
    root = home / "voices" / MODEL
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    os.chmod(home / "voices", 0o700)
    os.chmod(root, 0o700)
    (d / "reference.wav").write_bytes(_wav())
    (d / "voice.json").write_text(json.dumps({
        "name": name, "transcript": "Hello there.", "sample_rate": 24000,
        "duration_seconds": 2.0, "sha256": "cd" * 32,
        "created_at": "2026-11-20T00:00:00+00:00", **meta}))


def _client(*backends):
    reg = ModalityRegistry()
    for b, caps in backends:
        reg.register("audio/speech", b, manifest={
            "model_id": b.model_id, "modality": "audio/speech", "capabilities": caps})
    app = create_app(registry=reg, routers={
        "audio/speech": build_router(reg),
        "audio/speech:light": build_light_router(RegistryLightCatalog(reg)),
    })
    return TestClient(app)


SAVED = (Sentinel(), {"supports_saved_voices": True})
PRESET = (PresetTTS(), {})


def test_bundled_speech_package_exports_the_light_router():
    assert light_hooks()["audio/speech"].build_light_router is build_light_router


def test_list_has_data_categories_and_filters(home):
    _voice(home, "sage", category="narrator/epic", tags=["deep"])
    _voice(home, "guard", category="npc/guard", description="Gruff city guard")
    c = _client(SAVED)
    body = c.get(f"/v1/audio/speech/voices?model={MODEL}").json()
    assert body["object"] == "list" and body["model"] == MODEL
    assert body["voices"] == ["narrator", "guard", "sage"]
    assert [d["name"] for d in body["data"]] == body["voices"]
    assert {"path": "narrator", "count": 2} in body["categories"]
    assert {"path": "narrator/epic", "count": 1} in body["categories"]
    narrowed = c.get(f"/v1/audio/speech/voices?model={MODEL}&category=narrator&tag=deep").json()
    assert narrowed["voices"] == ["sage"]
    assert narrowed["categories"] == body["categories"]  # always over every voice
    assert c.get(f"/v1/audio/speech/voices?model={MODEL}&q=gruff").json()["voices"] == ["guard"]


def test_list_on_fresh_install_is_bundled_only_and_creates_nothing(home):
    body = _client(SAVED).get(f"/v1/audio/speech/voices?model={MODEL}").json()
    assert body["voices"] == ["narrator"]
    assert not (home / "voices").exists()


def test_get_one_and_audio_url(home):
    e = _client(SAVED).get(f"/v1/audio/speech/voices/narrator?model={MODEL}").json()
    assert e["object"] == "audio.voice" and e["read_only"] is True
    assert e["category"] == "narrator/neutral"
    assert e["audio_url"] == f"/v1/audio/speech/voices/narrator/audio?model={MODEL}"


def test_audio_returns_reference_wav(home):
    _voice(home, "guard")
    r = _client(SAVED).get(f"/v1/audio/speech/voices/guard/audio?model={MODEL}")
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/wav"
    assert r.headers["content-disposition"] == 'inline; filename="guard.wav"'
    assert r.content[:4] == b"RIFF"


def test_patch_sets_and_clears_fields(home):
    _voice(home, "guard", notes="old")
    c = _client(SAVED)
    r = c.patch("/v1/audio/speech/voices/guard", json={
        "model": MODEL, "category": "NPC/Guard", "tags": "gruff, male"})
    assert r.status_code == 200, r.text
    e = r.json()
    assert (e["category"], e["tags"], e["notes"]) == ("npc/guard", ["gruff", "male"], "old")
    assert e["updated_at"]
    cleared = c.patch("/v1/audio/speech/voices/guard", json={"model": MODEL, "notes": None}).json()
    assert cleared["notes"] is None and cleared["category"] == "npc/guard"


def test_patch_rule_violation_is_400(home):
    _voice(home, "guard")
    r = _client(SAVED).patch("/v1/audio/speech/voices/guard",
                             json={"model": MODEL, "category": "a/b/c/d/e"})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_voice_metadata"


def test_patch_unknown_field_is_422_envelope(home):
    _voice(home, "guard")
    r = _client(SAVED).patch("/v1/audio/speech/voices/guard",
                             json={"model": MODEL, "colour": "red"})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_request"


def test_patch_and_delete_bundled_are_409(home):
    c = _client(SAVED)
    r = c.patch("/v1/audio/speech/voices/narrator", json={"model": MODEL, "notes": "x"})
    assert r.status_code == 409 and r.json()["error"]["code"] == "voice_read_only"
    r = c.delete(f"/v1/audio/speech/voices/narrator?model={MODEL}")
    assert r.status_code == 409 and r.json()["error"]["code"] == "voice_read_only"


def test_delete_ok_then_unknown(home):
    _voice(home, "guard")
    c = _client(SAVED)
    r = c.delete(f"/v1/audio/speech/voices/guard?model={MODEL}")
    assert r.status_code == 200
    assert r.json() == {"object": "audio.voice.deleted", "model": MODEL,
                        "name": "guard", "deleted": True}
    r = c.delete(f"/v1/audio/speech/voices/guard?model={MODEL}")
    assert r.status_code == 404 and r.json()["error"]["code"] == "voice_not_found"


@pytest.mark.parametrize("bad", ["NARRATOR", "x" * 65, "nope", ".lock"])
def test_bad_names_are_404_voice_not_found(home, bad):
    _voice(home, "guard")
    c = _client(SAVED)
    for r in (c.get(f"/v1/audio/speech/voices/{bad}?model={MODEL}"),
              c.get(f"/v1/audio/speech/voices/{bad}/audio?model={MODEL}"),
              c.patch(f"/v1/audio/speech/voices/{bad}", json={"model": MODEL, "notes": "x"}),
              c.delete(f"/v1/audio/speech/voices/{bad}?model={MODEL}")):
        assert r.status_code == 404, r.text
        assert r.json()["error"]["code"] == "voice_not_found"


def test_encoded_traversal_never_reaches_a_voice(home):
    _voice(home, "guard")
    (home / "voices" / "secret.wav").write_bytes(b"RIFFsecret")
    c = _client(SAVED)
    for path in ("..%2Fsecret.wav", "..%2F..%2Fvoices%2Fsecret.wav"):
        r = c.get(f"/v1/audio/speech/voices/{path}/audio?model={MODEL}")
        assert r.status_code == 404
        assert b"secret" not in r.content


def test_corrupt_voice_is_500_voice_unreadable(home):
    _voice(home, "guard")
    (home / "voices" / MODEL / "guard" / "voice.json").write_text("{")
    c = _client(SAVED)
    assert "guard" not in c.get(f"/v1/audio/speech/voices?model={MODEL}").json()["voices"]
    r = c.get(f"/v1/audio/speech/voices/guard?model={MODEL}")
    assert r.status_code == 500 and r.json()["error"]["code"] == "voice_unreadable"


def test_preset_model_lists_backend_voices_with_filters(home):
    c = _client(PRESET)
    body = c.get("/v1/audio/speech/voices?model=kokoro").json()
    assert body["voices"] == ["af_heart", "bm_george"]
    assert body["data"][0] == {"object": "audio.voice", "model": "kokoro",
                               "name": "af_heart", "read_only": True}
    assert body["categories"] == []
    assert c.get("/v1/audio/speech/voices?model=kokoro&q=george").json()["voices"] == ["bm_george"]


def test_preset_model_list_unreadable_is_500_envelope(home):
    class Bad(PresetTTS):
        @property
        def voices(self):
            raise VoiceUnreadableError("/secret/path unreadable")

    r = _client((Bad(), {})).get("/v1/audio/speech/voices?model=kokoro")
    assert r.status_code == 500
    assert r.json()["error"]["code"] == "voice_unreadable"
    assert "/secret" not in r.text


def test_preset_model_other_routes_are_voice_saving_not_supported(home):
    c = _client(PRESET)
    for r in (c.get("/v1/audio/speech/voices/af_heart?model=kokoro"),
              c.get("/v1/audio/speech/voices/af_heart/audio?model=kokoro"),
              c.patch("/v1/audio/speech/voices/af_heart", json={"model": "kokoro", "notes": "x"}),
              c.delete("/v1/audio/speech/voices/af_heart?model=kokoro")):
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "voice_saving_not_supported"


def test_unknown_model_is_404_model_not_found(home):
    r = _client(SAVED).get("/v1/audio/speech/voices?model=nope")
    assert r.status_code == 404 and r.json()["error"]["code"] == "model_not_found"


def test_default_model_is_the_registry_default(home):
    body = _client(SAVED, PRESET).get("/v1/audio/speech/voices").json()
    assert body["model"] == MODEL and "narrator" in body["voices"]
