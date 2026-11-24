"""Route tests for ``POST /v1/music/midi``."""
from __future__ import annotations

import base64
import threading
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from muse.core.registry import ModalityRegistry
from muse.core.server import create_app
from muse.modalities.music_midi import (
    DISCOVERY,
    MODALITY,
    MODEL_OPTIONAL_PATHS,
    PROBE_DEFAULTS,
    build_router,
)
from muse.modalities.music_midi.runtimes.procedural import ProceduralMidiComposer
from muse.modalities.music_midi.smf import parse_smf


def _make_client(backend=None):
    if backend is None:
        backend = ProceduralMidiComposer(model_id="midi-composer")
    reg = ModalityRegistry()
    reg.register(
        MODALITY,
        backend,
        manifest={
            "model_id": backend.model_id,
            "modality": MODALITY,
            "capabilities": {"device": "cpu"},
        },
    )
    app = create_app(registry=reg, routers={MODALITY: build_router(reg)})
    return TestClient(app), backend


def test_discovery_and_model_optional_paths_surface():
    assert MODALITY == "music/midi"
    assert MODEL_OPTIONAL_PATHS == ("/v1/music/midi",)
    assert len(DISCOVERY["routes"]) == 1
    r = DISCOVERY["routes"][0]
    assert r["method"] == "POST"
    assert r["path"] == "/v1/music/midi"
    assert r["request_content_type"] == "application/json"
    assert r["response_content_type"] == "application/json"
    assert r["mcp_tool"] == "muse_generate_midi"
    assert r["file_fields"] == []
    assert r["media_fields"] == ["midi_b64"]


def test_post_midi_json_response_envelope_and_score():
    client, _ = _make_client()
    r = client.post(
        "/v1/music/midi",
        json={
            "prompt": "medieval village lute and flute",
            "key": "D",
            "mode": "dorian",
            "bpm": 110,
            "time_signature": "4/4",
            "bars": 8,
            "instruments": ["lute", "flute"],
            "layers": ["base", "combat"],
            "seed": 123,
            "temperature": 0.85,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "music.midi"
    assert body["model"] == "midi-composer"
    assert body["id"].startswith("midi-")
    assert isinstance(body["midi_b64"], str)
    raw = base64.b64decode(body["midi_b64"])
    assert raw[:4] == b"MThd"

    score = body["score"]
    assert score["bpm"] == 110
    assert score["time_signature"] == [4, 4]
    assert score["key"] == "D"
    assert score["mode"] == "dorian"
    assert score["bars"] == 8
    assert score["ppqn"] == 480
    assert len(score["tracks"]) >= 3
    layers = {t["layer"] for t in score["tracks"]}
    assert layers == {"base", "combat"}
    for tr in score["tracks"]:
        assert {"name", "layer", "instrument", "channel", "gm_program", "notes"} <= set(tr)
        for note in tr["notes"]:
            assert {"start_beat", "duration_beats", "pitch", "velocity"} <= set(note)

    # Round-trip the returned base64 MIDI bytes
    parsed = parse_smf(raw)
    assert parsed.bpm == 110
    assert parsed.bars == 8


def test_post_midi_raw_midi_response_format():
    client, _ = _make_client()
    r = client.post(
        "/v1/music/midi",
        json={
            "prompt": "calm harp melody",
            "bars": 4,
            "seed": 1,
            "response_format": "midi",
        },
    )
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/midi"
    assert 'filename="composition.mid"' in r.headers["content-disposition"]
    assert r.headers["x-muse-model"] == "midi-composer"
    assert r.content[:4] == b"MThd"


def test_post_midi_unknown_model_returns_404_envelope():
    client, _ = _make_client()
    r = client.post("/v1/music/midi", json={"prompt": "theme", "model": "missing"})
    assert r.status_code == 404
    body = r.json()
    assert body["error"]["code"] == "model_not_found"


def test_post_midi_validation_errors():
    client, _ = _make_client()
    # Empty prompt -> 422
    assert client.post("/v1/music/midi", json={"prompt": ""}).status_code == 422
    # BPM out of range -> 422
    assert client.post("/v1/music/midi", json={"prompt": "x", "bpm": 20}).status_code == 422
    assert client.post("/v1/music/midi", json={"prompt": "x", "bpm": 300}).status_code == 422
    # Bars out of range -> 422
    assert client.post("/v1/music/midi", json={"prompt": "x", "bars": 2}).status_code == 422
    assert client.post("/v1/music/midi", json={"prompt": "x", "bars": 128}).status_code == 422
    # Bad mode -> 422
    assert client.post("/v1/music/midi", json={"prompt": "x", "mode": "blues"}).status_code == 422
    # Bad time_signature -> 422
    assert client.post("/v1/music/midi", json={"prompt": "x", "time_signature": "7/8"}).status_code == 422
    # Bad temperature -> 422
    assert client.post("/v1/music/midi", json={"prompt": "x", "temperature": 3.0}).status_code == 422
    # Bad response_format -> 422
    assert client.post("/v1/music/midi", json={"prompt": "x", "response_format": "wav"}).status_code == 422
    # Bad key -> 400 OpenAI envelope
    r = client.post("/v1/music/midi", json={"prompt": "x", "key": "H#"})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_parameter"


def test_post_midi_internal_error_returns_500_envelope():
    backend = MagicMock()
    backend.model_id = "broken-midi"
    backend.generate.side_effect = RuntimeError("boom")
    client, _ = _make_client(backend)
    r = client.post("/v1/music/midi", json={"prompt": "test"})
    assert r.status_code == 500
    assert r.json()["error"]["code"] == "internal_error"


def test_post_midi_holds_inference_lock():
    class _LockAssertingComposer(ProceduralMidiComposer):
        def __init__(self):
            super().__init__(model_id="lock-midi")
            self._inference_lock = threading.Lock()
            self.lock_was_held = False

        def generate(self, prompt, **kwargs):
            self.lock_was_held = self._inference_lock.locked()
            return super().generate(prompt, **kwargs)

    backend = _LockAssertingComposer()
    client, _ = _make_client(backend)
    r = client.post("/v1/music/midi", json={"prompt": "village song", "bars": 4, "seed": 0})
    assert r.status_code == 200
    assert backend.lock_was_held


def test_probe_defaults_validates_mthd():
    composer = ProceduralMidiComposer()
    info = PROBE_DEFAULTS["call"](composer)
    assert info["bars"] == 4
    assert info["tracks"] >= 1
    assert info["bytes"] > 14
