"""The capabilities document: pure build + worker route."""
from __future__ import annotations

import json

import pytest
from fastapi import APIRouter
from fastapi.testclient import TestClient

from muse.core.capabilities import (
    ModelFacts,
    ModelState,
    UnknownFilter,
    build_document,
    filter_document,
    model_facts,
    render_llms_txt,
    server_block,
)
from muse.core.model_notes import ModelNotes
from muse.core.provenance import ArtifactComponent, ModelRevision

REV = "a" * 40
BASE_REV = "b" * 40

DISCOVERY = {
    "audio/speech": {
        "summary": "Text-to-speech.",
        "use_for": ["narration"],
        "not_for": ["music"],
        "routes": [{
            "method": "POST", "path": "/v1/audio/speech",
            "request_content_type": "application/json",
            "response_content_type": "audio/wav", "compat": "openai:audio.speech",
            "mcp_tool": "muse_speak", "file_fields": [], "media_fields": [],
            "example": {"model": "<model>", "input": "hi"},
        }],
    },
    "audio/transcription": {
        "summary": "Speech-to-text.",
        "use_for": ["transcripts"],
        "not_for": ["alignment"],
        "routes": [{
            "method": "POST", "path": "/v1/audio/transcriptions",
            "request_content_type": "multipart/form-data",
            "response_content_type": "application/json", "compat": None,
            "mcp_tool": "muse_transcribe", "file_fields": ["file"], "media_fields": [],
            "example": {"model": "<model>"},
        }],
    },
}


def _facts(model_id="kokoro-82m", modality="audio/speech", caps=None, components=None,
           digest=None, memory_gb=0.5) -> ModelFacts:
    rev = ModelRevision(
        model_id, REV,
        components if components is not None
        else (ArtifactComponent("weights", "hexgrad/Kokoro-82M", REV),),
        digest,
    )
    return ModelFacts(
        id=model_id, modality=modality, hf_repo="hexgrad/Kokoro-82M", license="Apache 2.0",
        revision=rev,
        notes=ModelNotes("Fast TTS.", ("fast",), ("English-centric",), "good"),
        capabilities=caps if caps is not None else {"voices": ["af_heart"], "sample_rate": 24000},
        memory_gb=memory_gb,
    )


def _state(**kw) -> ModelState:
    base = dict(loaded=False, last_cold_load_seconds=4.5, queue_depth=0,
                unservable_reason=None)
    base.update(kw)
    return ModelState(**base)


SERVER = server_block(version="1.2.3", node="box", base_url="http://box:8000")


def _doc(models, **kw):
    return build_document(server=SERVER, discovery=DISCOVERY, models=models, **kw)


def test_document_shape_and_filters():
    whisper = _facts("whisper-base", "audio/transcription", caps={"supports_translation": True})
    doc = _doc([(whisper, _state(loaded=True)), (_facts(), _state())])

    assert list(doc) == ["object", "server", "modalities", "models"]
    assert doc["object"] == "muse.capabilities"
    assert doc["server"] == {
        "version": "1.2.3",
        "node": "box",
        "links": {
            "openapi": "/openapi.json", "models": "/v1/models",
            "capabilities": "/v1/capabilities", "llms_txt": "/llms.txt",
        },
        "mcp": {
            "install": 'uvx --from "museq[mcp]" muse mcp',
            "env": {"MUSE_SERVER": "http://box:8000"},
        },
    }
    speech = next(m for m in doc["modalities"] if m["modality"] == "audio/speech")
    assert set(speech) == {"modality", "summary", "use_for", "not_for", "routes", "models"}
    assert speech["models"] == ["kokoro-82m"]
    assert speech["routes"] == DISCOVERY["audio/speech"]["routes"]

    # Models sorted by id; every key present.
    assert [m["id"] for m in doc["models"]] == ["kokoro-82m", "whisper-base"]
    kokoro = doc["models"][0]
    assert list(kokoro) == [
        "id", "modality", "hf_repo", "license", "revision", "components",
        "summary", "strengths", "limits", "quality_tier", "capabilities",
        "loaded", "memory_gb", "last_cold_load_seconds", "queue_depth",
        "unservable_reason",
    ]
    assert kokoro["revision"] == REV
    assert kokoro["components"] == [
        {"role": "weights", "repo": "hexgrad/Kokoro-82M", "revision": REV},
    ]
    assert kokoro["strengths"] == ["fast"] and kokoro["limits"] == ["English-centric"]
    assert kokoro["quality_tier"] == "good"
    assert kokoro["loaded"] is False and kokoro["last_cold_load_seconds"] == 4.5
    assert kokoro["queue_depth"] == 0 and kokoro["memory_gb"] == 0.5
    json.dumps(doc)  # serializable

    only_speech = _doc([(whisper, _state()), (_facts(), _state())], modality="audio/speech")
    assert [m["modality"] for m in only_speech["modalities"]] == ["audio/speech"]
    assert [m["id"] for m in only_speech["models"]] == ["kokoro-82m"]

    only_whisper = _doc([(whisper, _state()), (_facts(), _state())], model="whisper-base")
    assert [m["id"] for m in only_whisper["models"]] == ["whisper-base"]
    assert [m["modality"] for m in only_whisper["modalities"]] == ["audio/transcription"]
    assert only_whisper["modalities"][0]["models"] == ["whisper-base"]


def test_unknown_filters_raise():
    with pytest.raises(UnknownFilter):
        _doc([(_facts(), _state())], modality="video/nope")
    with pytest.raises(UnknownFilter):
        _doc([(_facts(), _state())], model="nope")
    # A model that exists but not in the requested modality.
    with pytest.raises(UnknownFilter):
        _doc([(_facts(), _state())], modality="audio/transcription", model="kokoro-82m")


def test_filter_document_is_pure_and_matches_build_filters():
    whisper = _facts("whisper-base", "audio/transcription")
    models = [(whisper, _state()), (_facts(), _state())]
    full = _doc(models)
    full["extra"] = ["kept"]
    before = json.dumps(full, sort_keys=True)

    by_mod = filter_document(full, modality="audio/speech")
    assert [m["id"] for m in by_mod["models"]] == ["kokoro-82m"]
    assert [m["modality"] for m in by_mod["modalities"]] == ["audio/speech"]
    assert by_mod["extra"] == ["kept"] and by_mod["server"] == full["server"]

    by_model = filter_document(full, model="whisper-base")
    assert by_model["modalities"] == _doc(models, model="whisper-base")["modalities"]
    assert [m["id"] for m in by_model["models"]] == ["whisper-base"]

    assert filter_document(full) == full
    assert json.dumps(full, sort_keys=True) == before  # input not mutated
    with pytest.raises(UnknownFilter):
        filter_document(full, modality="video/nope")
    with pytest.raises(UnknownFilter):
        filter_document(full, model="nope")
    with pytest.raises(UnknownFilter):
        filter_document(full, modality="audio/speech", model="whisper-base")


def test_modality_without_models_is_listed_with_empty_ids():
    doc = _doc([(_facts(), _state())])
    tr = next(m for m in doc["modalities"] if m["modality"] == "audio/transcription")
    assert tr["models"] == []
    # And it is a valid filter target.
    assert _doc([], modality="audio/transcription")["models"] == []


def test_model_of_undiscovered_modality_still_listed():
    doc = _doc([(_facts("x", "text/custom"), _state())])
    custom = next(m for m in doc["modalities"] if m["modality"] == "text/custom")
    assert custom == {
        "modality": "text/custom", "summary": None, "use_for": [], "not_for": [],
        "routes": [], "models": ["x"],
    }


def test_hidden_keys_never_appear():
    doc = _doc([(_facts(caps={"gguf_file": "x", "supports_tools": True}), _state())])
    assert doc["models"][0]["capabilities"] == {"supports_tools": True}


def test_state_none_means_loaded_and_unknown_cost():
    m = _doc([(_facts(), None)])["models"][0]
    assert m["loaded"] is True
    assert m["last_cold_load_seconds"] is None
    assert m["queue_depth"] is None
    assert m["unservable_reason"] is None


def test_multi_component_model_lists_every_component():
    comps = (
        ArtifactComponent("weights", "nerijs/pixel-art-xl", REV),
        ArtifactComponent("base", "stabilityai/sdxl-turbo", BASE_REV),
    )
    m = _doc([(_facts(components=comps, digest="d" * 16), _state())])["models"][0]
    assert m["components"] == [
        {"role": "weights", "repo": "nerijs/pixel-art-xl", "revision": REV},
        {"role": "base", "repo": "stabilityai/sdxl-turbo", "revision": BASE_REV},
    ]


def test_build_document_does_not_mutate_inputs():
    caps = {"supports_tools": True, "gguf_file": "x"}
    discovery = json.loads(json.dumps(DISCOVERY))
    doc = build_document(server=SERVER, discovery=discovery,
                         models=[(_facts(caps=caps), _state())])
    doc["modalities"][0]["routes"].append("junk")
    doc["models"][0]["capabilities"]["junk"] = 1
    assert discovery == DISCOVERY
    assert caps == {"supports_tools": True, "gguf_file": "x"}


def test_model_facts_from_manifest():
    manifest = {
        "model_id": "kokoro-82m", "modality": "audio/speech",
        "hf_repo": "hexgrad/Kokoro-82M", "license": "Apache 2.0",
        "summary": "Fast TTS.", "strengths": ["fast"], "limits": [],
        "quality_tier": "good",
        "capabilities": {"memory_gb": 0.5, "voices": ["af_heart"]},
    }
    rev = ModelRevision("kokoro-82m", REV, (), None)
    f = model_facts("kokoro-82m", manifest, revision=rev, memory_gb=0.75)
    assert f.modality == "audio/speech" and f.hf_repo == "hexgrad/Kokoro-82M"
    assert f.license == "Apache 2.0" and f.notes.quality_tier == "good"
    assert f.capabilities == {"memory_gb": 0.5, "voices": ["af_heart"]}
    assert f.memory_gb == 0.75 and f.revision is rev


def test_model_facts_tolerates_bad_types():
    f = model_facts("m", {"hf_repo": 3, "license": ["x"], "capabilities": "junk"},
                    revision=ModelRevision("m", None, (), None), memory_gb=None,
                    modality="audio/speech")
    assert f.hf_repo is None and f.license is None and f.capabilities == {}
    assert f.modality == "audio/speech"


def test_llms_txt_lists_modalities_and_links():
    text = render_llms_txt(_doc([(_facts(), _state())]))
    assert text.startswith("# muse")
    assert "/v1/capabilities" in text and "/openapi.json" in text
    assert "/llms.txt" not in text.split("\n", 1)[0]
    assert "audio/speech" in text and "Text-to-speech." in text
    assert "POST /v1/audio/speech" in text
    assert "audio/transcription" in text and "POST /v1/audio/transcriptions" in text
    assert "kokoro-82m" in text
    assert 'MUSE_SERVER=http://box:8000' in text
    assert "museq[mcp]" in text


def test_worker_app_serves_capabilities(tmp_path, monkeypatch):
    from muse.core.catalog import _reset_read_catalog_cache
    from muse.core.registry import ModalityRegistry
    from muse.core.server import create_app

    class FakeTTS:
        model_id = "fake-tts"

    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
    _reset_read_catalog_cache()
    reg = ModalityRegistry()
    reg.register("audio/speech", FakeTTS(), manifest={
        "model_id": "fake-tts", "modality": "audio/speech",
        "hf_repo": "org/fake-tts", "description": "A fake voice.",
        "capabilities": {"voices": ["v1"], "n_threads": 4, "memory_gb": 0.25},
    })
    app = create_app(registry=reg, routers={"audio/speech": APIRouter()})
    client = TestClient(app)

    r = client.get("/v1/capabilities")
    assert r.status_code == 200
    doc = r.json()
    assert doc["object"] == "muse.capabilities"
    assert doc["server"]["mcp"]["env"]["MUSE_SERVER"] == "http://testserver"
    assert [m["modality"] for m in doc["modalities"]] == ["audio/speech"]
    assert doc["modalities"][0]["models"] == ["fake-tts"]
    m = doc["models"][0]
    assert m["id"] == "fake-tts" and m["loaded"] is True
    assert m["summary"] == "A fake voice."
    assert m["capabilities"] == {"voices": ["v1"], "memory_gb": 0.25}
    assert m["memory_gb"] == 0.25
    assert m["queue_depth"] is None and m["last_cold_load_seconds"] is None

    assert client.get("/v1/capabilities", params={"model": "fake-tts"}).status_code == 200
    bad = client.get("/v1/capabilities", params={"model": "nope"})
    assert bad.status_code == 404
    assert bad.json()["error"]["code"] == "not_found"
