"""Speech timings, pronunciation routing, and take provenance contracts."""
import base64
import hashlib
import json
import re
import threading

import numpy as np
import pytest
from fastapi.testclient import TestClient

from muse.core.registry import ModalityRegistry
from muse.core.server import create_app
from muse.modalities.audio_speech import routes
from muse.modalities.audio_speech.codec import AudioFormatError, audio_to_wav_bytes
from muse.modalities.audio_speech.pronunciations import pronunciations_digest
from muse.modalities.audio_speech.protocol import AudioChunk, AudioResult
from muse.modalities.audio_speech.word_timing import TimedSpan


class FakeModel:
    model_id = "fake-timed"
    default_voice = "af_heart"

    def __init__(self):
        self._inference_lock = threading.Lock()
        self.calls = []
        self.audio = np.zeros(24000, dtype=np.float32)
        self.metadata = {"voice": "af_heart", "timing_calibration": None}
        self.failure = None
        self.spans = None

    def synthesize(self, text, **kwargs):
        self.calls.append((text, kwargs))
        if self.failure:
            raise self.failure
        spans = [TimedSpan(m.start(), m.end(), i * 0.1, (i + 1) * 0.1)
                 for i, m in enumerate(re.finditer(r"\S+", text))]
        return AudioResult(self.audio, 24000, {**self.metadata, "token_spans": (
            spans if self.spans is None else self.spans
        )})

    def synthesize_stream(self, text, **kwargs):
        self.calls.append((text, kwargs))
        yield AudioChunk(self.audio, 24000)


@pytest.fixture
def setup_client(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path / "catalog"))
    model = FakeModel()

    def make(capabilities=None):
        registry = ModalityRegistry()
        registry.register("audio/speech", model, manifest={"capabilities": (
            {"supports_word_timestamps": True} if capabilities is None else capabilities
        )})
        app = create_app(registry=registry, routers={"audio/speech": routes.build_router(registry)})
        return TestClient(app)

    return model, make


def post(client, **kwargs):
    return client.post("/v1/audio/speech", json={"input": "hello world", "model": "fake-timed", **kwargs})


def test_plain_request_body_unchanged(setup_client):
    model, make = setup_client
    response = post(make())
    assert response.status_code == 200
    assert response.content == audio_to_wav_bytes(model.audio, 24000)
    assert response.headers["content-type"] == "audio/wav"
    assert model.calls == [("hello world", {"voice": None, "speed": 1.0, "align": False})]
    assert response.headers["X-Muse-Speed"] == "1.0"
    assert response.headers["X-Muse-Voice"] == "af_heart"
    assert response.headers["X-Muse-Audio-SHA256"] == hashlib.sha256(response.content).hexdigest()
    assert "X-Muse-Pronunciations-Digest" not in response.headers


def test_timestamps_json_shape(setup_client):
    model, make = setup_client
    model.metadata["timing_calibration"] = {"offset_seconds": 0.01}
    response = post(make(), timestamps=True, speed=1.25)
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    data = response.json()
    assert set(data) == {"object", "model", "voice", "speed", "audio", "words", "timing"}
    assert data["object"] == "audio.speech"
    assert data["model"] == "fake-timed"
    assert data["voice"] == "af_heart"
    assert data["speed"] == 1.25
    assert set(data["audio"]) == {"format", "b64_json", "sha256", "sample_rate", "duration_seconds"}
    assert data["audio"]["format"] == "wav"
    assert data["audio"]["sample_rate"] == 24000
    assert data["audio"]["duration_seconds"] == 1.0
    assert data["audio"]["sha256"] == hashlib.sha256(base64.b64decode(data["audio"]["b64_json"])).hexdigest()
    assert response.headers["X-Muse-Audio-SHA256"] == data["audio"]["sha256"]
    assert len(data["words"]) == len("hello world".split())
    assert data["words"][0] == {"word": "hello", "char_start": 0, "char_end": 5,
                                "start": 0.0, "end": 0.1, "measured": True}
    assert data["timing"] == {"source": "native", "measured_words": 2, "estimated_words": 0,
                              "calibration": {"offset_seconds": 0.01}}
    assert model.calls[0][1]["timestamps"] is True


def test_timestamps_with_pronunciations_keyed_to_input(setup_client):
    model, make = setup_client
    raw = [{"term": "monoid", "respell": "mah noyd"}]
    response = post(make(), input="the monoid law", timestamps=True, pronunciations=raw)
    assert response.status_code == 200
    assert model.calls[0][0] == "the mah noyd law"
    data = response.json()
    assert [word["word"] for word in data["words"]] == ["the", "monoid", "law"]
    assert data["words"][1] == {"word": "monoid", "char_start": 4, "char_end": 10,
                                "start": 0.1, "end": 0.3, "measured": True}
    assert data["pronunciations"] == {"digest": pronunciations_digest(raw), "applied": [
        {"index": 0, "term": "monoid", "kind": "respell", "count": 1}]}
    assert response.headers["X-Muse-Pronunciations-Digest"] == pronunciations_digest(raw)


def test_timestamps_not_supported(setup_client):
    model, make = setup_client
    response = post(make({}), timestamps=True, pronunciations=[{}])
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "timestamps_not_supported"
    assert "/v1/audio/alignments" in response.json()["error"]["message"]
    assert model.calls == []


def test_timestamps_not_streamable(setup_client):
    _, make = setup_client
    response = post(make({}), timestamps=True, stream=True)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "timestamps_not_streamable"


@pytest.mark.parametrize("kinds,renderer", [([], True), (["ipa"], False)])
def test_ipa_not_supported_names_index(setup_client, kinds, renderer):
    model, make = setup_client
    if renderer:
        model.render_ipa = lambda original, ipa: f"[{original}](/{ipa}/)"
    response = post(make({"pronunciation_kinds": kinds}), pronunciations=[
        {"term": "hello", "respell": "hullo"}, {"term": "unused", "ipa": "xyz"}])
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "pronunciation_not_supported"
    assert "1" in response.json()["error"]["message"]
    assert model.calls == []


@pytest.mark.parametrize("raw,index", [({}, None), ([{}], 0), ([{"term": "x" * 5000, "respell": "x"}], 0),
    ([{"spans": [[0.5, 2]], "respell": "x"}], 0), ([{"spans": [[-1, 2]], "respell": "x"}], 0),
    ([{"term": "a", "respell": "b"}] * 10000, None)])
def test_invalid_pronunciation_400(setup_client, raw, index):
    model, make = setup_client
    response = post(make(), pronunciations=raw)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_pronunciation"
    if index is not None:
        assert str(index) in response.json()["error"]["message"]
    assert model.calls == []


def test_timestamps_opus_fallback_reports_wav(setup_client, monkeypatch):
    _, make = setup_client
    def fail(wav):
        raise AudioFormatError("encoder unavailable")
    monkeypatch.setattr(routes, "wav_bytes_to_opus", fail)
    response = post(make(), timestamps=True, response_format="opus")
    assert response.status_code == 200
    audio = response.json()["audio"]
    assert audio["format"] == "wav"
    assert base64.b64decode(audio["b64_json"]).startswith(b"RIFF")
    assert audio["sha256"] == hashlib.sha256(base64.b64decode(audio["b64_json"])).hexdigest()


def test_respell_on_model_without_kinds(setup_client):
    model, make = setup_client
    response = post(make({}), pronunciations=[{"term": "hello", "respell": "hullo"}])
    assert response.status_code == 200
    assert model.calls[0][0] == "hullo world"


def test_stream_with_pronunciations_sends_spoken_text_and_digest_header(setup_client):
    model, make = setup_client
    raw = [{"term": "hello", "respell": "hullo"}]
    response = post(make(), stream=True, pronunciations=raw, speed=1.5)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert model.calls == [("hullo world", {"voice": None, "speed": 1.5})]
    assert response.headers["X-Muse-Pronunciations-Digest"] == pronunciations_digest(raw)
    assert response.headers["X-Muse-Voice"] == "af_heart"
    assert response.headers["X-Muse-Speed"] == "1.5"
    assert "X-Muse-Audio-SHA256" not in response.headers


def test_align_flag_still_sets_x_alignment(setup_client):
    model, make = setup_client
    model.metadata["alignment"] = [{"start": 0.0, "end": 1.0}]
    response = post(make(), align=True)
    assert response.status_code == 200
    assert json.loads(response.headers["X-Alignment"]) == model.metadata["alignment"]
    assert model.calls[0][1] == {"voice": None, "speed": 1.0, "align": True}


@pytest.mark.parametrize("timestamps", [False, True])
def test_empty_pronunciations_are_recorded(setup_client, timestamps):
    _, make = setup_client
    response = post(make(), timestamps=timestamps, pronunciations=[])
    assert response.status_code == 200
    assert response.headers["X-Muse-Pronunciations-Digest"] == pronunciations_digest([])
    if timestamps:
        assert response.json()["pronunciations"] == {"digest": pronunciations_digest([]), "applied": []}


def test_ipa_link_spans_map_back_to_input(setup_client):
    model, make = setup_client
    model.render_ipa = lambda original, ipa: f"[{original}](/{ipa}/)"
    model.spans = [TimedSpan(0, 3, 0.0, 0.1), TimedSpan(5, 11, 0.1, 0.5),
                   TimedSpan(19, 22, 0.5, 0.8)]
    response = post(make({"supports_word_timestamps": True, "pronunciation_kinds": ["ipa"]}),
                    input="the monoid law", timestamps=True,
                    pronunciations=[{"term": "monoid", "ipa": "xyz"}])
    assert response.status_code == 200
    assert model.calls[0][0] == "the [monoid](/xyz/) law"
    assert response.json()["words"][1] == {"word": "monoid", "char_start": 4, "char_end": 10,
                                           "start": 0.1, "end": 0.5, "measured": True}
    assert response.json()["words"][2]["word"] == "law"
    assert response.json()["words"][2]["measured"] is True


def test_unmeasured_words_and_written_offsets(setup_client):
    model, make = setup_client
    model.spans = [TimedSpan(2, 7, 0.1, 0.3), TimedSpan(13, 18, 0.6, 0.9)]
    text = "  hello  —\n  world  "
    response = post(make(), input=text, timestamps=True)
    assert response.status_code == 200
    data = response.json()
    assert [word["word"] for word in data["words"]] == text.split()
    for word in data["words"]:
        assert text[word["char_start"]:word["char_end"]] == word["word"]
        assert 0 <= word["start"] <= word["end"] <= data["audio"]["duration_seconds"]
    assert data["timing"] == {"source": "native", "measured_words": 2, "estimated_words": 1,
                              "calibration": None}


def test_language_without_native_timestamps_returns_400(setup_client):
    model, make = setup_client
    model.failure = ValueError("Kokoro timestamps are supported only on English pipelines (a, b)")
    response = post(make(), timestamps=True)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "timestamps_not_supported"
    assert "/v1/audio/alignments" in response.json()["error"]["message"]


@pytest.mark.parametrize("actual,requested,default,expected", [
    ("actual", "requested", "default", "actual"),
    (None, "requested", "default", "requested"),
    (None, None, "default", "default"),
    (None, None, None, None),
])
def test_voice_header_resolves_provenance(setup_client, actual, requested, default, expected):
    model, make = setup_client
    model.metadata["voice"] = actual
    model.default_voice = default
    response = post(make(), voice=requested, timestamps=True)
    assert response.status_code == 200
    assert response.headers.get("X-Muse-Voice") == expected
    assert response.json()["voice"] == expected


def test_model_lookup_precedes_new_validation(setup_client):
    model, make = setup_client
    response = post(make(), model="missing", timestamps=True, stream=True, pronunciations={})
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "model_not_found"
    assert model.calls == []


def test_successful_opus_encoding_reports_actual_bytes(setup_client, monkeypatch):
    _, make = setup_client
    monkeypatch.setattr(routes, "wav_bytes_to_opus", lambda wav: b"OggS fake opus")
    response = post(make(), timestamps=True, response_format="opus")
    assert response.status_code == 200
    audio = response.json()["audio"]
    assert audio["format"] == "opus"
    assert base64.b64decode(audio["b64_json"]) == b"OggS fake opus"
    assert audio["sha256"] == hashlib.sha256(b"OggS fake opus").hexdigest()
