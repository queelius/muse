"""End-to-end: /v1/audio/embeddings through FastAPI + codec correctly.

Uses a fake AudioEmbeddingModel backend; no real weights, no real
librosa decode (mock returns canned vectors directly from the bytes
list).
"""
import pytest
from fastapi.testclient import TestClient

from muse.core.registry import ModalityRegistry
from muse.core.server import create_app
from muse.modalities.audio_embedding import (
    MODALITY,
    AudioEmbeddingResult,
    build_router,
)
from muse.modalities.audio_embedding.codec import base64_to_embedding


pytestmark = pytest.mark.slow


class _FakeAudioEmbedder:
    """Deterministic fake: returns a fixed-shape embedding sized by the
    number of input clips. Records every call so tests can assert
    pass-through fidelity (n_audio_clips preserved, dimensions echoed)."""

    def __init__(self, *, dim=768):
        self.calls = []
        self.model_id = "fake-audio-embedder"
        self._dim = dim

    @property
    def dimensions(self):
        return self._dim

    def embed(self, audio_bytes_list):
        self.calls.append({"n": len(audio_bytes_list)})
        embeddings = [
            [float(i) * 0.01 for i in range(self._dim)]
            for _ in audio_bytes_list
        ]
        return AudioEmbeddingResult(
            embeddings=embeddings,
            dimensions=self._dim,
            model_id=self.model_id,
            n_audio_clips=len(audio_bytes_list),
        )


@pytest.mark.timeout(10)
def test_e2e_audio_embeddings_full_request_response_cycle():
    fake = _FakeAudioEmbedder(dim=4)
    reg = ModalityRegistry()
    reg.register(MODALITY, fake, manifest={"model_id": "fake-audio-embedder"})
    app = create_app(registry=reg, routers={MODALITY: build_router(reg)})
    client = TestClient(app)

    r = client.post(
        "/v1/audio/embeddings",
        files={"file": ("a.wav", b"FAKEWAV", "audio/wav")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "list"
    assert body["model"] == "fake-audio-embedder"
    assert len(body["data"]) == 1
    assert len(body["data"][0]["embedding"]) == 4
    assert body["usage"]["prompt_tokens"] == 0
    assert body["usage"]["total_tokens"] == 0


@pytest.mark.timeout(10)
def test_e2e_audio_embeddings_batched_input_preserves_order():
    fake = _FakeAudioEmbedder(dim=4)
    reg = ModalityRegistry()
    reg.register(MODALITY, fake, manifest={"model_id": "fake-audio-embedder"})
    app = create_app(registry=reg, routers={MODALITY: build_router(reg)})
    client = TestClient(app)

    r = client.post(
        "/v1/audio/embeddings",
        files=[
            ("file", ("a.wav", b"AUDIOA", "audio/wav")),
            ("file", ("b.wav", b"AUDIOB", "audio/wav")),
            ("file", ("c.wav", b"AUDIOC", "audio/wav")),
        ],
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["data"]) == 3
    assert [d["index"] for d in body["data"]] == [0, 1, 2]


@pytest.mark.timeout(10)
def test_e2e_audio_embeddings_base64_encoding_roundtrip():
    fake = _FakeAudioEmbedder(dim=8)
    reg = ModalityRegistry()
    reg.register(MODALITY, fake, manifest={"model_id": "fake-audio-embedder"})
    app = create_app(registry=reg, routers={MODALITY: build_router(reg)})
    client = TestClient(app)

    r = client.post(
        "/v1/audio/embeddings",
        files={"file": ("a.wav", b"FAKEWAV", "audio/wav")},
        data={"encoding_format": "base64"},
    )
    body = r.json()
    enc = body["data"][0]["embedding"]
    assert isinstance(enc, str)
    decoded = base64_to_embedding(enc)
    assert len(decoded) == 8


@pytest.mark.timeout(10)
def test_e2e_audio_embeddings_400_on_empty_file():
    fake = _FakeAudioEmbedder()
    reg = ModalityRegistry()
    reg.register(MODALITY, fake, manifest={"model_id": "fake-audio-embedder"})
    app = create_app(registry=reg, routers={MODALITY: build_router(reg)})
    client = TestClient(app)

    r = client.post(
        "/v1/audio/embeddings",
        files={"file": ("a.wav", b"", "audio/wav")},
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_parameter"


@pytest.mark.timeout(10)
def test_e2e_audio_embeddings_404_on_unknown_model():
    fake = _FakeAudioEmbedder()
    reg = ModalityRegistry()
    reg.register(MODALITY, fake, manifest={"model_id": "fake-audio-embedder"})
    app = create_app(registry=reg, routers={MODALITY: build_router(reg)})
    client = TestClient(app)

    r = client.post(
        "/v1/audio/embeddings",
        files={"file": ("a.wav", b"FAKEWAV", "audio/wav")},
        data={"model": "nope"},
    )
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "model_not_found"


@pytest.mark.timeout(10)
def test_e2e_audio_embeddings_backend_receives_raw_bytes():
    """Backend gets the actual upload bytes, not UploadFile shells."""
    fake = _FakeAudioEmbedder(dim=4)
    reg = ModalityRegistry()
    reg.register(MODALITY, fake, manifest={"model_id": "fake-audio-embedder"})
    app = create_app(registry=reg, routers={MODALITY: build_router(reg)})
    client = TestClient(app)

    client.post(
        "/v1/audio/embeddings",
        files=[
            ("file", ("a.wav", b"AUDIOA", "audio/wav")),
            ("file", ("b.wav", b"AUDIOB", "audio/wav")),
        ],
    )
    assert len(fake.calls) == 1
    assert fake.calls[0]["n"] == 2
