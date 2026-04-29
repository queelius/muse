"""End-to-end: /v1/images/embeddings through FastAPI + codec correctly.

Uses a fake ImageEmbeddingModel backend; no real weights.
"""
import base64
import io

import pytest
from fastapi.testclient import TestClient

from muse.core.registry import ModalityRegistry
from muse.core.server import create_app
from muse.modalities.image_embedding import (
    MODALITY,
    ImageEmbeddingResult,
    build_router,
)
from muse.modalities.image_embedding.codec import base64_to_embedding


pytestmark = pytest.mark.slow

pytest.importorskip("PIL.Image")


def _png_data_url(width=32, height=32, color=(0, 128, 255)):
    from PIL import Image
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


class _FakeImageEmbedder:
    """Deterministic fake: returns a fixed-shape embedding sized by the
    number of input images. Records every call so tests can assert
    pass-through fidelity (n_images preserved, dimensions echoed)."""

    def __init__(self, *, dim=384):
        self.calls = []
        self.model_id = "fake-image-embedder"
        self._dim = dim

    @property
    def dimensions(self):
        return self._dim

    def embed(self, images, *, dimensions=None):
        self.calls.append({"n": len(images), "dimensions": dimensions})
        out_dim = dimensions if dimensions is not None and dimensions < self._dim else self._dim
        embeddings = [
            [float(i) * 0.01 for i in range(out_dim)]
            for _ in images
        ]
        return ImageEmbeddingResult(
            embeddings=embeddings,
            dimensions=out_dim,
            model_id=self.model_id,
            n_images=len(images),
        )


@pytest.mark.timeout(10)
def test_e2e_images_embeddings_full_request_response_cycle():
    fake = _FakeImageEmbedder(dim=4)
    reg = ModalityRegistry()
    reg.register(MODALITY, fake, manifest={"model_id": "fake-image-embedder"})
    app = create_app(registry=reg, routers={MODALITY: build_router(reg)})
    client = TestClient(app)

    r = client.post("/v1/images/embeddings", json={
        "input": _png_data_url(),
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "list"
    assert body["model"] == "fake-image-embedder"
    assert len(body["data"]) == 1
    assert len(body["data"][0]["embedding"]) == 4
    assert body["usage"]["prompt_tokens"] == 0
    assert body["usage"]["total_tokens"] == 0


@pytest.mark.timeout(10)
def test_e2e_images_embeddings_batched_input_preserves_order():
    fake = _FakeImageEmbedder(dim=4)
    reg = ModalityRegistry()
    reg.register(MODALITY, fake, manifest={"model_id": "fake-image-embedder"})
    app = create_app(registry=reg, routers={MODALITY: build_router(reg)})
    client = TestClient(app)

    r = client.post("/v1/images/embeddings", json={
        "input": [
            _png_data_url(color=(255, 0, 0)),
            _png_data_url(color=(0, 255, 0)),
            _png_data_url(color=(0, 0, 255)),
        ],
    })
    assert r.status_code == 200
    body = r.json()
    assert len(body["data"]) == 3
    assert [d["index"] for d in body["data"]] == [0, 1, 2]


@pytest.mark.timeout(10)
def test_e2e_images_embeddings_dimensions_truncation():
    """Backend respects request.dimensions; envelope echoes truncated dim."""
    fake = _FakeImageEmbedder(dim=384)
    reg = ModalityRegistry()
    reg.register(MODALITY, fake, manifest={"model_id": "fake-image-embedder"})
    app = create_app(registry=reg, routers={MODALITY: build_router(reg)})
    client = TestClient(app)

    r = client.post("/v1/images/embeddings", json={
        "input": _png_data_url(),
        "dimensions": 128,
    })
    body = r.json()
    assert len(body["data"][0]["embedding"]) == 128


@pytest.mark.timeout(10)
def test_e2e_images_embeddings_base64_encoding_roundtrip():
    fake = _FakeImageEmbedder(dim=8)
    reg = ModalityRegistry()
    reg.register(MODALITY, fake, manifest={"model_id": "fake-image-embedder"})
    app = create_app(registry=reg, routers={MODALITY: build_router(reg)})
    client = TestClient(app)

    r = client.post("/v1/images/embeddings", json={
        "input": _png_data_url(),
        "encoding_format": "base64",
    })
    body = r.json()
    enc = body["data"][0]["embedding"]
    assert isinstance(enc, str)
    decoded = base64_to_embedding(enc)
    assert len(decoded) == 8


@pytest.mark.timeout(10)
def test_e2e_images_embeddings_400_on_bad_input():
    fake = _FakeImageEmbedder()
    reg = ModalityRegistry()
    reg.register(MODALITY, fake, manifest={"model_id": "fake-image-embedder"})
    app = create_app(registry=reg, routers={MODALITY: build_router(reg)})
    client = TestClient(app)

    r = client.post("/v1/images/embeddings", json={
        "input": "data:image/png;base64,!!!corrupt!!!",
    })
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_parameter"


@pytest.mark.timeout(10)
def test_e2e_images_embeddings_404_on_unknown_model():
    fake = _FakeImageEmbedder()
    reg = ModalityRegistry()
    reg.register(MODALITY, fake, manifest={"model_id": "fake-image-embedder"})
    app = create_app(registry=reg, routers={MODALITY: build_router(reg)})
    client = TestClient(app)

    r = client.post("/v1/images/embeddings", json={
        "input": _png_data_url(),
        "model": "nope",
    })
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "model_not_found"
