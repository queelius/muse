"""Tests for /v1/embeddings FastAPI router."""
import base64
import struct

import pytest
from fastapi.testclient import TestClient

from muse.core.registry import ModalityRegistry
from muse.core.server import create_app
from muse.modalities.embedding_text.protocol import EmbeddingResult
from muse.modalities.embedding_text.routes import build_router


class FakeEmbeddingsModel:
    """Protocol-compatible stub that returns deterministic vectors."""
    model_id = "fake-embed"
    dimensions = 4

    def embed(self, input, *, dimensions=None, **_):
        texts = [input] if isinstance(input, str) else list(input)
        vectors = [[float(i), 0.0, 0.0, 0.0] for i in range(len(texts))]
        if dimensions is not None and dimensions < self.dimensions:
            vectors = [v[:dimensions] for v in vectors]
            out_dim = dimensions
        else:
            out_dim = self.dimensions
        return EmbeddingResult(
            embeddings=vectors,
            dimensions=out_dim,
            model_id=self.model_id,
            prompt_tokens=sum(len(t.split()) for t in texts),
        )


@pytest.fixture
def client():
    reg = ModalityRegistry()
    reg.register("embedding/text", FakeEmbeddingsModel())
    app = create_app(registry=reg, routers={"embedding/text": build_router(reg)})
    return TestClient(app)


def test_single_string_input_returns_one_embedding(client):
    r = client.post("/v1/embeddings", json={"input": "hello", "model": "fake-embed"})
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    assert len(body["data"]) == 1
    entry = body["data"][0]
    assert entry["object"] == "embedding"
    assert entry["index"] == 0
    assert entry["embedding"] == [0.0, 0.0, 0.0, 0.0]
    assert body["model"] == "fake-embed"


def test_list_input_returns_embeddings_in_order(client):
    r = client.post("/v1/embeddings", json={
        "input": ["first", "second", "third"],
        "model": "fake-embed",
    })
    assert r.status_code == 200
    data = r.json()["data"]
    assert len(data) == 3
    # FakeEmbeddingsModel returns [i, 0, 0, 0] for the i-th input
    for i, entry in enumerate(data):
        assert entry["index"] == i
        assert entry["embedding"][0] == float(i)


def test_response_includes_usage(client):
    r = client.post("/v1/embeddings", json={
        "input": ["hello world", "foo bar baz"],
        "model": "fake-embed",
    })
    body = r.json()
    assert "usage" in body
    assert body["usage"]["prompt_tokens"] == 5  # 2 + 3 split words
    assert body["usage"]["total_tokens"] == 5


def test_default_model_when_unspecified(client):
    r = client.post("/v1/embeddings", json={"input": "hello"})
    assert r.status_code == 200
    # Default model was the only registered one
    assert r.json()["model"] == "fake-embed"


def test_unknown_model_returns_openai_shape_404(client):
    r = client.post("/v1/embeddings", json={
        "input": "hi", "model": "no-such-model",
    })
    assert r.status_code == 404
    body = r.json()
    assert "error" in body
    assert "detail" not in body
    assert body["error"]["code"] == "model_not_found"


def test_encoding_format_base64_returns_base64_strings(client):
    r = client.post("/v1/embeddings", json={
        "input": "hello",
        "model": "fake-embed",
        "encoding_format": "base64",
    })
    assert r.status_code == 200
    entry = r.json()["data"][0]
    assert isinstance(entry["embedding"], str)
    # Decode and verify round-trip
    raw = base64.b64decode(entry["embedding"])
    assert len(raw) == 16  # 4 floats * 4 bytes
    vals = [struct.unpack("<f", raw[i:i+4])[0] for i in range(0, 16, 4)]
    for v, expected in zip(vals, [0.0, 0.0, 0.0, 0.0]):
        assert abs(v - expected) < 1e-6


def test_dimensions_truncation(client):
    r = client.post("/v1/embeddings", json={
        "input": "hello",
        "model": "fake-embed",
        "dimensions": 2,
    })
    assert r.status_code == 200
    emb = r.json()["data"][0]["embedding"]
    assert len(emb) == 2


def test_empty_input_string_rejected(client):
    r = client.post("/v1/embeddings", json={"input": "", "model": "fake-embed"})
    assert r.status_code in (400, 422)


def test_empty_input_list_rejected(client):
    r = client.post("/v1/embeddings", json={"input": [], "model": "fake-embed"})
    assert r.status_code in (400, 422)


def test_dimensions_out_of_range_rejected(client):
    r = client.post("/v1/embeddings", json={
        "input": "hi", "model": "fake-embed",
        "dimensions": 99999,
    })
    assert r.status_code in (400, 422)


def test_invalid_encoding_format_rejected(client):
    r = client.post("/v1/embeddings", json={
        "input": "hi", "model": "fake-embed",
        "encoding_format": "bogus",
    })
    assert r.status_code in (400, 422)


def test_user_field_accepted_and_ignored(client):
    """OpenAI allows optional `user` for abuse monitoring; we accept + ignore."""
    r = client.post("/v1/embeddings", json={
        "input": "hi", "model": "fake-embed",
        "user": "user-42",
    })
    assert r.status_code == 200


def test_oversized_batch_returns_400(monkeypatch):
    """Reject batches larger than MUSE_EMBEDDINGS_MAX_BATCH up front."""
    monkeypatch.setenv("MUSE_EMBEDDINGS_MAX_BATCH", "3")
    import importlib
    from muse.modalities.embedding_text import routes as routes_mod
    importlib.reload(routes_mod)

    reg = ModalityRegistry()
    reg.register("embedding/text", FakeEmbeddingsModel())
    app = create_app(
        registry=reg,
        routers={"embedding/text": routes_mod.build_router(reg)},
    )
    client = TestClient(app)

    r = client.post("/v1/embeddings", json={
        "input": ["a", "b", "c", "d"], "model": "fake-embed",
    })
    assert r.status_code == 400
    body = r.json()["error"]
    assert body["code"] == "invalid_parameter"
    assert "MUSE_EMBEDDINGS_MAX_BATCH=3" in body["message"]


def test_oversized_item_returns_400(monkeypatch):
    monkeypatch.setenv("MUSE_EMBEDDINGS_MAX_CHARS_PER_ITEM", "10")
    import importlib
    from muse.modalities.embedding_text import routes as routes_mod
    importlib.reload(routes_mod)

    reg = ModalityRegistry()
    reg.register("embedding/text", FakeEmbeddingsModel())
    app = create_app(
        registry=reg,
        routers={"embedding/text": routes_mod.build_router(reg)},
    )
    client = TestClient(app)

    r = client.post("/v1/embeddings", json={
        "input": "x" * 100, "model": "fake-embed",
    })
    assert r.status_code == 400
    assert "MUSE_EMBEDDINGS_MAX_CHARS_PER_ITEM=10" in r.json()["error"]["message"]


# ---------------- config-hierarchy migration (limits.embeddings_max_batch) ----------------


def test_max_batch_is_read_live_per_request_not_frozen_at_import(monkeypatch):
    """Finding 3 (v0.58.1 review): embedding_text/routes.py used to
    capture _MAX_BATCH as an import-time module constant, so a
    `muse config set` / env change was invisible to an already-running
    worker until it restarted. The cap must now be read per-request:
    change the env BETWEEN two requests (no module reload, no process
    restart) and confirm the second request honors the new, lower
    MUSE_EMBEDDINGS_MAX_BATCH cap."""
    from muse.core import config as cfg

    reg = ModalityRegistry()
    reg.register("embedding/text", FakeEmbeddingsModel())
    app = create_app(
        registry=reg, routers={"embedding/text": build_router(reg)},
    )
    client = TestClient(app)

    # Before any override: a 2-item batch is well under the default cap.
    r1 = client.post("/v1/embeddings", json={
        "input": ["a", "b"], "model": "fake-embed",
    })
    assert r1.status_code == 200, r1.text

    # Live reconfiguration -- no importlib.reload, no new client/app.
    monkeypatch.setenv("MUSE_EMBEDDINGS_MAX_BATCH", "1")
    cfg.reset_config()
    try:
        r2 = client.post("/v1/embeddings", json={
            "input": ["a", "b"], "model": "fake-embed",
        })
        assert r2.status_code == 400, r2.text
        assert "MUSE_EMBEDDINGS_MAX_BATCH=1" in r2.json()["error"]["message"]
    finally:
        monkeypatch.delenv("MUSE_EMBEDDINGS_MAX_BATCH", raising=False)
        cfg.reset_config()


def test_embed_backend_error_returns_500_envelope():
    """L14: a backend exception surfaces as the OpenAI 500 envelope."""
    class _RaisingModel:
        model_id = "fake-embed"
        dimensions = 4

        def embed(self, input, *, dimensions=None, **_):
            raise RuntimeError("model exploded")

    reg = ModalityRegistry()
    reg.register("embedding/text", _RaisingModel())
    app = create_app(registry=reg, routers={"embedding/text": build_router(reg)})
    client = TestClient(app, raise_server_exceptions=False)

    r = client.post("/v1/embeddings", json={"input": "hello", "model": "fake-embed"})
    assert r.status_code == 500
    assert r.json()["error"]["code"] == "internal_error"
