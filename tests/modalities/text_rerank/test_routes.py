"""Route tests for POST /v1/rerank."""
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from muse.core.registry import ModalityRegistry
from muse.core.server import create_app
from muse.modalities.text_rerank import (
    MODALITY,
    RerankResult,
    build_router,
)


def _make_client(backend, manifest=None):
    reg = ModalityRegistry()
    if manifest is None:
        manifest = {"model_id": "bge-reranker-v2-m3"}
    reg.register(MODALITY, backend, manifest=manifest)
    app = create_app(registry=reg, routers={MODALITY: build_router(reg)})
    return TestClient(app)


def _fake_backend(results):
    backend = MagicMock()
    backend.model_id = "bge-reranker-v2-m3"
    backend.rerank.return_value = results
    return backend


def test_rerank_returns_envelope_for_minimal_request():
    backend = _fake_backend([
        RerankResult(index=1, relevance_score=0.9, document_text="b"),
        RerankResult(index=0, relevance_score=0.1, document_text="a"),
    ])
    client = _make_client(backend)

    r = client.post("/v1/rerank", json={
        "query": "q",
        "documents": ["a", "b"],
    })
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "bge-reranker-v2-m3"
    assert body["id"].startswith("rrk-")
    assert body["meta"] == {"billed_units": {"search_units": 1}}
    assert len(body["results"]) == 2
    assert body["results"][0]["index"] == 1
    assert body["results"][0]["relevance_score"] == 0.9
    assert "document" not in body["results"][0]


def test_rerank_includes_documents_when_flag_true():
    backend = _fake_backend([
        RerankResult(index=0, relevance_score=0.7, document_text="alpha"),
    ])
    client = _make_client(backend)

    r = client.post("/v1/rerank", json={
        "query": "q",
        "documents": ["alpha"],
        "return_documents": True,
    })
    body = r.json()
    assert body["results"][0]["document"] == {"text": "alpha"}


def test_rerank_top_n_passed_to_backend():
    backend = _fake_backend([
        RerankResult(index=0, relevance_score=0.7, document_text="x"),
    ])
    client = _make_client(backend)

    r = client.post("/v1/rerank", json={
        "query": "q",
        "documents": ["x", "y", "z"],
        "top_n": 2,
    })
    assert r.status_code == 200
    args, kwargs = backend.rerank.call_args
    # rerank(query, documents, top_n) signature
    if "top_n" in kwargs:
        assert kwargs["top_n"] == 2
    else:
        assert args[2] == 2


def test_rerank_400_on_empty_query():
    backend = _fake_backend([])
    client = _make_client(backend)

    r = client.post("/v1/rerank", json={
        "query": "",
        "documents": ["a"],
    })
    assert r.status_code == 400
    body = r.json()
    assert "error" in body
    assert body["error"]["code"] == "invalid_parameter"


def test_rerank_400_on_empty_documents():
    backend = _fake_backend([])
    client = _make_client(backend)

    r = client.post("/v1/rerank", json={
        "query": "q",
        "documents": [],
    })
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_parameter"


def test_rerank_400_on_empty_document_string():
    backend = _fake_backend([])
    client = _make_client(backend)

    r = client.post("/v1/rerank", json={
        "query": "q",
        "documents": ["good", ""],
    })
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_parameter"


def test_rerank_400_on_top_n_zero():
    backend = _fake_backend([])
    client = _make_client(backend)

    r = client.post("/v1/rerank", json={
        "query": "q",
        "documents": ["a"],
        "top_n": 0,
    })
    assert r.status_code in (400, 422)


def test_rerank_404_on_unknown_model():
    backend = _fake_backend([])
    client = _make_client(backend)

    r = client.post("/v1/rerank", json={
        "query": "q",
        "documents": ["a"],
        "model": "nonexistent",
    })
    assert r.status_code == 404
    body = r.json()
    assert "error" in body
    assert body["error"]["code"] == "model_not_found"


def test_rerank_default_model_resolves_first_registered():
    backend = _fake_backend([
        RerankResult(index=0, relevance_score=0.5, document_text="x"),
    ])
    client = _make_client(backend)

    r = client.post("/v1/rerank", json={
        "query": "q",
        "documents": ["x"],
    })
    assert r.status_code == 200
    assert r.json()["model"] == "bge-reranker-v2-m3"
