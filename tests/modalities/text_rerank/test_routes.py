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


# H2 regression guard: backend.rerank must be called under _inference_lock.

def test_rerank_called_under_inference_lock():
    """backend.rerank must execute while _inference_lock is held (H2 fix)."""
    import threading

    class _LockAssertingBackend:
        model_id = "bge-reranker-v2-m3"

        def __init__(self):
            self._inference_lock = threading.Lock()
            self.lock_was_held = False

        def rerank(self, query, documents, top_n):
            self.lock_was_held = self._inference_lock.locked()
            return [RerankResult(index=0, relevance_score=0.9, document_text="x")]

    backend = _LockAssertingBackend()
    reg = ModalityRegistry()
    reg.register(MODALITY, backend, manifest={"model_id": backend.model_id})
    from muse.modalities.text_rerank import build_router as _build
    from muse.core.server import create_app as _create_app
    from fastapi.testclient import TestClient as _TC
    app = _create_app(registry=reg, routers={MODALITY: _build(reg)})
    client = _TC(app)
    r = client.post("/v1/rerank", json={"query": "q", "documents": ["x"]})
    assert r.status_code == 200, r.text
    assert backend.lock_was_held, (
        "backend.rerank was called without holding _inference_lock"
    )


def test_rerank_backend_error_returns_500_envelope():
    """L14: a backend exception must surface as the OpenAI 500 envelope,
    not a bare Starlette 500 that escapes the documented error contract."""
    backend = MagicMock()
    backend.model_id = "bge-reranker-v2-m3"
    backend.rerank.side_effect = RuntimeError("model exploded")
    reg = ModalityRegistry()
    reg.register(MODALITY, backend, manifest={"model_id": "bge-reranker-v2-m3"})
    app = create_app(registry=reg, routers={MODALITY: build_router(reg)})
    client = TestClient(app, raise_server_exceptions=False)

    r = client.post("/v1/rerank", json={"query": "q", "documents": ["a", "b"]})
    assert r.status_code == 500
    body = r.json()
    assert "error" in body
    assert body["error"]["code"] == "internal_error"


# ---------------- config-hierarchy migration (limits.rerank_*, multi-cap file) ----------------


def test_rerank_caps_are_import_time_consts_sourced_from_config(monkeypatch):
    """text_rerank/routes.py declares three import-time cap constants
    (_MAX_DOCUMENTS, _MAX_QUERY_CHARS, _MAX_DOC_CHARS) from the same
    limits.rerank_* keys. Prove each is sourced from muse.core.config,
    not raw os.environ, by reloading the module under a changed env
    and checking the frozen values."""
    from muse.core import config as cfg
    monkeypatch.setenv("MUSE_RERANK_MAX_DOCUMENTS", "17")
    monkeypatch.setenv("MUSE_RERANK_MAX_QUERY_CHARS", "18")
    monkeypatch.setenv("MUSE_RERANK_MAX_DOC_CHARS", "19")
    cfg.reset_config()
    assert cfg.get("limits.rerank_max_documents") == 17
    assert cfg.get("limits.rerank_max_query_chars") == 18
    assert cfg.get("limits.rerank_max_doc_chars") == 19

    import importlib
    from muse.modalities.text_rerank import routes as routes_mod
    importlib.reload(routes_mod)
    try:
        assert routes_mod._MAX_DOCUMENTS == 17
        assert routes_mod._MAX_QUERY_CHARS == 18
        assert routes_mod._MAX_DOC_CHARS == 19
    finally:
        monkeypatch.delenv("MUSE_RERANK_MAX_DOCUMENTS", raising=False)
        monkeypatch.delenv("MUSE_RERANK_MAX_QUERY_CHARS", raising=False)
        monkeypatch.delenv("MUSE_RERANK_MAX_DOC_CHARS", raising=False)
        cfg.reset_config()
        importlib.reload(routes_mod)
