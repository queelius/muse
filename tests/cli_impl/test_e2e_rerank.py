"""End-to-end: /v1/rerank through FastAPI + codec correctly.

Uses a fake RerankerModel backend; no real weights.
"""
import pytest
from fastapi.testclient import TestClient

from muse.core.registry import ModalityRegistry
from muse.core.server import create_app
from muse.modalities.text_rerank import (
    MODALITY,
    RerankResult,
    build_router,
)


pytestmark = pytest.mark.slow


class _FakeReranker:
    def __init__(self):
        self.called_with = None
        self.model_id = "fake-rerank"

    def rerank(self, query, documents, top_n=None):
        self.called_with = (query, documents, top_n)
        # Trivial rule: longer documents score higher (deterministic).
        scored = sorted(
            [(i, float(len(d))) for i, d in enumerate(documents)],
            key=lambda kv: kv[1], reverse=True,
        )
        if top_n is not None:
            scored = scored[:top_n]
        return [
            RerankResult(
                index=i, relevance_score=s, document_text=documents[i],
            )
            for i, s in scored
        ]


@pytest.mark.timeout(10)
def test_rerank_full_request_response_cycle():
    fake = _FakeReranker()
    reg = ModalityRegistry()
    reg.register(MODALITY, fake, manifest={"model_id": "fake-rerank"})
    app = create_app(registry=reg, routers={MODALITY: build_router(reg)})

    client = TestClient(app)
    r = client.post("/v1/rerank", json={
        "query": "what is muse?",
        "documents": ["short", "a longer document with more text", "mid-length"],
        "top_n": 2,
        "return_documents": True,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == "fake-rerank"
    assert body["id"].startswith("rrk-")
    assert body["meta"]["billed_units"]["search_units"] == 1
    # top_n=2 -> only 2 results
    assert len(body["results"]) == 2
    # Index 1 is the longest document; should be ranked first.
    assert body["results"][0]["index"] == 1
    assert body["results"][0]["document"] == {
        "text": "a longer document with more text",
    }
    # Scores are sorted descending.
    scores = [r["relevance_score"] for r in body["results"]]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.timeout(10)
def test_rerank_e2e_omits_documents_when_flag_false():
    fake = _FakeReranker()
    reg = ModalityRegistry()
    reg.register(MODALITY, fake, manifest={"model_id": "fake-rerank"})
    app = create_app(registry=reg, routers={MODALITY: build_router(reg)})

    client = TestClient(app)
    r = client.post("/v1/rerank", json={
        "query": "q",
        "documents": ["a", "bb"],
    })
    assert r.status_code == 200, r.text
    body = r.json()
    for row in body["results"]:
        assert "document" not in row
