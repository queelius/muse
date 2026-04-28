"""Integration tests for /v1/rerank against a live muse server.

Opt-in via MUSE_REMOTE_SERVER. Skipped when the server is unreachable
or no rerank model is loaded.

Targets the configurable rerank model id (default bge-reranker-v2-m3,
override via MUSE_RERANK_MODEL_ID).
"""
from __future__ import annotations

from muse.modalities.text_rerank import RerankClient


def test_protocol_basic_rerank(remote_url, rerank_model):
    """Hard claim: a rerank with 4 candidates returns 4 sorted results."""
    client = RerankClient(remote_url)
    out = client.rerank(
        query="what is muse?",
        documents=[
            "muse is an audio server",
            "muse is a generation server with multiple modalities",
            "purple cats are illegal in some places",
            "model serving with HTTP APIs",
        ],
        model=rerank_model,
    )
    assert "results" in out
    assert len(out["results"]) == 4
    scores = [r["relevance_score"] for r in out["results"]]
    assert scores == sorted(scores, reverse=True)
    indices = {r["index"] for r in out["results"]}
    assert indices == {0, 1, 2, 3}


def test_protocol_top_n_truncates(remote_url, rerank_model):
    client = RerankClient(remote_url)
    out = client.rerank(
        query="q",
        documents=["a", "b", "c", "d", "e"],
        top_n=2,
        model=rerank_model,
    )
    assert len(out["results"]) == 2


def test_protocol_return_documents_includes_text(remote_url, rerank_model):
    client = RerankClient(remote_url)
    out = client.rerank(
        query="q",
        documents=["alpha", "beta"],
        return_documents=True,
        model=rerank_model,
    )
    for row in out["results"]:
        assert "document" in row
        assert "text" in row["document"]


def test_protocol_return_documents_default_false(remote_url, rerank_model):
    client = RerankClient(remote_url)
    out = client.rerank(
        query="q",
        documents=["alpha", "beta"],
        model=rerank_model,
    )
    for row in out["results"]:
        assert "document" not in row


def test_protocol_envelope_meta_present(remote_url, rerank_model):
    """Cohere SDK compatibility: envelope must include meta.billed_units."""
    client = RerankClient(remote_url)
    out = client.rerank(
        query="q", documents=["a"], model=rerank_model,
    )
    assert "meta" in out
    assert "billed_units" in out["meta"]
    assert out["meta"]["billed_units"]["search_units"] == 1


def test_observe_relevance_score_for_clearly_relevant_doc(remote_url, rerank_model):
    """Spot-check that the model returns sensible relevance scores."""
    client = RerankClient(remote_url)
    out = client.rerank(
        query="capital of France",
        documents=[
            "Paris is the capital of France.",
            "Bananas are yellow.",
        ],
        model=rerank_model,
    )
    rows = sorted(
        out["results"], key=lambda r: r["relevance_score"], reverse=True,
    )
    # Top-scored row should be the Paris document (index 0).
    assert rows[0]["index"] == 0
