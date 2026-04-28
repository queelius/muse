"""Codec: list[RerankResult] + return_documents -> Cohere envelope."""
from muse.modalities.text_rerank import RerankResult
from muse.modalities.text_rerank.codec import encode_rerank_response


def _sample():
    return [
        RerankResult(index=3, relevance_score=0.97, document_text="alpha"),
        RerankResult(index=0, relevance_score=0.81, document_text="beta"),
    ]


def test_encode_envelope_minimum_shape():
    body = encode_rerank_response(
        _sample(), model_id="bge-reranker-v2-m3", return_documents=False,
    )
    assert body["model"] == "bge-reranker-v2-m3"
    assert body["id"].startswith("rrk-")
    assert body["meta"] == {"billed_units": {"search_units": 1}}
    assert len(body["results"]) == 2


def test_encode_envelope_omits_document_when_flag_false():
    body = encode_rerank_response(
        _sample(), model_id="m", return_documents=False,
    )
    for row in body["results"]:
        assert "document" not in row
        assert "index" in row
        assert "relevance_score" in row


def test_encode_envelope_includes_document_text_when_flag_true():
    body = encode_rerank_response(
        _sample(), model_id="m", return_documents=True,
    )
    assert body["results"][0]["document"] == {"text": "alpha"}
    assert body["results"][1]["document"] == {"text": "beta"}


def test_encode_preserves_input_order():
    """Codec is dumb: it preserves caller order. Sorting + truncation
    happen in the runtime."""
    rows = [
        RerankResult(index=2, relevance_score=0.1, document_text="c"),
        RerankResult(index=0, relevance_score=0.9, document_text="a"),
        RerankResult(index=1, relevance_score=0.5, document_text="b"),
    ]
    body = encode_rerank_response(rows, model_id="m", return_documents=False)
    indices = [r["index"] for r in body["results"]]
    assert indices == [2, 0, 1]


def test_encode_id_unique_per_call():
    a = encode_rerank_response(_sample(), model_id="m", return_documents=False)
    b = encode_rerank_response(_sample(), model_id="m", return_documents=False)
    assert a["id"] != b["id"]


def test_encode_empty_results_is_valid():
    body = encode_rerank_response([], model_id="m", return_documents=False)
    assert body["results"] == []
    assert body["model"] == "m"
    assert body["meta"] == {"billed_units": {"search_units": 1}}
