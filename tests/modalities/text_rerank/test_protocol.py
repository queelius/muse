"""Protocol + dataclass shape tests for text/rerank."""
from muse.modalities.text_rerank import (
    MODALITY,
    RerankResult,
    RerankerModel,
)


def test_modality_tag_is_mime_shaped():
    assert MODALITY == "text/rerank"


def test_rerank_result_minimal():
    r = RerankResult(index=0, relevance_score=0.9, document_text="hello")
    assert r.index == 0
    assert r.relevance_score == 0.9
    assert r.document_text == "hello"


def test_rerank_result_supports_negative_scores():
    """Cross-encoder logits can be negative pre-sigmoid; runtime may pass through."""
    r = RerankResult(index=2, relevance_score=-3.4, document_text="x")
    assert r.relevance_score == -3.4


def test_reranker_protocol_accepts_structural_impl():
    class Fake:
        def rerank(self, query, documents, top_n=None):
            return [RerankResult(index=0, relevance_score=1.0, document_text="x")]
    assert isinstance(Fake(), RerankerModel)


def test_reranker_protocol_rejects_missing_method():
    class Missing:
        pass
    assert not isinstance(Missing(), RerankerModel)
