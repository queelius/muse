"""Protocol + dataclasses for text/rerank."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class RerankResult:
    """One scored (query, document) pair from a rerank call.

    index: position of this document in the request's `documents` array.
           Stable so a client can map back from result row to input.
    relevance_score: float; higher means more relevant. Cross-encoders
           often emit raw logits or sigmoid-normalized scores in [0, 1].
           muse passes through whatever the runtime returns.
    document_text: original document. The codec uses this when the
           request asks `return_documents=True` and drops it otherwise.
           Holding it on the dataclass keeps the codec pure (no need
           to re-index the request).
    """
    index: int
    relevance_score: float
    document_text: str


@runtime_checkable
class RerankerModel(Protocol):
    """Structural protocol any reranker backend satisfies.

    CrossEncoderRuntime (the generic runtime) and the bundled
    bge-reranker-v2-m3 Model satisfy this without inheritance.
    """

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> list[RerankResult]:
        """Score query against each document; return sorted descending.

        When `top_n is None`, returns all documents in score-descending
        order. When set, returns the top-N.
        """
        ...
