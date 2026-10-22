"""text/rerank modality.

Public surface:
  - MODALITY: str (MIME-shaped tag; used by discover_modalities)
  - build_router(registry) -> APIRouter (mounted by the worker)
  - RerankResult dataclass
  - RerankerModel Protocol
  - RerankClient
  - PROBE_DEFAULTS

Wire contract (Cohere-compat):
  - POST /v1/rerank

This is muse's first modality with a Cohere-shape wire envelope rather
than OpenAI-compat. OpenAI has no rerank API; Cohere's /v1/rerank is
the de-facto standard, and downstream tooling (LangChain, LlamaIndex,
Haystack) expects it.
"""
from muse.modalities.text_rerank.client import RerankClient
from muse.modalities.text_rerank.protocol import (
    RerankResult,
    RerankerModel,
)
from muse.modalities.text_rerank.routes import build_router


MODALITY = "text/rerank"

DISCOVERY = {
    "summary": (
        "Reranking. Scores a list of documents against a query so the best "
        "matches come first."
    ),
    "use_for": [
        "ordering retrieved passages by relevance to a question",
        "second-stage ranking after vector search",
    ],
    "not_for": [
        "generating vectors (use embedding/text)",
        "answering questions (use chat/completion)",
    ],
    "routes": [
        {
            "method": "POST",
            "path": "/v1/rerank",
            "request_content_type": "application/json",
            "response_content_type": "application/json",
            "compat": "cohere:rerank",
            "mcp_tool": "muse_rerank",
            "file_fields": [],
            "media_fields": [],
            "example": {
                "model": "<model>",
                "query": "what is muse",
                "documents": ["a media server", "a fruit"],
                "top_n": 2,
            },
        },
    ],
}


# Per-modality probe defaults read by `muse models probe`.
PROBE_DEFAULTS = {
    "shape": "1 query, 4 documents",
    "call": lambda m: m.rerank(
        "what is muse?",
        [
            "muse is an audio server",
            "muse is a server",
            "purple cats",
            "model serving",
        ],
        None,
    ),
}


__all__ = [
    "MODALITY",
    "DISCOVERY",
    "PROBE_DEFAULTS",
    "build_router",
    "RerankResult",
    "RerankerModel",
    "RerankClient",
]
