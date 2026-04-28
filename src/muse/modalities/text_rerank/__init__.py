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
    "PROBE_DEFAULTS",
    "build_router",
    "RerankResult",
    "RerankerModel",
    "RerankClient",
]
