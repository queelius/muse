"""Embedding text modality: text-to-vector.

Wire contract: POST /v1/embeddings with {input (str | list[str]), model,
encoding_format? ('float' | 'base64'), dimensions?} returns list of
embedding vectors in OpenAI-compatible shape.

Models declaring `modality = "embedding/text"` in their MANIFEST and
satisfying the EmbeddingsModel protocol plug into this modality.
"""
from muse.modalities.embedding_text.client import EmbeddingsClient
from muse.modalities.embedding_text.protocol import (
    EmbeddingResult,
    EmbeddingsModel,
)
from muse.modalities.embedding_text.routes import build_router

MODALITY = "embedding/text"

DISCOVERY = {
    "summary": (
        "Text embeddings. Turns text into vectors for semantic search, "
        "clustering and retrieval."
    ),
    "use_for": ["indexing documents for semantic search", "measuring similarity between texts"],
    "not_for": [
        "ranking a candidate list against a query (use text/rerank)",
        "image or audio vectors",
    ],
    "routes": [
        {
            "method": "POST",
            "path": "/v1/embeddings",
            "request_content_type": "application/json",
            "response_content_type": "application/json",
            "compat": "openai:embeddings",
            "mcp_tool": "muse_embed_text",
            "file_fields": [],
            "media_fields": [],
            "example": {"model": "<model>", "input": ["alpha", "beta"]},
        },
    ],
}

# Per-modality probe defaults read by `muse models probe`.
PROBE_DEFAULTS = {
    "shape": "1 short string",
    "call": lambda m: m.embed(["probe text"]),
}

__all__ = [
    "MODALITY",
    "DISCOVERY",
    "PROBE_DEFAULTS",
    "build_router",
    "EmbeddingsClient",
    "EmbeddingResult",
    "EmbeddingsModel",
]
