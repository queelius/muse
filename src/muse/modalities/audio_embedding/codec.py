"""Encoding helpers for /v1/audio/embeddings responses.

The wire format mirrors `/v1/embeddings` exactly: float lists when
`encoding_format="float"`, base64-encoded little-endian float32 bytes
when `encoding_format="base64"`. We re-export the existing
embedding_text helpers so the byte layout is bit-identical and
OpenAI SDK clients round-trip cleanly via
`np.frombuffer(decoded_bytes, dtype="<f4")`.
"""
from __future__ import annotations

from muse.modalities.embedding_text.codec import (
    base64_to_embedding,
    embedding_to_base64,
)


__all__ = ["embedding_to_base64", "base64_to_embedding"]
