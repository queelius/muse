"""Wire codec: ChatResult/ChatChunk <-> OpenAI JSON wire shapes + SSE."""
from __future__ import annotations

import json

from muse.modalities.chat_completion.protocol import ChatChunk, ChatResult


DONE_SENTINEL = "[DONE]"
"""Sentinel sent as the last SSE `data:` line in a streaming response.
Clients stop reading at this marker. OpenAI uses the exact same string."""


def chunk_to_openai_dict(
    chunk: ChatChunk, *, system_fingerprint: str | None = None,
) -> dict:
    """Map a ChatChunk to the OpenAI ChatCompletionChunk wire dict."""
    out = {
        "id": chunk.id,
        "object": "chat.completion.chunk",
        "created": chunk.created,
        "model": chunk.model_id,
        "choices": [{
            "index": chunk.choice_index,
            "delta": chunk.delta,
            "finish_reason": chunk.finish_reason,
        }],
    }
    if system_fingerprint:
        out["system_fingerprint"] = system_fingerprint
    return out


def chunk_to_sse_data(
    chunk: ChatChunk, *, system_fingerprint: str | None = None,
) -> str:
    """JSON-encode a chunk ready for an SSE `data:` line.

    Returns only the JSON body; the transport layer (EventSourceResponse
    in routes.py) handles the `data:` field and event separator.
    """
    return json.dumps(
        chunk_to_openai_dict(chunk, system_fingerprint=system_fingerprint),
        separators=(",", ":"),
    )


def result_to_openai_dict(
    result: ChatResult, *, system_fingerprint: str | None = None,
) -> dict:
    """Map a ChatResult to the OpenAI ChatCompletion wire dict."""
    out = {
        "id": result.id,
        "object": "chat.completion",
        "created": result.created,
        "model": result.model_id,
        "choices": [{
            "index": c.index,
            "message": c.message,
            "finish_reason": c.finish_reason,
        } for c in result.choices],
        "usage": result.usage,
    }
    if system_fingerprint:
        out["system_fingerprint"] = system_fingerprint
    return out
