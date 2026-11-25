"""Wire codec: ChatResult/ChatChunk & CompletionResult/CompletionChunk <-> JSON/SSE."""
from __future__ import annotations

import json
from typing import Any

from muse.modalities.chat_completion.protocol import (
    ChatChunk,
    ChatResult,
    CompletionChunk,
    CompletionResult,
)


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


def completion_result_to_openai_dict(
    result: CompletionResult, *, system_fingerprint: str | None = None,
) -> dict:
    """Map a CompletionResult to the OpenAI text_completion wire dict."""
    out: dict[str, Any] = {
        "id": result.id,
        "object": "text_completion",
        "created": result.created,
        "model": result.model_id,
        "choices": [{
            "text": c.text,
            "index": c.index,
            "logprobs": c.logprobs,
            "finish_reason": c.finish_reason,
        } for c in result.choices],
        "usage": result.usage,
    }
    if system_fingerprint:
        out["system_fingerprint"] = system_fingerprint
    return out


def completion_chunk_to_openai_dict(
    chunk: CompletionChunk, *, system_fingerprint: str | None = None,
) -> dict:
    """Map a CompletionChunk to the OpenAI text_completion streaming dict."""
    out: dict[str, Any] = {
        "id": chunk.id,
        "object": "text_completion",
        "created": chunk.created,
        "model": chunk.model_id,
        "choices": [{
            "text": chunk.text,
            "index": chunk.choice_index,
            "logprobs": None,
            "finish_reason": chunk.finish_reason,
        }],
    }
    if system_fingerprint:
        out["system_fingerprint"] = system_fingerprint
    return out


def completion_chunk_to_sse_data(
    chunk: CompletionChunk, *, system_fingerprint: str | None = None,
) -> str:
    """JSON-encode a CompletionChunk for an OpenAI SSE `data:` line."""
    return json.dumps(
        completion_chunk_to_openai_dict(
            chunk, system_fingerprint=system_fingerprint,
        ),
        separators=(",", ":"),
    )


def completion_result_to_llama_dict(result: CompletionResult) -> dict:
    """Map a CompletionResult to native llama-server `/completion` JSON."""
    choice = result.choices[0] if result.choices else None
    text = choice.text if choice is not None else ""
    finish_reason = choice.finish_reason if choice is not None else None
    meta = result.metadata or {}
    usage = result.usage or {}

    stopped_eos = (
        bool(meta["stopped_eos"])
        if "stopped_eos" in meta
        else finish_reason == "stop"
    )
    stopped_limit = (
        bool(meta["stopped_limit"])
        if "stopped_limit" in meta
        else finish_reason == "length"
    )
    tokens_evaluated = int(
        meta.get("tokens_evaluated", usage.get("prompt_tokens", 0)) or 0
    )
    tokens_predicted = int(
        meta.get("tokens_predicted", usage.get("completion_tokens", 0)) or 0
    )
    timings = meta.get("timings") if isinstance(meta.get("timings"), dict) else {}

    out: dict[str, Any] = {
        "content": text,
        "stop": True,
        "stopped_eos": stopped_eos,
        "stopped_limit": stopped_limit,
        "tokens_evaluated": tokens_evaluated,
        "tokens_predicted": tokens_predicted,
        "timings": timings,
        "model": result.model_id,
    }
    for extra_key in ("stopped_word", "stopping_word", "stop_type"):
        if extra_key in meta:
            out[extra_key] = meta[extra_key]
    return out


def completion_chunk_to_llama_sse_data(chunk: CompletionChunk) -> str:
    """JSON-encode a CompletionChunk for native llama-server `/completion` SSE."""
    is_stop = chunk.finish_reason is not None
    meta = chunk.metadata or {}
    out: dict[str, Any] = {
        "content": chunk.text,
        "stop": is_stop,
    }
    if "tokens_predicted" in meta:
        out["tokens_predicted"] = meta["tokens_predicted"]
    if "tokens_evaluated" in meta:
        out["tokens_evaluated"] = meta["tokens_evaluated"]
    if "timings" in meta:
        out["timings"] = meta["timings"]
    if is_stop:
        out["stopped_eos"] = (
            bool(meta["stopped_eos"])
            if "stopped_eos" in meta
            else chunk.finish_reason == "stop"
        )
        out["stopped_limit"] = (
            bool(meta["stopped_limit"])
            if "stopped_limit" in meta
            else chunk.finish_reason == "length"
        )
        out["model"] = chunk.model_id
    return json.dumps(out, separators=(",", ":"))
