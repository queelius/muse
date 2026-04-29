"""Inference text tools.

Five tools wrapping the text modalities:
  - muse_chat        /v1/chat/completions
  - muse_summarize   /v1/summarize
  - muse_rerank      /v1/rerank
  - muse_classify    /v1/moderations
  - muse_embed_text  /v1/embeddings

Each tool returns the muse server's full JSON envelope as a TextContent
block; the LLM picks fields it needs from there.
"""
from __future__ import annotations

import json
from typing import Any

from mcp.types import Tool

from muse.mcp.client import MuseClient
from muse.mcp.tools import INFERENCE_TOOLS, ToolEntry


def _json_block(payload: Any) -> dict:
    return {"type": "text", "text": json.dumps(payload, indent=2)}


# ---- handlers ----

def _handle_chat(client: MuseClient, args: dict) -> list[dict]:
    body = {k: v for k, v in args.items() if v is not None}
    out = client.chat(**body)
    return [_json_block(out)]


def _handle_summarize(client: MuseClient, args: dict) -> list[dict]:
    body = {k: v for k, v in args.items() if v is not None}
    out = client.summarize(**body)
    return [_json_block(out)]


def _handle_rerank(client: MuseClient, args: dict) -> list[dict]:
    body = {k: v for k, v in args.items() if v is not None}
    out = client.rerank(**body)
    return [_json_block(out)]


def _handle_classify(client: MuseClient, args: dict) -> list[dict]:
    body = {k: v for k, v in args.items() if v is not None}
    out = client.classify(**body)
    return [_json_block(out)]


def _handle_embed_text(client: MuseClient, args: dict) -> list[dict]:
    body = {k: v for k, v in args.items() if v is not None}
    out = client.embed_text(**body)
    # Return the full envelope; vectors are big so prefer not to log
    # them but the LLM still needs the dimensions for downstream tools.
    return [_json_block(out)]


# ---- tool definitions ----

INFERENCE_TOOLS.extend([
    ToolEntry(
        tool=Tool(
            name="muse_chat",
            description=(
                "Generate a chat completion using a language model. "
                "Use this to ask muse-hosted LLMs questions, run them "
                "on tool calls, or get free-form text responses. Returns "
                "the OpenAI-shape ChatCompletion dict."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "model": {
                        "type": "string",
                        "description": (
                            "Catalog id (e.g. 'qwen3.5-9b-q4'); omit to "
                            "use the default chat/completion model."
                        ),
                    },
                    "messages": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": (
                            "OpenAI-shape message list with role + content."
                        ),
                    },
                    "temperature": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 2.0,
                    },
                    "max_tokens": {"type": "integer", "minimum": 1},
                    "top_p": {"type": "number"},
                    "tools": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "OpenAI-shape function/tool definitions.",
                    },
                    "tool_choice": {
                        "description": (
                            "'auto' | 'none' | a specific tool selector dict."
                        ),
                    },
                    "response_format": {"type": "object"},
                },
                "required": ["messages"],
            },
        ),
        handler=_handle_chat,
    ),
    ToolEntry(
        tool=Tool(
            name="muse_summarize",
            description=(
                "Condense a long document into a shorter summary. Use "
                "this when the user wants the key points of a passage "
                "without the LLM rereading and re-paraphrasing it. "
                "Returns the Cohere-shape envelope with `summary`."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to summarize.",
                    },
                    "model": {"type": "string"},
                    "length": {
                        "type": "string",
                        "enum": ["auto", "short", "medium", "long"],
                    },
                    "format": {
                        "type": "string",
                        "enum": ["auto", "paragraph", "bullets"],
                    },
                },
                "required": ["text"],
            },
        ),
        handler=_handle_summarize,
    ),
    ToolEntry(
        tool=Tool(
            name="muse_rerank",
            description=(
                "Rerank a list of documents by relevance to a query "
                "using a cross-encoder. Use this in retrieval pipelines "
                "to improve top-k precision after a vector search. "
                "Returns the Cohere-shape envelope with sorted indices "
                "and relevance scores."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "documents": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of candidate documents.",
                    },
                    "model": {"type": "string"},
                    "top_n": {"type": "integer", "minimum": 1},
                    "return_documents": {"type": "boolean", "default": False},
                },
                "required": ["query", "documents"],
            },
        ),
        handler=_handle_rerank,
    ),
    ToolEntry(
        tool=Tool(
            name="muse_classify",
            description=(
                "Classify text against a moderation or sentiment model. "
                "Use this for content moderation, intent detection, or "
                "any task framed as multi-label classification over "
                "fixed categories. Returns OpenAI-shape Moderation."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "input": {
                        "description": (
                            "Text to classify. Accepts a single string "
                            "or an array of strings for batch."
                        ),
                    },
                    "model": {"type": "string"},
                    "threshold": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                },
                "required": ["input"],
            },
        ),
        handler=_handle_classify,
    ),
    ToolEntry(
        tool=Tool(
            name="muse_embed_text",
            description=(
                "Compute dense vector embeddings for one or more text "
                "strings. Use this for semantic search, clustering, or "
                "similarity comparisons. Returns OpenAI-shape Embeddings "
                "with float vectors."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "input": {
                        "description": (
                            "String or array of strings to embed."
                        ),
                    },
                    "model": {"type": "string"},
                    "dimensions": {"type": "integer", "minimum": 1},
                    "encoding_format": {
                        "type": "string",
                        "enum": ["float", "base64"],
                        "default": "float",
                    },
                },
                "required": ["input"],
            },
        ),
        handler=_handle_embed_text,
    ),
])
