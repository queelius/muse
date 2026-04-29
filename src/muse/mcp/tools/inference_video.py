"""Inference video tools.

One tool wrapping the video modality:
  - muse_generate_video  /v1/video/generations

Video bytes are big and don't fit MCP's content-block model cleanly
(no VideoContent type in the SDK). The handler returns the muse server's
JSON envelope verbatim so the LLM client can decide how to render it
(typically a download link or embedded video tag if the client supports
inline media).
"""
from __future__ import annotations

import json
from typing import Any

from mcp.types import Tool

from muse.mcp.client import MuseClient
from muse.mcp.tools import INFERENCE_TOOLS, ToolEntry


def _json_block(payload: Any) -> dict:
    return {"type": "text", "text": json.dumps(payload, indent=2)}


def _filter_keys(args: dict) -> dict:
    return {k: v for k, v in args.items() if v is not None}


# ---- handler ----

def _handle_generate_video(client: MuseClient, args: dict) -> list[dict]:
    body = _filter_keys(args)
    body.setdefault("response_format", "mp4")
    out = client.generate_video(**body)
    return [_json_block({
        "model": out.get("model"),
        "format": body["response_format"],
        "duration_seconds": out.get("duration_seconds"),
        "fps": out.get("fps"),
        "size": out.get("size"),
        "data": out.get("data"),
    })]


# ---- tool definition ----

INFERENCE_TOOLS.extend([
    ToolEntry(
        tool=Tool(
            name="muse_generate_video",
            description=(
                "Generate a short video clip from a text prompt "
                "(e.g. Wan, CogVideoX). Use when the user wants a "
                "moving picture, not a still image. GPU-required and "
                "can take minutes per clip. Returns a JSON envelope "
                "with base64-encoded video bytes (mp4 or webm) or a "
                "list of base64 frames (frames_b64)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "model": {"type": "string"},
                    "duration_seconds": {"type": "number"},
                    "fps": {"type": "integer"},
                    "size": {"type": "string"},
                    "seed": {"type": "integer"},
                    "negative_prompt": {"type": "string"},
                    "steps": {"type": "integer"},
                    "guidance": {"type": "number"},
                    "response_format": {
                        "type": "string",
                        "enum": ["mp4", "webm", "frames_b64"],
                        "default": "mp4",
                    },
                    "n": {"type": "integer", "default": 1},
                },
                "required": ["prompt"],
            },
        ),
        handler=_handle_generate_video,
    ),
])
