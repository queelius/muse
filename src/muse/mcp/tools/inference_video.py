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

from mcp.types import Tool

from muse.mcp.client import MuseClient
from muse.mcp.tools import INFERENCE_TOOLS, ToolEntry
from muse.mcp.tools._common import json_block, request_body


# ---- handler ----

def _handle_generate_video(client: MuseClient, args: dict) -> list[dict]:
    body = request_body(args)
    body.setdefault("response_format", "mp4")
    # The route's envelope is {data, model, metadata: {frames, fps,
    # duration_seconds, format, size}}; pass it through unchanged.
    return [json_block(client.generate_video(**body))]


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
                "required": ["model", "prompt"],
            },
        ),
        handler=_handle_generate_video,
    ),
])
