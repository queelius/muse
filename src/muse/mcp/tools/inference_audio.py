"""Inference audio tools.

Five tools wrapping the audio modalities:
  - muse_speak           /v1/audio/speech
  - muse_transcribe      /v1/audio/transcriptions  (audio binary input)
  - muse_generate_music  /v1/audio/music
  - muse_generate_sfx    /v1/audio/sfx
  - muse_embed_audio     /v1/audio/embeddings      (audio binary input)

Audio-output handlers (speak, music, sfx) return one AudioContent block
plus a TextContent summary. Audio-input handlers (transcribe,
embed_audio) resolve a binary input via the standard b64 / url / path
trio and forward via multipart.
"""
from __future__ import annotations

import json
from typing import Any

from mcp.types import Tool

from muse.mcp.binary_io import (
    binary_input_schema,
    pack_audio_output,
    resolve_binary_input,
)
from muse.mcp.client import MuseClient
from muse.mcp.tools import INFERENCE_TOOLS, ToolEntry


def _json_block(payload: Any) -> dict:
    return {"type": "text", "text": json.dumps(payload, indent=2)}


def _filter_keys(args: dict, prefixes: tuple[str, ...]) -> dict:
    out = {}
    for k, v in args.items():
        if any(k.startswith(p) for p in prefixes):
            continue
        if v is None:
            continue
        out[k] = v
    return out


# ---- handlers ----

def _handle_speak(client: MuseClient, args: dict) -> list[dict]:
    body = _filter_keys(args, prefixes=())
    audio = client.speak(**body)
    return [
        pack_audio_output(audio, mime="audio/wav"),
        _json_block({
            "model": body.get("model"),
            "size_bytes": len(audio),
            "format": body.get("response_format", "wav"),
        }),
    ]


def _handle_transcribe(client: MuseClient, args: dict) -> list[dict]:
    audio = resolve_binary_input(
        b64=args.get("audio_b64"),
        url=args.get("audio_url"),
        path=args.get("audio_path"),
        field_name="audio",
    )
    body = _filter_keys(args, prefixes=("audio_",))
    out = client.transcribe(audio=audio, **body)
    if isinstance(out, bytes):
        # raw text or vtt/srt body returned as bytes
        return [_json_block({"text": out.decode("utf-8", errors="replace")})]
    return [_json_block(out)]


def _handle_generate_music(client: MuseClient, args: dict) -> list[dict]:
    body = _filter_keys(args, prefixes=())
    audio = client.generate_music(**body)
    return [
        pack_audio_output(audio, mime="audio/wav"),
        _json_block({
            "model": body.get("model"),
            "duration": body.get("duration"),
            "size_bytes": len(audio),
        }),
    ]


def _handle_generate_sfx(client: MuseClient, args: dict) -> list[dict]:
    body = _filter_keys(args, prefixes=())
    audio = client.generate_sfx(**body)
    return [
        pack_audio_output(audio, mime="audio/wav"),
        _json_block({
            "model": body.get("model"),
            "duration": body.get("duration"),
            "size_bytes": len(audio),
        }),
    ]


def _handle_embed_audio(client: MuseClient, args: dict) -> list[dict]:
    audio = resolve_binary_input(
        b64=args.get("audio_b64"),
        url=args.get("audio_url"),
        path=args.get("audio_path"),
        field_name="audio",
    )
    body = _filter_keys(args, prefixes=("audio_",))
    out = client.embed_audio(audio=audio, **body)
    return [_json_block(out)]


# ---- tool definitions ----

INFERENCE_TOOLS.extend([
    ToolEntry(
        tool=Tool(
            name="muse_speak",
            description=(
                "Synthesize speech from text using a TTS model. Use "
                "when the user wants the LLM to 'read aloud' a passage. "
                "Returns an MCP AudioContent block (audio/wav)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "Text to speak.",
                    },
                    "model": {"type": "string"},
                    "voice": {"type": "string"},
                    "response_format": {
                        "type": "string",
                        "enum": ["wav", "opus", "mp3"],
                        "default": "wav",
                    },
                    "speed": {"type": "number"},
                },
                "required": ["input"],
            },
        ),
        handler=_handle_speak,
    ),
    ToolEntry(
        tool=Tool(
            name="muse_transcribe",
            description=(
                "Transcribe spoken audio to text using Whisper-family "
                "models. Use when the user wants captions, transcripts, "
                "or speech-to-text output. Provide audio_b64 (base64), "
                "audio_url (URL), or audio_path (local file)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "model": {"type": "string"},
                    "language": {
                        "type": "string",
                        "description": "ISO-639-1 (e.g. 'en') or omit for auto.",
                    },
                    "prompt": {"type": "string"},
                    "response_format": {
                        "type": "string",
                        "enum": ["json", "text", "srt", "verbose_json", "vtt"],
                        "default": "json",
                    },
                    "temperature": {"type": "number"},
                    "word_timestamps": {"type": "boolean"},
                    **binary_input_schema("audio"),
                },
            },
        ),
        handler=_handle_transcribe,
    ),
    ToolEntry(
        tool=Tool(
            name="muse_generate_music",
            description=(
                "Generate a music clip from a text prompt. Use when the "
                "user wants original music (e.g. 'ambient pad with "
                "chimes'). Returns an MCP AudioContent block."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "model": {"type": "string"},
                    "duration": {"type": "number"},
                    "seed": {"type": "integer"},
                    "response_format": {
                        "type": "string",
                        "enum": ["wav", "ogg", "mp3"],
                        "default": "wav",
                    },
                    "steps": {"type": "integer"},
                    "guidance": {"type": "number"},
                    "negative_prompt": {"type": "string"},
                },
                "required": ["prompt"],
            },
        ),
        handler=_handle_generate_music,
    ),
    ToolEntry(
        tool=Tool(
            name="muse_generate_sfx",
            description=(
                "Generate a sound-effect clip from a text prompt "
                "(e.g. 'thunder', 'door creak'). Use for short "
                "non-musical audio events. Returns an AudioContent block."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "model": {"type": "string"},
                    "duration": {"type": "number"},
                    "seed": {"type": "integer"},
                    "response_format": {
                        "type": "string",
                        "enum": ["wav", "ogg", "mp3"],
                        "default": "wav",
                    },
                    "steps": {"type": "integer"},
                    "guidance": {"type": "number"},
                    "negative_prompt": {"type": "string"},
                },
                "required": ["prompt"],
            },
        ),
        handler=_handle_generate_sfx,
    ),
    ToolEntry(
        tool=Tool(
            name="muse_embed_audio",
            description=(
                "Compute dense vector embeddings for an audio clip "
                "(CLAP, MERT, wav2vec). Use for music / audio "
                "similarity, classification heads, or zero-shot tagging. "
                "Provide audio_b64 / audio_url / audio_path."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "model": {"type": "string"},
                    "encoding_format": {
                        "type": "string",
                        "enum": ["float", "base64"],
                        "default": "float",
                    },
                    **binary_input_schema("audio"),
                },
            },
        ),
        handler=_handle_embed_audio,
    ),
])
