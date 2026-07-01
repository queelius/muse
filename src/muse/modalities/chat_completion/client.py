"""HTTP client for /v1/chat/completions. Mirrors the OpenAI SDK shape."""
from __future__ import annotations

import json
import logging
import os
from typing import Iterator

import httpx


logger = logging.getLogger(__name__)


class ChatStreamError(RuntimeError):
    """Raised when a chat stream carries a mid-stream `event: error` frame.

    `error` is the parsed OpenAI error envelope's inner dict
    (``{"code", "message", "type"}``) when the payload is JSON, else a
    ``{"message": <raw>}`` fallback.
    """

    def __init__(self, error: dict):
        self.error = error
        super().__init__(error.get("message", "chat stream error"))


class ChatClient:
    """Client for the chat/completion modality.

    Non-streaming: `chat(model, messages, **kwargs)` returns the full
    OpenAI ChatCompletion dict.

    Streaming: `chat_stream(model, messages, **kwargs)` yields each
    ChatCompletionChunk dict as it arrives, stopping at the [DONE]
    sentinel.
    """

    def __init__(self, base_url: str | None = None, timeout: float = 300.0) -> None:
        self.base_url = (
            base_url
            or os.environ.get("MUSE_SERVER")
            or "http://localhost:8000"
        ).rstrip("/")
        self.timeout = timeout

    def chat(self, *, model: str | None = None, messages: list[dict], **kwargs) -> dict:
        body = {"model": model, "messages": messages, "stream": False, **kwargs}
        body = {k: v for k, v in body.items() if v is not None}
        r = httpx.post(
            f"{self.base_url}/v1/chat/completions",
            json=body,
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def chat_stream(
        self,
        *,
        model: str | None = None,
        messages: list[dict],
        **kwargs,
    ) -> Iterator[dict]:
        body = {"model": model, "messages": messages, "stream": True, **kwargs}
        body = {k: v for k, v in body.items() if v is not None}
        with httpx.stream(
            "POST",
            f"{self.base_url}/v1/chat/completions",
            json=body,
            timeout=self.timeout,
        ) as r:
            r.raise_for_status()
            # Track the current SSE `event:` type. The server signals a
            # mid-stream backend failure with an `event: error` frame whose
            # `data:` line carries the OpenAI error envelope; without honoring
            # the event type we would yield that envelope as a normal chunk
            # and callers iterating chunk["choices"] would KeyError (L6).
            event_type: str | None = None
            for line in r.iter_lines():
                if not line:
                    event_type = None  # blank line terminates an SSE event
                    continue
                if line.startswith("event: "):
                    event_type = line[len("event: "):].strip()
                    continue
                if not line.startswith("data: "):
                    continue
                payload = line[len("data: "):]
                if event_type == "error":
                    try:
                        envelope = json.loads(payload)
                    except json.JSONDecodeError:
                        envelope = {"error": {"message": payload}}
                    err = envelope.get("error") if isinstance(envelope, dict) else None
                    raise ChatStreamError(err if isinstance(err, dict) else {"message": payload})
                if payload == "[DONE]":
                    return
                try:
                    yield json.loads(payload)
                except json.JSONDecodeError as e:
                    logger.warning("malformed SSE chunk: %s (%s)", payload, e)
