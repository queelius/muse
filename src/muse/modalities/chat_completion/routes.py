"""/v1/chat/completions router.

Two call shapes:
  - stream=False (default): non-streaming. Calls ChatModel.chat() once,
    returns OpenAI ChatCompletion JSON.
  - stream=True: SSE. Producer thread calls ChatModel.chat_stream() and
    pushes ChatChunk items into an asyncio.Queue; the response iterator
    reads from the queue and serializes to SSE `data:` lines plus a
    final `data: [DONE]` sentinel.

Thread + queue pattern matches the audio.speech streaming code so we
do not buffer tokens on the server. Every token dispatches as produced.
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from sse_starlette.sse import EventSourceResponse

from muse.core.errors import ModelNotFoundError
from muse.core.registry import ModalityRegistry
from muse.modalities.chat_completion.codec import (
    DONE_SENTINEL,
    chunk_to_sse_data,
    result_to_openai_dict,
)
from muse.modalities.image_generation.image_input import decode_image_input


logger = logging.getLogger(__name__)

MODALITY = "chat/completion"


class ChatCompletionRequest(BaseModel):
    """OpenAI-shape request. Most fields are passthrough to the backend."""
    model: str | None = None
    messages: list[dict] = Field(..., min_length=1)
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stop: str | list[str] | None = None
    seed: int | None = None
    tools: list[dict] | None = None
    tool_choice: str | dict | None = None
    response_format: dict | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
    extra_body: dict | None = None

    @field_validator("messages")
    @classmethod
    def _non_empty_messages(cls, v: list[dict]) -> list[dict]:
        if not v:
            raise ValueError("messages must be non-empty")
        return v

    def backend_kwargs(self) -> dict:
        """Dict of kwargs to forward to ChatModel.chat()/chat_stream().

        Omits `model` (routing metadata) and `stream` (handled by the
        route, not the backend). extra_body spreads in raw.
        """
        out: dict[str, Any] = {}
        for key in (
            "temperature", "top_p", "max_tokens", "stop", "seed",
            "tools", "tool_choice", "response_format", "logprobs",
            "top_logprobs",
        ):
            val = getattr(self, key)
            if val is not None:
                out[key] = val
        if self.extra_body:
            out.update(self.extra_body)
        return out


async def _decode_image_parts(messages: list[dict], model: Any) -> list[dict]:
    """Walk messages; validate + decode any image_url parts; return rewritten list.

    Pre-dispatch step that runs before ChatModel.chat()/chat_stream() to:
      - Detect image_url parts in any message.content list
      - Reject capability mismatches (vision_not_supported, too_many_images)
      - Validate part shape (invalid_content_part, unsupported_content_type)
      - Decode each url via decode_image_input (data: or http(s)://)
      - Rewrite the part as {type: image, image: <PIL.Image>} so the
        backend consumes a uniform muse-internal shape

    Raises ValueError with structured codes (code: human-readable) that
    the route translates to 400 responses with OpenAI-shape error envelopes.

    Returns the original `messages` list unchanged when no image_url parts
    are found, preserving byte-identical behaviour for text-only requests.
    """
    has_any_image = False
    new_messages: list[dict] = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            new_messages.append(msg)
            continue
        image_count = 0
        new_content: list[Any] = []
        for part in content:
            if not isinstance(part, dict):
                new_content.append(part)
                continue
            ptype = part.get("type")
            if ptype == "text":
                new_content.append(part)
            elif ptype == "image_url":
                has_any_image = True
                image_count += 1
                url = (part.get("image_url") or {}).get("url")
                if not url:
                    raise ValueError(
                        "invalid_content_part: image_url part missing required url field"
                    )
                # Capability check before decode so we don't waste a
                # fetch on a request that's about to 400.
                if not getattr(model, "supports_vision", False):
                    raise ValueError(
                        f"vision_not_supported: model "
                        f"{getattr(model, 'model_id', '?')} does not "
                        f"support vision input; pick a model with "
                        f"supports_vision=true"
                    )
                try:
                    img = await decode_image_input(url)
                except ValueError as ve:
                    raise ValueError(
                        f"invalid_image: could not decode image: {ve}"
                    ) from ve
                new_content.append({"type": "image", "image": img})
            elif ptype is None:
                new_content.append(part)
            else:
                raise ValueError(
                    f"unsupported_content_type: content type {ptype!r} "
                    f"not supported; allowed: text, image_url"
                )
        if image_count > 1 and not getattr(model, "supports_multi_image", False):
            raise ValueError(
                f"too_many_images: model "
                f"{getattr(model, 'model_id', '?')} accepts only 1 "
                f"image per message; got {image_count}"
            )
        new_messages.append({**msg, "content": new_content})
    return new_messages if has_any_image else messages


def _image_decode_error_response(ve: ValueError) -> JSONResponse:
    """Translate a ValueError from _decode_image_parts into an OpenAI-shape
    400 JSONResponse.  ValueError messages are formatted as "code: text"."""
    msg = str(ve)
    code, _, detail = msg.partition(": ")
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "code": code,
                "message": detail or msg,
                "type": "invalid_request_error",
            },
        },
    )


def build_router(registry: ModalityRegistry) -> APIRouter:
    router = APIRouter(prefix="/v1", tags=["chat/completion"])

    def _get_model(model_id: str | None):
        try:
            return registry.get(MODALITY, model_id)
        except KeyError:
            raise ModelNotFoundError(
                model_id=model_id or "<default>",
                modality=MODALITY,
            )

    @router.post("/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        model = _get_model(req.model)

        # Pre-dispatch: decode image_url content parts and validate
        # capability flags BEFORE branching on stream/non-stream so that
        # errors always surface as 400 (never as mid-stream SSE errors).
        try:
            messages = await _decode_image_parts(req.messages, model)
        except ValueError as ve:
            return _image_decode_error_response(ve)

        kwargs = req.backend_kwargs()

        # If the request asks for tool calling, warn when the loaded
        # model isn't known to support it. Tool-call quality is a
        # property of the model + chat_format combination; muse doesn't
        # block the request, but a warning lets the user know structured
        # tool_calls may not appear and the model may emit raw text in
        # `content` instead.
        if req.tools is not None:
            supports = getattr(model, "supports_tools", None)
            if supports is False:
                logger.warning(
                    "model %s is not known to support tool calling; "
                    "structured tool_calls may not appear in the response",
                    getattr(model, "model_id", "?"),
                )
            elif supports is None:
                logger.warning(
                    "tool support for model %s is unknown; "
                    "structured tool_calls may not appear in the response. "
                    "If you know this model works, set "
                    "capabilities.chat_format in its manifest "
                    "(see docs/CHAT_COMPLETION.md)",
                    getattr(model, "model_id", "?"),
                )

        if not req.stream:
            result = await asyncio.to_thread(model.chat, messages, **kwargs)
            return result_to_openai_dict(result)

        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _producer():
            # call_soon_threadsafe + put_nowait avoids blocking the
            # producer thread on the event loop's scheduler. The earlier
            # run_coroutine_threadsafe(...).result() pattern would
            # deadlock if the loop was busy. Mirrors audio_speech.
            try:
                for chunk in model.chat_stream(messages, **kwargs):
                    loop.call_soon_threadsafe(queue.put_nowait, chunk)
            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, e)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        threading.Thread(target=_producer, daemon=True).start()

        async def _events():
            while True:
                item = await queue.get()
                if item is None:
                    yield {"data": DONE_SENTINEL}
                    return
                if isinstance(item, Exception):
                    # Surface the failure to the client as an SSE
                    # `event: error` carrying the OpenAI error envelope.
                    # Without this, a backend crash mid-stream is
                    # indistinguishable from a clean empty completion.
                    logger.error("chat_stream backend error: %s", item)
                    err_payload = {"error": {
                        "code": "internal",
                        "message": str(item),
                        "type": "server_error",
                    }}
                    yield {"event": "error", "data": json.dumps(err_payload)}
                    yield {"data": DONE_SENTINEL}
                    return
                yield {"data": chunk_to_sse_data(item)}

        return EventSourceResponse(_events())

    return router
