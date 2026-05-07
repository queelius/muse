"""VLM-extended chat/completions route tests.

Exercises the new pre-dispatch step: capability gating, image decoding,
content-shape validation. Text-only requests must remain byte-identical
to v0.41.x behavior (regression watchdog).
"""
import base64
from io import BytesIO
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from muse.core.registry import ModalityRegistry
from muse.modalities.chat_completion.routes import build_router


def _make_data_url(size=(8, 8), color="red"):
    img = Image.new("RGB", size, color)
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


class _FakeChatModel:
    def __init__(self, model_id, supports_vision=False, supports_multi_image=False):
        self.model_id = model_id
        self.supports_vision = supports_vision
        self.supports_multi_image = supports_multi_image
        self.received_messages = None

    def chat(self, messages, **kwargs):
        self.received_messages = messages
        from muse.modalities.chat_completion.protocol import (
            ChatChoice, ChatResult,
        )
        return ChatResult(
            id="chatcmpl-test",
            model_id=self.model_id,
            created=0,
            choices=[ChatChoice(
                index=0,
                message={"role": "assistant", "content": "ok"},
                finish_reason="stop",
            )],
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        )


def _client_with_model(model):
    registry = ModalityRegistry()
    registry.register("chat/completion", model)
    app = FastAPI()
    app.include_router(build_router(registry))
    return TestClient(app)


def test_text_only_request_byte_identical():
    """v0.41.x regression watchdog: pure-text requests do not trip the
    new pre-dispatch step, do not touch decode_image_input, and reach
    the backend unchanged."""
    model = _FakeChatModel("text-only", supports_vision=False)
    client = _client_with_model(model)
    r = client.post("/v1/chat/completions", json={
        "model": "text-only",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 200
    assert model.received_messages == [{"role": "user", "content": "hi"}]


def test_vision_capability_mismatch_returns_400():
    model = _FakeChatModel("text-only", supports_vision=False)
    client = _client_with_model(model)
    data_url = _make_data_url()
    r = client.post("/v1/chat/completions", json={
        "model": "text-only",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "describe"},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }],
    })
    assert r.status_code == 400
    body = r.json()
    assert body["error"]["code"] == "vision_not_supported"


def test_multi_image_capability_mismatch_returns_400():
    model = _FakeChatModel(
        "vlm-single", supports_vision=True, supports_multi_image=False,
    )
    client = _client_with_model(model)
    data_url = _make_data_url()
    r = client.post("/v1/chat/completions", json={
        "model": "vlm-single",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "compare"},
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }],
    })
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "too_many_images"


def test_invalid_content_part_missing_url_returns_400():
    model = _FakeChatModel("vlm", supports_vision=True, supports_multi_image=True)
    client = _client_with_model(model)
    r = client.post("/v1/chat/completions", json={
        "model": "vlm",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {}},  # missing url
            ],
        }],
    })
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_content_part"


def test_unsupported_content_type_returns_400():
    model = _FakeChatModel("vlm", supports_vision=True, supports_multi_image=True)
    client = _client_with_model(model)
    r = client.post("/v1/chat/completions", json={
        "model": "vlm",
        "messages": [{
            "role": "user",
            "content": [{"type": "video_url", "video_url": {"url": "x"}}],
        }],
    })
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "unsupported_content_type"


def test_malformed_data_url_returns_400_invalid_image():
    model = _FakeChatModel("vlm", supports_vision=True, supports_multi_image=True)
    client = _client_with_model(model)
    r = client.post("/v1/chat/completions", json={
        "model": "vlm",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:notavalidurl"}},
            ],
        }],
    })
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_image"


def test_valid_image_decoded_and_forwarded():
    """Happy path: image_url part is decoded, rewritten to {type: image,
    image: <PIL>}, and the backend sees the rewritten messages."""
    model = _FakeChatModel("vlm", supports_vision=True, supports_multi_image=True)
    client = _client_with_model(model)
    data_url = _make_data_url((16, 16), "blue")
    r = client.post("/v1/chat/completions", json={
        "model": "vlm",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "what?"},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }],
    })
    assert r.status_code == 200
    msg = model.received_messages[0]
    assert msg["content"][0] == {"type": "text", "text": "what?"}
    assert msg["content"][1]["type"] == "image"
    img = msg["content"][1]["image"]
    assert img.size == (16, 16)


def test_multi_image_supported_when_capability_true():
    model = _FakeChatModel(
        "multi-vlm", supports_vision=True, supports_multi_image=True,
    )
    client = _client_with_model(model)
    data_url = _make_data_url()
    r = client.post("/v1/chat/completions", json={
        "model": "multi-vlm",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "compare"},
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }],
    })
    assert r.status_code == 200
    images = [p for p in model.received_messages[0]["content"] if p["type"] == "image"]
    assert len(images) == 2


def test_legacy_string_content_unaffected():
    """OpenAI's older `content: "string"` shape: pass through unchanged."""
    model = _FakeChatModel("text", supports_vision=False)
    client = _client_with_model(model)
    r = client.post("/v1/chat/completions", json={
        "model": "text",
        "messages": [{"role": "user", "content": "plain text"}],
    })
    assert r.status_code == 200
    assert model.received_messages == [{"role": "user", "content": "plain text"}]


def test_capability_error_fires_before_sse_stream_opens():
    """When stream=true is set, capability errors MUST surface as a
    pre-stream 400, never as an event-error mid-stream."""
    model = _FakeChatModel("text-only", supports_vision=False)
    client = _client_with_model(model)
    data_url = _make_data_url()
    r = client.post("/v1/chat/completions", json={
        "model": "text-only",
        "stream": True,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }],
    })
    # Pre-stream 400 (not text/event-stream).
    assert r.status_code == 400
    assert "text/event-stream" not in r.headers.get("content-type", "")
