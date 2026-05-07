# tests/integration/test_remote_vlm.py
"""Integration tests for VLM modality.

Opt-in via MUSE_REMOTE_SERVER. The fixtures auto-skip when the server
isn't reachable or smolvlm-256m-instruct isn't loaded.
"""
import base64
import os
from io import BytesIO

import pytest
from PIL import Image


pytestmark = pytest.mark.skipif(
    not os.environ.get("MUSE_REMOTE_SERVER"),
    reason="MUSE_REMOTE_SERVER not set; integration tests skipped",
)


VLM_MODEL_ID = os.environ.get("MUSE_VLM_MODEL_ID", "smolvlm-256m-instruct")


def _data_url_for_color(color="red", size=8):
    img = Image.new("RGB", (size, size), color)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@pytest.fixture(scope="module")
def client():
    from openai import OpenAI
    base_url = os.environ["MUSE_REMOTE_SERVER"].rstrip("/") + "/v1"
    return OpenAI(base_url=base_url, api_key="not-used")


@pytest.fixture(scope="module")
def vlm_loaded(client):
    """Skip if the configured VLM model isn't enabled+loaded on the server."""
    models = {m.id for m in client.models.list().data}
    if VLM_MODEL_ID not in models:
        pytest.skip(f"VLM model {VLM_MODEL_ID} not on server")


def test_protocol_vision_capability_gating(client):
    """Sending an image to a non-VLM chat model returns 400."""
    text_model = os.environ.get("MUSE_CHAT_MODEL_ID", "qwen3.5-4b-q4")
    from openai import BadRequestError
    with pytest.raises(BadRequestError) as exc:
        client.chat.completions.create(
            model=text_model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "?"},
                    {"type": "image_url", "image_url": {
                        "url": _data_url_for_color()
                    }},
                ],
            }],
        )
    assert "vision_not_supported" in str(exc.value)


def test_protocol_data_url_image(client, vlm_loaded):
    """A small data-URL image to a VLM yields a non-empty response."""
    r = client.chat.completions.create(
        model=VLM_MODEL_ID,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "What color is this image?"},
                {"type": "image_url", "image_url": {
                    "url": _data_url_for_color("blue")
                }},
            ],
        }],
        max_tokens=40,
    )
    assert r.choices[0].message.content


def test_protocol_text_only_with_vlm_works(client, vlm_loaded):
    """A pure-text request to a VLM model still works."""
    r = client.chat.completions.create(
        model=VLM_MODEL_ID,
        messages=[{"role": "user", "content": "say hi"}],
        max_tokens=20,
    )
    assert r.choices[0].message.content


def test_protocol_legacy_string_content_unaffected(client):
    """`content: "string"` shape works on text chat (regression watchdog)."""
    text_model = os.environ.get("MUSE_CHAT_MODEL_ID", "qwen3.5-4b-q4")
    r = client.chat.completions.create(
        model=text_model,
        messages=[{"role": "user", "content": "respond with only 'hi'"}],
        max_tokens=10,
    )
    assert r.choices[0].message.content


def test_protocol_multi_image_rejected_when_unsupported(client):
    """Sending 2 images to llava-1.5-7b (supports_multi_image=False)
    returns 400 too_many_images. Skip if llava-1.5-7b not loaded."""
    models = {m.id for m in client.models.list().data}
    if "llava-1.5-7b" not in models:
        pytest.skip("llava-1.5-7b not on server")
    from openai import BadRequestError
    with pytest.raises(BadRequestError) as exc:
        client.chat.completions.create(
            model="llava-1.5-7b",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "compare"},
                    {"type": "image_url", "image_url": {"url": _data_url_for_color("red")}},
                    {"type": "image_url", "image_url": {"url": _data_url_for_color("blue")}},
                ],
            }],
        )
    assert "too_many_images" in str(exc.value)


def test_observe_smolvlm_describes_simple_image(client, vlm_loaded):
    """Watchdog: not a hard assertion. Logs the description so a human
    can spot quality drift across runs."""
    r = client.chat.completions.create(
        model=VLM_MODEL_ID,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe what you see."},
                {"type": "image_url", "image_url": {
                    "url": _data_url_for_color("red", size=64)
                }},
            ],
        }],
        max_tokens=80,
    )
    print(f"\n[observed VLM description]: {r.choices[0].message.content}")


def test_observe_streaming_with_image(client, vlm_loaded):
    """Streaming response with an image input. Watchdog only."""
    stream = client.chat.completions.create(
        model=VLM_MODEL_ID,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe briefly."},
                {"type": "image_url", "image_url": {
                    "url": _data_url_for_color("green", size=32)
                }},
            ],
        }],
        max_tokens=40,
        stream=True,
    )
    pieces = []
    for chunk in stream:
        delta = chunk.choices[0].delta
        if getattr(delta, "content", None):
            pieces.append(delta.content)
    assert "".join(pieces)
