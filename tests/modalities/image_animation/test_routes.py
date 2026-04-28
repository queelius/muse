"""Tests for POST /v1/images/animations."""
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from muse.core.registry import ModalityRegistry
from muse.core.server import create_app
from muse.modalities.image_animation import (
    MODALITY, AnimationResult, build_router,
)


class RecordingModel:
    """Captures the kwargs each generate() received."""
    model_id = "fake-anim"
    def __init__(self, capabilities=None):
        self._caps = capabilities or {}
        self.last_kwargs = None
    def generate(self, prompt, **kwargs):
        self.last_kwargs = kwargs
        # Return a 4-frame "animation"
        frames = [Image.new("RGB", (64, 64), (i*50, 100, 150)) for i in range(4)]
        return AnimationResult(
            frames=frames, fps=8, width=64, height=64, seed=-1,
            metadata={"prompt": prompt},
        )


@pytest.fixture
def client_text_only():
    backend = RecordingModel()
    reg = ModalityRegistry()
    reg.register(MODALITY, backend, manifest={
        "model_id": "fake-anim",
        "capabilities": {
            "supports_text_to_animation": True,
            "supports_image_to_animation": False,
        },
    })
    app = create_app(registry=reg, routers={MODALITY: build_router(reg)})
    return TestClient(app), backend


@pytest.fixture
def client_img2vid():
    backend = RecordingModel()
    reg = ModalityRegistry()
    reg.register(MODALITY, backend, manifest={
        "model_id": "fake-anim",
        "capabilities": {
            "supports_text_to_animation": True,
            "supports_image_to_animation": True,
        },
    })
    app = create_app(registry=reg, routers={MODALITY: build_router(reg)})
    return TestClient(app), backend


def test_post_returns_webp_by_default(client_text_only):
    client, _backend = client_text_only
    r = client.post("/v1/images/animations", json={
        "prompt": "a cat playing", "model": "fake-anim",
    })
    assert r.status_code == 200
    body = r.json()
    assert "data" in body
    assert len(body["data"]) == 1
    import base64
    asset = base64.b64decode(body["data"][0]["b64_json"])
    assert asset[:4] == b"RIFF"
    assert asset[8:12] == b"WEBP"
    assert body["metadata"]["format"] == "webp"


def test_post_response_format_gif(client_text_only):
    client, _ = client_text_only
    r = client.post("/v1/images/animations", json={
        "prompt": "x", "model": "fake-anim", "response_format": "gif",
    })
    assert r.status_code == 200
    import base64
    asset = base64.b64decode(r.json()["data"][0]["b64_json"])
    assert asset[:6] in (b"GIF87a", b"GIF89a")


def test_post_response_format_frames_b64(client_text_only):
    client, _ = client_text_only
    r = client.post("/v1/images/animations", json={
        "prompt": "x", "model": "fake-anim", "response_format": "frames_b64",
    })
    assert r.status_code == 200
    body = r.json()
    assert len(body["data"]) == 4  # 4 frames
    for entry in body["data"]:
        import base64
        png = base64.b64decode(entry["b64_json"])
        assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_post_image_with_text_only_model_returns_400(client_text_only):
    """Model with supports_image_to_animation=False rejects image input."""
    client, _ = client_text_only
    r = client.post("/v1/images/animations", json={
        "prompt": "x", "model": "fake-anim",
        "image": "data:image/png;base64,iVBORw0KGgo=",
    })
    assert r.status_code == 400
    assert "image-to-animation" in r.json()["error"]["message"].lower()


def test_post_image_with_img2vid_model_passes_through(client_img2vid):
    """Model with supports_image_to_animation=True accepts image input."""
    import base64, io
    img = Image.new("RGB", (64, 64), (255, 0, 0))
    buf = io.BytesIO(); img.save(buf, format="PNG")
    data_url = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

    client, backend = client_img2vid
    r = client.post("/v1/images/animations", json={
        "prompt": "x", "model": "fake-anim",
        "image": data_url, "strength": 0.6,
    })
    assert r.status_code == 200
    # Backend received a PIL image as init_image
    assert backend.last_kwargs.get("init_image") is not None
    assert backend.last_kwargs.get("strength") == 0.6


def test_post_unknown_model_returns_404(client_text_only):
    client, _ = client_text_only
    r = client.post("/v1/images/animations", json={
        "prompt": "x", "model": "nonexistent",
    })
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "model_not_found"


def test_post_loop_default_true(client_text_only):
    """Default loop=true encodes infinite-loop WebP."""
    client, _ = client_text_only
    r1 = client.post("/v1/images/animations", json={
        "prompt": "x", "model": "fake-anim",
    })
    r2 = client.post("/v1/images/animations", json={
        "prompt": "x", "model": "fake-anim", "loop": False,
    })
    assert r1.json()["data"][0]["b64_json"] != r2.json()["data"][0]["b64_json"]


def test_post_invalid_response_format_returns_400(client_text_only):
    client, _ = client_text_only
    r = client.post("/v1/images/animations", json={
        "prompt": "x", "model": "fake-anim", "response_format": "avi",
    })
    assert r.status_code in (400, 422)


def test_post_frames_out_of_range_rejected(client_text_only):
    client, _ = client_text_only
    r = client.post("/v1/images/animations", json={
        "prompt": "x", "model": "fake-anim", "frames": 999,
    })
    assert r.status_code in (400, 422)
