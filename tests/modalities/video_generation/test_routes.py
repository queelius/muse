"""Tests for POST /v1/video/generations."""
from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from muse.core.registry import ModalityRegistry
from muse.core.server import create_app
from muse.modalities.video_generation import (
    MODALITY,
    VideoResult,
    build_router,
)


class RecordingModel:
    """Captures the kwargs each generate() received."""

    model_id = "fake-vid"

    def __init__(self):
        self.last_kwargs = None

    def generate(self, prompt, **kwargs):
        self.last_kwargs = kwargs
        # Return a 5-frame "video" at 720x480.
        frames = [
            Image.new("RGB", (720, 480), (i * 40, 100, 150)) for i in range(5)
        ]
        return VideoResult(
            frames=frames, fps=5, width=720, height=480,
            duration_seconds=1.0, seed=-1, metadata={"prompt": prompt},
        )


def _fake_imageio_factory(format_marker: bytes):
    """Build a fake imageio module that writes deterministic bytes."""
    fake_imageio = type("ii", (), {})()

    def fake_mimwrite(buf, frames, *, fps, codec, format, **_):
        buf.write(format_marker + f":{codec}:{fps}:{len(frames)}".encode())

    fake_imageio.mimwrite = fake_mimwrite
    return fake_imageio


@pytest.fixture
def client_default():
    backend = RecordingModel()
    reg = ModalityRegistry()
    reg.register(MODALITY, backend, manifest={
        "model_id": "fake-vid",
        "capabilities": {
            "default_duration_seconds": 1.0,
            "default_fps": 5,
            "default_size": (720, 480),
        },
    })
    app = create_app(registry=reg, routers={MODALITY: build_router(reg)})
    return TestClient(app), backend


def test_post_returns_mp4_by_default(client_default):
    client, _ = client_default
    fake = _fake_imageio_factory(b"FAKEMP4")
    with patch(
        "muse.modalities.video_generation.codec._try_import_imageio",
        return_value=fake,
    ):
        r = client.post(
            "/v1/video/generations",
            json={"prompt": "a cat", "model": "fake-vid"},
        )
    assert r.status_code == 200
    body = r.json()
    assert "data" in body
    assert len(body["data"]) == 1
    assert body["model"] == "fake-vid"
    assert body["metadata"]["format"] == "mp4"
    assert body["metadata"]["fps"] == 5
    import base64
    asset = base64.b64decode(body["data"][0]["b64_json"])
    assert asset.startswith(b"FAKEMP4")
    assert b"h264" in asset


def test_post_response_format_webm(client_default):
    client, _ = client_default
    fake = _fake_imageio_factory(b"FAKEWEBM")
    with patch(
        "muse.modalities.video_generation.codec._try_import_imageio",
        return_value=fake,
    ):
        r = client.post(
            "/v1/video/generations",
            json={
                "prompt": "x", "model": "fake-vid",
                "response_format": "webm",
            },
        )
    assert r.status_code == 200
    body = r.json()
    assert body["metadata"]["format"] == "webm"
    import base64
    asset = base64.b64decode(body["data"][0]["b64_json"])
    assert asset.startswith(b"FAKEWEBM")
    assert b"vp9" in asset


def test_post_response_format_frames_b64(client_default):
    client, _ = client_default
    r = client.post(
        "/v1/video/generations",
        json={
            "prompt": "x", "model": "fake-vid",
            "response_format": "frames_b64",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["data"]) == 5  # 5 frames
    for entry in body["data"]:
        import base64
        png = base64.b64decode(entry["b64_json"])
        assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert body["metadata"]["format"] == "frames_b64"


def test_post_n_2_returns_two_videos_for_mp4(client_default):
    client, _ = client_default
    fake = _fake_imageio_factory(b"FAKEMP4")
    with patch(
        "muse.modalities.video_generation.codec._try_import_imageio",
        return_value=fake,
    ):
        r = client.post(
            "/v1/video/generations",
            json={"prompt": "x", "model": "fake-vid", "n": 2},
        )
    assert r.status_code == 200
    body = r.json()
    assert len(body["data"]) == 2


def test_post_unknown_model_returns_404(client_default):
    client, _ = client_default
    r = client.post(
        "/v1/video/generations",
        json={"prompt": "x", "model": "nonexistent"},
    )
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "model_not_found"


def test_post_missing_prompt_returns_422(client_default):
    client, _ = client_default
    r = client.post(
        "/v1/video/generations",
        json={"model": "fake-vid"},
    )
    assert r.status_code == 422


def test_post_duration_out_of_range_returns_422(client_default):
    client, _ = client_default
    r = client.post(
        "/v1/video/generations",
        json={"prompt": "x", "model": "fake-vid", "duration_seconds": 100.0},
    )
    assert r.status_code in (400, 422)


def test_post_invalid_response_format_returns_422(client_default):
    client, _ = client_default
    r = client.post(
        "/v1/video/generations",
        json={"prompt": "x", "model": "fake-vid", "response_format": "avi"},
    )
    assert r.status_code in (400, 422)


def test_post_malformed_size_returns_422(client_default):
    client, _ = client_default
    r = client.post(
        "/v1/video/generations",
        json={"prompt": "x", "model": "fake-vid", "size": "garbage"},
    )
    assert r.status_code in (400, 422)


def test_post_n_too_large_returns_422(client_default):
    """n is capped at 2; n=3 should fail validation."""
    client, _ = client_default
    r = client.post(
        "/v1/video/generations",
        json={"prompt": "x", "model": "fake-vid", "n": 3},
    )
    assert r.status_code in (400, 422)


def test_post_size_parsed_into_width_and_height(client_default):
    """size='WxH' is split and forwarded as width + height to backend."""
    client, backend = client_default
    fake = _fake_imageio_factory(b"FAKEMP4")
    with patch(
        "muse.modalities.video_generation.codec._try_import_imageio",
        return_value=fake,
    ):
        r = client.post(
            "/v1/video/generations",
            json={"prompt": "x", "model": "fake-vid", "size": "832x480"},
        )
    assert r.status_code == 200
    assert backend.last_kwargs["width"] == 832
    assert backend.last_kwargs["height"] == 480


def test_post_seed_offset_per_n(client_default):
    """For n>1, each result gets seed + offset."""
    backend = RecordingModel()
    reg = ModalityRegistry()
    reg.register(MODALITY, backend, manifest={"model_id": "fake-vid"})
    app = create_app(
        registry=reg, routers={MODALITY: build_router(reg)},
    )
    client = TestClient(app)
    fake = _fake_imageio_factory(b"FAKEMP4")
    seeds_seen = []

    original_generate = backend.generate

    def tracked_generate(prompt, **kwargs):
        seeds_seen.append(kwargs.get("seed"))
        return original_generate(prompt, **kwargs)

    backend.generate = tracked_generate

    with patch(
        "muse.modalities.video_generation.codec._try_import_imageio",
        return_value=fake,
    ):
        r = client.post(
            "/v1/video/generations",
            json={"prompt": "x", "model": "fake-vid", "n": 2, "seed": 100},
        )
    assert r.status_code == 200
    assert seeds_seen == [100, 101]


def test_post_unsupported_format_returns_400_when_imageio_absent(
    client_default,
):
    """If imageio isn't installed, mp4/webm requests get a clean 400."""
    client, _ = client_default
    with patch(
        "muse.modalities.video_generation.codec._try_import_imageio",
        return_value=None,
    ):
        r = client.post(
            "/v1/video/generations",
            json={"prompt": "x", "model": "fake-vid", "response_format": "mp4"},
        )
    assert r.status_code == 400
    assert "imageio" in r.json()["error"]["message"]
