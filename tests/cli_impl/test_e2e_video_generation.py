"""End-to-end: /v1/video/generations through FastAPI + codec correctly.

Uses a fake VideoGenerationModel backend; no real weights. The fake
backend returns a small frame list which the codec encodes into mp4
(via mocked imageio), webm (also mocked), or per-frame PNG.
"""
from __future__ import annotations

import base64
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


pytestmark = pytest.mark.slow


class FakeVideoBackend:
    """A fake backend that returns a small synthetic video clip."""

    model_id = "fake-vid"

    def generate(self, prompt, **kwargs):
        n_frames = 5
        fps = 5
        frames = [
            Image.new("RGB", (64, 48), (i * 30, 100, 150))
            for i in range(n_frames)
        ]
        return VideoResult(
            frames=frames,
            fps=fps,
            width=64,
            height=48,
            duration_seconds=round(n_frames / fps, 3),
            seed=kwargs.get("seed", -1) if kwargs.get("seed") is not None else -1,
            metadata={"prompt": prompt},
        )


def _build_client():
    reg = ModalityRegistry()
    reg.register(
        MODALITY,
        FakeVideoBackend(),
        manifest={
            "model_id": "fake-vid",
            "modality": MODALITY,
            "capabilities": {
                "default_duration_seconds": 1.0,
                "default_fps": 5,
                "default_size": (64, 48),
                "device": "cpu",
            },
        },
    )
    app = create_app(registry=reg, routers={MODALITY: build_router(reg)})
    return TestClient(app)


def _fake_imageio(format_marker: bytes):
    fake = type("ii", (), {})()

    def fake_mimwrite(buf, frames, *, fps, codec, format, **_):
        buf.write(
            format_marker + f":{codec}:{fps}:{len(frames)}".encode()
        )

    fake.mimwrite = fake_mimwrite
    return fake


@pytest.mark.timeout(10)
def test_video_e2e_default_mp4_envelope():
    client = _build_client()
    fake = _fake_imageio(b"FAKEMP4")
    with patch(
        "muse.modalities.video_generation.codec._try_import_imageio",
        return_value=fake,
    ):
        r = client.post(
            "/v1/video/generations",
            json={"prompt": "a cat in a field", "model": "fake-vid"},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == "fake-vid"
    assert body["metadata"]["format"] == "mp4"
    assert body["metadata"]["frames"] == 5
    assert body["metadata"]["fps"] == 5
    assert body["metadata"]["size"] == [64, 48]
    asset = base64.b64decode(body["data"][0]["b64_json"])
    assert asset.startswith(b"FAKEMP4")
    assert b"h264" in asset


@pytest.mark.timeout(10)
def test_video_e2e_webm_envelope():
    client = _build_client()
    fake = _fake_imageio(b"FAKEWEBM")
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
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["metadata"]["format"] == "webm"
    asset = base64.b64decode(body["data"][0]["b64_json"])
    assert asset.startswith(b"FAKEWEBM")
    assert b"vp9" in asset


@pytest.mark.timeout(10)
def test_video_e2e_frames_b64_envelope():
    client = _build_client()
    r = client.post(
        "/v1/video/generations",
        json={
            "prompt": "x", "model": "fake-vid",
            "response_format": "frames_b64",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["metadata"]["format"] == "frames_b64"
    assert len(body["data"]) == 5
    for entry in body["data"]:
        png = base64.b64decode(entry["b64_json"])
        assert png[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.mark.timeout(10)
def test_video_e2e_n_2_returns_two_videos():
    client = _build_client()
    fake = _fake_imageio(b"FAKEMP4")
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


@pytest.mark.timeout(10)
def test_video_e2e_unknown_model_returns_404():
    client = _build_client()
    r = client.post(
        "/v1/video/generations",
        json={"prompt": "x", "model": "nope"},
    )
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "model_not_found"


@pytest.mark.timeout(10)
def test_video_e2e_validation_errors():
    """Out-of-range duration / fps / steps should be rejected by pydantic."""
    client = _build_client()
    for body in [
        {"prompt": "x", "model": "fake-vid", "duration_seconds": 100.0},
        {"prompt": "x", "model": "fake-vid", "fps": 200},
        {"prompt": "x", "model": "fake-vid", "steps": 999},
        {"prompt": "x", "model": "fake-vid", "size": "garbage"},
        {"prompt": "x", "model": "fake-vid", "n": 5},
    ]:
        r = client.post("/v1/video/generations", json=body)
        assert r.status_code in (400, 422), (
            f"expected validation error for {body}, got {r.status_code}"
        )


@pytest.mark.timeout(10)
def test_video_e2e_envelope_metadata_carries_duration():
    client = _build_client()
    r = client.post(
        "/v1/video/generations",
        json={
            "prompt": "x", "model": "fake-vid",
            "response_format": "frames_b64",
        },
    )
    body = r.json()
    assert body["metadata"]["duration_seconds"] == 1.0  # 5 frames @ 5fps
