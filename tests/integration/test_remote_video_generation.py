"""End-to-end /v1/video/generations against a running muse server. Opt-in.

Requires:
- MUSE_REMOTE_SERVER set
- The target server has the video_model loaded
  (default wan2-1-t2v-1-3b; override via MUSE_VIDEO_MODEL_ID)

These tests validate the wire contract for the video/generation
modality introduced in v0.27.0:
- POST /v1/video/generations returns a JSON envelope with `data` (list
  of `{b64_json}` entries), `model`, and `metadata` (frames, fps,
  duration_seconds, format, size).
- response_format defaults to mp4; webm and frames_b64 are also valid.

All tests are heavy (real GPU diffusion) and slow (30s+ each on a
12GB GPU for Wan 1.3B). They are marked slow and gated behind
MUSE_REMOTE_SERVER.

Naming convention:
  - test_protocol_*: claims muse should always satisfy
  - test_observe_*: probes that record what a particular model did. May
    xfail on weak models or constrained hardware.
"""
from __future__ import annotations

import base64
import os

import httpx
import pytest


pytestmark = pytest.mark.slow


@pytest.fixture(scope="session")
def video_model(remote_health) -> str:
    """The video/generation model id integration tests should target.

    Defaults to wan2-1-t2v-1-3b. Override via MUSE_VIDEO_MODEL_ID:

      MUSE_VIDEO_MODEL_ID=cogvideox-2b pytest tests/integration/

    Skips the test if the chosen model isn't loaded on the server.
    """
    model_id = os.environ.get("MUSE_VIDEO_MODEL_ID", "wan2-1-t2v-1-3b")
    loaded = remote_health.get("models") or []
    if model_id not in loaded:
        pytest.skip(
            f"muse server doesn't have {model_id!r} loaded "
            f"(loaded: {loaded}); pull and restart to enable this test"
        )
    return model_id


def _post_video(
    remote_url: str, body: dict, timeout: float = 600.0,
) -> httpx.Response:
    """POST to /v1/video/generations and return the raw response.

    Video generation is heavy: a single 2-second clip at 10 steps on
    a 12GB GPU takes 30-90s. Timeout is generous.
    """
    return httpx.post(
        f"{remote_url}/v1/video/generations",
        json=body,
        timeout=timeout,
    )


def test_protocol_returns_mp4_envelope(remote_url, video_model):
    """Default response_format=mp4 returns one mp4 asset."""
    r = _post_video(remote_url, {
        "prompt": "a flag waving in the wind",
        "model": video_model,
        "duration_seconds": 2.0,
        "steps": 10,
    })
    assert r.status_code == 200, (
        f"expected 200, got {r.status_code}: {r.text[:500]}"
    )
    body = r.json()
    assert "data" in body and len(body["data"]) == 1
    asset = base64.b64decode(body["data"][0]["b64_json"])
    # mp4 ftyp box at offset 4
    assert b"ftyp" in asset[:32], (
        f"expected mp4 ftyp box near the head, got {asset[:32]!r}"
    )
    assert body["metadata"]["format"] == "mp4"
    assert body["metadata"]["frames"] > 0
    assert body["metadata"]["fps"] > 0
    assert isinstance(body["metadata"]["size"], list)
    assert len(body["metadata"]["size"]) == 2
    assert body["model"] == video_model


def test_protocol_response_format_frames_b64(remote_url, video_model):
    """frames_b64 returns one PNG per frame.

    The number of entries should match metadata.frames.
    """
    r = _post_video(remote_url, {
        "prompt": "a butterfly fluttering",
        "model": video_model,
        "duration_seconds": 2.0,
        "steps": 10,
        "response_format": "frames_b64",
    })
    assert r.status_code == 200, (
        f"expected 200, got {r.status_code}: {r.text[:500]}"
    )
    body = r.json()
    assert body["metadata"]["format"] == "frames_b64"
    assert len(body["data"]) > 0
    png_magic = b"\x89PNG\r\n\x1a\n"
    for i, entry in enumerate(body["data"]):
        frame = base64.b64decode(entry["b64_json"])
        assert frame[:8] == png_magic, (
            f"frame {i} is not a PNG: header={frame[:8]!r}"
        )


def test_protocol_unknown_model_returns_404(remote_url):
    """Unknown model id returns OpenAI-shape 404."""
    r = _post_video(
        remote_url,
        {
            "prompt": "x",
            "model": "nonexistent-video-model-zzz",
            "duration_seconds": 2.0,
            "steps": 5,
        },
        timeout=30.0,
    )
    assert r.status_code == 404
    body = r.json()
    assert body["error"]["code"] == "model_not_found"


def test_protocol_metadata_carries_duration_seconds(remote_url, video_model):
    """metadata.duration_seconds reflects actual rendered clip length."""
    r = _post_video(remote_url, {
        "prompt": "a snowflake falling",
        "model": video_model,
        "duration_seconds": 2.0,
        "steps": 10,
        "response_format": "frames_b64",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["metadata"]["duration_seconds"] > 0
    # Reported duration should be within an order of magnitude of the request
    assert 0.5 <= body["metadata"]["duration_seconds"] <= 30.0


def test_protocol_validation_rejects_out_of_range(remote_url):
    """Pydantic validation rejects out-of-range parameters."""
    r = _post_video(
        remote_url,
        {
            "prompt": "x",
            "model": "wan2-1-t2v-1-3b",
            "duration_seconds": 100.0,
        },
        timeout=10.0,
    )
    assert r.status_code in (400, 422)
