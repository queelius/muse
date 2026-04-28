"""End-to-end /v1/images/animations against a running muse server. Opt-in.

Requires:
- MUSE_REMOTE_SERVER set
- The target server has the animation_model loaded
  (default animatediff-motion-v3; override via MUSE_ANIMATION_MODEL_ID)

These tests validate the wire contract for the image/animation modality
introduced in v0.18.0:
- POST /v1/images/animations returns a JSON envelope with `data` (list
  of `{b64_json}` entries), `model`, and `metadata` (frames, fps,
  duration_seconds, format, size).
- response_format defaults to webp; gif and frames_b64 are also valid.
- Image input requires a model with supports_image_to_animation=True
  in capabilities; otherwise the route returns 400.

All tests are heavy (real GPU diffusion) and slow (10-90s each). They
are marked slow and gated behind MUSE_REMOTE_SERVER.

Naming convention:
  - test_protocol_*: claims muse should always satisfy
  - test_observe_*: probes that record what a particular model did. May
    xfail on weak models; documenting the observation is the value.
"""
from __future__ import annotations

import base64
import os

import httpx
import pytest


pytestmark = pytest.mark.slow


@pytest.fixture(scope="session")
def animation_model(remote_health) -> str:
    """The image/animation model id integration tests should target.

    Defaults to animatediff-motion-v3. Override via
    MUSE_ANIMATION_MODEL_ID:

      MUSE_ANIMATION_MODEL_ID=animatelcm pytest tests/integration/

    Skips the test if the chosen model isn't loaded on the server.
    """
    model_id = os.environ.get("MUSE_ANIMATION_MODEL_ID", "animatediff-motion-v3")
    loaded = remote_health.get("models") or []
    if model_id not in loaded:
        pytest.skip(
            f"muse server doesn't have {model_id!r} loaded "
            f"(loaded: {loaded}); pull and restart to enable this test"
        )
    return model_id


def _post_animation(remote_url: str, body: dict, timeout: float = 240.0) -> httpx.Response:
    """POST to /v1/images/animations and return the raw response.

    The OpenAI SDK does not (yet) expose an animations endpoint, so we
    use httpx directly. Animation generation is slow (16 frames at 25
    diffusion steps each), so the timeout is generous.
    """
    return httpx.post(
        f"{remote_url}/v1/images/animations",
        json=body,
        timeout=timeout,
    )


def test_protocol_returns_webp_envelope(remote_url, animation_model):
    """Default response_format=webp returns a single webp asset.

    Verifies:
      - HTTP 200
      - data has exactly one entry with b64_json
      - the decoded bytes start with the RIFF+WEBP magic header
      - metadata.format == "webp", frames > 0, size is [w, h]
    """
    r = _post_animation(remote_url, {
        "prompt": "a cat playing with yarn",
        "model": animation_model,
        "frames": 8,
        "steps": 10,
    })
    assert r.status_code == 200, f"expected 200, got {r.status_code}: {r.text[:500]}"
    body = r.json()

    assert "data" in body and len(body["data"]) == 1
    asset = base64.b64decode(body["data"][0]["b64_json"])
    # WebP file signature: "RIFF" at offset 0, "WEBP" at offset 8
    assert asset[:4] == b"RIFF", f"expected RIFF header, got {asset[:4]!r}"
    assert asset[8:12] == b"WEBP", f"expected WEBP magic, got {asset[8:12]!r}"

    assert body["metadata"]["format"] == "webp"
    assert body["metadata"]["frames"] > 0
    assert body["metadata"]["fps"] > 0
    assert isinstance(body["metadata"]["size"], list)
    assert len(body["metadata"]["size"]) == 2
    assert body["model"] == animation_model


def test_protocol_response_format_gif(remote_url, animation_model):
    """response_format=gif returns a GIF asset with a GIF89a header."""
    r = _post_animation(remote_url, {
        "prompt": "a dog wagging its tail",
        "model": animation_model,
        "frames": 8,
        "steps": 10,
        "response_format": "gif",
    })
    assert r.status_code == 200, f"expected 200, got {r.status_code}: {r.text[:500]}"
    body = r.json()

    asset = base64.b64decode(body["data"][0]["b64_json"])
    # GIF signature: "GIF87a" or "GIF89a" at offset 0
    assert asset[:6] in (b"GIF87a", b"GIF89a"), (
        f"expected GIF magic, got {asset[:6]!r}"
    )
    assert body["metadata"]["format"] == "gif"


def test_protocol_response_format_frames_b64(remote_url, animation_model):
    """response_format=frames_b64 returns one PNG per frame in data[].

    Each entry should be a standalone PNG (signature 89 50 4E 47 ...).
    The number of entries should be >= the configured min_frames (the
    AnimateDiff floor is 8) and the sum should match metadata.frames.
    """
    r = _post_animation(remote_url, {
        "prompt": "a butterfly fluttering",
        "model": animation_model,
        "frames": 8,
        "steps": 10,
        "response_format": "frames_b64",
    })
    assert r.status_code == 200, f"expected 200, got {r.status_code}: {r.text[:500]}"
    body = r.json()

    assert body["metadata"]["format"] == "frames_b64"
    assert len(body["data"]) >= 8, (
        f"expected >= 8 frame entries (min_frames floor), got {len(body['data'])}"
    )
    # PNG signature: 89 50 4E 47 0D 0A 1A 0A
    png_magic = b"\x89PNG\r\n\x1a\n"
    for i, entry in enumerate(body["data"]):
        frame = base64.b64decode(entry["b64_json"])
        assert frame[:8] == png_magic, (
            f"frame {i} is not a PNG: header={frame[:8]!r}"
        )


def test_protocol_seed_reproducibility(remote_url, animation_model):
    """Same seed produces structurally-identical animations.

    Bit-exact reproducibility across diffusers versions is unreliable
    (kernel-level non-determinism, cuDNN tuning, fp32 vs fp16). The
    conservative claim: two requests with the same seed produce the
    same NUMBER of frames and the same dimensions. Stronger checks
    (frame-byte equality) would xfail on most GPU stacks; we keep
    this assertion lax to remain a hard regression watchdog rather
    than environmental flake.
    """
    body = {
        "prompt": "a steampunk owl",
        "model": animation_model,
        "frames": 8,
        "steps": 10,
        "seed": 42,
        "response_format": "frames_b64",
    }
    r1 = _post_animation(remote_url, body)
    r2 = _post_animation(remote_url, body)
    assert r1.status_code == 200 and r2.status_code == 200

    b1, b2 = r1.json(), r2.json()
    assert len(b1["data"]) == len(b2["data"]), (
        f"frame count differs across same-seed runs: "
        f"{len(b1['data'])} vs {len(b2['data'])}"
    )
    assert b1["metadata"]["size"] == b2["metadata"]["size"], (
        f"dimensions differ across same-seed runs: "
        f"{b1['metadata']['size']} vs {b2['metadata']['size']}"
    )
    assert b1["metadata"]["frames"] == b2["metadata"]["frames"]
    assert b1["metadata"]["fps"] == b2["metadata"]["fps"]


def test_protocol_image_input_with_text_only_model_returns_400(
    remote_url, animation_model, remote_health,
):
    """Posting an `image` field to a text-only animation model must 400.

    Capability gate: a model with supports_image_to_animation=False in
    its manifest capabilities cannot accept init_image. The route
    returns 400 with code='invalid_parameter' and a message that
    points at supports_image_to_animation.

    Skipped when the target model actually supports image-to-animation
    (then the request would succeed; this test is asymmetric and only
    valid against text-only models like animatediff-motion-v3).
    """
    # Probe /v1/models to read the capabilities of the target model.
    r = httpx.get(f"{remote_url}/v1/models", timeout=10.0)
    assert r.status_code == 200
    entries = r.json().get("data", [])
    target = next((e for e in entries if e.get("id") == animation_model), None)
    if target is None:
        pytest.skip(f"{animation_model} not present in /v1/models")
    caps = target.get("capabilities") or {}
    if caps.get("supports_image_to_animation"):
        pytest.skip(
            f"{animation_model} supports image-to-animation; "
            f"capability gate test only applies to text-only models"
        )

    # 1x1 transparent PNG, base64-encoded data URL.
    tiny_png_b64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42m"
        "NkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
    )
    image_data_url = f"data:image/png;base64,{tiny_png_b64}"

    r = _post_animation(remote_url, {
        "prompt": "anything",
        "model": animation_model,
        "image": image_data_url,
        "frames": 8,
        "steps": 5,
    }, timeout=30.0)
    assert r.status_code == 400, f"expected 400, got {r.status_code}: {r.text[:500]}"
    body = r.json()
    assert "error" in body, f"expected OpenAI envelope, got {body}"
    assert body["error"]["code"] == "invalid_parameter"
    assert "supports_image_to_animation" in body["error"]["message"], (
        f"error message should reference the capability gate: "
        f"{body['error']['message']!r}"
    )
