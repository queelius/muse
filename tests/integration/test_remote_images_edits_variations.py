"""End-to-end /v1/images/edits + /v1/images/variations against a running muse server. Opt-in.

Requires:
- MUSE_REMOTE_SERVER set (see tests/integration/conftest.py)
- The target server has an image/generation model loaded that
  advertises supports_inpainting and supports_variations
  (defaults to sd-turbo; override via MUSE_IMAGE_MODEL_ID)

These tests validate the wire contract for v0.21.0:
- POST /v1/images/edits multipart with image + mask + prompt returns
  the OpenAI envelope with data: [{b64_json, revised_prompt}, ...]
- POST /v1/images/variations multipart with image only returns the
  OpenAI envelope with data: [{b64_json}, ...] (no revised_prompt)

All tests are slow (real GPU diffusion, 1-30s per call) and gated
behind MUSE_REMOTE_SERVER.

Naming convention:
  - test_protocol_*: hard claims muse should always satisfy
  - test_observe_*: probes that record what a particular model did
"""
from __future__ import annotations

import base64
import io
import os

import httpx
import pytest


pytestmark = pytest.mark.slow


@pytest.fixture(scope="session")
def image_model(remote_health) -> str:
    """The image/generation model id integration tests should target.

    Defaults to sd-turbo. Override via MUSE_IMAGE_MODEL_ID:

      MUSE_IMAGE_MODEL_ID=sdxl-turbo pytest tests/integration/

    Skips the test if the chosen model isn't loaded on the server.
    """
    model_id = os.environ.get("MUSE_IMAGE_MODEL_ID", "sd-turbo")
    loaded = remote_health.get("models") or []
    if model_id not in loaded:
        pytest.skip(
            f"muse server doesn't have {model_id!r} loaded "
            f"(loaded: {loaded}); pull and restart to enable this test"
        )
    return model_id


def _png_bytes(width=64, height=64, color=(0, 128, 255)) -> bytes:
    """Build minimal PNG bytes via PIL."""
    try:
        from PIL import Image
    except ImportError:
        pytest.skip("Pillow not installed in the test environment")
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _mask_bytes(width=64, height=64) -> bytes:
    """Build a fully-white grayscale PNG mask (regenerate the entire image)."""
    try:
        from PIL import Image
    except ImportError:
        pytest.skip("Pillow not installed in the test environment")
    img = Image.new("L", (width, height), 255)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_protocol_edits_returns_png_envelope(remote_url, image_model):
    """POST /v1/images/edits with image+mask+prompt returns a valid PNG envelope.

    Verifies:
      - HTTP 200
      - data has exactly one entry with b64_json + revised_prompt
      - the decoded bytes start with the PNG magic header
      - revised_prompt echoes the request prompt
    """
    src = _png_bytes(64, 64, color=(255, 0, 0))
    msk = _mask_bytes(64, 64)

    r = httpx.post(
        f"{remote_url}/v1/images/edits",
        files={
            "image": ("scene.png", src, "image/png"),
            "mask": ("mask.png", msk, "image/png"),
        },
        data={
            "prompt": "a quiet meadow at dusk",
            "model": image_model,
            "n": "1",
            "size": "64x64",
        },
        timeout=240.0,
    )
    assert r.status_code == 200, f"expected 200, got {r.status_code}: {r.text[:500]}"
    body = r.json()
    assert "data" in body and len(body["data"]) == 1
    entry = body["data"][0]
    assert "b64_json" in entry
    assert entry["revised_prompt"] == "a quiet meadow at dusk"
    decoded = base64.b64decode(entry["b64_json"])
    assert decoded[:8] == b"\x89PNG\r\n\x1a\n"


def test_protocol_variations_returns_png_envelope(remote_url, image_model):
    """POST /v1/images/variations with image only returns a valid PNG envelope.

    Verifies:
      - HTTP 200
      - data has exactly one entry with b64_json (NO revised_prompt)
      - the decoded bytes start with the PNG magic header
    """
    src = _png_bytes(64, 64, color=(0, 200, 80))

    r = httpx.post(
        f"{remote_url}/v1/images/variations",
        files={"image": ("scene.png", src, "image/png")},
        data={
            "model": image_model,
            "n": "1",
            "size": "64x64",
        },
        timeout=240.0,
    )
    assert r.status_code == 200, f"expected 200, got {r.status_code}: {r.text[:500]}"
    body = r.json()
    assert "data" in body and len(body["data"]) == 1
    entry = body["data"][0]
    assert "b64_json" in entry
    assert "revised_prompt" not in entry
    decoded = base64.b64decode(entry["b64_json"])
    assert decoded[:8] == b"\x89PNG\r\n\x1a\n"


def test_protocol_variations_n_returns_multiple_entries(remote_url, image_model):
    """Variations honors n; n=2 returns 2 entries."""
    src = _png_bytes(64, 64, color=(50, 50, 200))

    r = httpx.post(
        f"{remote_url}/v1/images/variations",
        files={"image": ("scene.png", src, "image/png")},
        data={
            "model": image_model,
            "n": "2",
            "size": "64x64",
        },
        timeout=300.0,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["data"]) == 2
    for entry in body["data"]:
        assert "b64_json" in entry
        decoded = base64.b64decode(entry["b64_json"])
        assert decoded[:8] == b"\x89PNG\r\n\x1a\n"


def test_observe_variations_differs_from_input(remote_url, image_model):
    """Observation: variations of a flat-color image differ from the input.

    A pure-color 64x64 PNG is a degenerate input; the variations
    pipeline should not return literally the same bytes. We can't
    assert pixel-distance without loading PIL on the server side, but
    output bytes inequality is a reasonable lower bar.
    """
    src = _png_bytes(64, 64, color=(40, 200, 60))

    r = httpx.post(
        f"{remote_url}/v1/images/variations",
        files={"image": ("scene.png", src, "image/png")},
        data={
            "model": image_model,
            "n": "1",
            "size": "64x64",
        },
        timeout=240.0,
    )
    assert r.status_code == 200
    out = base64.b64decode(r.json()["data"][0]["b64_json"])
    assert out != src, "variations returned input bytes verbatim"
