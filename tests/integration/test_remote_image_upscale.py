"""Integration tests for /v1/images/upscale (opt-in via MUSE_REMOTE_SERVER).

Skipped automatically unless `MUSE_REMOTE_SERVER` is set AND the server
has the requested upscale model loaded. Override the model id via
MUSE_UPSCALE_MODEL_ID; default is `stable-diffusion-x4-upscaler`.
"""
import base64
import io

import httpx
import pytest
from PIL import Image


pytestmark = pytest.mark.slow


def _png_bytes(width=128, height=128, color=(40, 40, 100)):
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_protocol_upscale_returns_envelope(remote_url, upscale_model):
    """Hard claim: the upscale route always returns a valid envelope."""
    src = _png_bytes(128, 128)
    r = httpx.post(
        f"{remote_url}/v1/images/upscale",
        files={"image": ("src.png", src, "image/png")},
        data={"model": upscale_model, "scale": "4", "prompt": ""},
        timeout=600.0,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "data" in body
    assert "created" in body
    assert isinstance(body["created"], int)
    entry = body["data"][0]
    assert "b64_json" in entry
    decoded = base64.b64decode(entry["b64_json"])
    assert decoded[:8] == b"\x89PNG\r\n\x1a\n"


def test_observe_upscale_increases_resolution(remote_url, upscale_model):
    """Observation: 4x upscale yields a 4x larger image.

    Hard claim for SD x4 since the model is fixed-scale, but kept as
    `observe_*` since other (future) curated upscalers may not have
    exact 4x output (e.g. AuraSR can do 2x).
    """
    src = _png_bytes(128, 128)
    r = httpx.post(
        f"{remote_url}/v1/images/upscale",
        files={"image": ("src.png", src, "image/png")},
        data={"model": upscale_model, "scale": "4"},
        timeout=600.0,
    )
    assert r.status_code == 200
    decoded = base64.b64decode(r.json()["data"][0]["b64_json"])
    out_img = Image.open(io.BytesIO(decoded))
    assert out_img.size == (512, 512)


def test_protocol_upscale_unsupported_scale_returns_400(remote_url, upscale_model):
    """SD x4 only supports scale=4. Asking for scale=8 must return 400."""
    src = _png_bytes(64, 64)
    r = httpx.post(
        f"{remote_url}/v1/images/upscale",
        files={"image": ("src.png", src, "image/png")},
        data={"model": upscale_model, "scale": "8"},
        timeout=60.0,
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_parameter"


def test_protocol_upscale_revised_prompt_echoes(remote_url, upscale_model):
    src = _png_bytes(64, 64)
    r = httpx.post(
        f"{remote_url}/v1/images/upscale",
        files={"image": ("src.png", src, "image/png")},
        data={"model": upscale_model, "scale": "4", "prompt": "razor sharp"},
        timeout=600.0,
    )
    assert r.status_code == 200
    assert r.json()["data"][0]["revised_prompt"] == "razor sharp"
