"""End-to-end: /v1/images/upscale through FastAPI + codec correctly.

Uses a fake ImageUpscaleModel backend; no real weights. The fake
upscaler returns a 4x larger image of a fixed color, which the codec
encodes to PNG and decodes back end-to-end.
"""
import base64
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from muse.core.registry import ModalityRegistry
from muse.core.server import create_app
from muse.modalities.image_upscale import (
    MODALITY,
    UpscaleResult,
    build_router,
)


pytestmark = pytest.mark.slow

pytest.importorskip("PIL.Image")


def _png_bytes(width=32, height=32, color=(5, 10, 15)):
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class FakeUpscaler:
    model_id = "fake-upscaler"
    supported_scales = [4]

    def upscale(self, image, *, scale=None, **kwargs):
        ow, oh = image.size
        scl = scale or 4
        out = Image.new("RGB", (ow * scl, oh * scl), (10, 200, 50))
        return UpscaleResult(
            image=out,
            original_width=ow, original_height=oh,
            upscaled_width=ow * scl, upscaled_height=oh * scl,
            scale=scl, seed=-1,
            metadata={"prompt": kwargs.get("prompt") or ""},
        )


def _build_client():
    reg = ModalityRegistry()
    reg.register(
        MODALITY, FakeUpscaler(),
        manifest={
            "model_id": "fake-upscaler",
            "modality": MODALITY,
            "capabilities": {"supported_scales": [4], "default_scale": 4},
        },
    )
    app = create_app(registry=reg, routers={MODALITY: build_router(reg)})
    return TestClient(app)


@pytest.mark.timeout(10)
def test_upscale_e2e_multipart_b64_json():
    client = _build_client()
    r = client.post(
        "/v1/images/upscale",
        files={"image": ("src.png", _png_bytes(64, 64), "image/png")},
        data={"model": "fake-upscaler", "scale": "4", "prompt": "neat"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "data" in body
    decoded = base64.b64decode(body["data"][0]["b64_json"])
    assert decoded[:8] == b"\x89PNG\r\n\x1a\n"
    out_img = Image.open(io.BytesIO(decoded))
    # 64x64 source upscaled by 4x = 256x256
    assert out_img.size == (256, 256)


@pytest.mark.timeout(10)
def test_upscale_e2e_multipart_url_response_format():
    client = _build_client()
    r = client.post(
        "/v1/images/upscale",
        files={"image": ("src.png", _png_bytes(32, 32), "image/png")},
        data={
            "model": "fake-upscaler", "scale": "4",
            "response_format": "url",
        },
    )
    assert r.status_code == 200
    url = r.json()["data"][0]["url"]
    assert url.startswith("data:image/png;base64,")
    # Decode the data URL back into bytes and confirm 4x upscale
    payload = url.split(",", 1)[1]
    decoded = base64.b64decode(payload)
    out_img = Image.open(io.BytesIO(decoded))
    assert out_img.size == (128, 128)  # 32 * 4


@pytest.mark.timeout(10)
def test_upscale_e2e_revised_prompt_echoes_input():
    client = _build_client()
    r = client.post(
        "/v1/images/upscale",
        files={"image": ("src.png", _png_bytes(32, 32), "image/png")},
        data={
            "model": "fake-upscaler", "scale": "4",
            "prompt": "high detail clarity",
        },
    )
    body = r.json()
    assert body["data"][0]["revised_prompt"] == "high detail clarity"


@pytest.mark.timeout(10)
def test_upscale_e2e_n_creates_multiple_entries():
    client = _build_client()
    r = client.post(
        "/v1/images/upscale",
        files={"image": ("src.png", _png_bytes(32, 32), "image/png")},
        data={"model": "fake-upscaler", "scale": "4", "n": "2"},
    )
    assert r.status_code == 200
    assert len(r.json()["data"]) == 2


@pytest.mark.timeout(10)
def test_upscale_e2e_unsupported_scale_returns_400():
    client = _build_client()
    r = client.post(
        "/v1/images/upscale",
        files={"image": ("src.png", _png_bytes(32, 32), "image/png")},
        data={"model": "fake-upscaler", "scale": "8"},
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_parameter"


@pytest.mark.timeout(10)
def test_upscale_e2e_unknown_model_returns_404():
    client = _build_client()
    r = client.post(
        "/v1/images/upscale",
        files={"image": ("src.png", _png_bytes(32, 32), "image/png")},
        data={"model": "nope", "scale": "4"},
    )
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "model_not_found"
