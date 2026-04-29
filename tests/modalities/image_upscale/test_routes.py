"""Tests for /v1/images/upscale router."""
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from muse.core.registry import ModalityRegistry
from muse.core.server import create_app
from muse.modalities.image_upscale.protocol import UpscaleResult
from muse.modalities.image_upscale.routes import build_router


class RecordingUpscaleModel:
    """A fake upscaler that records calls and returns a 4x-larger image."""

    def __init__(self, model_id="fake-upscale", supported_scales=(4,)):
        self.model_id = model_id
        self.supported_scales = list(supported_scales)
        self.calls: list[dict] = []

    def upscale(self, image, *, scale=None, prompt=None,
                negative_prompt=None, steps=None, guidance=None,
                seed=None, **kwargs):
        self.calls.append({
            "image": image, "scale": scale, "prompt": prompt,
            "negative_prompt": negative_prompt, "steps": steps,
            "guidance": guidance, "seed": seed, **kwargs,
        })
        ow, oh = image.size
        scl = scale or 4
        out = Image.new("RGB", (ow * scl, oh * scl), (60, 60, 60))
        return UpscaleResult(
            image=out,
            original_width=ow, original_height=oh,
            upscaled_width=ow * scl, upscaled_height=oh * scl,
            scale=scl, seed=seed if seed is not None else -1,
            metadata={"prompt": prompt or ""},
        )


def _png_bytes(width=64, height=64, color=(0, 128, 255)):
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def model():
    return RecordingUpscaleModel(supported_scales=(4,))


@pytest.fixture
def client(model):
    reg = ModalityRegistry()
    reg.register(
        "image/upscale", model,
        manifest={
            "model_id": model.model_id,
            "modality": "image/upscale",
            "capabilities": {
                "supported_scales": [4],
                "default_scale": 4,
            },
        },
    )
    app = create_app(
        registry=reg,
        routers={"image/upscale": build_router(reg)},
    )
    return TestClient(app)


def test_post_upscale_returns_envelope(client, model):
    r = client.post(
        "/v1/images/upscale",
        files={"image": ("src.png", _png_bytes(64, 64), "image/png")},
        data={"model": "fake-upscale", "scale": "4"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "data" in body
    assert len(body["data"]) == 1
    entry = body["data"][0]
    assert "b64_json" in entry
    import base64
    decoded = base64.b64decode(entry["b64_json"])
    assert decoded[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(model.calls) == 1


def test_post_upscale_response_format_url(client):
    r = client.post(
        "/v1/images/upscale",
        files={"image": ("src.png", _png_bytes(), "image/png")},
        data={
            "model": "fake-upscale", "scale": "4",
            "response_format": "url",
        },
    )
    assert r.status_code == 200
    url = r.json()["data"][0]["url"]
    assert url.startswith("data:image/png;base64,")


def test_post_upscale_n_creates_multiple_entries(client):
    r = client.post(
        "/v1/images/upscale",
        files={"image": ("src.png", _png_bytes(), "image/png")},
        data={"model": "fake-upscale", "scale": "4", "n": "3"},
    )
    assert r.status_code == 200
    assert len(r.json()["data"]) == 3


def test_post_upscale_revised_prompt_echoes(client):
    r = client.post(
        "/v1/images/upscale",
        files={"image": ("src.png", _png_bytes(), "image/png")},
        data={
            "model": "fake-upscale", "scale": "4",
            "prompt": "make it crisp",
        },
    )
    assert r.status_code == 200
    assert r.json()["data"][0]["revised_prompt"] == "make it crisp"


def test_post_upscale_revised_prompt_null_when_no_prompt(client):
    r = client.post(
        "/v1/images/upscale",
        files={"image": ("src.png", _png_bytes(), "image/png")},
        data={"model": "fake-upscale", "scale": "4"},
    )
    assert r.status_code == 200
    assert r.json()["data"][0]["revised_prompt"] is None


def test_post_upscale_unknown_model_returns_404(client):
    r = client.post(
        "/v1/images/upscale",
        files={"image": ("src.png", _png_bytes(), "image/png")},
        data={"model": "no-such", "scale": "4"},
    )
    assert r.status_code == 404
    body = r.json()
    assert "error" in body
    assert body["error"]["code"] == "model_not_found"


def test_post_upscale_unsupported_scale_returns_400(client):
    r = client.post(
        "/v1/images/upscale",
        files={"image": ("src.png", _png_bytes(), "image/png")},
        data={"model": "fake-upscale", "scale": "8"},
    )
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["code"] == "invalid_parameter"
    assert "scales" in err["message"].lower()


def test_post_upscale_empty_image_returns_400(client):
    r = client.post(
        "/v1/images/upscale",
        files={"image": ("src.png", b"", "image/png")},
        data={"model": "fake-upscale", "scale": "4"},
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_parameter"


def test_post_upscale_malformed_image_returns_400(client):
    r = client.post(
        "/v1/images/upscale",
        files={"image": ("src.png", b"not an image", "image/png")},
        data={"model": "fake-upscale", "scale": "4"},
    )
    assert r.status_code == 400


def test_post_upscale_oversize_input_returns_400(client, monkeypatch):
    monkeypatch.setenv("MUSE_UPSCALE_MAX_INPUT_SIDE", "32")
    r = client.post(
        "/v1/images/upscale",
        files={"image": ("src.png", _png_bytes(64, 64), "image/png")},
        data={"model": "fake-upscale", "scale": "4"},
    )
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["code"] == "invalid_parameter"
    assert (
        "too large" in err["message"].lower()
        or "max" in err["message"].lower()
    )


def test_post_upscale_n_over_limit_rejected(client):
    r = client.post(
        "/v1/images/upscale",
        files={"image": ("src.png", _png_bytes(), "image/png")},
        data={"model": "fake-upscale", "scale": "4", "n": "100"},
    )
    assert r.status_code in (400, 422)


def test_post_upscale_passes_prompt_steps_guidance_seed_to_backend(client, model):
    r = client.post(
        "/v1/images/upscale",
        files={"image": ("src.png", _png_bytes(), "image/png")},
        data={
            "model": "fake-upscale", "scale": "4",
            "prompt": "high detail",
            "negative_prompt": "blurry",
            "steps": "15", "guidance": "7.5", "seed": "42",
        },
    )
    assert r.status_code == 200
    call = model.calls[0]
    assert call["prompt"] == "high detail"
    assert call["negative_prompt"] == "blurry"
    assert call["steps"] == 15
    assert call["guidance"] == 7.5
    assert call["seed"] == 42


def test_post_upscale_response_includes_created_unix_timestamp(client):
    r = client.post(
        "/v1/images/upscale",
        files={"image": ("src.png", _png_bytes(), "image/png")},
        data={"model": "fake-upscale", "scale": "4"},
    )
    assert r.status_code == 200
    created = r.json()["created"]
    assert isinstance(created, int)
    assert created > 1577836800  # 2020-01-01 UTC


def test_post_upscale_invalid_response_format_returns_400(client):
    r = client.post(
        "/v1/images/upscale",
        files={"image": ("src.png", _png_bytes(), "image/png")},
        data={
            "model": "fake-upscale", "scale": "4",
            "response_format": "wat",
        },
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_parameter"


def test_post_upscale_steps_out_of_range_returns_400(client):
    r = client.post(
        "/v1/images/upscale",
        files={"image": ("src.png", _png_bytes(), "image/png")},
        data={"model": "fake-upscale", "scale": "4", "steps": "999"},
    )
    assert r.status_code == 400


def test_post_upscale_guidance_out_of_range_returns_400(client):
    r = client.post(
        "/v1/images/upscale",
        files={"image": ("src.png", _png_bytes(), "image/png")},
        data={"model": "fake-upscale", "scale": "4", "guidance": "100.0"},
    )
    assert r.status_code == 400


def test_post_upscale_seed_offset_increments_for_n_over_one(client, model):
    r = client.post(
        "/v1/images/upscale",
        files={"image": ("src.png", _png_bytes(), "image/png")},
        data={
            "model": "fake-upscale", "scale": "4",
            "n": "3", "seed": "100",
        },
    )
    assert r.status_code == 200
    seeds = [c["seed"] for c in model.calls]
    assert seeds == [100, 101, 102]
