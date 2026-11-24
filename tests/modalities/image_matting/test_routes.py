"""HTTP contract tests for ``POST /v1/images/matting``."""
from __future__ import annotations

import base64
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from muse.core.registry import ModalityRegistry
from muse.core.server import create_app
from muse.modalities.image_matting import (
    DISCOVERY,
    MODALITY,
    MODEL_OPTIONAL_PATHS,
    build_router,
)
from muse.modalities.image_matting.runtimes.sprite_matte import SpriteMatte


def _sprite_img(size: int = 48, bg=(255, 255, 255), fg=(30, 110, 200)) -> Image.Image:
    arr = np.empty((size, size, 3), dtype=np.uint8)
    arr[...] = bg
    arr[12:36, 12:36] = fg
    return Image.fromarray(arr)


def _png(img: Image.Image, name: str = "in.png"):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return (name, buf.getvalue(), "image/png")


def _decode(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64)))


def _err(r) -> str:
    return r.json()["error"]["code"]


@pytest.fixture
def client():
    reg = ModalityRegistry()
    model = SpriteMatte(model_id="sprite-matte", hf_repo="", local_dir=None)
    reg.register(
        MODALITY,
        model,
        manifest={
            "model_id": "sprite-matte",
            "modality": MODALITY,
            "capabilities": {"device": "cpu", "memory_gb": 0.5},
        },
    )
    app = create_app(registry=reg, routers={MODALITY: build_router(reg)})
    return TestClient(app)


def test_discovery_and_model_optional_paths():
    assert MODALITY == "image/matting"
    assert MODEL_OPTIONAL_PATHS == ("/v1/images/matting",)
    [route] = DISCOVERY["routes"]
    assert route["method"] == "POST"
    assert route["path"] == "/v1/images/matting"
    assert route["mcp_tool"] == "muse_matte_image"
    assert route["file_fields"] == ["image"]
    assert route["media_fields"] == ["data[].b64_json"]


def test_matting_json_envelope(client):
    r = client.post(
        "/v1/images/matting",
        files={"image": _png(_sprite_img())},
        data={"model": "sprite-matte", "background": "auto", "defringe": "false"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "image.matting"
    assert body["model"] == "sprite-matte"
    [entry] = body["data"]
    assert entry["width"] == 48 and entry["height"] == 48
    assert entry["opaque_fraction"] == pytest.approx(0.25)
    assert entry["transparent_fraction"] == pytest.approx(0.75)

    img = _decode(entry["b64_json"])
    assert img.mode == "RGBA" and img.size == (48, 48)
    rgba = np.asarray(img)
    assert rgba[0, 0, 3] == 0
    assert rgba[24, 24, 3] == 255


def test_matting_default_model(client):
    r = client.post("/v1/images/matting", files={"image": _png(_sprite_img())})
    assert r.status_code == 200, r.text
    assert r.json()["model"] == "sprite-matte"


def test_matting_png_response_with_crop_and_hard_alpha(client):
    r = client.post(
        "/v1/images/matting",
        files={"image": _png(_sprite_img(48))},
        data={
            "background": "white",
            "hard_alpha": "true",
            "alpha_threshold": "128",
            "crop": "true",
            "response_format": "png",
        },
    )
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "image/png"
    img = Image.open(io.BytesIO(r.content))
    assert img.mode == "RGBA" and img.size == (24, 24)
    alpha = np.asarray(img)[..., 3]
    assert set(np.unique(alpha).tolist()) <= {0, 255}


def test_matting_enclosed_white_and_all_form_fields(client):
    arr = np.full((64, 64, 3), 255, dtype=np.uint8)
    arr[12:52, 12:52] = (40, 80, 120)
    arr[24:40, 24:40] = (250, 250, 250)  # enclosed white pocket

    r = client.post(
        "/v1/images/matting",
        files={"image": _png(Image.fromarray(arr))},
        data={
            "background": "white",
            "bg_threshold": "236",
            "color_tolerance": "24",
            "hard_alpha": "true",
            "alpha_threshold": "128",
            "defringe": "true",
            "fringe_threshold": "212",
            "clear_enclosed_white": "true",
            "crop": "false",
            "response_format": "json",
        },
    )
    assert r.status_code == 200, r.text
    img = _decode(r.json()["data"][0]["b64_json"])
    rgba = np.asarray(img)
    assert (rgba[24:40, 24:40, 3] == 0).all()
    assert rgba[16, 16, 3] == 255


@pytest.mark.parametrize(
    "field,value",
    [
        ("background", "purple"),
        ("response_format", "jpeg"),
        ("bg_threshold", "-1"),
        ("bg_threshold", "256"),
        ("color_tolerance", "300"),
        ("alpha_threshold", "-5"),
        ("fringe_threshold", "256"),
    ],
)
def test_matting_bad_parameters_400(client, field, value):
    r = client.post(
        "/v1/images/matting",
        files={"image": _png(_sprite_img(16))},
        data={field: value},
    )
    assert r.status_code == 400, (field, value, r.text)
    assert _err(r) == "invalid_parameter"


def test_matting_bad_parameter_wins_over_bad_image(client):
    r = client.post(
        "/v1/images/matting",
        files={"image": ("bad.png", b"not-a-png", "image/png")},
        data={"background": "invalid"},
    )
    assert r.status_code == 400 and _err(r) == "invalid_parameter"


def test_matting_undecodable_image_400(client):
    r = client.post(
        "/v1/images/matting",
        files={"image": ("bad.png", b"not-an-image", "image/png")},
    )
    assert r.status_code == 400 and _err(r) == "invalid_image"


def test_matting_byte_cap_413(client, monkeypatch):
    monkeypatch.setenv("MUSE_IMAGE_INPUT_MAX_BYTES", "50")
    r = client.post("/v1/images/matting", files={"image": _png(_sprite_img())})
    assert r.status_code == 413 and _err(r) == "payload_too_large"


def test_matting_unknown_model_404(client):
    r = client.post(
        "/v1/images/matting",
        files={"image": _png(_sprite_img(16))},
        data={"model": "does-not-exist"},
    )
    assert r.status_code == 404


def _on_event_loop() -> bool:
    import asyncio

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def test_matting_decodes_and_computes_off_event_loop(client, monkeypatch):
    import muse.modalities._common.images as image_input

    seen: dict[str, list[bool]] = {"decode": [], "matte": []}
    real_decode = image_input._bytes_to_pil
    real_matte = SpriteMatte.matte

    def spy_decode(*a, **k):
        seen["decode"].append(_on_event_loop())
        return real_decode(*a, **k)

    def spy_matte(self, *a, **k):
        seen["matte"].append(_on_event_loop())
        return real_matte(self, *a, **k)

    monkeypatch.setattr(image_input, "_bytes_to_pil", spy_decode)
    monkeypatch.setattr(SpriteMatte, "matte", spy_matte)

    r = client.post("/v1/images/matting", files={"image": _png(_sprite_img(32))})
    assert r.status_code == 200, r.text
    assert seen["decode"] == [False]
    assert seen["matte"] == [False]
