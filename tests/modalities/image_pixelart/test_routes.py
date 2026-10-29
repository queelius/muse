"""HTTP contract of /v1/images/pixelize and /v1/images/autotile.

The real ``PixelTools`` runtime serves the routes (it has no weights and no
heavy deps) behind a ``ModalityRegistry`` + ``create_app``.
"""
from __future__ import annotations

import base64
import io
import json
import zipfile

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from muse.core.registry import ModalityRegistry
from muse.core.server import create_app
from muse.modalities.image_pixelart import build_router
from muse.modalities.image_pixelart.palettes import NAMED_PALETTES
from muse.modalities.image_pixelart.runtime import PixelTools


def sprite(w=32, h=32, seed=0) -> Image.Image:
    """A random-colour 1:1 sprite with a 2 px transparent border."""
    rng = np.random.default_rng(seed)
    pal = np.array([[200, 40, 40], [40, 160, 60], [40, 60, 200], [230, 200, 60]],
                   dtype=np.uint8)
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[..., :3] = pal[rng.integers(0, len(pal), size=(h, w))]
    arr[..., 3] = 255
    arr[:2] = 0
    arr[-2:] = 0
    arr[:, :2] = 0
    arr[:, -2:] = 0
    return Image.fromarray(arr)


def upscaled_sprite(k: int, w=32, h=32) -> Image.Image:
    s = sprite(w, h)
    return s.resize((w * k, h * k), Image.Resampling.NEAREST)


def texture(side=48, rgb=(70, 140, 60), seed=3) -> Image.Image:
    rng = np.random.default_rng(seed)
    a = np.empty((side, side, 3), dtype=np.int16)
    a[...] = rgb
    a += rng.integers(-30, 31, size=a.shape).astype(np.int16)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def png(img: Image.Image, name="in.png"):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return (name, buf.getvalue(), "image/png")


def decode(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64)))


def assert_binary_alpha(img: Image.Image):
    alpha = np.asarray(img.convert("RGBA"))[..., 3]
    assert set(np.unique(alpha).tolist()) <= {0, 255}


def err(r) -> str:
    return r.json()["error"]["code"]


@pytest.fixture
def client():
    reg = ModalityRegistry()
    model = PixelTools(model_id="pixel-tools", hf_repo="", local_dir=None)
    reg.register("image/pixelart", model, manifest={
        "model_id": "pixel-tools", "modality": "image/pixelart",
        "capabilities": {"device": "cpu", "memory_gb": 1.5},
    })
    app = create_app(registry=reg, routers={"image/pixelart": build_router(reg)})
    return TestClient(app)


# ------------------------------------------------------------------ pixelize


def test_pixelize_json_envelope(client):
    r = client.post("/v1/images/pixelize", files={"image": png(upscaled_sprite(4))},
                    data={"model": "pixel-tools", "palette": "db32"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "image.pixelize"
    assert body["model"] == "pixel-tools"
    assert body["id"].startswith("pixelize-")
    assert body["grid"]["cell"] == 4 and body["grid"]["detected"] is True
    assert set(body["grid"]) == {"cell", "offset_x", "offset_y", "confidence", "detected"}
    assert body["size"] == [32, 32]
    assert body["palette"] == list(NAMED_PALETTES["db32"])
    assert isinstance(body["background_removed"], bool)
    assert isinstance(body["metadata"], dict)
    [entry] = body["data"]
    img = decode(entry["b64_json"])
    assert img.size == (32, 32) and img.mode == "RGBA"
    assert_binary_alpha(img)


def test_pixelize_default_model(client):
    r = client.post("/v1/images/pixelize", files={"image": png(upscaled_sprite(4))})
    assert r.status_code == 200, r.text
    assert r.json()["model"] == "pixel-tools"


def test_pixelize_png_response(client):
    r = client.post("/v1/images/pixelize", files={"image": png(upscaled_sprite(4))},
                    data={"model": "pixel-tools", "response_format": "png",
                          "preview_scale": "3"})
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "image/png"
    img = Image.open(io.BytesIO(r.content))
    assert img.size == (96, 96)
    assert_binary_alpha(img)
    # nearest-neighbour preview: each art pixel is a solid 3x3 block
    a = np.asarray(img.convert("RGBA"))
    assert (a[::3, ::3] == a[1::3, 1::3]).all() and (a[::3, ::3] == a[2::3, 2::3]).all()


def test_pixelize_explicit_grid_and_target_size(client):
    r = client.post("/v1/images/pixelize", files={"image": png(upscaled_sprite(4))},
                    data={"grid": "4", "palette": "none"})
    assert r.status_code == 200, r.text
    assert r.json()["grid"]["cell"] == 4 and r.json()["size"] == [32, 32]
    r = client.post("/v1/images/pixelize", files={"image": png(upscaled_sprite(4))},
                    data={"target_size": "16x16", "palette": "pico-8"})
    assert r.status_code == 200, r.text
    assert r.json()["size"] == [16, 16]


def test_pixelize_sheet_and_flags(client):
    r = client.post("/v1/images/pixelize", files={"image": png(upscaled_sprite(4, 64, 32))},
                    data={"sheet": "2x1", "palette": "auto:8", "outline": "dark",
                          "seamless": "true", "cleanup": "false", "background": "none"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["metadata"]["sheet"] == [2, 1]
    assert body["metadata"]["cleanup_applied"] is False
    assert body["metadata"]["seamless"] is True


@pytest.mark.parametrize("field,value", [
    ("palette", "not-a-palette"),
    ("palette", "auto:1"),
    ("grid", "zero"),
    ("grid", "0"),
    ("target_size", "16by16"),
    ("sheet", "2x"),
    ("background", "purple"),
    ("outline", "#12345"),
    ("alpha_threshold", "256"),
    ("response_format", "jpeg"),
])
def test_pixelize_bad_parameters_400(client, field, value):
    r = client.post("/v1/images/pixelize", files={"image": png(upscaled_sprite(4))},
                    data={field: value})
    assert r.status_code == 400, (field, value, r.text)
    assert err(r) == "invalid_parameter"


def test_pixelize_bad_palette_400(client):
    r = client.post("/v1/images/pixelize", files={"image": png(upscaled_sprite(4))},
                    data={"model": "pixel-tools", "palette": "rainbow"})
    assert r.status_code == 400
    assert err(r) == "invalid_parameter"
    assert "rainbow" in r.json()["error"]["message"]


def test_pixelize_grid_and_target_size_conflict_400(client):
    r = client.post("/v1/images/pixelize", files={"image": png(upscaled_sprite(4))},
                    data={"grid": "4", "target_size": "16x16"})
    assert r.status_code == 400 and err(r) == "invalid_parameter"


def test_pixelize_oversize_413(client, monkeypatch):
    monkeypatch.setenv("MUSE_PIXELART_MAX_INPUT_SIDE", "64")
    r = client.post("/v1/images/pixelize", files={"image": png(upscaled_sprite(4))})
    assert r.status_code == 413
    assert err(r) == "input_too_large"
    assert "MUSE_PIXELART_MAX_INPUT_SIDE" in r.json()["error"]["message"]


def test_pixelize_byte_cap_413(client, monkeypatch):
    monkeypatch.setenv("MUSE_IMAGE_INPUT_MAX_BYTES", "100")
    r = client.post("/v1/images/pixelize", files={"image": png(upscaled_sprite(4))})
    assert r.status_code == 413
    assert err(r) == "payload_too_large"


def test_pixelize_undecodable_image_400(client):
    r = client.post("/v1/images/pixelize",
                    files={"image": ("x.png", b"not an image", "image/png")})
    assert r.status_code == 400 and err(r) == "invalid_image"


@pytest.mark.parametrize("scale", ["0", "17"])
def test_pixelize_preview_scale_bounds(client, scale):
    r = client.post("/v1/images/pixelize", files={"image": png(upscaled_sprite(4))},
                    data={"preview_scale": scale})
    assert r.status_code == 400 and err(r) == "invalid_parameter"


def test_pixelize_unknown_model_404(client):
    r = client.post("/v1/images/pixelize", files={"image": png(upscaled_sprite(4))},
                    data={"model": "nope"})
    assert r.status_code == 404


# ------------------------------------------------------------------ autotile


def test_autotile_from_fill_json(client):
    T = 16
    r = client.post("/v1/images/autotile", files={"fill": png(texture())},
                    data={"model": "pixel-tools", "tile_size": str(T), "name": "grass"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "image.autotile" and body["model"] == "pixel-tools"
    assert body["id"].startswith("autotile-")
    assert body["tile_size"] == T
    files = body["files"]
    a2 = decode(files["a2"]["b64_json"])
    blob = decode(files["blob47"]["b64_json"])
    assert a2.size == (2 * T, 3 * T)
    assert blob.size == (8 * T, 6 * T)
    assert_binary_alpha(a2)
    assert_binary_alpha(blob)
    assert "res://grass_blob47.png" in files["godot_tres"]
    assert files["godot_tres"].startswith("[gd_resource")
    assert 'source="grass_blob47.png"' in files["tiled_tsx"]
    assert files["tiled_tsx"].startswith("<?xml")
    layout = body["layout"]
    assert len(layout) == 47
    assert [t["index"] for t in layout] == list(range(47))
    first = layout[9]
    assert first["x"] == 1 and first["y"] == 1 and first["unit"] == "tile"
    assert first["pixel_x"] == T and first["pixel_y"] == T
    assert 0 <= first["mask"] <= 255
    meta = body["metadata"]
    assert meta["source"] == "fill" and meta["base"] is False
    assert meta["mask_bits"]["N"] == 1 and meta["mask_bits"]["NW"] == 128
    assert meta["dropped_shape"] == 47


def test_autotile_from_fill_with_base_is_opaque(client):
    r = client.post("/v1/images/autotile",
                    files={"fill": png(texture()), "base": png(texture(rgb=(120, 90, 50)))},
                    data={"tile_size": "24", "palette": "db32"})
    assert r.status_code == 200, r.text
    a2 = np.asarray(decode(r.json()["files"]["a2"]["b64_json"]).convert("RGBA"))
    assert (a2[..., 3] == 255).all()
    assert r.json()["metadata"]["base"] is True


def test_autotile_default_tile_size_is_48(client):
    r = client.post("/v1/images/autotile", files={"fill": png(texture())})
    assert r.status_code == 200, r.text
    assert r.json()["tile_size"] == 48
    assert decode(r.json()["files"]["a2"]["b64_json"]).size == (96, 144)


def _a2_block(T=16) -> Image.Image:
    r_img = Image.new("RGBA", (2 * T, 3 * T), (90, 150, 70, 255))
    a = np.array(r_img)
    a[: T // 2, : T // 2, 3] = 100  # semi-transparent: must be hardened
    return Image.fromarray(a)


def test_autotile_from_a2_zip(client):
    T = 16
    r = client.post("/v1/images/autotile", files={"a2": png(_a2_block(T))},
                    data={"tile_size": str(T), "name": "grass", "response_format": "zip"})
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "application/zip"
    assert r.headers["content-disposition"] == 'attachment; filename="grass.zip"'
    z = zipfile.ZipFile(io.BytesIO(r.content))
    assert sorted(z.namelist()) == sorted([
        "grass_a2.png", "grass_blob47.png", "grass.tres", "grass.tsx", "grass_layout.json",
    ])
    a2 = Image.open(io.BytesIO(z.read("grass_a2.png")))
    assert a2.size == (2 * T, 3 * T)
    assert_binary_alpha(a2)  # Ruling G: imported A2 is alpha-hardened at 128
    assert np.asarray(a2.convert("RGBA"))[0, 0, 3] == 0
    layout = json.loads(z.read("grass_layout.json"))
    assert len(layout) == 47 and layout[0]["unit"] == "tile"
    assert "res://grass_blob47.png" in z.read("grass.tres").decode()


def test_autotile_a2_ignores_style_fields(client):
    r = client.post("/v1/images/autotile", files={"a2": png(_a2_block(16))},
                    data={"tile_size": "16", "roughness": "0.9", "outline": "none"})
    assert r.status_code == 200, r.text
    meta = r.json()["metadata"]
    assert meta["source"] == "a2"
    assert sorted(meta["ignored"]) == ["outline", "roughness"]


def test_autotile_wrong_a2_size_400(client):
    r = client.post("/v1/images/autotile",
                    files={"a2": png(Image.new("RGBA", (100, 144), (1, 2, 3, 255)))},
                    data={"tile_size": "48"})
    assert r.status_code == 400 and err(r) == "invalid_parameter"
    assert "96x144" in r.json()["error"]["message"]


def test_autotile_requires_exactly_one_source(client):
    r = client.post("/v1/images/autotile", data={"tile_size": "16"})
    assert r.status_code == 400 and err(r) == "invalid_parameter"
    r = client.post("/v1/images/autotile",
                    files={"a2": png(_a2_block(16)), "fill": png(texture())},
                    data={"tile_size": "16"})
    assert r.status_code == 400 and err(r) == "invalid_parameter"


def test_autotile_base_needs_fill(client):
    r = client.post("/v1/images/autotile",
                    files={"a2": png(_a2_block(16)), "base": png(texture())},
                    data={"tile_size": "16"})
    assert r.status_code == 400 and err(r) == "invalid_parameter"


@pytest.mark.parametrize("size", ["20", "64", "8"])
def test_autotile_tile_size_must_be_allowed(client, size):
    r = client.post("/v1/images/autotile", files={"fill": png(texture())},
                    data={"tile_size": size})
    assert r.status_code == 400 and err(r) == "invalid_parameter"


@pytest.mark.parametrize("field,value", [
    ("name", "Bad Name"),
    ("name", "../x"),
    ("roughness", "1.5"),
    ("outline", "violet"),
    ("palette", "nope"),
    ("response_format", "tar"),
    ("edge_width", "40"),
    ("corner_radius", "99"),
])
def test_autotile_bad_parameters_400(client, field, value):
    r = client.post("/v1/images/autotile", files={"fill": png(texture())},
                    data={"tile_size": "16", field: value})
    assert r.status_code == 400, (field, value, r.text)
    assert err(r) == "invalid_parameter"


def test_autotile_library_no_room_is_400_not_500(client):
    # build_a2 raises "no room" for T=16, edge_width 4, roughness 1.0.
    r = client.post("/v1/images/autotile", files={"fill": png(texture())},
                    data={"tile_size": "16", "edge_width": "4", "roughness": "1.0"})
    assert r.status_code == 400 and err(r) == "invalid_parameter"
    assert "room" in r.json()["error"]["message"]


def test_autotile_fill_oversize_413(client, monkeypatch):
    monkeypatch.setenv("MUSE_PIXELART_MAX_INPUT_SIDE", "32")
    r = client.post("/v1/images/autotile", files={"fill": png(texture(side=48))},
                    data={"tile_size": "16"})
    assert r.status_code == 413 and err(r) == "input_too_large"


def test_autotile_seeded_deterministic(client):
    def run():
        r = client.post("/v1/images/autotile", files={"fill": png(texture())},
                        data={"tile_size": "16", "seed": "7", "roughness": "0.5",
                              "response_format": "zip"})
        assert r.status_code == 200, r.text
        return r.content

    assert run() == run()
