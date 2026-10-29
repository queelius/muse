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


def test_pixelize_preview_over_side_cap_413(client, monkeypatch):
    # 32x32 art at preview_scale 3 is a 96 px preview: over a 95 px cap.
    import muse.modalities.image_pixelart.pixelize as pz

    def boom(*a, **k):  # the preview check must run before the grid vote
        raise AssertionError("_snap ran for an over-cap preview")

    monkeypatch.setattr(pz, "_snap", boom)
    monkeypatch.setenv("MUSE_PIXELART_MAX_INPUT_SIDE", "95")
    r = client.post("/v1/images/pixelize", files={"image": png(upscaled_sprite(2))},
                    data={"grid": "2", "preview_scale": "3"})
    assert r.status_code == 413, r.text
    assert err(r) == "input_too_large"
    msg = r.json()["error"]["message"]
    assert "96x96" in msg and "95" in msg
    assert "limits.pixelart_max_input_side" in msg and "MUSE_PIXELART_MAX_INPUT_SIDE" in msg


def test_pixelize_preview_at_side_cap_ok(client, monkeypatch):
    monkeypatch.setenv("MUSE_PIXELART_MAX_INPUT_SIDE", "96")
    r = client.post("/v1/images/pixelize", files={"image": png(upscaled_sprite(2))},
                    data={"grid": "2", "preview_scale": "3", "response_format": "png"})
    assert r.status_code == 200, r.text
    assert Image.open(io.BytesIO(r.content)).size == (96, 96)


def test_pixelize_target_size_preview_over_cap_rejected_early(client, monkeypatch):
    import muse.modalities.image_pixelart.pixelize as pz

    def boom(*a, **k):
        raise AssertionError("background removal ran for an over-cap preview")

    monkeypatch.setattr(pz, "_remove_background", boom)
    monkeypatch.setenv("MUSE_PIXELART_MAX_INPUT_SIDE", "64")
    r = client.post("/v1/images/pixelize", files={"image": png(upscaled_sprite(2))},
                    data={"target_size": "40x20", "preview_scale": "2"})
    assert r.status_code == 413 and err(r) == "input_too_large"
    assert "80x40" in r.json()["error"]["message"]


def test_pixelize_sheet_preview_over_cap_413(client, monkeypatch):
    # A 2x1 sheet of 32x32 cells at grid 2 is 32x16 art; scale 4 -> 128x64.
    monkeypatch.setenv("MUSE_PIXELART_MAX_INPUT_SIDE", "127")
    r = client.post("/v1/images/pixelize", files={"image": png(upscaled_sprite(2, 32, 16))},
                    data={"grid": "2", "sheet": "2x1", "preview_scale": "4",
                          "background": "none"})
    assert r.status_code == 413 and err(r) == "input_too_large"
    assert "128x64" in r.json()["error"]["message"]


def test_pixelize_grid_larger_than_image_400(client):
    r = client.post("/v1/images/pixelize", files={"image": png(sprite(64, 64))},
                    data={"grid": "100"})
    assert r.status_code == 400 and err(r) == "invalid_parameter"
    assert r.json()["error"]["message"] == "grid 100 exceeds image size 64x64"


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


def test_autotile_negative_seed_400_names_field(client):
    r = client.post("/v1/images/autotile", files={"fill": png(texture())},
                    data={"tile_size": "16", "seed": "-1"})
    assert r.status_code == 400 and err(r) == "invalid_parameter"
    assert "seed" in r.json()["error"]["message"]


def test_autotile_seeded_deterministic(client):
    def run():
        r = client.post("/v1/images/autotile", files={"fill": png(texture())},
                        data={"tile_size": "16", "seed": "7", "roughness": "0.5",
                              "response_format": "zip"})
        assert r.status_code == 200, r.text
        return r.content

    assert run() == run()


# ------------------------------------------------------------- remove_enclosed


def ring_sprite() -> Image.Image:
    """Grey ring on magenta; the 16x16 hole is magenta enclosed by the ring,
    plus a single magenta highlight pixel inside the ring band."""
    a = np.zeros((64, 64, 4), dtype=np.uint8)
    a[..., :3] = (255, 0, 255)
    a[..., 3] = 255
    a[16:48, 16:48, :3] = (120, 120, 130)
    a[24:40, 24:40, :3] = (255, 0, 255)
    a[18, 18, :3] = (255, 0, 255)
    return Image.fromarray(a)


@pytest.mark.parametrize("flag,hole_alpha", [("true", 0), ("false", 255), (None, 255)])
def test_pixelize_remove_enclosed(client, flag, hole_alpha):
    data = {"grid": "1", "palette": "none", "response_format": "json"}
    if flag is not None:
        data["remove_enclosed"] = flag
    r = client.post("/v1/images/pixelize", files={"image": png(ring_sprite())}, data=data)
    assert r.status_code == 200, r.text
    body = r.json()
    out = np.asarray(decode(body["data"][0]["b64_json"]).convert("RGBA"))
    assert np.all(out[24:40, 24:40, 3] == hole_alpha)
    assert out[18, 18, 3] == 255  # a 1-cell highlight is never punched out
    assert np.all(out[:16, :, 3] == 0)
    assert body["metadata"]["remove_enclosed"] is (flag == "true")


def test_pixelize_remove_enclosed_bad_type_rejected_like_other_bools(client):
    def code(field):
        r = client.post("/v1/images/pixelize", files={"image": png(ring_sprite())},
                        data={field: "maybe"})
        return r.status_code
    assert code("remove_enclosed") == code("seamless")
    assert code("remove_enclosed") in (400, 422)


def test_pixelize_remove_enclosed_in_openapi(client):
    spec = client.get("/openapi.json").json()
    schemas = spec["components"]["schemas"]
    ref = spec["paths"]["/v1/images/pixelize"]["post"]["requestBody"]["content"][
        "multipart/form-data"]["schema"]["$ref"].rsplit("/", 1)[-1]
    prop = schemas[ref]["properties"]["remove_enclosed"]
    assert prop["type"] == "boolean" and prop["default"] is False
    assert "enclosed" in prop["description"]


# ------------------------------------------- option validation before decode


@pytest.mark.parametrize("field,value", [("outline", "#12345"), ("outline", "bright"),
                                         ("background", "purple")])
def test_pixelize_bad_option_wins_over_undecodable_image(client, field, value):
    r = client.post("/v1/images/pixelize",
                    files={"image": ("x.png", b"not an image", "image/png")},
                    data={field: value})
    assert r.status_code == 400 and err(r) == "invalid_parameter", r.text
    assert field in r.json()["error"]["message"]


@pytest.mark.parametrize("field,value", [("outline", "#12345"), ("background", "purple")])
def test_pixelize_bad_option_wins_over_oversized_image(client, monkeypatch, field, value):
    monkeypatch.setenv("MUSE_IMAGE_INPUT_MAX_BYTES", "100")
    r = client.post("/v1/images/pixelize", files={"image": png(upscaled_sprite(4))},
                    data={field: value})
    assert r.status_code == 400 and err(r) == "invalid_parameter", r.text
    monkeypatch.setenv("MUSE_PIXELART_MAX_INPUT_SIDE", "64")
    monkeypatch.delenv("MUSE_IMAGE_INPUT_MAX_BYTES")
    r = client.post("/v1/images/pixelize", files={"image": png(upscaled_sprite(4))},
                    data={field: value})
    assert r.status_code == 400 and err(r) == "invalid_parameter", r.text


def test_pixelize_hex_options_accepted(client):
    r = client.post("/v1/images/pixelize", files={"image": png(upscaled_sprite(4))},
                    data={"outline": "#102030", "background": " #FF00ff "})
    assert r.status_code == 200, r.text


# ----------------------------------------------------------- off the event loop


def _on_event_loop() -> bool:
    import asyncio

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


@pytest.fixture
def loop_spy(monkeypatch):
    """Record, per call, whether image decoding and the pixel work ran on a
    thread with a running event loop (the handler's) or in a worker thread."""
    import muse.modalities.image_generation.image_input as image_input

    seen: dict[str, list[bool]] = {"decode": [], "pixelize": [], "autotile": []}
    real_decode = image_input._bytes_to_pil

    def decode(*a, **k):
        seen["decode"].append(_on_event_loop())
        return real_decode(*a, **k)

    monkeypatch.setattr(image_input, "_bytes_to_pil", decode)
    for name in ("pixelize", "autotile"):
        real = getattr(PixelTools, name)

        def spy(self, *a, _real=real, _name=name, **k):
            seen[_name].append(_on_event_loop())
            return _real(self, *a, **k)

        monkeypatch.setattr(PixelTools, name, spy)
    return seen


def test_pixelize_decodes_and_computes_off_the_event_loop(client, loop_spy):
    r = client.post("/v1/images/pixelize", files={"image": png(upscaled_sprite(4))})
    assert r.status_code == 200, r.text
    assert loop_spy["decode"] == [False]
    assert loop_spy["pixelize"] == [False]


def test_autotile_decodes_and_computes_off_the_event_loop(client, loop_spy):
    r = client.post("/v1/images/autotile", data={"tile_size": "16"},
                    files={"fill": png(texture(32)), "base": png(texture(32, (140, 100, 60)))})
    assert r.status_code == 200, r.text
    assert loop_spy["decode"] == [False, False]
    assert loop_spy["autotile"] == [False]
