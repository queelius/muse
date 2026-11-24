"""Unit and HTTP contract tests for sprite-sheet baking (POST /v1/images/spritesheet)."""
from __future__ import annotations

import asyncio
import base64
import io
import json
import zipfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from muse.core.registry import ModalityRegistry
from muse.core.server import create_app
from muse.modalities.image_pixelart import PixelartClient, build_router
from muse.modalities.image_pixelart.palettes import NAMED_PALETTES, hex_to_rgb, parse_palette
from muse.modalities.image_pixelart.runtime import PixelTools
from muse.modalities.image_pixelart.spritesheet import (
    DIRECTION_8_NAMES,
    SpritesheetOptions,
    SpritesheetTooLargeError,
    bake_spritesheet,
    filter_stray_components,
    fit_and_align,
    flood_key_background,
    mirror_5to8,
    slice_grid,
)


def _png_upload(img: Image.Image, name: str = "sheet.png"):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return (name, buf.getvalue(), "image/png")


def _decode_b64(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64)))


def _assert_binary_alpha(img: Image.Image) -> None:
    alpha = np.asarray(img.convert("RGBA"))[..., 3]
    assert set(np.unique(alpha).tolist()) <= {0, 255}


def _make_3x2_white_sheet(cell_w: int = 64, cell_h: int = 64) -> Image.Image:
    """Synthetic 3x2 directional actor sheet on solid white, with asymmetric
    shapes in the first 5 cells (each with a distinct marker on the right side
    so horizontal mirroring is verifiable) and a dummy 6th cell."""
    arr = np.full((cell_h * 2, cell_w * 3, 4), 255, dtype=np.uint8)
    colors = [
        (190, 45, 45),   # 0: up
        (45, 165, 65),   # 1: north_east
        (45, 75, 200),   # 2: side
        (210, 150, 40),  # 3: south_east
        (150, 50, 180),  # 4: down
        (80, 80, 80),    # 5: unused 6th cell
    ]
    for idx, rgb in enumerate(colors):
        r, c = divmod(idx, 3)
        y0, x0 = r * cell_h, c * cell_w
        # Main body
        arr[y0 + 12:y0 + 52, x0 + 18:x0 + 44, :3] = rgb
        # Asymmetric right-pointing arm / sword (makes flips distinguishable)
        arr[y0 + 22:y0 + 30, x0 + 44:x0 + 54, :3] = (30, 30, 30)
        # 1 px light-grey anti-aliased halo on the top edge
        arr[y0 + 11, x0 + 18:x0 + 44, :3] = (224, 224, 224)
        # Enclosed pure-white pocket between legs (4x8 pixels)
        arr[y0 + 40:y0 + 48, x0 + 28:x0 + 34, :3] = (255, 255, 255)
        # Tiny 2 px white eye glint (must survive enclosed-white cleanup)
        arr[y0 + 18:y0 + 19, x0 + 26:x0 + 28, :3] = (255, 255, 255)
        # Tiny 2x2 stray speck in top-left corner of cell
        arr[y0 + 3:y0 + 5, x0 + 3:x0 + 5, :3] = (40, 40, 40)
    return Image.fromarray(arr)


@pytest.fixture
def client():
    reg = ModalityRegistry()
    model = PixelTools(model_id="pixel-tools", hf_repo="", local_dir=None)
    reg.register("image/pixelart", model, manifest={
        "model_id": "pixel-tools",
        "modality": "image/pixelart",
        "capabilities": {"device": "cpu", "memory_gb": 1.5, "supports_spritesheet": True},
    })
    app = create_app(registry=reg, routers={"image/pixelart": build_router(reg)})
    return TestClient(app)


# ------------------------------------------------------------------ slice_grid


def test_slice_grid_exact_and_row_major():
    arr = np.zeros((40, 60, 4), dtype=np.uint8)
    for idx in range(6):
        r, c = divmod(idx, 3)
        arr[r * 20:(r + 1) * 20, c * 20:(c + 1) * 20] = (idx * 40, 100, 200, 255)
    cells = slice_grid(Image.fromarray(arr), 3, 2)
    assert len(cells) == 6
    for idx, cell in enumerate(cells):
        assert cell.size == (20, 20)
        assert cell.mode == "RGBA"
        assert tuple(np.asarray(cell)[10, 10]) == (idx * 40, 100, 200, 255)


def test_slice_grid_uneven_dimensions_cover_full_image():
    img = Image.new("RGB", (100, 50), (10, 20, 30))
    cells = slice_grid(img, 3, 2)
    assert len(cells) == 6
    assert sum(c.size[0] for c in cells[:3]) == 100
    assert cells[0].size[1] + cells[3].size[1] == 50


def test_slice_grid_rejects_invalid_grid_or_tiny_image():
    with pytest.raises(ValueError):
        slice_grid(Image.new("RGBA", (10, 10)), 0, 2)
    with pytest.raises(ValueError):
        slice_grid(Image.new("RGBA", (2, 2)), 3, 2)


# -------------------------------------------------------- flood_key_background


def test_flood_key_background_white_enclosed_and_defringe():
    arr = np.full((48, 48, 4), 255, dtype=np.uint8)
    # Dark blue body
    arr[10:38, 12:36, :3] = (40, 60, 160)
    # 1 px light-grey anti-aliased fringe along top of body (above fringe_threshold=212)
    arr[9, 12:36, :3] = (225, 225, 225)
    # Enclosed white pocket (6x6)
    arr[20:26, 20:26, :3] = (255, 255, 255)
    # 2 px white eye highlight (smaller than enclosed threshold -> preserved)
    arr[14, 16:18, :3] = (255, 255, 255)

    out = np.asarray(flood_key_background(Image.fromarray(arr), mode="white"))
    assert out[0, 0, 3] == 0                       # outer white keyed
    assert (out[9, 12:36, 3] == 0).all()           # 1 px halo defringed
    assert (out[20:26, 20:26, 3] == 0).all()       # enclosed white pocket cleared
    assert (out[14, 16:18, 3] == 255).all()        # 2 px eye highlight kept
    assert out[12, 14, 3] == 255                   # body kept

    # With clear_enclosed_white=False, the 6x6 enclosed pocket remains opaque
    out_keep = np.asarray(flood_key_background(
        Image.fromarray(arr), mode="white", clear_enclosed_white=False,
    ))
    assert (out_keep[20:26, 20:26, 3] == 255).all()
    assert out_keep[0, 0, 3] == 0


def test_flood_key_background_auto_handles_rgba_and_solid_color():
    # Already-RGBA sprite with white clothes must NOT lose its white pixels
    rgba = np.zeros((32, 32, 4), dtype=np.uint8)
    rgba[8:24, 8:24] = (255, 255, 255, 255)
    out = np.asarray(flood_key_background(Image.fromarray(rgba), mode="auto"))
    assert np.array_equal(out, rgba)

    # Solid magenta background with green sprite
    mag = np.zeros((32, 32, 4), dtype=np.uint8)
    mag[..., :3] = (255, 0, 255)
    mag[..., 3] = 255
    mag[8:24, 8:24, :3] = (40, 180, 60)
    out_mag = np.asarray(flood_key_background(Image.fromarray(mag), mode="auto"))
    assert out_mag[0, 0, 3] == 0
    assert (out_mag[8:24, 8:24, 3] == 255).all()


def test_flood_key_background_none_is_noop():
    arr = np.full((16, 16, 4), 255, dtype=np.uint8)
    out = np.asarray(flood_key_background(Image.fromarray(arr), mode="none"))
    assert np.array_equal(out, arr)


# ----------------------------------------------------- filter_stray_components


def test_filter_stray_components_specks_bleed_and_diagonal_connection():
    arr = np.zeros((64, 64, 4), dtype=np.uint8)
    # Main body: 20x20 = 400 px
    arr[20:40, 20:40] = (180, 50, 50, 255)
    # Diagonally attached 2x2 antenna (8-connected to corner (20, 20) -> part of main body)
    arr[18:20, 18:20] = (180, 50, 50, 255)
    # Valid detached interior part (floating orb): 5x5 = 25 px (6.25% >= 3%, not touching border)
    arr[6:11, 30:35] = (50, 200, 220, 255)
    # Tiny stray speck: 2x2 = 4 px (1% < 3%)
    arr[50:52, 50:52] = (40, 40, 40, 255)
    # Neighbor-cell border bleed: 6x6 = 36 px (9% < 15%, touches left border x=0)
    arr[25:31, 0:6] = (90, 90, 90, 255)

    out = np.asarray(filter_stray_components(
        Image.fromarray(arr), min_component_ratio=0.03, border_bleed_ratio=0.15,
    ))
    assert (out[20:40, 20:40, 3] == 255).all()  # main body kept
    assert (out[18:20, 18:20, 3] == 255).all()  # 8-connected diagonal antenna kept
    assert (out[6:11, 30:35, 3] == 255).all()   # valid floating orb kept
    assert (out[50:52, 50:52, 3] == 0).all()    # tiny speck removed
    assert (out[25:31, 0:6, 3] == 0).all()      # border bleed removed


# --------------------------------------------------------------- fit_and_align


def test_fit_and_align_bottom_and_center_and_hard_alpha():
    # Tall sprite: 20x40 off-center in a 100x100 cell, with semi-transparent edge
    arr = np.zeros((100, 100, 4), dtype=np.uint8)
    arr[10:50, 5:25] = (60, 140, 220, 255)
    arr[10:50, 25, :] = (60, 140, 220, 100)  # below alpha_threshold=128

    bottom = fit_and_align(
        Image.fromarray(arr), frame_size=(48, 48), align="bottom",
        bottom_padding=2, resample="nearest", alpha_threshold=128, hard_alpha=True,
    )
    assert bottom.size == (48, 48)
    _assert_binary_alpha(bottom)
    ba = np.asarray(bottom)
    ys, xs = np.nonzero(ba[..., 3] > 0)
    # Fit height is 44 -> 20x40 scales to 22x44; bottom rests at y = 48 - 2 - 1 = 45
    assert int(ys.max()) == 45
    assert int(ys.min()) == 2
    # Horizontally centered: (48 - 22) // 2 = 13..34
    assert int(xs.min()) == 13 and int(xs.max()) == 34

    centered = fit_and_align(
        Image.fromarray(arr), frame_size=(48, 48), fit_size=(20, 20),
        align="center", resample="lanczos", hard_alpha=True,
    )
    assert centered.size == (48, 48)
    _assert_binary_alpha(centered)
    ca = np.asarray(centered)
    cys, cxs = np.nonzero(ca[..., 3] > 0)
    # 20x40 into 20x20 -> 10x20 centered at x=19..28, y=14..33
    assert int(cys.min()) == 14 and int(cys.max()) == 33
    assert int(cxs.min()) == 19 and int(cxs.max()) == 28


def test_fit_and_align_empty_cell_returns_transparent():
    empty = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    out = fit_and_align(empty, (48, 48))
    assert out.size == (48, 48)
    assert np.asarray(out)[..., 3].sum() == 0


# ----------------------------------------------------------------- mirror_5to8


def test_mirror_5to8_order_and_horizontal_flips():
    cells = []
    for i in range(6):
        a = np.zeros((16, 16, 4), dtype=np.uint8)
        a[4:12, 2:6] = (i * 40 + 20, 100, 150, 255)  # left-weighted bar
        cells.append(Image.fromarray(a))

    out = mirror_5to8(cells)
    assert len(out) == 8
    # [up, north_east, side, south_east, down, south_west, left, north_west]
    for idx in range(5):
        assert np.array_equal(np.asarray(out[idx]), np.asarray(cells[idx]))
    # SE (3) -> SW (5), E/side (2) -> left (6), NE (1) -> NW (7)
    assert np.array_equal(np.asarray(out[5]), np.asarray(cells[3])[:, ::-1])
    assert np.array_equal(np.asarray(out[6]), np.asarray(cells[2])[:, ::-1])
    assert np.array_equal(np.asarray(out[7]), np.asarray(cells[1])[:, ::-1])

    with pytest.raises(ValueError):
        mirror_5to8(cells[:4])


# ------------------------------------------------------------ bake_spritesheet


def test_bake_spritesheet_5to8_and_shared_palette():
    sheet = _make_3x2_white_sheet()
    res = bake_spritesheet(
        sheet,
        SpritesheetOptions(
            grid=(3, 2),
            frame_size=(48, 48),
            mirror_mode="5to8",
            background="auto",
            palette=parse_palette("pico-8"),
        ),
    )
    assert res.frame_size == (48, 48)
    assert res.output_grid == (4, 2)
    assert res.atlas_size == (192, 96)
    assert res.atlas.size == (192, 96)
    assert len(res.cells) == 8
    assert [c.name for c in res.layout] == list(DIRECTION_8_NAMES)
    assert res.palette == list(NAMED_PALETTES["pico-8"])
    _assert_binary_alpha(res.atlas)

    # Every opaque pixel across the atlas belongs strictly to pico-8
    allowed = {hex_to_rgb(c) for c in NAMED_PALETTES["pico-8"]}
    atlas_arr = np.asarray(res.atlas)
    op_rgb = atlas_arr[atlas_arr[..., 3] > 0, :3]
    assert len(op_rgb) > 0
    assert {tuple(int(v) for v in p) for p in np.unique(op_rgb, axis=0)} <= allowed

    # Mirrored cells match pixel-for-pixel flips
    assert np.array_equal(np.asarray(res.cells[5]), np.asarray(res.cells[3])[:, ::-1])
    assert np.array_equal(np.asarray(res.cells[6]), np.asarray(res.cells[2])[:, ::-1])
    assert np.array_equal(np.asarray(res.cells[7]), np.asarray(res.cells[1])[:, ::-1])

    # Every cell has positive opaque pixels and bottom anchor at y = 48 - 2 = 46
    for i, cell_meta in enumerate(res.layout):
        assert cell_meta.index == i
        assert cell_meta.x == (i % 4) * 48
        assert cell_meta.y == (i // 4) * 48
        assert cell_meta.width == 48 and cell_meta.height == 48
        assert cell_meta.anchor_x == 24 and cell_meta.anchor_y == 46
        assert cell_meta.opaque_pixels > 0


def test_bake_spritesheet_auto_palette_shared_across_frames():
    sheet = _make_3x2_white_sheet()
    res = bake_spritesheet(
        sheet,
        SpritesheetOptions(
            grid=(3, 2),
            frame_size=(32, 32),
            mirror_mode="none",
            palette=parse_palette("auto:4"),
            output_cols=2,
        ),
    )
    assert res.output_grid == (2, 3)
    assert res.atlas_size == (64, 96)
    assert len(res.palette) <= 4
    atlas_arr = np.asarray(res.atlas)
    op_rgb = atlas_arr[atlas_arr[..., 3] > 0, :3]
    uniq = np.unique(op_rgb, axis=0)
    assert len(uniq) <= 4


def test_bake_spritesheet_max_output_side_raises():
    sheet = _make_3x2_white_sheet(16, 16)
    with pytest.raises(SpritesheetTooLargeError):
        bake_spritesheet(
            sheet,
            SpritesheetOptions(
                grid=(3, 2), frame_size=(48, 48), mirror_mode="5to8", max_output_side=100,
            ),
        )


# ----------------------------------------------------------- HTTP route tests


def test_route_spritesheet_json_5to8(client):
    r = client.post(
        "/v1/images/spritesheet",
        files={"image": _png_upload(_make_3x2_white_sheet())},
        data={
            "grid": "3x2",
            "frame_size": "48x48",
            "mirror_mode": "5to8",
            "palette": "db32",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "image.spritesheet"
    assert body["model"] == "pixel-tools"
    assert body["id"].startswith("spritesheet-")
    assert body["frame_size"] == [48, 48]
    assert body["atlas_size"] == [192, 96]
    assert body["output_grid"] == [4, 2]
    assert body["palette"] == list(NAMED_PALETTES["db32"])
    assert len(body["cells"]) == 8
    assert [c["name"] for c in body["cells"]] == list(DIRECTION_8_NAMES)
    expected_keys = {
        "index", "name", "x", "y", "width", "height",
        "anchor_x", "anchor_y", "opaque_pixels",
    }
    for c in body["cells"]:
        assert set(c) == expected_keys
        assert c["opaque_pixels"] > 0

    [entry] = body["data"]
    atlas = _decode_b64(entry["b64_json"])
    assert atlas.size == (192, 96) and atlas.mode == "RGBA"
    _assert_binary_alpha(atlas)


def test_route_spritesheet_png_and_zip(client):
    sheet = _make_3x2_white_sheet()
    r_png = client.post(
        "/v1/images/spritesheet",
        files={"image": _png_upload(sheet)},
        data={"grid": "3x2", "frame_size": "32x32", "mirror_mode": "5to8",
              "response_format": "png"},
    )
    assert r_png.status_code == 200, r_png.text
    assert r_png.headers["content-type"] == "image/png"
    img = Image.open(io.BytesIO(r_png.content))
    assert img.size == (128, 64)
    _assert_binary_alpha(img)

    r_zip = client.post(
        "/v1/images/spritesheet",
        files={"image": _png_upload(sheet)},
        data={"grid": "3x2", "frame_size": "32x32", "mirror_mode": "5to8",
              "response_format": "zip"},
    )
    assert r_zip.status_code == 200, r_zip.text
    assert r_zip.headers["content-type"] == "application/zip"
    z = zipfile.ZipFile(io.BytesIO(r_zip.content))
    expected_files = [
        "atlas.png",
        "layout.json",
        *[f"cells/cell_{i:02d}.png" for i in range(8)],
    ]
    assert sorted(z.namelist()) == sorted(expected_files)
    z_atlas = Image.open(io.BytesIO(z.read("atlas.png")))
    assert z_atlas.size == (128, 64)
    c0 = Image.open(io.BytesIO(z.read("cells/cell_00.png")))
    assert c0.size == (32, 32)
    layout = json.loads(z.read("layout.json"))
    assert layout["atlas_size"] == [128, 64]
    assert len(layout["cells"]) == 8
    assert layout["cells"][0]["name"] == "up"


@pytest.mark.parametrize("field,value", [
    ("grid", "3by2"),
    ("grid", "0x2"),
    ("frame_size", "bad"),
    ("fit_size", "64x64"),       # exceeds default frame_size 48x48
    ("mirror_mode", "4to8"),
    ("background", "magenta"),
    ("bg_threshold", "256"),
    ("fringe_threshold", "-1"),
    ("align", "top"),
    ("bottom_padding", "48"),    # >= frame height 48
    ("bottom_padding", "-1"),
    ("resample", "bilinear"),
    ("alpha_threshold", "300"),
    ("palette", "not-a-palette"),
    ("output_cols", "0"),
    ("response_format", "webp"),
])
def test_route_spritesheet_bad_parameters_400(client, field, value):
    r = client.post(
        "/v1/images/spritesheet",
        files={"image": _png_upload(_make_3x2_white_sheet(32, 32))},
        data={field: value},
    )
    assert r.status_code == 400, (field, value, r.text)
    assert r.json()["error"]["code"] == "invalid_parameter"


def test_route_spritesheet_5to8_needs_5_cells_400(client):
    r = client.post(
        "/v1/images/spritesheet",
        files={"image": _png_upload(_make_3x2_white_sheet(32, 32))},
        data={"grid": "2x2", "mirror_mode": "5to8"},
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_parameter"


def test_route_spritesheet_bad_option_wins_over_undecodable_upload(client):
    r = client.post(
        "/v1/images/spritesheet",
        files={"image": ("bad.png", b"not-a-png", "image/png")},
        data={"mirror_mode": "nope"},
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_parameter"


def test_route_spritesheet_oversize_input_and_output_413(client, monkeypatch):
    monkeypatch.setenv("MUSE_PIXELART_MAX_INPUT_SIDE", "100")
    # Output 4*48 = 192 > 100 -> rejected early before decode
    r_out = client.post(
        "/v1/images/spritesheet",
        files={"image": ("bad.png", b"not-even-decoded", "image/png")},
        data={"grid": "3x2", "frame_size": "48x48", "mirror_mode": "5to8"},
    )
    assert r_out.status_code == 413
    assert r_out.json()["error"]["code"] == "input_too_large"

    # Input 192x128 > 100, output 4*16 = 64 <= 100 -> rejected on decode
    r_in = client.post(
        "/v1/images/spritesheet",
        files={"image": _png_upload(_make_3x2_white_sheet(64, 64))},
        data={"grid": "3x2", "frame_size": "16x16", "mirror_mode": "5to8"},
    )
    assert r_in.status_code == 413
    assert r_in.json()["error"]["code"] == "input_too_large"


def test_route_spritesheet_unknown_model_404(client):
    r = client.post(
        "/v1/images/spritesheet",
        files={"image": _png_upload(_make_3x2_white_sheet(32, 32))},
        data={"model": "missing-model"},
    )
    assert r.status_code == 404


def test_route_spritesheet_runs_off_event_loop(client, monkeypatch):
    import muse.modalities._common.images as image_input

    def on_loop() -> bool:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return False
        return True

    seen = {"decode": [], "spritesheet": []}
    real_decode = image_input._bytes_to_pil
    real_sheet = PixelTools.spritesheet

    def spy_decode(*a, **k):
        seen["decode"].append(on_loop())
        return real_decode(*a, **k)

    def spy_sheet(self, *a, **k):
        seen["spritesheet"].append(on_loop())
        return real_sheet(self, *a, **k)

    monkeypatch.setattr(image_input, "_bytes_to_pil", spy_decode)
    monkeypatch.setattr(PixelTools, "spritesheet", spy_sheet)

    r = client.post(
        "/v1/images/spritesheet",
        files={"image": _png_upload(_make_3x2_white_sheet(32, 32))},
    )
    assert r.status_code == 200, r.text
    assert seen["decode"] == [False]
    assert seen["spritesheet"] == [False]


# ---------------------------------------------------------------------- client


def test_pixelart_client_spritesheet():
    c = PixelartClient(base_url="http://localhost:8000")
    mock_resp = MagicMock(status_code=200, content=b"ZIPBYTES")
    mock_resp.json.return_value = {"object": "image.spritesheet"}

    with patch("muse.modalities.image_pixelart.client.requests.post", return_value=mock_resp) as post:
        out_json = c.spritesheet(b"PNG", grid="3x2", mirror_mode="5to8")
        assert out_json == {"object": "image.spritesheet"}
        assert post.call_args.args[0] == "http://localhost:8000/v1/images/spritesheet"

        out_zip = c.spritesheet(b"PNG", response_format="zip")
        assert out_zip == b"ZIPBYTES"

        out_png = c.spritesheet(b"PNG", response_format="png")
        assert out_png == b"ZIPBYTES"
