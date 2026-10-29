"""Tests for the pixel_compare contact-sheet script (fake httpx transport)."""
from __future__ import annotations

import base64
import io
import json

import httpx
import pytest
from PIL import Image

from scripts.bench.pixel_compare import load_cases, paint_map, run, write_index


def _png(w=8, h=8, colour=(200, 40, 40, 255)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGBA", (w, h), colour).save(buf, "PNG")
    return buf.getvalue()


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def _layout(t=4):
    """47 fake blob tiles on an 8-wide sheet; masks distinct, mask 0 first."""
    from muse.modalities.image_pixelart.autotile import blob_masks
    return [{"index": i, "x": i % 8, "y": i // 8, "unit": "tile", "mask": m}
            for i, m in enumerate(blob_masks())]


def _sheet(t=4) -> bytes:
    return _png(8 * t, 6 * t, (30, 160, 50, 255))


def _handler(seen):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        path = request.url.path
        if path == "/v1/images/generations":
            return httpx.Response(200, json={"data": [{"b64_json": _b64(_png(32, 32))}]})
        if path == "/v1/images/pixelize":
            return httpx.Response(200, json={
                "data": [{"b64_json": _b64(_png(8, 8))}],
                "grid": {"cell": 4, "confidence": 0.9, "detected": True},
                "size": [8, 8], "palette": ["#ff0000", "#00ff00"]})
        if path == "/v1/images/autotile":
            return httpx.Response(200, json={
                "tile_size": 4,
                "files": {"a2": {"b64_json": _b64(_png(8, 12))},
                          "blob47": {"b64_json": _b64(_sheet())},
                          "godot_tres": "[gd_resource]", "tiled_tsx": "<tileset/>"},
                "layout": _layout()})
        return httpx.Response(404)
    return handler


def _client(seen):
    return httpx.Client(transport=httpx.MockTransport(_handler(seen)))


def test_runs_sprite_case_and_writes_outputs(tmp_path):
    cases = [{"id": "knight", "kind": "sprite", "prompt": "a knight", "model": "klein-pixel-art",
              "pixelize": [{"palette": "db32"}, {"palette": "auto:16", "outline": "dark"}]}]
    seen = []
    results = run(cases, "http://muse.test", tmp_path, _client(seen))
    assert len(results) == 2
    assert [r["variant"] for r in results] == ["palette=db32", "palette=auto:16, outline=dark"]
    assert all(r["error"] is None and r["grid"]["cell"] == 4 for r in results)
    assert (tmp_path / "knight" / "raw.png").exists()
    assert (tmp_path / results[1]["pixelized"]).exists()
    gen = json.loads(seen[0].content)
    assert gen["model"] == "klein-pixel-art" and gen["response_format"] == "b64_json"
    pix = seen[1]
    assert pix.url.path == "/v1/images/pixelize"
    assert b'name="palette"' in pix.content and b'name="image"' in pix.content
    assert b"pixel-tools" in pix.content
    index = write_index(results, tmp_path).read_text()
    assert "knight/v1.png" in index and "palette=db32" in index
    assert "image-rendering:pixelated" in index


def test_autotile_case_paints_map(tmp_path):
    cases = [{"id": "grass", "kind": "autotile", "fill_prompt": "grass", "base_prompt": "dirt",
              "model": "klein-pixel-art", "autotile": {"tile_size": 4, "name": "grass"}}]
    seen = []
    results = run(cases, "http://muse.test", tmp_path, _client(seen))
    assert len(results) == 1 and results[0]["error"] is None
    assert [r.url.path for r in seen] == [
        "/v1/images/generations", "/v1/images/generations", "/v1/images/autotile"]
    assert b'name="fill"' in seen[2].content and b'name="base"' in seen[2].content
    map_img = Image.open(tmp_path / results[0]["map"])
    assert map_img.size == (16 * 4, 12 * 4)
    # Unfilled cells show the generated base (red), not a blob-tile crop.
    assert (200, 40, 40, 255) in set(map_img.convert("RGBA").getdata())
    assert (tmp_path / "grass" / "tileset.tres").read_text() == "[gd_resource]"
    index = write_index(results, tmp_path).read_text()
    assert "grass/map.png" in index and "grass/blob47.png" in index


def test_paint_map_uses_mask_tiles():
    layout = _layout()
    # Distinct colour per tile so wrong lookups are visible.
    sheet = Image.new("RGBA", (8 * 4, 6 * 4))
    for e in layout:
        sheet.paste((e["index"] * 5 % 256, 99, 7, 255), (e["x"] * 4, e["y"] * 4, e["x"] * 4 + 4, e["y"] * 4 + 4))
    buf = io.BytesIO()
    sheet.save(buf, "PNG")
    img = Image.open(io.BytesIO(paint_map(buf.getvalue(), layout, 4, seed=3)))
    assert img.size == (64, 48)
    full = {e["mask"]: e["index"] for e in layout}[255]
    assert any(img.getpixel((x, y))[0] == full * 5 % 256
               for x in range(0, 64, 4) for y in range(0, 48, 4))


def test_server_error_recorded_run_continues(tmp_path):
    def handler(request):
        if request.url.path == "/v1/images/generations":
            body = json.loads(request.content)
            if "bad" in body["prompt"]:
                return httpx.Response(500, text="boom")
            return httpx.Response(200, json={"data": [{"b64_json": _b64(_png())}]})
        return httpx.Response(400, text="nope")
    client = httpx.Client(transport=httpx.MockTransport(handler))
    cases = [{"id": "a", "kind": "sprite", "prompt": "bad", "model": "m", "pixelize": [{}]},
             {"id": "b", "kind": "sprite", "prompt": "ok", "model": "m", "pixelize": [{}]}]
    results = run(cases, "http://muse.test", tmp_path, client)
    assert "HTTP 500" in results[0]["error"] and "HTTP 400" in results[1]["error"]


def test_case_file_validation(tmp_path):
    def load(text):
        p = tmp_path / "c.yaml"
        p.write_text(text)
        return load_cases(p)

    with pytest.raises(SystemExit, match="unknown kind 'video'"):
        load("- {id: x, kind: video, model: m, prompt: p}")
    with pytest.raises(SystemExit, match="needs prompt"):
        load("- {id: x, kind: sprite, model: m}")
    with pytest.raises(SystemExit, match="needs fill_prompt"):
        load("- {id: x, kind: autotile, model: m}")
    with pytest.raises(SystemExit, match="duplicate id"):
        load("- {id: x, kind: sprite, model: m, prompt: p}\n- {id: x, kind: sprite, model: m, prompt: p}")
    ok = load("- {id: x, kind: sprite, model: m, prompt: p, pixelize: {palette: db32}}")
    assert ok[0]["pixelize"] == [{"palette": "db32"}]


def test_shipped_case_file_is_valid():
    from pathlib import Path
    cases = load_cases(Path(__file__).resolve().parents[2] / "scripts/bench/pixel_cases.yaml")
    sprites = [c for c in cases if c["kind"] == "sprite"]
    # Generated sprites have no consistent grid, so each also gets a sized
    # variant (grid=auto keeps 512x512 on most of them).
    assert len(sprites) == 8 and all(
        [(v["palette"], v.get("target_size")) for v in c["pixelize"]]
        == [("db32", None), ("auto:16", None), ("db32", "64x64")] for c in sprites)
    tiles = [c for c in cases if c["kind"] == "autotile"]
    assert len(tiles) == 1
    # Without "seamless" klein drew a dirt clearing ringed by trees, not a texture.
    assert all("seamless" in c[k] for c in tiles for k in ("fill_prompt", "base_prompt"))


def test_shipped_walk_case_file_is_valid():
    from pathlib import Path
    cases = load_cases(Path(__file__).resolve().parents[2] / "scripts/bench/pixel_walk_cases.yaml")
    assert [c["id"] for c in cases] == [
        "walk-knight", "walk-witch", "walk-goblin", "walk-elf-ranger"]
    for c in cases:
        assert c["model"] == "klein-pixel-walk" and c["kind"] == "sprite"
        # The model's prompt_prefix adds the card's 4x4 template sentence.
        assert "4 by 4" not in c["prompt"]
        assert all(v["sheet"] == "4x4" and v["palette"] == "auto:24"
                   for v in c["pixelize"])
        assert c["pixelize"][0]["grid"] == 4


def _cells(img, t=4):
    """{(col, row): set of pixels} for a painted map."""
    out = {}
    for r in range(img.height // t):
        for c in range(img.width // t):
            out[(c, r)] = {img.getpixel((c * t + dx, r * t + dy))
                           for dx in range(t) for dy in range(t)}
    return out


def test_paint_map_unfilled_cells_show_the_base_texture():
    # The isolated (mask 0) tile's corner holds part of the fill blob and its
    # outline, so tiling it as "base" painted the blob's corner into every
    # unfilled cell. With the base texture given, unfilled cells are pure base.
    layout = _layout()
    sheet = Image.new("RGBA", (8 * 4, 6 * 4), (30, 160, 50, 255))
    buf = io.BytesIO()
    sheet.save(buf, "PNG")
    base = _png(32, 32, (120, 80, 40, 255))
    img = Image.open(io.BytesIO(paint_map(buf.getvalue(), layout, 4, seed=3,
                                          base_png=base))).convert("RGBA")
    cells = _cells(img)
    base_cells = [k for k, px in cells.items() if px == {(120, 80, 40, 255)}]
    fill_cells = [k for k, px in cells.items() if px == {(30, 160, 50, 255)}]
    assert base_cells and fill_cells
    assert len(base_cells) + len(fill_cells) == len(cells)


def test_paint_map_without_base_leaves_unfilled_cells_transparent():
    layout = _layout()
    sheet = Image.new("RGBA", (8 * 4, 6 * 4), (30, 160, 50, 255))
    buf = io.BytesIO()
    sheet.save(buf, "PNG")
    img = Image.open(io.BytesIO(paint_map(buf.getvalue(), layout, 4, seed=3))).convert("RGBA")
    cells = _cells(img)
    empty = [k for k, px in cells.items() if all(p[3] == 0 for p in px)]
    assert empty
    assert all(px == {(30, 160, 50, 255)} for k, px in cells.items() if k not in empty)


def _nearest_widths(index: str) -> list[int]:
    import re

    return [int(w) for w in re.findall(r"style='width:(\d+)px'", index)
            if int(w) not in (192, 384)]


def test_nearest_column_uses_the_variant_preview_scale(tmp_path):
    """The nearest column shows the art at the variant's preview_scale, so a
    preview_scale 8 variant is 8 x its art width; 4x when it sets none."""
    cases = [{"id": "knight", "kind": "sprite", "prompt": "a knight", "model": "m",
              "pixelize": [{"palette": "db32"}, {"palette": "db32", "preview_scale": 8},
                           {"palette": "db32", "preview_scale": 2}]}]
    results = run(cases, "http://muse.test", tmp_path, _client([]))
    assert [r.get("preview_scale") for r in results] == [4, 8, 2]
    index = write_index(results, tmp_path).read_text()
    assert _nearest_widths(index) == [8 * 4, 8 * 8, 8 * 2]  # art is 8 px wide
