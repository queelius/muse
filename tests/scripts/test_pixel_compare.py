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
    assert len(sprites) == 8 and all(
        [v["palette"] for v in c["pixelize"]] == ["db32", "auto:16"] for c in sprites)
    assert sum(c["kind"] == "autotile" for c in cases) == 1
