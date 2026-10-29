"""The generated image/pixelart MCP tools write files a game engine can load.

MCP saves each JSON media output under a content-hash name, which would break
the Godot ``.tres`` / Tiled ``.tsx`` reference to ``<name>_blob47.png``. The
autotile tool therefore defaults to ``response_format=zip`` (DISCOVERY
``mcp_defaults``), whose five files keep consistent names. These tests run the
generated tools against the real routes and the real ``PixelTools`` runtime
(the MCP client's transport forwards to an in-process app).
"""
from __future__ import annotations

import re
import zipfile

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from muse.core.registry import ModalityRegistry
from muse.core.server import create_app
from muse.mcp.generated import call_generated, generate_tools
from muse.modalities.image_pixelart import build_router
from muse.modalities.image_pixelart.runtime import PixelTools


@pytest.fixture
def app_client():
    reg = ModalityRegistry()
    reg.register("image/pixelart", PixelTools(model_id="image-pixelart-model"), manifest={
        "model_id": "image-pixelart-model", "modality": "image/pixelart",
        "capabilities": {"device": "cpu", "memory_gb": 1.5},
    })
    return TestClient(create_app(registry=reg, routers={"image/pixelart": build_router(reg)}))


@pytest.fixture
def forwarding(fake_http, app_client):
    """A MuseClient whose inference requests hit the real pixelart routes."""
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = request.read()
        seen.append(body)
        r = app_client.post(request.url.path, content=body,
                            headers={"content-type": request.headers["content-type"]})
        return httpx.Response(r.status_code, content=r.content, headers=dict(r.headers))

    client = fake_http(handler)
    client.seen = seen
    return client


def _gen(caps_doc, openapi_doc, name):
    return next(g for g in generate_tools(caps_doc, openapi_doc) if g.name == name)


def _structured(blocks):
    [s] = [b for b in blocks if b["type"] == "structured"]
    return s["data"]


def _texture(path, side=32):
    rng = np.random.default_rng(0)
    a = np.empty((side, side, 3), dtype=np.int16)
    a[...] = (70, 140, 60)
    a += rng.integers(-30, 31, size=a.shape).astype(np.int16)
    Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).save(path)


def test_autotile_tool_defaults_to_zip(caps_doc, openapi_doc):
    gen = _gen(caps_doc, openapi_doc, "muse_autotile")
    props = gen.tool.inputSchema["properties"]
    assert props["response_format"]["default"] == "zip"
    assert {"a2_path", "fill_path", "base_path"} <= set(props)
    assert not {"a2_path", "fill_path", "base_path"} & set(gen.tool.inputSchema["required"])
    assert 'response_format="zip"' in gen.tool.description


def test_autotile_tool_zip_unzips_into_a_loadable_set(caps_doc, openapi_doc, forwarding,
                                                     media_env):
    _texture(media_env / "grass.png")
    gen = _gen(caps_doc, openapi_doc, "muse_autotile")
    blocks = call_generated(forwarding, gen, {
        "model": "image-pixelart-model", "fill_path": "grass.png",
        "tile_size": 16, "name": "grass",
    })
    assert b'name="response_format"\r\n\r\nzip' in forwarding.seen[-1]
    data = _structured(blocks)
    [f] = data["files"]
    assert f["mime"] == "application/zip" and f["path"].endswith(".zip")
    assert data["response"] is None

    # Unzip into a "project folder": every file the tilesets reference exists.
    project = media_env / "project"
    zipfile.ZipFile(f["path"]).extractall(project)
    names = sorted(p.name for p in project.iterdir())
    assert names == ["grass.tres", "grass.tsx", "grass_a2.png", "grass_blob47.png",
                     "grass_layout.json"]
    tres = (project / "grass.tres").read_text()
    [res] = re.findall(r'path="res://([^"]+)"', tres)
    tsx = (project / "grass.tsx").read_text()
    [src] = re.findall(r'<image source="([^"]+)"', tsx)
    assert res == src == "grass_blob47.png"
    sheet = Image.open(project / res)
    assert sheet.size == (8 * 16, 6 * 16)
    assert set(np.unique(np.asarray(sheet.convert("RGBA"))[..., 3]).tolist()) <= {0, 255}


def test_autotile_tool_json_still_writes_every_file(caps_doc, openapi_doc, forwarding,
                                                   media_env):
    """Explicit json: the two PNGs and both tileset texts land as files (under
    content-hash names, which is why zip is the tool's default)."""
    _texture(media_env / "grass.png")
    gen = _gen(caps_doc, openapi_doc, "muse_autotile")
    blocks = call_generated(forwarding, gen, {
        "model": "image-pixelart-model", "fill_path": "grass.png",
        "tile_size": 16, "response_format": "json",
    })
    data = _structured(blocks)
    mimes = sorted(f["mime"] for f in data["files"])
    assert mimes == ["application/xml", "image/png", "image/png", "text/plain"]
    files = data["response"]["files"]
    assert files["a2"]["b64_json"].endswith(".png")
    assert open(files["godot_tres"], encoding="utf-8").read().startswith("[gd_resource")
    assert open(files["tiled_tsx"], encoding="utf-8").read().startswith("<?xml")


def test_pixelize_tool_writes_png(caps_doc, openapi_doc, forwarding, media_env):
    s = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
    a = np.array(s)
    a[4:12, 4:12] = (200, 40, 40, 255)
    a[6:10, 6:10] = (40, 60, 200, 255)
    Image.fromarray(a).resize((64, 64), Image.Resampling.NEAREST).save(media_env / "s.png")
    gen = _gen(caps_doc, openapi_doc, "muse_pixelize_image")
    assert gen.image_output
    blocks = call_generated(forwarding, gen, {
        "model": "image-pixelart-model", "image_path": "s.png", "palette": "pico-8",
        "grid": "4",
    })
    data = _structured(blocks)
    [f] = data["files"]
    assert f["mime"] == "image/png"
    out = Image.open(f["path"])
    assert out.size == (16, 16)
    assert data["response"]["grid"]["cell"] == 4
    assert data["response"]["data"][0]["b64_json"] == f["path"]
    assert set(np.unique(np.asarray(out.convert("RGBA"))[..., 3]).tolist()) <= {0, 255}
