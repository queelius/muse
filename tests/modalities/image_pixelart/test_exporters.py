"""Exporters: Godot 4 TileSet (.tres), Tiled tileset (.tsx), the zip bundle,
and the file-name slug.

Field names were checked against Godot 4.3 ``scene/resources/2d/tile_set.cpp``
(TileSet terrain_set_N/..., TileSetAtlasSource ``x:y/alt/...``, TileData
``terrains_peering_bit/<CELL_NEIGHBOR name>``) and Tiled's ``mapwriter.cpp`` /
TMX docs (wangset ``type``, wangid order top, top-right, ..., top-left).
"""
from __future__ import annotations

import io
import json
import re
import xml.etree.ElementTree as ET
import zipfile

import pytest
from PIL import Image

from muse.modalities.image_pixelart.autotile import (
    SHEET_COLUMNS,
    SHEET_ROWS,
    EdgeStyle,
    build_a2,
    expand_blob47,
)
from muse.modalities.image_pixelart.exporters import (
    bundle_zip,
    godot_tileset_tres,
    safe_name,
    tiled_tileset_tsx,
)

T = 16
BITS = {"top_side": 1, "top_right_corner": 2, "right_side": 4, "bottom_right_corner": 8,
        "bottom_side": 16, "bottom_left_corner": 32, "left_side": 64, "top_left_corner": 128}


@pytest.fixture(scope="module")
def tiles():
    a2 = build_a2(Image.new("RGB", (T, T), (80, 150, 70)), None, T, EdgeStyle(), None)
    _, tiles = expand_blob47(a2, T)
    return tiles


def decode(mask: int) -> set[str]:
    return {name for name, bit in BITS.items() if mask & bit}


# ------------------------------------------------------------------------ Godot


def test_godot_tres_parses_and_peering_matches_mask(tiles):
    text = godot_tileset_tres("grass", T, tiles)
    assert text.startswith('[gd_resource type="TileSet"')
    assert 'format=3' in text.splitlines()[0]
    assert 'path="res://grass_blob47.png"' in text
    assert '[sub_resource type="TileSetAtlasSource"' in text
    assert f"texture_region_size = Vector2i({T}, {T})" in text
    assert f"tile_size = Vector2i({T}, {T})" in text
    assert "terrain_set_0/mode = 0" in text
    assert 'terrain_set_0/terrain_0/name = "grass"' in text
    assert re.search(r"^terrain_set_0/terrain_0/color = Color\([^)]*\)$", text, re.M)
    assert re.search(r'^sources/0 = SubResource\("[^"]+"\)$', text, re.M)

    # Every line is a section header, blank, or ``key = value``.
    for line in text.splitlines():
        assert (not line or re.fullmatch(r"\[[a-z_]+( [a-z_]+=(\"[^\"]*\"|\d+))*\]", line)
                or re.fullmatch(r"[A-Za-z0-9_:/]+ = .+", line)), line

    # The ext_resource id the atlas source references exists.
    ext = re.search(r'\[ext_resource type="Texture2D" path="[^"]+" id="([^"]+)"\]', text)
    assert ext and f'texture = ExtResource("{ext.group(1)}")' in text
    sub = re.search(r'\[sub_resource type="TileSetAtlasSource" id="([^"]+)"\]', text)
    assert sub and f'sources/0 = SubResource("{sub.group(1)}")' in text
    load_steps = int(re.search(r"load_steps=(\d+)", text.splitlines()[0]).group(1))
    assert load_steps == text.count("[ext_resource") + text.count("[sub_resource") + 1

    created = set(re.findall(r"^(\d+):(\d+)/0 = 0$", text, re.M))
    assert created == {(str(t.x), str(t.y)) for t in tiles}
    assert len(created) == 47
    peer: dict[tuple[str, str], set[str]] = {c: set() for c in created}
    for x, y, name, val in re.findall(
            r"^(\d+):(\d+)/0/terrains_peering_bit/(\w+) = (-?\d+)$", text, re.M):
        assert name in BITS and val == "0"
        peer[(x, y)].add(name)
    for t in tiles:
        key = (str(t.x), str(t.y))
        assert f"{t.x}:{t.y}/0/terrain_set = 0" in text
        assert f"{t.x}:{t.y}/0/terrain = 0" in text
        assert peer[key] == decode(t.mask), t


def test_godot_tres_rejects_bad_name(tiles):
    with pytest.raises(ValueError):
        godot_tileset_tres("Grass Path!", T, tiles)


# ------------------------------------------------------------------------ Tiled


def test_tiled_tsx_parses_and_wangids_match_mask(tiles):
    root = ET.fromstring(tiled_tileset_tsx("grass", T, tiles))
    assert root.tag == "tileset"
    assert root.get("name") == "grass"
    assert root.get("tilewidth") == str(T) and root.get("tileheight") == str(T)
    assert root.get("columns") == str(SHEET_COLUMNS)
    assert root.get("tilecount") == str(SHEET_COLUMNS * SHEET_ROWS)
    img = root.find("image")
    assert img.get("source") == "grass_blob47.png"
    assert img.get("width") == str(SHEET_COLUMNS * T)
    assert img.get("height") == str(SHEET_ROWS * T)
    ws = root.find("wangsets/wangset")
    assert ws.get("type") == "mixed"
    assert len(root.findall("wangsets/wangset")) == 1
    colors = ws.findall("wangcolor")
    assert len(colors) == 1 and re.fullmatch(r"#[0-9a-f]{6}", colors[0].get("color"))
    wangtiles = ws.findall("wangtile")
    assert len(wangtiles) == 47
    by_id = {int(w.get("tileid")): w.get("wangid") for w in wangtiles}
    for t in tiles:
        wid = [int(v) for v in by_id[t.y * SHEET_COLUMNS + t.x].split(",")]
        assert len(wid) == 8
        # Tiled order: top, top-right, right, bottom-right, bottom, bottom-left,
        # left, top-left == our bit order N, NE, E, SE, S, SW, W, NW.
        assert wid == [(t.mask >> i) & 1 for i in range(8)], t


# -------------------------------------------------------------------------- zip


def test_bundle_zip_names(tiles):
    layout = [{"index": t.index, "x": t.x, "y": t.y, "mask": t.mask} for t in tiles]
    tres = godot_tileset_tres("grass", T, tiles)
    tsx = tiled_tileset_tsx("grass", T, tiles)
    data = bundle_zip("grass", b"A2PNG", b"BLOBPNG", tres, tsx, layout)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        assert sorted(z.namelist()) == sorted([
            "grass_a2.png", "grass_blob47.png", "grass.tres", "grass.tsx",
            "grass_layout.json"])
        assert z.read("grass_a2.png") == b"A2PNG"
        assert z.read("grass_blob47.png") == b"BLOBPNG"
        assert z.read("grass.tres").decode() == tres
        assert z.read("grass.tsx").decode() == tsx
        assert json.loads(z.read("grass_layout.json")) == layout
    # Deterministic bytes (fixed timestamps).
    assert data == bundle_zip("grass", b"A2PNG", b"BLOBPNG", tres, tsx, layout)


def test_bundle_zip_rejects_bad_name():
    with pytest.raises(ValueError):
        bundle_zip("../evil", b"", b"", "", "", [])


# --------------------------------------------------------------------- safe_name


@pytest.mark.parametrize("name", ["grass_path", "terrain", "a", "dirt-2", "x" * 40])
def test_safe_name_ok(name):
    assert safe_name(name) == name


@pytest.mark.parametrize("name", ["Grass Path!", "", "x" * 41, "../x", "Grass", "a.b",
                                  "a/b", None, 3, "grass\n", "grass\r", "\ngrass"])
def test_safe_name_rejects(name):
    with pytest.raises(ValueError):
        safe_name(name)
