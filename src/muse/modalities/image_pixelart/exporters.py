"""Engine exporters for the blob-47 sheet: Godot 4 TileSet, Tiled tileset, zip.

Pure text / stdlib. Formats verified against upstream sources:

- Godot 4 (``scene/resources/2d/tile_set.cpp``, 4.3-stable): TileSet keys
  ``tile_size``, ``terrain_set_N/mode`` (0 = TERRAIN_MODE_MATCH_CORNERS_AND_SIDES),
  ``terrain_set_N/terrain_M/name`` and ``/color``, ``sources/ID``;
  TileSetAtlasSource keys ``texture``, ``texture_region_size`` and per tile
  ``x:y/alt`` with TileData ``terrain_set``, ``terrain`` and
  ``terrains_peering_bit/<name>`` where name is a ``CELL_NEIGHBOR_ENUM_TO_TEXT``
  entry (``right_side``, ``bottom_right_corner``, ...). Unset peering bits
  default to -1 and are omitted, as Godot itself saves them.
- Tiled (``src/libtiled/mapwriter.cpp`` and the TMX reference): ``<wangset
  name type tile>`` with ``type`` corner / edge / mixed, ``<wangcolor name
  color tile probability>``, ``<wangtile tileid wangid>`` where wangid lists
  8 colour indexes (0 = unset, 1 = first colour) in the order top, top-right,
  right, bottom-right, bottom, bottom-left, left, top-left.

Both files reference the sheet as ``<name>_blob47.png`` in the same folder
(Godot: ``res://<name>_blob47.png``, i.e. unzipped at the project root).
"""
from __future__ import annotations

import io
import json
import re
import zipfile
from xml.sax.saxutils import quoteattr

from muse.modalities.image_pixelart.autotile import SHEET_COLUMNS, SHEET_ROWS, BlobTile

_NAME_RE = re.compile(r"^[a-z0-9_-]{1,40}$")

# Mask bit -> Godot square-tile peering bit, in Godot's CellNeighbor enum
# order (the order its saved files list them in).
_GODOT_PEERING = (
    (4, "right_side"),
    (8, "bottom_right_corner"),
    (16, "bottom_side"),
    (32, "bottom_left_corner"),
    (64, "left_side"),
    (128, "top_left_corner"),
    (1, "top_side"),
    (2, "top_right_corner"),
)

_TERRAIN_RGB = (0.36, 0.62, 0.3)  # editor colour of the one terrain
_TERRAIN_HEX = "#5c9e4d"
_ZIP_TIME = (1980, 1, 1, 0, 0, 0)  # fixed, so the bundle bytes are reproducible


def safe_name(name: str) -> str:
    """Return ``name`` if it is a slug of ``[a-z0-9_-]{1,40}``; else ValueError."""
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise ValueError("name must be 1-40 characters of a-z, 0-9, '_' or '-'")
    return name


def _check_tiles(tile_size: int, tiles: list[BlobTile]) -> None:
    if not isinstance(tile_size, int) or isinstance(tile_size, bool) or tile_size < 1:
        raise ValueError("tile_size must be a positive integer")
    for t in tiles:
        if not (0 <= t.x < SHEET_COLUMNS and 0 <= t.y < SHEET_ROWS and 0 <= t.mask <= 255):
            raise ValueError(f"tile {t} is outside the {SHEET_COLUMNS}x{SHEET_ROWS} sheet")


def godot_tileset_tres(name: str, tile_size: int, tiles: list[BlobTile]) -> str:
    """A Godot 4 ``TileSet`` resource: one atlas source on ``<name>_blob47.png``,
    one terrain set (match corners and sides) with one terrain, and each
    tile's peering bits from its mask."""
    name = safe_name(name)
    _check_tiles(tile_size, tiles)
    t = int(tile_size)
    tex_id, src_id = "1_blob47", "TileSetAtlasSource_blob47"
    lines = [
        '[gd_resource type="TileSet" load_steps=3 format=3]',
        "",
        f'[ext_resource type="Texture2D" path="res://{name}_blob47.png" id="{tex_id}"]',
        "",
        f'[sub_resource type="TileSetAtlasSource" id="{src_id}"]',
        f'texture = ExtResource("{tex_id}")',
        f"texture_region_size = Vector2i({t}, {t})",
    ]
    for tile in sorted(tiles, key=lambda b: (b.y, b.x)):
        key = f"{tile.x}:{tile.y}/0"
        lines += [f"{key} = 0", f"{key}/terrain_set = 0", f"{key}/terrain = 0"]
        lines += [f"{key}/terrains_peering_bit/{bit_name} = 0"
                  for bit, bit_name in _GODOT_PEERING if tile.mask & bit]
    r, g, b = _TERRAIN_RGB
    lines += [
        "",
        "[resource]",
        f"tile_size = Vector2i({t}, {t})",
        "terrain_set_0/mode = 0",
        f'terrain_set_0/terrain_0/name = "{name}"',
        f"terrain_set_0/terrain_0/color = Color({r}, {g}, {b}, 1)",
        f'sources/0 = SubResource("{src_id}")',
        "",
    ]
    return "\n".join(lines)


def tiled_tileset_tsx(name: str, tile_size: int, tiles: list[BlobTile]) -> str:
    """A Tiled 1.10+ external tileset on ``<name>_blob47.png`` with one
    ``mixed`` wang set; each tile's wangid comes from its mask (our bit order
    N, NE, E, SE, S, SW, W, NW is Tiled's wangid order)."""
    name = safe_name(name)
    _check_tiles(tile_size, tiles)
    t = int(tile_size)
    full = next((b for b in tiles if b.mask == 255), None)
    rep = full.y * SHEET_COLUMNS + full.x if full is not None else -1
    q = quoteattr
    out = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<tileset version="1.10" tiledversion="1.10.2" name={q(name)} '
        f'tilewidth="{t}" tileheight="{t}" tilecount="{SHEET_COLUMNS * SHEET_ROWS}" '
        f'columns="{SHEET_COLUMNS}">',
        f' <image source={q(name + "_blob47.png")} width="{SHEET_COLUMNS * t}" '
        f'height="{SHEET_ROWS * t}"/>',
        " <wangsets>",
        f'  <wangset name={q(name)} type="mixed" tile="{rep}">',
        f'   <wangcolor name={q(name)} color="{_TERRAIN_HEX}" tile="{rep}" probability="1"/>',
    ]
    for tile in sorted(tiles, key=lambda b: (b.y, b.x)):
        wangid = ",".join(str((tile.mask >> i) & 1) for i in range(8))
        out.append(f'   <wangtile tileid="{tile.y * SHEET_COLUMNS + tile.x}" wangid="{wangid}"/>')
    out += ["  </wangset>", " </wangsets>", "</tileset>", ""]
    return "\n".join(out)


def bundle_zip(name: str, a2_png: bytes, blob_png: bytes, tres: str, tsx: str,
               layout: list[dict]) -> bytes:
    """Zip the five files with consistent names (and fixed timestamps)."""
    name = safe_name(name)
    files = [
        (f"{name}_a2.png", a2_png),
        (f"{name}_blob47.png", blob_png),
        (f"{name}.tres", tres.encode("utf-8")),
        (f"{name}.tsx", tsx.encode("utf-8")),
        (f"{name}_layout.json", (json.dumps(layout, indent=2) + "\n").encode("utf-8")),
    ]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for fname, data in files:
            info = zipfile.ZipInfo(fname, date_time=_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, data)
    return buf.getvalue()
