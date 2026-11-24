"""Image pixel-art modality (package ``image_pixelart``): real pixel art from
soft or off-grid images, and game-ready autotiles.

Wire contract (both multipart/form-data, CPU only):

- ``POST /v1/images/pixelize``: ``image`` + grid / palette / background /
  alpha / cleanup / outline / seamless / sheet / preview fields -> JSON
  (``object: "image.pixelize"``) or raw ``image/png``.
- ``POST /v1/images/autotile``: ``a2`` file OR ``fill`` file (+ optional
  ``base``) + edge style -> JSON (``object: "image.autotile"``) or a zip of
  ``<name>_a2.png``, ``<name>_blob47.png``, ``<name>.tres``, ``<name>.tsx``
  and ``<name>_layout.json``.
- ``POST /v1/images/spritesheet``: ``image`` + grid / frame_size / mirror_mode /
  background / cleanup / alignment / palette fields -> JSON
  (``object: "image.spritesheet"``), raw ``image/png``, or a zip of
  ``atlas.png``, ``cells/cell_XX.png`` and ``layout.json``.

Library modules (Pillow + numpy only, imported by the runtime and inside the
route handlers, never at package import):

- ``palettes``: the named retro palettes, the palette-spec grammar
  (``pico-8`` / ``auto:N`` / ``none`` / ``#rrggbb,...``), OKLab conversion and
  nearest-colour mapping.
- ``pixelize``: grid detection and snapping, palette mapping, hard alpha,
  cleanup, outline, seamless wrap and shared-palette sheets.
- ``autotile``: RPG Maker MZ/MV A2 ground autotile blocks drawn from a fill
  (+ base) texture and an edge style, A2 validation, and blob-47 expansion
  (RPG Maker MV's ``FLOOR_AUTOTILE_TABLE``, MIT, with attribution).
- ``spritesheet``: grid slicing, border flood-key background removal + defringe,
  8-connected stray component filtering, tight fit & alignment, shared-palette
  locking, and 5-to-8-direction mirroring.
- ``exporters``: Godot 4 ``TileSet`` (.tres), Tiled tileset (.tsx), the zip
  bundle and the file-name slug.

Bundled model: ``pixel-tools`` (weightless; ``runtime.PixelTools``).
"""
from muse.modalities.image_pixelart.client import PixelartClient
from muse.modalities.image_pixelart.protocol import (
    AutotileRequest,
    AutotileResult,
    ImagePixelartModel,
)
from muse.modalities.image_pixelart.routes import build_router

MODALITY = "image/pixelart"

# One weightless bundled model serves all routes, so `curl -F image=@x.png`
# without `-F model=...` reaches it: the gateway resolves the first enabled
# image/pixelart model for these paths (AST-read; keep it a literal).
MODEL_OPTIONAL_PATHS = (
    "/v1/images/pixelize",
    "/v1/images/autotile",
    "/v1/images/spritesheet",
)

DISCOVERY = {
    "summary": (
        "Deterministic pixel-art tools (CPU, no weights). Pixelize snaps an image "
        "onto a pixel grid with a hard palette and binary alpha. Generated "
        "sprites (e.g. from klein-pixel-art) have no consistent grid, so pass "
        "target_size (64x64 for items and tiles, 96x96 for characters); "
        "grid=auto is for real pixel art that was upscaled. A 4x4 walk sheet "
        "(klein-pixel-walk) takes sheet=4x4 grid=4 palette=auto:24 for 32x32 "
        "frames. Autotile turns a fill texture (or an RPG Maker A2 block) into "
        "an A2 block, a 47-tile blob sheet and Godot 4 / Tiled tilesets (the "
        "MCP tool returns them as one zip so the tilesets' file references hold). "
        "Spritesheet slices a grid sheet (such as a 3x2 directional actor sheet), "
        "keys the background, filters stray specks/border bleed, aligns each frame, "
        "locks a shared palette, and can mirror 5 views into an 8-direction 4x2 atlas."
    ),
    "use_for": [
        "cleaning up generated pixel art (grid snap, palette, hard alpha)",
        "retro palettes (pico-8, db16, db32, endesga-32, nes, gameboy)",
        "seamless tiles and shared-palette sprite sheets",
        "slicing, aligning, palette-locking, and 5-to-8-direction mirroring of sprite sheets",
        "RPG Maker A2 autotiles and blob-47 terrain sets for Godot or Tiled",
    ],
    "not_for": [
        "generating new images (use image/generation)",
        "enlarging an image with new detail (use image/upscale)",
        "photographs you want to keep photographic",
    ],
    "routes": [
        {
            "method": "POST",
            "path": "/v1/images/pixelize",
            "request_content_type": "multipart/form-data",
            "response_content_type": "application/json",
            "compat": None,
            "mcp_tool": "muse_pixelize_image",
            "file_fields": ["image"],
            "media_fields": ["data[].b64_json"],
            "example": {"model": "<model>", "target_size": "64x64", "palette": "db32"},
        },
        {
            "method": "POST",
            "path": "/v1/images/autotile",
            "request_content_type": "multipart/form-data",
            "response_content_type": "application/json",
            "compat": None,
            "mcp_tool": "muse_autotile",
            "file_fields": ["a2", "fill", "base"],
            "media_fields": ["files.a2.b64_json", "files.blob47.b64_json"],
            "text_media_fields": {
                "files.godot_tres": "text/plain",
                "files.tiled_tsx": "application/xml",
            },
            # MCP writes each output under a content-hash name, which would
            # break the .tres/.tsx -> <name>_blob47.png reference; the zip
            # keeps the five files' names consistent.
            "mcp_defaults": {"response_format": "zip"},
            "example": {"model": "<model>", "tile_size": 48, "name": "grass"},
        },
        {
            "method": "POST",
            "path": "/v1/images/spritesheet",
            "request_content_type": "multipart/form-data",
            "response_content_type": "application/json",
            "compat": None,
            "mcp_tool": "muse_bake_spritesheet",
            "file_fields": ["image"],
            "media_fields": ["data[].b64_json"],
            "example": {
                "model": "<model>",
                "grid": "3x2",
                "frame_size": "48x48",
                "mirror_mode": "5to8",
            },
        },
    ],
}


def _probe_call(model):
    """Pixelize a synthetic 4x upscaled 32x32 sprite (must detect grid 4),
    autotile a synthetic fill at T=16, and bake a 3x2 sheet with 5-to-8 mirroring.
    PIL/numpy imported lazily."""
    import numpy as np
    from PIL import Image

    from muse.modalities.image_pixelart.palettes import parse_palette
    from muse.modalities.image_pixelart.pixelize import PixelizeOptions
    from muse.modalities.image_pixelart.spritesheet import SpritesheetOptions

    rng = np.random.default_rng(0)
    colours = np.array([[200, 40, 40], [40, 160, 60], [40, 60, 200], [230, 200, 60]],
                       dtype=np.uint8)
    art = np.zeros((32, 32, 4), dtype=np.uint8)
    art[..., :3] = colours[rng.integers(0, len(colours), size=(32, 32))]
    art[2:-2, 2:-2, 3] = 255
    art[art[..., 3] == 0] = 0
    sprite = Image.fromarray(art).resize((128, 128), Image.Resampling.NEAREST)
    result = model.pixelize(sprite, PixelizeOptions(
        grid=None, target_size=None, palette=parse_palette("db32"), background="auto"))
    assert result.grid.cell == 4, f"probe expected grid 4, detected {result.grid.cell}"

    fill = np.empty((32, 32, 3), dtype=np.int16)
    fill[...] = (70, 140, 60)
    fill += rng.integers(-30, 31, size=fill.shape).astype(np.int16)
    tile = model.autotile(AutotileRequest(
        tile_size=16, fill=Image.fromarray(np.clip(fill, 0, 255).astype(np.uint8))))
    assert len(tile.tiles) == 47

    raw_sheet = np.full((64, 96, 4), 255, dtype=np.uint8)
    for idx in range(5):
        r, c = divmod(idx, 3)
        raw_sheet[r * 32 + 6:r * 32 + 26, c * 32 + 8:c * 32 + 24, :3] = colours[idx % len(colours)]
    sheet = model.spritesheet(
        Image.fromarray(raw_sheet),
        SpritesheetOptions(grid=(3, 2), frame_size=(16, 16), mirror_mode="5to8"),
    )
    assert len(sheet.cells) == 8 and sheet.atlas.size == (64, 32)
    return {"grid": result.grid.cell, "tiles": len(tile.tiles), "spritesheet_frames": len(sheet.cells)}


PROBE_DEFAULTS = {
    "shape": "pixelize a 4x upscaled 32x32 sprite (grid 4) + autotile a fill at T=16 + 3x2->8-dir spritesheet",
    "call": _probe_call,
}

__all__ = [
    "MODALITY",
    "MODEL_OPTIONAL_PATHS",
    "DISCOVERY",
    "PROBE_DEFAULTS",
    "build_router",
    "AutotileRequest",
    "AutotileResult",
    "ImagePixelartModel",
    "PixelartClient",
]
