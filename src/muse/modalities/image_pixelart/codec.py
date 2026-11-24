"""Wire encoding for image/pixelart: PNG bytes, the JSON envelopes, the zip.

PNG is always lossless RGBA (pixel art must survive byte for byte). Nothing
here imports PIL or numpy at module load; images arrive as PIL objects.
"""
from __future__ import annotations

import base64
import io
import json
import uuid
import zipfile
from typing import Any

_ZIP_TIME = (1980, 1, 1, 0, 0, 0)


def png_bytes(image: Any) -> bytes:
    """Encode a PIL image as PNG (RGBA kept as-is, no resampling)."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def encode_pixelize(result: Any, model_id: str) -> dict:
    """``{id, object: "image.pixelize", model, data, grid, size, palette,
    background_removed, metadata}``."""
    g = result.grid
    return {
        "id": f"pixelize-{uuid.uuid4().hex}",
        "object": "image.pixelize",
        "model": model_id,
        "data": [{"b64_json": _b64(png_bytes(result.image))}],
        "grid": {
            "cell": int(g.cell),
            "offset_x": int(g.offset_x),
            "offset_y": int(g.offset_y),
            "confidence": float(g.confidence),
            "detected": bool(g.detected),
        },
        "size": [int(result.size[0]), int(result.size[1])],
        "palette": list(result.palette),
        "background_removed": bool(result.background_removed),
        "metadata": dict(result.metadata),
    }


def encode_autotile(result: Any, model_id: str, tile_size: int, layout: list[dict]) -> dict:
    """``{id, object: "image.autotile", model, tile_size, files, layout,
    metadata}``; ``files`` holds the two PNGs as base64 and the two tileset
    texts, which reference ``<name>_blob47.png`` (see ``metadata.files``)."""
    return {
        "id": f"autotile-{uuid.uuid4().hex}",
        "object": "image.autotile",
        "model": model_id,
        "tile_size": int(tile_size),
        "files": {
            "a2": {"b64_json": _b64(png_bytes(result.a2))},
            "blob47": {"b64_json": _b64(png_bytes(result.blob47))},
            "godot_tres": result.godot_tres,
            "tiled_tsx": result.tiled_tsx,
        },
        "layout": layout,
        "metadata": dict(result.metadata),
    }


def autotile_zip(result: Any, name: str, layout: list[dict]) -> bytes:
    """The five-file bundle (``<name>_a2.png``, ``<name>_blob47.png``,
    ``<name>.tres``, ``<name>.tsx``, ``<name>_layout.json``)."""
    from muse.modalities.image_pixelart.exporters import bundle_zip

    return bundle_zip(name, png_bytes(result.a2), png_bytes(result.blob47),
                      result.godot_tres, result.tiled_tsx, layout)


def _spritesheet_cells_layout(result: Any) -> list[dict]:
    return [
        {
            "index": int(c.index),
            "name": c.name,
            "x": int(c.x),
            "y": int(c.y),
            "width": int(c.width),
            "height": int(c.height),
            "anchor_x": int(c.anchor_x),
            "anchor_y": int(c.anchor_y),
            "opaque_pixels": int(c.opaque_pixels),
        }
        for c in result.layout
    ]


def encode_spritesheet(result: Any, model_id: str) -> dict:
    """``{id, object: "image.spritesheet", model, data, frame_size, atlas_size,
    output_grid, cells, palette, metadata}``."""
    return {
        "id": f"spritesheet-{uuid.uuid4().hex}",
        "object": "image.spritesheet",
        "model": model_id,
        "data": [{"b64_json": _b64(png_bytes(result.atlas))}],
        "frame_size": [int(result.frame_size[0]), int(result.frame_size[1])],
        "atlas_size": [int(result.atlas_size[0]), int(result.atlas_size[1])],
        "output_grid": [int(result.output_grid[0]), int(result.output_grid[1])],
        "cells": _spritesheet_cells_layout(result),
        "palette": list(result.palette),
        "metadata": dict(result.metadata),
    }


def spritesheet_zip(result: Any) -> bytes:
    """Bundle ``atlas.png``, ``cells/cell_XX.png``, and ``layout.json`` with
    reproducible timestamps."""
    layout_doc = {
        "frame_size": [int(result.frame_size[0]), int(result.frame_size[1])],
        "atlas_size": [int(result.atlas_size[0]), int(result.atlas_size[1])],
        "output_grid": [int(result.output_grid[0]), int(result.output_grid[1])],
        "palette": list(result.palette),
        "metadata": dict(result.metadata),
        "cells": _spritesheet_cells_layout(result),
    }
    files: list[tuple[str, bytes]] = [
        ("atlas.png", png_bytes(result.atlas)),
        ("layout.json", (json.dumps(layout_doc, indent=2) + "\n").encode("utf-8")),
    ]
    for i, cell_img in enumerate(result.cells):
        files.append((f"cells/cell_{i:02d}.png", png_bytes(cell_img)))

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for fname, data in files:
            info = zipfile.ZipInfo(fname, date_time=_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, data)
    return buf.getvalue()
