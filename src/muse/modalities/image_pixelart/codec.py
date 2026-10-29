"""Wire encoding for image/pixelart: PNG bytes, the JSON envelopes, the zip.

PNG is always lossless RGBA (pixel art must survive byte for byte). Nothing
here imports PIL or numpy at module load; images arrive as PIL objects.
"""
from __future__ import annotations

import base64
import io
import uuid
from typing import Any


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
