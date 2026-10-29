"""``PixelTools``: the deterministic, weightless pixel-art runtime.

No weights and no model: it wraps the Pillow + numpy library modules
(``pixelize``, ``autotile``, ``exporters``) behind the
``ImagePixelartModel`` protocol. The bundled ``pixel-tools`` script aliases
it as ``Model``. CPU only; ``device`` is accepted and ignored.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from muse.modalities.image_pixelart.protocol import AutotileRequest, AutotileResult

if TYPE_CHECKING:  # pragma: no cover
    from muse.modalities.image_pixelart.pixelize import PixelizeOptions, PixelizeResult

# numpy, Pillow and the pixel library are imported inside the methods, never
# at module load: Pillow is not a base muse dependency, so script discovery
# (and `muse pull pixel-tools`) must work on a host without it.


# The blob mask's bit order (clockwise from north), as documented in
# ``autotile.BlobTile``; returned with every autotile result.
MASK_BITS = {"N": 1, "NE": 2, "E": 4, "SE": 8, "S": 16, "SW": 32, "W": 64, "NW": 128}


def layout_entries(tiles, tile_size: int) -> list[dict]:
    """The blob-47 layout as data: ``x`` / ``y`` in tiles (``unit``), plus
    the same position in pixels, and the 8-bit neighbour ``mask``."""
    t = int(tile_size)
    return [
        {"index": b.index, "x": b.x, "y": b.y, "unit": "tile",
         "pixel_x": b.x * t, "pixel_y": b.y * t, "mask": b.mask}
        for b in tiles
    ]


class PixelTools:
    """Pixelize and autotile, with no weights and no device."""

    def __init__(self, *, model_id: str = "pixel-tools", hf_repo: str = "",
                 local_dir: str | None = None, device: str = "cpu", **_: Any) -> None:
        self.model_id = model_id
        self.hf_repo = hf_repo
        self.local_dir = local_dir
        self.device = "cpu"

    def pixelize(self, image: Any, opts: "PixelizeOptions") -> "PixelizeResult":
        from muse.modalities.image_pixelart.pixelize import pixelize

        return pixelize(image, opts)

    def autotile(self, request: AutotileRequest) -> AutotileResult:
        import numpy as np
        from PIL import Image

        from muse.modalities.image_pixelart.autotile import (
            DROPPED_SHAPE,
            SHEET_COLUMNS,
            SHEET_ROWS,
            EdgeStyle,
            _hard_alpha,
            build_a2,
            expand_blob47,
            validate_a2,
        )
        from muse.modalities.image_pixelart.exporters import (
            godot_tileset_tres,
            safe_name,
            tiled_tileset_tsx,
        )
        from muse.modalities.image_pixelart.pixelize import PixelizeOptions, pixelize

        name = safe_name(request.name)
        t = int(request.tile_size)
        if (request.a2 is None) == (request.fill is None):
            raise ValueError("exactly one of a2 or fill is required")
        if request.a2 is not None and request.base is not None:
            raise ValueError("base is only used with fill, not with an a2 block")
        palette = request.palette
        mapped = palette is not None and palette.kind != "none"
        if request.a2 is not None:
            validate_a2(request.a2, t)
            # Imported blocks keep their colours; alpha is hardened at 128
            # (the blob sheet is built from the same hardened pixels).
            a2 = Image.fromarray(_hard_alpha(np.array(request.a2.convert("RGBA"),
                                                      dtype=np.uint8)))
            if mapped:
                a2 = pixelize(a2, PixelizeOptions(grid=1, target_size=None, palette=palette,
                                                  background="none")).image.convert("RGBA")
            source = "a2"
            style = None
        else:
            style = request.style if request.style is not None else EdgeStyle()
            a2 = build_a2(request.fill, request.base, t, style, palette)
            source = "fill"
        sheet, tiles = expand_blob47(a2, t)
        metadata: dict = {
            "source": source,
            "base": request.base is not None,
            "palette_kind": palette.kind if palette is not None else None,
            "sheet_columns": SHEET_COLUMNS,
            "sheet_rows": SHEET_ROWS,
            "tile_count": len(tiles),
            "dropped_shape": DROPPED_SHAPE,
            "mask_bits": dict(MASK_BITS),
            "layout_units": {"x": "tile", "y": "tile", "pixel_x": "px", "pixel_y": "px"},
            "files": {
                "a2": f"{name}_a2.png",
                "blob47": f"{name}_blob47.png",
                "godot_tres": f"{name}.tres",
                "tiled_tsx": f"{name}.tsx",
                "layout": f"{name}_layout.json",
            },
        }
        if style is not None:
            metadata["edge_style"] = {
                "edge_width": style.edge_width, "outline": style.outline,
                "highlight": style.highlight, "roughness": style.roughness,
                "corner_radius": style.corner_radius, "seed": style.seed,
            }
        return AutotileResult(
            a2=a2,
            blob47=sheet,
            tiles=tiles,
            godot_tres=godot_tileset_tres(name, t, tiles),
            tiled_tsx=tiled_tileset_tsx(name, t, tiles),
            metadata=metadata,
        )
