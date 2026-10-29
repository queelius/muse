"""Muse image/pixelart modality protocol.

``ImagePixelartModel`` is the backend contract: ``pixelize`` (soft or
off-grid image -> real pixel art) and ``autotile`` (fill texture or A2 block
-> RPG Maker A2 block, blob-47 sheet and Godot / Tiled tilesets).

Import-light: images are typed ``Any`` and the library types are imported
only for type checking, so importing this module loads neither PIL nor numpy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover
    from muse.modalities.image_pixelart.autotile import BlobTile, EdgeStyle
    from muse.modalities.image_pixelart.palettes import PaletteSpec
    from muse.modalities.image_pixelart.pixelize import PixelizeOptions, PixelizeResult


@dataclass
class AutotileRequest:
    """One autotile job. Exactly one of ``a2`` (an existing 2T x 3T A2 block)
    or ``fill`` (a fill texture, with an optional ``base`` texture) is set.

    ``style`` (an ``EdgeStyle``) applies only to ``fill``; None means the
    defaults. ``palette`` (a ``PaletteSpec``) palette-maps the A2 block when
    set and not ``none``. ``name`` is the file-name slug.
    """

    tile_size: int
    a2: Any = None
    fill: Any = None
    base: Any = None
    style: "EdgeStyle | None" = None
    palette: "PaletteSpec | None" = None
    name: str = "terrain"


@dataclass
class AutotileResult:
    """``a2`` is the A2 block (RGBA, 2T x 3T), ``blob47`` the 8 x 6 tile
    sheet (RGBA), ``tiles`` its layout (``x`` / ``y`` in tile units), and the
    two tileset texts reference ``<name>_blob47.png`` in the same folder."""

    a2: Any
    blob47: Any
    tiles: "list[BlobTile]"
    godot_tres: str
    tiled_tsx: str
    metadata: dict = field(default_factory=dict)


@runtime_checkable
class ImagePixelartModel(Protocol):
    """Protocol for pixel-art tool backends."""

    @property
    def model_id(self) -> str: ...

    def pixelize(self, image: Any, opts: "PixelizeOptions") -> "PixelizeResult": ...

    def autotile(self, request: AutotileRequest) -> AutotileResult: ...
