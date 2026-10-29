"""Image pixel-art modality (package ``image_pixelart``): real pixel art from
soft or off-grid images.

Library modules (Pillow + numpy only, imported by the runtime, never by
``muse --help``):

- ``palettes``: the named retro palettes, the palette-spec grammar
  (``pico-8`` / ``auto:N`` / ``none`` / ``#rrggbb,...``), OKLab conversion and
  nearest-colour mapping.
- ``pixelize``: grid detection and snapping, palette mapping, hard alpha,
  cleanup, outline, seamless wrap and shared-palette sheets.
- ``autotile``: RPG Maker MZ/MV A2 ground autotile blocks drawn from a fill
  (+ base) texture and an edge style, A2 validation, and blob-47 expansion
  (RPG Maker MV's ``FLOOR_AUTOTILE_TABLE``, MIT, with attribution).
- ``exporters``: Godot 4 ``TileSet`` (.tres), Tiled tileset (.tsx), the zip
  bundle and the file-name slug.

The modality's wire contract (``MODALITY``, ``build_router``, ``DISCOVERY``)
is added with the routes; until then this package is not a live modality.
"""
