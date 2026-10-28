"""Image pixel-art modality (package ``image_pixelart``): real pixel art from
soft or off-grid images.

Library modules (Pillow + numpy only, imported by the runtime, never by
``muse --help``):

- ``palettes``: the named retro palettes, the palette-spec grammar
  (``pico-8`` / ``auto:N`` / ``none`` / ``#rrggbb,...``), OKLab conversion and
  nearest-colour mapping.
- ``pixelize``: grid detection and snapping, palette mapping, hard alpha,
  cleanup, outline, seamless wrap and shared-palette sheets.

The modality's wire contract (``MODALITY``, ``build_router``, ``DISCOVERY``)
is added with the routes; until then this package is not a live modality.
"""
