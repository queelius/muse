"""pixel-tools: deterministic pixel-art tools for `image/pixelart` (no weights).

Serves `/v1/images/pixelize` (grid snap, retro palettes, binary alpha,
cleanup, outline, seamless, sprite sheets) and `/v1/images/autotile` (RPG
Maker A2 blocks, the 47-tile blob set, Godot 4 and Tiled tilesets). Pure
Pillow + numpy on the CPU: `muse pull pixel-tools` installs the venv only and
records a code receipt (the script's SHA-256) instead of an HF revision.

The Model class is the generic `PixelTools` runtime.
"""
from muse.modalities.image_pixelart.runtime import PixelTools as Model


MANIFEST = {
    "model_id": "pixel-tools",
    "modality": "image/pixelart",
    "weights": "none",
    "description": (
        "Pixel-art tools: grid snap + palettes (pixelize) and RPG Maker A2 / "
        "blob-47 autotiles with Godot and Tiled export. CPU, no weights, MIT."
    ),
    "summary": "Turns soft or AI-made pixel art into real pixel art, and fill textures into game-ready autotiles",
    "strengths": [
        "Detects the art grid and snaps every pixel to it; alpha only 0 or 255",
        "Retro palettes (pico-8, db16, db32, endesga-32, nes, gameboy), auto:N or custom",
        "A2 autotile + 47-tile blob sheet with Godot 4 .tres and Tiled .tsx; CPU, no weights",
    ],
    "limits": [
        "Cleans up existing art; it does not draw new art (pair with image/generation)",
        "Grid detection falls back to grid 1 when no period is confident",
        "Autotile edges are drawn from analytic masks, not hand-painted",
    ],
    "quality_tier": "good",
    "license": "MIT",
    "pip_extras": (
        "Pillow>=9.1.0",
        "numpy",
    ),
    "system_packages": (),
    "capabilities": {
        "device": "cpu",
        # Measured peak ~1.3 GB pixelizing a 4096 px input (the default cap).
        "memory_gb": 1.5,
        "supports_pixelize": True,
        "supports_autotile": True,
        "supported_tile_sizes": [16, 24, 32, 48],
    },
}
