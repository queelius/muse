"""Protocol + dataclasses for 3d/generation.

Two routes share one MIME tag (mirroring audio_generation's music+sfx
split): `/v1/3d/generations` (text-to-3d) and `/v1/3d/from-image`
(image-to-3d). Capability flags `supports_text_to_3d` and
`supports_image_to_3d` on the manifest gate which route a given
backend accepts.

The Protocol declares both methods as abstract so static type-checkers
see a uniform interface; routes do an explicit capability-flag check
before invoking, so backends that only implement one direction satisfy
the contract by raising NotImplementedError on the unsupported method
(or by simply not declaring the capability flag).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class Generation3DResult:
    """One generated 3D asset.

    `glb_bytes` is the binary glTF blob (embedded textures, web-friendly).
    `format` is reserved for future expansion (USDZ, OBJ-archive); for
    v0.41.0 it is always "glb".
    """
    glb_bytes: bytes
    model_id: str
    format: str = "glb"


@runtime_checkable
class Generation3DBackend(Protocol):
    """Structural protocol any 3D-generation backend satisfies.

    Both methods are declared. Backends that only implement one
    direction (e.g., TripoSR is image-to-3d only) raise
    NotImplementedError on the unsupported direction; the route layer
    short-circuits on the capability flag before calling, so this
    fallback rarely fires in practice.
    """

    def image_to_3d(
        self, image_path: str, **kwargs,
    ) -> list[Generation3DResult]:
        """Generate one or more 3D assets from a single image."""
        ...

    def text_to_3d(
        self, prompt: str, **kwargs,
    ) -> list[Generation3DResult]:
        """Generate one or more 3D assets from a text prompt."""
        ...
