"""Protocols + dataclasses for 3d/generation.

Two routes share one MIME tag (mirroring audio_generation's music+sfx
split): `/v1/3d/generations` (text-to-3d) and `/v1/3d/from-image`
(image-to-3d). Capability flags `supports_text_to_3d` and
`supports_image_to_3d` on the manifest gate which route a given
backend accepts.

The protocol is split into two single-direction protocols
(`ImageTo3DBackend`, `TextTo3DBackend`), both `@runtime_checkable`.
This mirrors the asymmetric reality: TripoSR (the bundled default) is
image-to-3d only; Shap-E is text-to-3d only; TRELLIS does both. With
a unified two-method protocol every single-direction backend would
fail isinstance checks even though it is perfectly correct against
its own capability flag. Splitting the protocol keeps the static
type-check honest.

`Generation3DBackend = Union[ImageTo3DBackend, TextTo3DBackend]` is
preserved as a typing alias for callers that don't care which
direction generated a result (e.g., consumers of `Generation3DResult`
itself).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Union, runtime_checkable


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
class ImageTo3DBackend(Protocol):
    """Structural protocol for backends that take an image as input.

    Examples: TripoSR, Wonder3D, Hunyuan3D-2 (img-mode), TRELLIS (mixed).
    """

    def image_to_3d(
        self, image_path: str, **kwargs,
    ) -> list[Generation3DResult]:
        """Generate one or more 3D assets from a single image."""
        ...


@runtime_checkable
class TextTo3DBackend(Protocol):
    """Structural protocol for backends that take a text prompt as input.

    Examples: Shap-E, TRELLIS (mixed).
    """

    def text_to_3d(
        self, prompt: str, **kwargs,
    ) -> list[Generation3DResult]:
        """Generate one or more 3D assets from a text prompt."""
        ...


# Union alias for callers that accept either direction. The Union form
# preserves both runtime-checkable components for downstream isinstance
# usage like `isinstance(obj, Generation3DBackend)` (which falls back to
# member-by-member checks at runtime via typing.Union semantics).
Generation3DBackend = Union[ImageTo3DBackend, TextTo3DBackend]
