"""3d/generation modality.

Two routes share one MIME tag:

  POST /v1/3d/generations  - text-to-3d (JSON body)
  POST /v1/3d/from-image   - image-to-3d (multipart/form-data)

Capability flags `supports_text_to_3d` and `supports_image_to_3d` on
each manifest gate which route a model accepts; mismatch returns 400
before the runtime is invoked.

Output format is GLB (binary glTF, embedded textures, web-friendly).
The `format` field on each data element is reserved as a string so
future formats (USDZ, OBJ-archive) are additive.

Bundled default: TripoSR (image-to-3d only, ~120MB, MIT, CUDA).
Curated coverage: TRELLIS (both routes), Hunyuan3D-2, Wonder3D, Shap-E.

Python package directory is `model_3d_generation/` because Python
identifiers cannot start with a digit; the MIME tag remains
`3d/generation`.
"""
from muse.modalities.model_3d_generation.client import Generation3DClient
from muse.modalities.model_3d_generation.protocol import (
    Generation3DBackend,
    Generation3DResult,
    ImageTo3DBackend,
    TextTo3DBackend,
)
from muse.modalities.model_3d_generation.routes import build_router


MODALITY = "3d/generation"


def _probe_call(model):
    """Probe-default body: a tiny synthetic 256x256 RGB image, written
    to a temp PNG, then run through `image_to_3d`.

    Mirrors audio_classification's probe-with-temp-file pattern: the
    temp file is unlinked unconditionally so a failed call (transient
    OOM, model error) does not leak a PNG under /tmp. Probe targets
    the image-to-3d direction because every plausible v0.41.0 model
    declares `supports_image_to_3d` even when text-to-3d is also
    available; image-to-3d is the dominant 3D direction in 2026.
    """
    import os
    import tempfile

    from PIL import Image

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        path = f.name
    try:
        Image.new("RGB", (256, 256), color=(128, 128, 128)).save(path, "PNG")
        return model.image_to_3d(path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


PROBE_DEFAULTS = {
    "shape": "1 small synthetic 256x256 image",
    "call": _probe_call,
}


__all__ = [
    "MODALITY",
    "PROBE_DEFAULTS",
    "build_router",
    "Generation3DBackend",
    "Generation3DClient",
    "Generation3DResult",
    "ImageTo3DBackend",
    "TextTo3DBackend",
]
