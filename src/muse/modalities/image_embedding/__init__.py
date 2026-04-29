"""image/embedding modality.

Public surface:
  - MODALITY: str (MIME-shaped tag; used by discover_modalities)
  - build_router(registry) -> APIRouter (mounted by the worker)
  - ImageEmbeddingResult dataclass
  - ImageEmbeddingModel Protocol
  - ImageEmbeddingsClient (HTTP client; exported in Task D)
  - PROBE_DEFAULTS

Wire contract (OpenAI-shape, mirroring /v1/embeddings):
  - POST /v1/images/embeddings

Each `input` entry is a data: URL or http(s):// URL pointing at a
PNG/JPEG/WEBP image. Image decoding uses the shared
`decode_image_input` helper from the image_generation modality so all
muse routes accept the same image-input shapes.

Cross-modal note: CLIP/SigLIP can also embed text. The current modality
exposes only image embedding. Future work may add a text route at
`/v1/images/embeddings/text` or thread a `text` field through the
existing route guarded by a `supports_text_embeddings_too` capability
flag. Out of scope for v0.23.0.
"""
from muse.modalities.image_embedding.client import ImageEmbeddingsClient
from muse.modalities.image_embedding.protocol import (
    ImageEmbeddingModel,
    ImageEmbeddingResult,
)
from muse.modalities.image_embedding.routes import build_router


MODALITY = "image/embedding"


def _make_probe_image():
    """Build a 224x224 mid-gray PIL image used by `muse models probe`.

    Lazy-imports PIL so the module loads without it; if PIL is missing
    the probe will fail with a clear error, but discovery still works.
    """
    from PIL import Image
    return Image.new("RGB", (224, 224), (128, 128, 128))


# Per-modality probe defaults read by `muse models probe`.
PROBE_DEFAULTS = {
    "shape": "1 image, 224x224",
    "call": lambda m: m.embed([_make_probe_image()]),
}


__all__ = [
    "MODALITY",
    "PROBE_DEFAULTS",
    "build_router",
    "ImageEmbeddingResult",
    "ImageEmbeddingModel",
    "ImageEmbeddingsClient",
]
