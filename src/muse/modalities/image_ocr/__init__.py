"""image/ocr modality.

Public surface:
  - MODALITY: str (MIME-shaped tag; used by discover_modalities)
  - build_router(registry) -> APIRouter (mounted by the worker)
  - OcrResult dataclass
  - OcrModel Protocol
  - OcrClient (HTTP client for /v1/images/ocr)

Wire contract: POST /v1/images/ocr (multipart/form-data in, JSON out).
Mirrors /v1/audio/transcriptions: one file upload, plain text response
with optional usage block.

Bundled: trocr-base-printed (English printed-text OCR, 334M, MIT).
Curated: nougat-base, texteller, trocr-large-handwritten.

Excludes image-text-to-text VLMs (those are reserved for the future
#97 image/description modality).
"""
from muse.modalities.image_ocr.client import OcrClient
from muse.modalities.image_ocr.protocol import OcrModel, OcrResult
from muse.modalities.image_ocr.routes import build_router


MODALITY = "image/ocr"


def _probe_call(model):
    """Probe-default body: small white image, capped tokens.

    Imported lazily by `muse models probe` via PROBE_DEFAULTS["call"];
    PIL is a runtime dep of any OCR model so it's safe to import here.
    """
    from PIL import Image
    img = Image.new("RGB", (64, 64), (255, 255, 255))
    return model.ocr(img, max_new_tokens=8)


# Per-modality probe defaults read by `muse models probe`.
PROBE_DEFAULTS = {
    "shape": "1 small (64x64) white image, max_new_tokens=8",
    "call": _probe_call,
}


__all__ = [
    "MODALITY",
    "PROBE_DEFAULTS",
    "build_router",
    "OcrModel",
    "OcrResult",
    "OcrClient",
]
