"""text/classification modality.

Public surface:
  - MODALITY: str (MIME-shaped tag; used by discover_modalities)
  - build_router(registry) -> APIRouter (mounted by the worker)
  - ClassificationResult dataclass
  - TextClassifierModel Protocol
  - ModerationsClient

Wire contract (OpenAI-compat):
  - POST /v1/moderations

Modality vs URL: this is muse's first modality whose MIME tag
(text/classification) is broader than its primary URL (/v1/moderations).
Future routes (e.g., /v1/text/classifications for sentiment) can share
the same runtime + dataclasses without a new modality package.
"""
from muse.modalities.text_classification.client import ModerationsClient
from muse.modalities.text_classification.protocol import (
    ClassificationResult,
    TextClassifierModel,
)
from muse.modalities.text_classification.routes import build_router


MODALITY = "text/classification"


# Per-modality probe defaults read by `muse models probe`.
PROBE_DEFAULTS = {
    "shape": "1 short string",
    "call": lambda m: m.classify(["probe text"]),
}


__all__ = [
    "MODALITY",
    "PROBE_DEFAULTS",
    "build_router",
    "ClassificationResult",
    "TextClassifierModel",
    "ModerationsClient",
]
