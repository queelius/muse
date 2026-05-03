"""text/classification modality.

Public surface:
  - MODALITY: str (MIME-shaped tag; used by discover_modalities)
  - build_router(registry) -> APIRouter (mounted by the worker)
  - ClassificationResult dataclass
  - TextClassifierModel Protocol (fine-tuned classifiers)
  - ZeroShotClassifierModel Protocol (NLI heads)
  - ModerationsClient (HTTP client for /v1/moderations)
  - ClassificationsClient (HTTP client for /v1/text/classifications)

Wire contracts (OpenAI-compat):
  - POST /v1/moderations: reduces classifier scores to flag/categories
  - POST /v1/text/classifications: returns full label distribution;
    capability-gated dispatch between fine-tuned classifier and
    zero-shot NLI runtime

Modality vs URL: text/classification's MIME tag is broader than its
primary URL (/v1/moderations). The /v1/text/classifications route
(added v0.35.0) shares the same runtime layer; capability flags
(supports_classification, supports_zero_shot) gate which models
each route accepts.
"""
from muse.modalities.text_classification.client import (
    ClassificationsClient,
    ModerationsClient,
)
from muse.modalities.text_classification.protocol import (
    ClassificationResult,
    TextClassifierModel,
    ZeroShotClassifierModel,
)
from muse.modalities.text_classification.routes import build_router


MODALITY = "text/classification"


# Per-modality probe defaults read by `muse models probe`. The probe
# uses the classify() method which both runtimes share. Zero-shot-only
# models (no classify) get probed via classify_zero_shot when the
# probe worker reads supports_zero_shot from capabilities.
PROBE_DEFAULTS = {
    "shape": "1 short string",
    "call": lambda m: (
        m.classify(["probe text"])
        if hasattr(m, "classify") else
        m.classify_zero_shot(
            ["probe text"], candidate_labels=["positive", "negative"],
        )
    ),
}


__all__ = [
    "MODALITY",
    "PROBE_DEFAULTS",
    "build_router",
    "ClassificationResult",
    "TextClassifierModel",
    "ZeroShotClassifierModel",
    "ModerationsClient",
    "ClassificationsClient",
]
