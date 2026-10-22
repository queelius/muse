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

DISCOVERY = {
    "summary": (
        "Text classification. Moderation, sentiment and zero-shot labelling "
        "of text with scored labels."
    ),
    "use_for": [
        "moderating text for policy categories",
        "labelling text with sentiment or your own candidate labels",
    ],
    "not_for": [
        "typed decisions with explicit criteria (use decision/evaluation)",
        "audio classification (use audio/classification)",
    ],
    "routes": [
        {
            "method": "POST",
            "path": "/v1/moderations",
            "request_content_type": "application/json",
            "response_content_type": "application/json",
            "compat": "openai:moderations",
            "mcp_tool": "muse_moderate",
            "file_fields": [],
            "media_fields": [],
            "example": {"model": "<model>", "input": "some text to check"},
        },
        {
            "method": "POST",
            "path": "/v1/text/classifications",
            "request_content_type": "application/json",
            "response_content_type": "application/json",
            "compat": None,
            "mcp_tool": "muse_classify",
            "file_fields": [],
            "media_fields": [],
            "example": {"model": "<model>", "input": "I love this", "top_k": 3},
        },
    ],
}


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
    "DISCOVERY",
    "PROBE_DEFAULTS",
    "build_router",
    "ClassificationResult",
    "TextClassifierModel",
    "ZeroShotClassifierModel",
    "ModerationsClient",
    "ClassificationsClient",
]
