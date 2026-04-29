"""text/summarization modality.

Public surface:
  - MODALITY: str (MIME-shaped tag; used by discover_modalities)
  - build_router(registry) -> APIRouter (mounted by the worker)
  - SummarizationResult dataclass
  - SummarizationModel Protocol
  - SummarizationClient
  - PROBE_DEFAULTS

Wire contract (Cohere-compat):
  - POST /v1/summarize

This is muse's second modality with a Cohere-shape wire envelope rather
than OpenAI-compat (after text/rerank). OpenAI has no summarization
API; Cohere's /v1/summarize was the de-facto reference, and downstream
tooling expects the {text, length, format, model} -> {id, model,
summary, usage} shape.
"""
from muse.modalities.text_summarization.client import SummarizationClient
from muse.modalities.text_summarization.protocol import (
    SummarizationModel,
    SummarizationResult,
)
from muse.modalities.text_summarization.routes import build_router


MODALITY = "text/summarization"


# Per-modality probe defaults read by `muse models probe`.
PROBE_DEFAULTS = {
    "shape": "200 word input, medium length output",
    "call": lambda m: m.summarize(
        "muse is a model-agnostic multi-modality generation server. It hosts text, "
        "image, audio, and video models behind a unified HTTP API that mirrors OpenAI "
        "where possible. Each modality is a self-contained plugin: it declares its MIME "
        "tag, contributes a build_router function, and the discovery layer wires it in. "
        "Models are pulled into per-model venvs so that conflicting dependencies between "
        "different model families never break each other.",
        length="medium",
    ),
}


__all__ = [
    "MODALITY",
    "PROBE_DEFAULTS",
    "build_router",
    "SummarizationResult",
    "SummarizationModel",
    "SummarizationClient",
]
