"""text/translation modality.

Public surface:
  - MODALITY: str (MIME-shaped tag; used by discover_modalities)
  - build_router(registry) -> APIRouter (mounted by the worker)
  - MODEL_OPTIONAL_PATHS: tuple[str, ...] (paths where the gateway may
    resolve a default model when the request omits `model`)
  - TranslationResult dataclass
  - TranslationBackend Protocol
  - TranslateClient
  - PROBE_DEFAULTS

Wire contract (LibreTranslate-compat):
  - POST /v1/translate and POST /translate (alias)
  - GET /languages

This is muse's twentieth modality and its first LibreTranslate-compat
one. See docs/superpowers/specs/2026-07-09-text-translation-design.md
for the full design.

Task 1 shipped the package skeleton: protocol, codec, client, and the
language-name table. Task 3 (this revision) adds the real FastAPI router
mounting POST /v1/translate, POST /translate (alias, identical handler),
and GET /languages -- see routes.py.
"""
from fastapi import APIRouter

from muse.modalities.text_translation.client import TranslateClient
from muse.modalities.text_translation.protocol import (
    TranslationBackend,
    TranslationResult,
    UnsupportedLanguageError,
)
from muse.modalities.text_translation.routes import build_router as _build_router


MODALITY = "text/translation"


# Paths on which the gateway may resolve a default model when the
# request omits `model` (LibreTranslate clients never send one).
MODEL_OPTIONAL_PATHS = ("/v1/translate", "/translate", "/languages")


# Per-modality probe defaults read by `muse models probe`.
PROBE_DEFAULTS = {
    "shape": "q='The weather is nice today.' en->es",
    "call": lambda m: m.translate(
        ["The weather is nice today."], source="en", target="es",
    ),
}


def build_router(registry) -> APIRouter:
    """Mount POST /v1/translate, POST /translate (alias), GET /languages.

    Thin re-export of routes.build_router; kept as the __init__-level
    name so `discover_modalities` (which imports this package and reads
    module-level `build_router`) finds it without reaching into routes.py.
    """
    return _build_router(registry)


__all__ = [
    "MODALITY",
    "MODEL_OPTIONAL_PATHS",
    "PROBE_DEFAULTS",
    "build_router",
    "TranslationResult",
    "TranslationBackend",
    "UnsupportedLanguageError",
    "TranslateClient",
]
