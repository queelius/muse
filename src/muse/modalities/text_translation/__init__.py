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

Task 1 (this file) ships the package skeleton only: protocol, codec,
client, and the language-name table. `build_router` is a stub; Task 3
replaces it with the real FastAPI router mounting POST /v1/translate,
POST /translate, and GET /languages.

The stub returns an empty `APIRouter()` rather than raising
NotImplementedError. `muse.cli_impl.worker.run_worker` unconditionally
calls `build_router(registry)` for EVERY discovered modality (not just
loaded ones) so that an empty registry still gets the OpenAI 404
envelope instead of FastAPI's default -- so a raising stub would break
every `run_worker` invocation (and its test coverage) the moment this
package is discoverable, not just once Task 3's real routes are hit.
Mirrors the 3d/generation skeleton precedent (model_3d_generation/routes.py,
commit 8b406bd): a real, callable router object now; NotImplementedError
deferred to the level that actually needs it (there, per-route handlers;
here, whichever real route Task 3 adds).
"""
from fastapi import APIRouter

from muse.modalities.text_translation.client import TranslateClient
from muse.modalities.text_translation.protocol import (
    TranslationBackend,
    TranslationResult,
    UnsupportedLanguageError,
)


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
    """Stub for Task 1: an empty router with no routes mounted.

    Task 3 replaces this with the real router (POST /v1/translate,
    POST /translate, GET /languages). Kept non-raising so
    `run_worker`'s unconditional build_router(registry) call over
    every discovered modality does not break before Task 3 lands.
    """
    return APIRouter()


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
