"""FastAPI routes for /v1/audio/music and /v1/audio/sfx.

Both routes share the same body shape and codec; the only difference
is the capability key consulted. Music routes gate on
`capabilities.supports_music`; SFX routes gate on
`capabilities.supports_sfx`. When the flag is missing, the default is
True (assume the model supports the kind unless stated otherwise).

Replaced incrementally: Task A leaves a stub; Task C wires up the full
handlers.
"""
from __future__ import annotations

from fastapi import APIRouter

from muse.core.registry import ModalityRegistry


MODALITY = "audio/generation"


def build_router(registry: ModalityRegistry) -> APIRouter:
    """Stub router; Task C replaces with real handlers."""
    return APIRouter()
