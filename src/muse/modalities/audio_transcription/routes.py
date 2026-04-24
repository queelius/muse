"""Stub routes for audio/transcription.

Replaced in Task 3 with the real FastAPI multipart endpoints
(/v1/audio/transcriptions and /v1/audio/translations). For now this
exists only so the modality's build_router() shim in __init__.py has
something to import, which keeps run_worker's "mount all discovered
modality routers" path working between Task 1 (protocol) and Task 3
(routes).
"""
from __future__ import annotations

from fastapi import APIRouter

from muse.core.registry import ModalityRegistry


def build_router(registry: ModalityRegistry) -> APIRouter:
    """Empty router placeholder; Task 3 adds the two OpenAI-shape endpoints."""
    return APIRouter()
