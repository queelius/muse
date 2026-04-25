"""Stub routes for text/classification.

Replaced in Task 3 with the real /v1/moderations endpoint. For now
this exists only so the modality's build_router() in __init__.py has
something to import, which keeps run_worker's "mount all discovered
modality routers" path working between Task 1 (protocol) and Task 3
(routes).
"""
from fastapi import APIRouter

from muse.core.registry import ModalityRegistry


def build_router(registry: ModalityRegistry) -> APIRouter:
    """Empty router placeholder; Task 3 adds POST /v1/moderations."""
    return APIRouter()
