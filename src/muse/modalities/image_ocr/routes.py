"""FastAPI routes for /v1/images/ocr.

Stub: replaced in Task C. Defined here so `__init__.py` can import
`build_router` without an ImportError during the staged build.
"""
from __future__ import annotations

from fastapi import APIRouter

from muse.core.registry import ModalityRegistry


MODALITY = "image/ocr"


def build_router(registry: ModalityRegistry) -> APIRouter:
    return APIRouter()
