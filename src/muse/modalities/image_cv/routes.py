"""FastAPI routes for image/cv (stub; real handlers in Tasks E/F/G).

Defined here so __init__.py can import build_router during the staged
build without an ImportError. The real depth/keypoints/detect handlers
land in subsequent commits.
"""
from __future__ import annotations

from fastapi import APIRouter

from muse.core.registry import ModalityRegistry


MODALITY = "image/cv"


def build_router(registry: ModalityRegistry) -> APIRouter:
    return APIRouter()
