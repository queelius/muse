"""Stub routes for text/summarization.

Replaced in Task C with the real /v1/summarize endpoint. For now this
exists so build_router(registry) is importable, keeping discovery
clean between Task A (protocol + codec) and Task C (full route).
"""
from __future__ import annotations

from fastapi import APIRouter

from muse.core.registry import ModalityRegistry


def build_router(registry: ModalityRegistry) -> APIRouter:
    """Empty placeholder. Task C adds POST /v1/summarize."""
    return APIRouter()
