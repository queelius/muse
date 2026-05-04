"""Encoding for /v1/images/ocr responses.

Pure functions: OcrResult to OpenAI envelope dict. Tested without
FastAPI.
"""
from __future__ import annotations

import uuid
from typing import Any

from muse.modalities.image_ocr.protocol import OcrResult


def encode_ocr(result: OcrResult) -> dict[str, Any]:
    """Build the OpenAI-shape envelope for /v1/images/ocr.

    Mirrors /v1/audio/transcriptions: {id, model, text, usage}.
    """
    return {
        "id": f"ocr-{uuid.uuid4().hex[:24]}",
        "model": result.model_id,
        "text": result.text,
        "usage": {"completion_tokens": result.completion_tokens},
    }
