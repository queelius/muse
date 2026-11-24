"""Wire encoding for image/matting: lossless RGBA PNG bytes and JSON envelope.

Nothing here imports PIL or numpy at module load; images arrive as PIL objects.
"""
from __future__ import annotations

import base64
import io
from typing import Any


def png_bytes(image: Any) -> bytes:
    """Encode a PIL image as lossless PNG (preserving RGBA alpha)."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def encode_matting(result: Any, model_id: str) -> dict[str, Any]:
    """Encode a ``MattingResult`` into the ``image.matting`` JSON envelope."""
    return {
        "object": "image.matting",
        "model": model_id,
        "data": [
            {
                "b64_json": _b64(png_bytes(result.image)),
                "width": int(result.width),
                "height": int(result.height),
                "transparent_fraction": float(result.transparent_fraction),
                "opaque_fraction": float(result.opaque_fraction),
            }
        ],
    }
