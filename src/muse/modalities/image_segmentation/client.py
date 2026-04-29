"""HTTP client for /v1/images/segment.

Multipart upload of an image plus mode-aware form fields. Points and
boxes are Python lists at the public API; the client serializes them
via ``json.dumps`` before posting as Form fields, mirroring the wire
contract documented in ``routes.py``.

The default ``response_format`` is ``"png_b64"`` so each returned mask
is a base64-encoded PNG string. Pass ``mask_format="rle"`` for
COCO-style RLE dicts; downstream tooling like pycocotools and
FiftyOne consumes that shape directly.
"""
from __future__ import annotations

import json
import os
from typing import Any

import requests


def _resolve_base_url(base_url: str | None) -> str:
    base = base_url or os.environ.get("MUSE_SERVER", "http://localhost:8000")
    return base.rstrip("/")


class ImageSegmentationClient:
    """Thin HTTP client against the muse images.segment endpoint."""

    def __init__(self, base_url: str | None = None, timeout: float = 600.0) -> None:
        self.base_url = _resolve_base_url(base_url)
        self.timeout = timeout

    def segment(
        self,
        *,
        image: bytes,
        model: str | None = None,
        mode: str = "auto",
        prompt: str | None = None,
        points: list[list[int]] | None = None,
        boxes: list[list[int]] | None = None,
        mask_format: str = "png_b64",
        max_masks: int = 16,
    ) -> dict[str, Any]:
        """Segment ``image`` using ``mode``. Returns the parsed JSON envelope."""
        files = {"image": ("image.png", image, "image/png")}
        data: list[tuple[str, str]] = [
            ("mode", mode),
            ("mask_format", mask_format),
            ("max_masks", str(max_masks)),
        ]
        if model is not None:
            data.append(("model", model))
        if prompt is not None:
            data.append(("prompt", prompt))
        if points is not None:
            data.append(("points", json.dumps(points)))
        if boxes is not None:
            data.append(("boxes", json.dumps(boxes)))

        r = requests.post(
            f"{self.base_url}/v1/images/segment",
            files=files, data=data, timeout=self.timeout,
        )
        if r.status_code != 200:
            raise RuntimeError(
                f"server returned {r.status_code}: {r.text[:500]}"
            )
        return r.json()
