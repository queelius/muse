"""HTTP clients for image/cv routes.

Three clients (DepthClient, KeypointClient, ObjectDetectionClient)
share construction (server_url, MUSE_SERVER fallback, timeout). Each
posts multipart/form-data to its route and returns the parsed JSON
envelope.

The image-input handling (bytes / path / file-like / PIL.Image) is
factored into _to_bytes (mirrors OcrClient).
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import requests

from muse.core import config


def _to_bytes(image: Any) -> bytes:
    """Coerce bytes / path / file-like / PIL.Image to bytes (PNG)."""
    if isinstance(image, bytes):
        return image
    if isinstance(image, (str, Path)):
        return Path(image).read_bytes()
    if hasattr(image, "read"):
        return image.read()
    if hasattr(image, "save"):
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()
    raise TypeError(f"unsupported image type: {type(image).__name__}")


class _CVClientBase:
    """Shared construction for DepthClient/KeypointClient/ObjectDetectionClient."""

    def __init__(
        self,
        server_url: str | None = None,
        *,
        timeout: float = 300.0,
    ) -> None:
        url = server_url or config.get("client.server_url")
        self.server_url = url.rstrip("/")
        self._timeout = timeout


class DepthClient(_CVClientBase):
    """HTTP client for /v1/images/depth."""

    def estimate_depth(
        self,
        image: bytes | str | Path | Any,
        *,
        model: str | None = None,
        response_format: str = "png16",
    ) -> dict[str, Any]:
        files = {"image": ("image.png", _to_bytes(image), "image/png")}
        data: dict[str, Any] = {"response_format": response_format}
        if model is not None:
            data["model"] = model
        r = requests.post(
            f"{self.server_url}/v1/images/depth",
            files=files, data=data, timeout=self._timeout,
        )
        r.raise_for_status()
        return r.json()


class KeypointClient(_CVClientBase):
    """HTTP client for /v1/images/keypoints."""

    def detect_keypoints(
        self,
        image: bytes | str | Path | Any,
        *,
        model: str | None = None,
        threshold: float | None = None,
    ) -> dict[str, Any]:
        files = {"image": ("image.png", _to_bytes(image), "image/png")}
        data: dict[str, Any] = {}
        if model is not None:
            data["model"] = model
        if threshold is not None:
            data["threshold"] = str(threshold)
        r = requests.post(
            f"{self.server_url}/v1/images/keypoints",
            files=files, data=data, timeout=self._timeout,
        )
        r.raise_for_status()
        return r.json()


class ObjectDetectionClient(_CVClientBase):
    """HTTP client for /v1/images/detect."""

    def detect_objects(
        self,
        image: bytes | str | Path | Any,
        *,
        model: str | None = None,
        threshold: float | None = None,
        max_detections: int | None = None,
    ) -> dict[str, Any]:
        files = {"image": ("image.png", _to_bytes(image), "image/png")}
        data: dict[str, Any] = {}
        if model is not None:
            data["model"] = model
        if threshold is not None:
            data["threshold"] = str(threshold)
        if max_detections is not None:
            data["max_detections"] = str(max_detections)
        r = requests.post(
            f"{self.server_url}/v1/images/detect",
            files=files, data=data, timeout=self._timeout,
        )
        r.raise_for_status()
        return r.json()
