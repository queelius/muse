"""HTTP clients for image/cv routes.

Three clients (DepthClient, KeypointClient, ObjectDetectionClient)
share construction (server_url, MUSE_SERVER fallback, timeout). Each
posts multipart/form-data to its route and returns the parsed JSON
envelope.

The image-input handling (bytes / path / file-like / PIL.Image) is
shared via muse.modalities._common.client.upload_bytes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import requests

from muse.modalities._common.client import resolve_server_url, upload_bytes


class _CVClientBase:
    """Shared construction for DepthClient/KeypointClient/ObjectDetectionClient."""

    def __init__(
        self,
        server_url: str | None = None,
        *,
        timeout: float = 300.0,
    ) -> None:
        self.server_url = resolve_server_url(server_url)
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
        files = {"image": ("image.png", upload_bytes(image, kind="image", allow_pil=True), "image/png")}
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
        files = {"image": ("image.png", upload_bytes(image, kind="image", allow_pil=True), "image/png")}
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
        files = {"image": ("image.png", upload_bytes(image, kind="image", allow_pil=True), "image/png")}
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
