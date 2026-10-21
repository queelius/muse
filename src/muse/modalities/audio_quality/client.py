"""HTTP client for ``POST /v1/audio/quality``."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import requests

from muse.modalities._common.client import resolve_server_url, upload_bytes


class AudioQualityClient:
    """Thin multipart client for audio-quality assessment."""

    def __init__(
        self,
        server_url: str | None = None,
        *,
        timeout: float = 300.0,
    ) -> None:
        self.server_url = resolve_server_url(server_url)
        self._timeout = timeout

    def assess(
        self,
        audio: bytes | str | Path | Any,
        *,
        model: str | None = None,
        filename: str = "audio.wav",
        content_type: str = "audio/wav",
    ) -> dict[str, Any]:
        """Assess bytes, a path, or a readable file-like object."""
        files = {"file": (filename, upload_bytes(audio, kind="audio"), content_type)}
        data: dict[str, str] = {}
        if model is not None:
            data["model"] = model
        response = requests.post(
            f"{self.server_url}/v1/audio/quality",
            files=files,
            data=data,
            timeout=self._timeout,
        )
        response.raise_for_status()
        return response.json()
