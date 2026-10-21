"""HTTP client for /v1/audio/classifications."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import requests

from muse.modalities._common.client import resolve_server_url, upload_bytes


class AudioClassificationsClient:
    """HTTP client for the audio/classification modality."""

    def __init__(
        self,
        server_url: str | None = None,
        *,
        timeout: float = 300.0,
    ) -> None:
        self.server_url = resolve_server_url(server_url)
        self._timeout = timeout

    def classify(
        self,
        audio: bytes | str | Path | Any,
        *,
        model: str | None = None,
        top_k: int | None = None,
        filename: str = "audio.wav",
        content_type: str = "audio/wav",
    ) -> dict[str, Any]:
        """Send a classification request.

        `audio` accepts bytes, a path-like, or a file-like object with
        .read(). The path's contents (or bytes) are forwarded as the
        multipart `file` field.
        """
        files = {"file": (filename, upload_bytes(audio, kind="audio"), content_type)}
        data: dict[str, Any] = {}
        if model is not None:
            data["model"] = model
        if top_k is not None:
            data["top_k"] = str(top_k)
        r = requests.post(
            f"{self.server_url}/v1/audio/classifications",
            files=files, data=data, timeout=self._timeout,
        )
        r.raise_for_status()
        return r.json()
