"""HTTP client for /v1/audio/classifications."""
from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

import requests


_DEFAULT_SERVER = "http://localhost:8000"


def _to_bytes(audio: Any) -> bytes:
    """Coerce bytes / path / file-like to bytes."""
    if isinstance(audio, bytes):
        return audio
    if isinstance(audio, (str, Path)):
        return Path(audio).read_bytes()
    if hasattr(audio, "read"):
        return audio.read()
    raise TypeError(f"unsupported audio type: {type(audio).__name__}")


class AudioClassificationsClient:
    """HTTP client for the audio/classification modality."""

    def __init__(
        self,
        server_url: str | None = None,
        *,
        timeout: float = 300.0,
    ) -> None:
        url = (
            server_url
            or os.environ.get("MUSE_SERVER")
            or _DEFAULT_SERVER
        )
        self.server_url = url.rstrip("/")
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
        files = {"file": (filename, _to_bytes(audio), content_type)}
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
