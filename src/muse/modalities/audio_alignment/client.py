"""HTTP client for ``POST /v1/audio/alignments``."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import requests

from muse.core import config


def _to_bytes(audio: bytes | str | Path | Any) -> bytes:
    if isinstance(audio, bytes):
        return audio
    if isinstance(audio, (str, Path)):
        return Path(audio).read_bytes()
    if hasattr(audio, "read"):
        return audio.read()
    raise TypeError(f"unsupported audio type: {type(audio).__name__}")


class AudioAlignmentClient:
    """Thin multipart client for reference-text forced alignment."""

    def __init__(
        self,
        server_url: str | None = None,
        *,
        timeout: float = 300.0,
    ) -> None:
        url = server_url or config.get("client.server_url")
        self.server_url = url.rstrip("/")
        self._timeout = timeout

    def align(
        self,
        audio: bytes | str | Path | Any,
        text: str,
        *,
        model: str | None = None,
        language: str | None = None,
        filename: str = "audio.wav",
        content_type: str = "audio/wav",
    ) -> dict[str, Any]:
        """Align reference text against bytes, a path, or a readable file."""
        files = {"file": (filename, _to_bytes(audio), content_type)}
        data = {"text": text}
        if model is not None:
            data["model"] = model
        if language is not None:
            data["language"] = language
        response = requests.post(
            f"{self.server_url}/v1/audio/alignments",
            files=files,
            data=data,
            timeout=self._timeout,
        )
        response.raise_for_status()
        return response.json()
