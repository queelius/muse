"""HTTP client for /v1/audio/embeddings.

Mirrors the shape of TranscriptionClient (multipart) plus
ImageEmbeddingsClient (envelope-aware return types). Server URL public
attribute, MUSE_SERVER env fallback, requests + raise_for_status.

Helper `embed(audio: bytes | list[bytes], ...)` returns
`list[list[float]]`. The lower-level `embed_envelope(...)` returns
the full OpenAI-shape envelope when callers need `usage`, `model`, or
per-entry indices.
"""
from __future__ import annotations

import os
from typing import Any, Union

import requests

from muse.modalities.audio_embedding.codec import base64_to_embedding


_DEFAULT_SERVER = "http://localhost:8000"


class AudioEmbeddingsClient:
    """Thin HTTP client against muse's /v1/audio/embeddings endpoint.

    The `embed(...)` helper accepts either:
      - a bytes object (single audio file)
      - a list of bytes (batched embedding via repeated `file` parts)

    Returns `list[list[float]]` regardless of wire format. The full
    envelope (with `usage`, `model`, etc.) is available via
    `embed_envelope(...)`.
    """

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
        self.timeout = timeout

    def embed(
        self,
        audio: Union[bytes, list[bytes]],
        *,
        model: str | None = None,
        encoding_format: str = "float",
        filename: str = "audio.wav",
        content_type: str = "audio/wav",
    ) -> list[list[float]]:
        """Embed audio clip(s); return vectors as list[list[float]].

        `audio` is bytes (single clip) or list of bytes (batch). When
        batching, the same `filename` and `content_type` apply to all
        parts; servers don't use these for routing (librosa sniffs the
        bytes), so the defaults are fine for any audio format.
        """
        if encoding_format not in ("float", "base64"):
            raise ValueError(
                f"encoding_format must be 'float' or 'base64', "
                f"got {encoding_format!r}"
            )

        envelope = self.embed_envelope(
            audio,
            model=model,
            encoding_format=encoding_format,
            filename=filename,
            content_type=content_type,
        )
        entries = envelope["data"]
        if encoding_format == "base64":
            return [base64_to_embedding(e["embedding"]) for e in entries]
        return [e["embedding"] for e in entries]

    def embed_envelope(
        self,
        audio: Union[bytes, list[bytes]],
        *,
        model: str | None = None,
        encoding_format: str = "float",
        filename: str = "audio.wav",
        content_type: str = "audio/wav",
    ) -> dict[str, Any]:
        """Send the request and return the full OpenAI-shape envelope.

        Useful when callers need `usage`, `model`, or per-entry indices
        rather than just the float lists.
        """
        if encoding_format not in ("float", "base64"):
            raise ValueError(
                f"encoding_format must be 'float' or 'base64', "
                f"got {encoding_format!r}"
            )

        files = self._build_files(audio, filename=filename, content_type=content_type)
        data: list[tuple[str, str]] = [
            ("encoding_format", encoding_format),
        ]
        if model is not None:
            data.append(("model", model))

        r = requests.post(
            f"{self.server_url}/v1/audio/embeddings",
            files=files, data=data, timeout=self.timeout,
        )
        if r.status_code != 200:
            raise RuntimeError(f"server returned {r.status_code}: {r.text[:500]}")
        return r.json()

    @staticmethod
    def _build_files(
        audio: Union[bytes, list[bytes]],
        *,
        filename: str,
        content_type: str,
    ) -> list[tuple[str, tuple[str, bytes, str]]]:
        """Build the multipart `files` payload for one or more clips.

        Returns a list of (field_name, (filename, bytes, content_type))
        tuples. Repeating the field name is how multipart represents a
        list of UploadFile objects on the server side.
        """
        if isinstance(audio, bytes):
            return [("file", (filename, audio, content_type))]
        if isinstance(audio, list):
            if not all(isinstance(a, bytes) for a in audio):
                raise TypeError("audio list must contain only bytes objects")
            return [
                ("file", (filename, raw, content_type)) for raw in audio
            ]
        raise TypeError(
            f"audio must be bytes or list of bytes; got {type(audio).__name__}"
        )
