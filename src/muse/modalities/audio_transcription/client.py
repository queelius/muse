"""HTTP client for /v1/audio/transcriptions and /v1/audio/translations.

Parallel to EmbeddingsClient / SpeechClient / GenerationsClient /
ChatClient. Uses `requests` with a multipart/form-data body. Response
type tracks `response_format`: json returns the plain string (text
field), text/srt/vtt return the raw string body, verbose_json returns
the full dict.

Base URL precedence: explicit constructor arg > MUSE_SERVER env var >
http://localhost:8000. Trailing slashes stripped.
"""
from __future__ import annotations

import os
from typing import Any

import requests


_DEFAULT_SERVER = "http://localhost:8000"


class TranscriptionClient:
    """Minimal HTTP client for the audio/transcription modality."""

    def __init__(
        self,
        *,
        server_url: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        url = (
            server_url
            or os.environ.get("MUSE_SERVER")
            or _DEFAULT_SERVER
        )
        self.server_url = url.rstrip("/")
        self._timeout = timeout

    def transcribe(
        self,
        *,
        audio: bytes,
        filename: str,
        model: str,
        language: str | None = None,
        prompt: str | None = None,
        response_format: str = "json",
        temperature: float = 0.0,
        word_timestamps: bool = False,
        vad_filter: bool = False,
    ) -> str | dict[str, Any]:
        """Transcribe audio in its source language.

        Return type depends on `response_format`:
          - json (default): returns the transcript `str` (just the text field)
          - text: returns the raw transcript `str`
          - srt, vtt: returns the subtitle file contents as `str`
          - verbose_json: returns the full `dict` (task, language, duration, text, segments, ...)
        """
        return self._post(
            "/v1/audio/transcriptions",
            audio=audio, filename=filename, model=model,
            language=language, prompt=prompt,
            response_format=response_format, temperature=temperature,
            word_timestamps=word_timestamps, vad_filter=vad_filter,
        )

    def translate(
        self,
        *,
        audio: bytes,
        filename: str,
        model: str,
        prompt: str | None = None,
        response_format: str = "json",
        temperature: float = 0.0,
        word_timestamps: bool = False,
        vad_filter: bool = False,
    ) -> str | dict[str, Any]:
        """Translate source-language audio to an English transcript.

        Same return-type contract as `transcribe`; `language` form field
        is implicitly dropped.
        """
        return self._post(
            "/v1/audio/translations",
            audio=audio, filename=filename, model=model,
            language=None, prompt=prompt,
            response_format=response_format, temperature=temperature,
            word_timestamps=word_timestamps, vad_filter=vad_filter,
        )

    def _post(
        self,
        path: str,
        *,
        audio: bytes,
        filename: str,
        model: str,
        language: str | None,
        prompt: str | None,
        response_format: str,
        temperature: float,
        word_timestamps: bool,
        vad_filter: bool,
    ) -> str | dict[str, Any]:
        files = {"file": (filename, audio)}
        # Data as a LIST of tuples so bracketed alias
        # `timestamp_granularities[]` can repeat cleanly. A dict would
        # silently drop duplicates on the same key.
        data: list[tuple[str, str]] = [
            ("model", model),
            ("response_format", response_format),
            ("temperature", str(temperature)),
            ("vad_filter", "true" if vad_filter else "false"),
        ]
        if language is not None:
            data.append(("language", language))
        if prompt is not None:
            data.append(("prompt", prompt))
        if word_timestamps:
            data.append(("timestamp_granularities[]", "word"))

        r = requests.post(
            f"{self.server_url}{path}",
            files=files, data=data, timeout=self._timeout,
        )
        r.raise_for_status()

        ct = r.headers.get("content-type", "")
        if "json" in ct:
            j = r.json()
            # json format returns {"text": "..."}; verbose_json returns a full dict.
            if isinstance(j, dict) and set(j.keys()) == {"text"}:
                return j["text"]
            return j
        return r.text
