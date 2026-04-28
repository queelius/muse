"""HTTP clients for the audio/generation modality.

Two clients, one per route. The split mirrors the URL split: a user
calling MusicClient.generate is hitting /v1/audio/music; SFXClient is
hitting /v1/audio/sfx. The server enforces per-model capability gates.

Both clients return raw audio bytes (matching /v1/audio/speech): the
response Content-Type indicates the encoding (audio/wav by default).
"""
from __future__ import annotations

import os
from typing import Any

import requests


_DEFAULT_SERVER = "http://localhost:8000"


class _AudioGenerationClient:
    """Shared base. Subclasses provide their own route on top."""

    _route: str = ""

    def __init__(
        self,
        server_url: str | None = None,
        *,
        timeout: float = 300.0,
    ) -> None:
        url = server_url or os.environ.get("MUSE_SERVER") or _DEFAULT_SERVER
        self.server_url = url.rstrip("/")
        self._timeout = timeout

    def _post(
        self,
        prompt: str,
        *,
        model: str | None,
        duration: float | None,
        seed: int | None,
        response_format: str,
        steps: int | None,
        guidance: float | None,
        negative_prompt: str | None,
    ) -> bytes:
        body: dict[str, Any] = {"prompt": prompt}
        if model is not None:
            body["model"] = model
        if duration is not None:
            body["duration"] = duration
        if seed is not None:
            body["seed"] = seed
        if response_format != "wav":
            body["response_format"] = response_format
        if steps is not None:
            body["steps"] = steps
        if guidance is not None:
            body["guidance"] = guidance
        if negative_prompt is not None:
            body["negative_prompt"] = negative_prompt
        r = requests.post(
            f"{self.server_url}{self._route}",
            json=body,
            timeout=self._timeout,
        )
        r.raise_for_status()
        return r.content


class MusicClient(_AudioGenerationClient):
    """Client for POST /v1/audio/music.

    Servers gate this route on the loaded model's
    `capabilities.supports_music` flag.
    """
    _route = "/v1/audio/music"

    def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        duration: float | None = None,
        seed: int | None = None,
        response_format: str = "wav",
        steps: int | None = None,
        guidance: float | None = None,
        negative_prompt: str | None = None,
    ) -> bytes:
        return self._post(
            prompt,
            model=model, duration=duration, seed=seed,
            response_format=response_format,
            steps=steps, guidance=guidance, negative_prompt=negative_prompt,
        )


class SFXClient(_AudioGenerationClient):
    """Client for POST /v1/audio/sfx.

    Servers gate this route on the loaded model's
    `capabilities.supports_sfx` flag.
    """
    _route = "/v1/audio/sfx"

    def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        duration: float | None = None,
        seed: int | None = None,
        response_format: str = "wav",
        steps: int | None = None,
        guidance: float | None = None,
        negative_prompt: str | None = None,
    ) -> bytes:
        return self._post(
            prompt,
            model=model, duration=duration, seed=seed,
            response_format=response_format,
            steps=steps, guidance=guidance, negative_prompt=negative_prompt,
        )
