"""HTTP clients for the audio/generation modality.

Two clients, one per route. The split mirrors the URL split: a user
calling MusicClient.generate is hitting /v1/audio/music; SFXClient is
hitting /v1/audio/sfx. The server enforces per-model capability gates.

Both clients return raw audio bytes (matching /v1/audio/speech): the
response Content-Type indicates the encoding (audio/wav by default).
"""
from __future__ import annotations

from typing import Any

import requests

from muse.modalities._common.client import resolve_server_url


class _AudioGenerationClient:
    """Shared base. Subclasses provide their own route on top."""

    _route: str = ""

    def __init__(
        self,
        server_url: str | None = None,
        *,
        timeout: float = 300.0,
    ) -> None:
        self.server_url = resolve_server_url(server_url)
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
        lyrics: str | None = None,
        trim_silence: bool = False,
        loop_fade_s: float = 0.0,
        target_peak_dbfs: float | None = None,
        mono: bool = False,
        sample_rate: int | None = None,
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
        if lyrics is not None:
            body["lyrics"] = lyrics
        if trim_silence:
            body["trim_silence"] = True
        if loop_fade_s:
            body["loop_fade_s"] = loop_fade_s
        if target_peak_dbfs is not None:
            body["target_peak_dbfs"] = target_peak_dbfs
        if mono:
            body["mono"] = True
        if sample_rate is not None:
            body["sample_rate"] = sample_rate
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
        lyrics: str | None = None,
        trim_silence: bool = False,
        loop_fade_s: float = 0.0,
        target_peak_dbfs: float | None = None,
        mono: bool = False,
        sample_rate: int | None = None,
    ) -> bytes:
        return self._post(
            prompt,
            model=model, duration=duration, seed=seed,
            response_format=response_format,
            steps=steps, guidance=guidance, negative_prompt=negative_prompt,
            lyrics=lyrics,
            trim_silence=trim_silence,
            loop_fade_s=loop_fade_s,
            target_peak_dbfs=target_peak_dbfs,
            mono=mono,
            sample_rate=sample_rate,
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
        lyrics: str | None = None,
        trim_silence: bool = False,
        loop_fade_s: float = 0.0,
        target_peak_dbfs: float | None = None,
        mono: bool = False,
        sample_rate: int | None = None,
    ) -> bytes:
        return self._post(
            prompt,
            model=model, duration=duration, seed=seed,
            response_format=response_format,
            steps=steps, guidance=guidance, negative_prompt=negative_prompt,
            lyrics=lyrics,
            trim_silence=trim_silence,
            loop_fade_s=loop_fade_s,
            target_peak_dbfs=target_peak_dbfs,
            mono=mono,
            sample_rate=sample_rate,
        )
