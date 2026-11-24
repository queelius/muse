"""HTTP client for the ``music/midi`` modality (``POST /v1/music/midi``)."""
from __future__ import annotations

from typing import Any

import requests

from muse.modalities._common.client import resolve_server_url


class MidiClient:
    """Client for ``POST /v1/music/midi``.

    Returns a parsed JSON dict when ``response_format="json"`` (default)
    or raw ``audio/midi`` SMF Type 1 bytes when ``response_format="midi"``.
    """

    _route = "/v1/music/midi"

    def __init__(
        self,
        server_url: str | None = None,
        *,
        timeout: float = 120.0,
    ) -> None:
        self.server_url = resolve_server_url(server_url)
        self._timeout = timeout

    def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        key: str | None = None,
        mode: str | None = None,
        bpm: int | None = None,
        time_signature: str = "4/4",
        bars: int = 16,
        instruments: list[str] | None = None,
        layers: list[str] | None = None,
        seed: int | None = None,
        temperature: float = 0.9,
        response_format: str = "json",
    ) -> dict[str, Any] | bytes:
        body: dict[str, Any] = {
            "prompt": prompt,
            "time_signature": time_signature,
            "bars": bars,
            "temperature": temperature,
            "response_format": response_format,
        }
        if model is not None:
            body["model"] = model
        if key is not None:
            body["key"] = key
        if mode is not None:
            body["mode"] = mode
        if bpm is not None:
            body["bpm"] = bpm
        if instruments is not None:
            body["instruments"] = list(instruments)
        if layers is not None:
            body["layers"] = list(layers)
        if seed is not None:
            body["seed"] = seed

        resp = requests.post(
            f"{self.server_url}{self._route}",
            json=body,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        if response_format == "midi":
            return resp.content
        return resp.json()

    def generate_bytes(
        self,
        prompt: str,
        *,
        model: str | None = None,
        key: str | None = None,
        mode: str | None = None,
        bpm: int | None = None,
        time_signature: str = "4/4",
        bars: int = 16,
        instruments: list[str] | None = None,
        layers: list[str] | None = None,
        seed: int | None = None,
        temperature: float = 0.9,
    ) -> bytes:
        """Convenience wrapper requesting raw ``audio/midi`` SMF bytes."""
        out = self.generate(
            prompt,
            model=model,
            key=key,
            mode=mode,
            bpm=bpm,
            time_signature=time_signature,
            bars=bars,
            instruments=instruments,
            layers=layers,
            seed=seed,
            temperature=temperature,
            response_format="midi",
        )
        assert isinstance(out, bytes)
        return out
