"""HTTP client for ``POST /v1/audio/process``."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import requests

from muse.modalities._common.client import resolve_server_url, upload_bytes


def _form_fields(fields: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in fields.items():
        if value is None:
            continue
        if isinstance(value, bool):
            out[key] = "true" if value else "false"
        else:
            out[key] = str(value)
    return out


class AudioProcessClient:
    """Thin multipart client for ``POST /v1/audio/process``."""

    def __init__(
        self,
        server_url: str | None = None,
        *,
        timeout: float = 300.0,
    ) -> None:
        self.server_url = resolve_server_url(server_url)
        self._timeout = timeout

    def process(
        self,
        audio: bytes | str | Path | Any,
        *,
        model: str | None = None,
        trim_silence: bool | None = None,
        trim_threshold_db: float | None = None,
        pad_lead_ms: float | None = None,
        pad_tail_ms: float | None = None,
        fade_in_ms: float | None = None,
        fade_out_ms: float | None = None,
        loop_fade_s: float | None = None,
        target_peak_dbfs: float | None = None,
        mono: bool | None = None,
        sample_rate: int | None = None,
        output_format: str = "wav",
        response_format: str = "json",
        filename: str = "audio.wav",
        content_type: str = "audio/wav",
    ) -> dict[str, Any] | bytes:
        """Process audio bytes, a file path, or a readable binary stream.

        Returns the parsed JSON dict (with ``b64_audio`` and ``metrics``) when
        ``response_format="json"``, or raw encoded audio bytes when
        ``response_format="audio"``.
        """
        files = {"file": (filename, upload_bytes(audio, kind="audio"), content_type)}
        data = _form_fields({
            "model": model,
            "trim_silence": trim_silence,
            "trim_threshold_db": trim_threshold_db,
            "pad_lead_ms": pad_lead_ms,
            "pad_tail_ms": pad_tail_ms,
            "fade_in_ms": fade_in_ms,
            "fade_out_ms": fade_out_ms,
            "loop_fade_s": loop_fade_s,
            "target_peak_dbfs": target_peak_dbfs,
            "mono": mono,
            "sample_rate": sample_rate,
            "output_format": output_format if output_format != "wav" else None,
            "response_format": response_format if response_format != "json" else None,
        })
        response = requests.post(
            f"{self.server_url}/v1/audio/process",
            files=files,
            data=data,
            timeout=self._timeout,
        )
        response.raise_for_status()
        if response_format == "audio":
            return response.content
        return response.json()
