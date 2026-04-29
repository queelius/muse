"""HTTP client for /v1/video/generations.

By default returns the encoded video bytes (mp4/webm) for the
configured response_format. For response_format='frames_b64', returns
list[bytes] (one PNG per frame).
"""
from __future__ import annotations

import base64
import os
from typing import Any

import requests


class VideoGenerationClient:
    """Client for muse's video/generation modality.

    Defaults to MUSE_SERVER env var (else http://localhost:8000).
    Default timeout is generous (600s) because video models can take
    minutes to render even short clips.
    """

    def __init__(
        self, server_url: str | None = None, timeout: float = 600.0,
    ) -> None:
        server_url = server_url or os.environ.get(
            "MUSE_SERVER", "http://localhost:8000",
        )
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        duration_seconds: float | None = None,
        fps: int | None = None,
        size: str | None = None,
        seed: int | None = None,
        negative_prompt: str | None = None,
        steps: int | None = None,
        guidance: float | None = None,
        response_format: str = "mp4",
        n: int = 1,
    ) -> Any:
        """Generate video(s).

        Returns:
          - For n=1 with mp4/webm: bytes (the encoded video).
          - For n>1 with mp4/webm: list[bytes] (one per video).
          - For frames_b64 (any n): list[bytes] (one PNG per frame;
            frames from multiple videos are appended in order).
        """
        body: dict = {
            "prompt": prompt, "n": n, "response_format": response_format,
        }
        for k, v in [
            ("model", model),
            ("duration_seconds", duration_seconds),
            ("fps", fps),
            ("size", size),
            ("seed", seed),
            ("negative_prompt", negative_prompt),
            ("steps", steps),
            ("guidance", guidance),
        ]:
            if v is not None:
                body[k] = v

        r = requests.post(
            f"{self.server_url}/v1/video/generations",
            json=body, timeout=self.timeout,
        )
        if r.status_code != 200:
            raise RuntimeError(
                f"server returned {r.status_code}: {r.text[:500]}"
            )
        payload = r.json()
        data = payload["data"]
        if response_format == "frames_b64":
            return [base64.b64decode(e["b64_json"]) for e in data]
        if n == 1:
            return base64.b64decode(data[0]["b64_json"])
        return [base64.b64decode(e["b64_json"]) for e in data]
