"""HTTP client for /v1/images/animations.

By default returns the encoded animation bytes (webp/gif/mp4) for the
configured response_format. For response_format='frames_b64', returns
list[bytes] (one PNG per frame).
"""
from __future__ import annotations

import base64
import os
from typing import Any

import requests


class AnimationsClient:
    def __init__(self, server_url: str | None = None, timeout: float = 300.0) -> None:
        server_url = server_url or os.environ.get(
            "MUSE_SERVER", "http://localhost:8000",
        )
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout

    def animate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        n: int = 1,
        frames: int | None = None,
        fps: int | None = None,
        loop: bool | None = None,
        negative_prompt: str | None = None,
        steps: int | None = None,
        guidance: float | None = None,
        seed: int | None = None,
        image: str | None = None,
        strength: float | None = None,
        response_format: str = "webp",
        size: str | None = None,
    ) -> Any:
        body: dict = {"prompt": prompt, "n": n, "response_format": response_format}
        for k, v in [
            ("model", model), ("frames", frames), ("fps", fps), ("loop", loop),
            ("negative_prompt", negative_prompt), ("steps", steps),
            ("guidance", guidance), ("seed", seed), ("image", image),
            ("strength", strength), ("size", size),
        ]:
            if v is not None:
                body[k] = v

        r = requests.post(
            f"{self.server_url}/v1/images/animations",
            json=body, timeout=self.timeout,
        )
        if r.status_code != 200:
            raise RuntimeError(f"server returned {r.status_code}: {r.text[:500]}")
        payload = r.json()
        data = payload["data"]
        if response_format == "frames_b64":
            return [base64.b64decode(e["b64_json"]) for e in data]
        return base64.b64decode(data[0]["b64_json"])
