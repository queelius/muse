"""HTTP client for text/translation (/v1/translate, /languages).

Mirrors ChatClient's structure (httpx, base_url public attribute,
MUSE_SERVER env fallback via muse.core.config) rather than requests,
since translate() unwraps the LibreTranslate envelope to return a bare
str/list[str] instead of passing the envelope through unchanged.
"""
from __future__ import annotations

from typing import Any

import httpx

from muse.core import config


class TranslateClient:
    """Minimal HTTP client for the text/translation modality."""

    def __init__(self, base_url: str | None = None, timeout: float = 120.0) -> None:
        self.base_url = (base_url or config.get("client.server_url")).rstrip("/")
        self.timeout = timeout

    def translate(
        self,
        q: str | list[str],
        source: str,
        target: str,
        model: str | None = None,
    ) -> str | list[str]:
        """Translate `q` from `source` to `target`.

        Returns a str when `q` was a str, a list[str] when `q` was a
        list (mirroring the LibreTranslate-shape response's own
        scalar/list symmetry).
        """
        body: dict[str, Any] = {"q": q, "source": source, "target": target}
        if model is not None:
            body["model"] = model

        r = httpx.post(
            f"{self.base_url}/v1/translate",
            json=body, timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()["translatedText"]

    def languages(self) -> list[dict]:
        """Fetch the LibreTranslate-shape /languages list."""
        r = httpx.get(f"{self.base_url}/languages", timeout=self.timeout)
        r.raise_for_status()
        return r.json()
