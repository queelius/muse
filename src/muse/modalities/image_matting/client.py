"""HTTP client for ``POST /v1/images/matting``."""
from __future__ import annotations

from typing import Any

import requests

from muse.modalities._common.client import resolve_server_url


def _form(fields: dict[str, Any]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for k, v in fields.items():
        if v is None:
            continue
        if isinstance(v, bool):
            v = "true" if v else "false"
        out.append((k, str(v)))
    return out


class MattingClient:
    """Thin HTTP client against the muse ``image/matting`` endpoint."""

    def __init__(self, base_url: str | None = None, timeout: float = 300.0) -> None:
        self.base_url = resolve_server_url(base_url)
        self.timeout = timeout

    def matte(
        self,
        image: bytes,
        *,
        model: str | None = None,
        response_format: str = "json",
        **fields: Any,
    ) -> dict[str, Any] | bytes:
        """Remove background / extract RGBA foreground from image bytes.

        Returns the JSON envelope by default, or raw PNG bytes when
        ``response_format="png"``. Extra keyword arguments map to the route's
        form fields (``background``, ``bg_threshold``, ``color_tolerance``,
        ``hard_alpha``, ``alpha_threshold``, ``defringe``, ``fringe_threshold``,
        ``clear_enclosed_white``, ``crop``).
        """
        r = requests.post(
            f"{self.base_url}/v1/images/matting",
            files={"image": ("image.png", image, "image/png")},
            data=_form(
                {"model": model, "response_format": response_format, **fields}
            ),
            timeout=self.timeout,
        )
        if r.status_code != 200:
            raise RuntimeError(f"server returned {r.status_code}: {r.text[:500]}")
        return r.content if response_format == "png" else r.json()
