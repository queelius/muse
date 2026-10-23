"""MuseClient: the thin HTTP side of ``muse mcp``.

Reads the server's discovery documents (``/v1/capabilities``,
``/openapi.json``), makes one generic ``call(route, ...)`` per generated
tool, and delegates admin operations to ``AdminClient``. Light by design:
httpx and ``muse.core.config`` only.
"""
from __future__ import annotations

from typing import Any

import httpx

from muse.admin.client import AdminClient
from muse.core import config

_DISCOVERY_TIMEOUT = 10.0


class MuseClient:
    """HTTP client of one muse server (a node or a federation coordinator).

    ``transport`` is an optional ``httpx`` transport (tests pass an
    ``httpx.MockTransport``); every request goes through it.
    """

    def __init__(
        self,
        server_url: str | None = None,
        admin_token: str | None = None,
        timeout: float = 600.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.server_url = (server_url or config.get("client.server_url")).rstrip("/")
        self.admin_token = admin_token or config.get("admin.token")
        self.timeout = timeout
        self.transport = transport
        self.admin = AdminClient(
            base_url=self.server_url,
            token=self.admin_token,
            timeout=30.0,
        )

    def _get_json(self, path: str, params: dict | None = None) -> Any:
        with httpx.Client(timeout=_DISCOVERY_TIMEOUT, transport=self.transport) as c:
            r = c.get(f"{self.server_url}{path}", params=params)
        r.raise_for_status()
        return r.json()

    # Probes and discovery

    def health(self) -> dict:
        return self._get_json("/health")

    def list_models(self) -> dict:
        return self._get_json("/v1/models")

    def capabilities(self, *, modality: str | None = None,
                     model: str | None = None) -> dict:
        params = {k: v for k, v in (("modality", modality), ("model", model)) if v}
        return self._get_json("/v1/capabilities", params or None)

    def openapi(self) -> dict:
        return self._get_json("/openapi.json")

    # Inference

    def call(
        self,
        route: dict,
        *,
        json: Any = None,
        data: dict | None = None,
        files: dict | list | None = None,
        params: dict | None = None,
    ) -> httpx.Response:
        """One request to a DISCOVERY route; raises on an error status.

        ``json`` sends a JSON body; ``data`` + ``files`` send multipart;
        ``params`` are query parameters (GET routes).
        """
        kwargs: dict[str, Any] = {"params": params}
        if json is not None:
            kwargs["json"] = json
        if files:
            kwargs["files"] = files
        if data:
            kwargs["data"] = data
        with httpx.Client(timeout=self.timeout, transport=self.transport) as c:
            r = c.request(route["method"].upper(),
                          f"{self.server_url}{route['path']}", **kwargs)
        r.raise_for_status()
        return r

    # Resolver search (admin-style discovery)

    def search_models(
        self,
        *,
        query: str,
        modality: str | None = None,
        max_size_gb: float | None = None,
        limit: int = 20,
    ) -> dict:
        """Run resolver.search() locally and return rows as plain dicts.

        Pinning resolver search to local execution (rather than going
        through muse) keeps this useful when the LLM is connected to a
        muse server that doesn't yet have the desired model. The HF
        resolver registers itself on import.
        """
        import muse.core.resolvers_hf  # noqa: F401  (registers HFResolver)
        from muse.core.resolvers import search as _resolver_search

        rows = list(_resolver_search(
            query, backend=None, modality=modality, limit=limit,
        ))
        if max_size_gb is not None:
            rows = [r for r in rows if r.size_gb is None or r.size_gb <= max_size_gb]
        return {
            "results": [
                {
                    "uri": r.uri,
                    "modality": r.modality,
                    "size_gb": r.size_gb,
                    "downloads": r.downloads,
                    "license": r.license,
                    "description": r.description,
                }
                for r in rows
            ],
        }
