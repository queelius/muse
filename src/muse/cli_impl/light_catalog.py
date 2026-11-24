"""The gateway's light catalog (muse.core.light).

Light routes mounted on the gateway resolve their model here, from
catalog.json and the model's manifest, without the director, a worker or a
cold load. The model errors match the proxy's for the same request:
400 ``model_required`` / ``invalid_model``, 503 ``catalog_unavailable``,
404 ``model_not_found``, 409 ``model_disabled``. A request a light route
cannot answer from disk (a preset-voice listing) is forwarded through the
director like any proxied request.
"""
from __future__ import annotations

from typing import Any

from fastapi import Request, Response

from muse.core import catalog as _catalog
from muse.core.catalog import CatalogError
from muse.core.light import LightModel, LightRequestError


class GatewayLightCatalog:
    def __init__(self, state: Any, timeout: float, cancellable_paths: frozenset[str]):
        self._state = state
        self._timeout = timeout
        self._cancellable = cancellable_paths

    def resolve(self, modality: str, model_id: str | None) -> LightModel:
        from muse.cli_impl import gateway as gw

        if model_id is None:
            raise LightRequestError(
                400, "model_required",
                "request is missing a `model` field (required for gateway routing)")
        if not gw._valid_model_id(model_id):
            raise LightRequestError(
                400, "invalid_model",
                "`model` must be a non-empty string of at most "
                f"{gw._MAX_MODEL_ID_CHARS} characters without control characters")
        try:
            entry = _catalog._read_catalog().get(model_id)
        except CatalogError as exc:
            raise LightRequestError(
                503, "catalog_unavailable",
                f"model catalog is temporarily unavailable: {exc}") from exc
        if not isinstance(entry, dict):
            raise LightRequestError(
                404, "model_not_found", f"model {model_id!r} is not in the catalog")
        if not bool(entry.get("enabled", True)):
            raise LightRequestError(
                409, "model_disabled",
                f"model {model_id!r} is disabled; enable it before requesting it")
        try:
            manifest = _catalog.get_manifest(model_id)
        except (KeyError, CatalogError):
            raise LightRequestError(
                404, "model_not_found", f"model {model_id!r} is not in the catalog") from None
        if not isinstance(manifest, dict) or manifest.get("modality") != modality:
            raise LightRequestError(
                404, "model_not_found",
                f"Model {model_id!r} is not available for modality {modality!r}")
        caps = manifest.get("capabilities")
        return LightModel(model_id=model_id, capabilities=caps if isinstance(caps, dict) else {})

    async def forward(self, request: Request, model_id: str) -> Response:
        from muse.cli_impl import gateway as gw

        return await gw._route_via_director(
            request, request.url.path.lstrip("/"), model_id, self._state,
            self._timeout, cancellable_paths=self._cancellable,
        )
