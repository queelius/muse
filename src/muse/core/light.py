"""Light routes: model-free routes served without loading a model.

A modality package may export ``build_light_router(catalog) -> APIRouter``
beside ``build_router``. Its routes answer from disk or manifests (voice
listings, reference clips, metadata edits), so they never need the model in
memory. Both processes mount it:

- a worker over ``RegistryLightCatalog`` (its own registry; the loaded
  backend rides along on ``LightModel.backend``);
- the gateway, ahead of its catch-all proxy, over a catalog that resolves
  models from catalog.json and can ``forward`` a request the light route
  cannot answer itself (``muse.cli_impl.light_catalog``).

A light route declares ``"light": true`` in its DISCOVERY entry.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

from fastapi import Request, Response
from fastapi.responses import JSONResponse

from muse.core.errors import ModelNotFoundError, error_response


@dataclass(frozen=True)
class LightModel:
    """The model a light request names."""

    model_id: str
    capabilities: Mapping[str, Any] = field(default_factory=dict)
    # The in-process model in a worker; None in the gateway.
    backend: Any = None


class LightRequestError(Exception):
    """A light request that cannot proceed (an OpenAI-envelope error)."""

    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


class LightCatalog(Protocol):
    def resolve(self, modality: str, model_id: str | None) -> LightModel:
        """The model ``model_id`` names (blocking); raises LightRequestError."""
        ...

    async def forward(self, request: Request, model_id: str) -> Response:
        """Serve ``request`` the normal way (through the model's worker)."""
        ...


class RegistryLightCatalog:
    """A worker's light catalog: models come from its own registry."""

    def __init__(self, registry: Any):
        self._registry = registry

    def resolve(self, modality: str, model_id: str | None) -> LightModel:
        try:
            backend = self._registry.get_or_404(modality, model_id)
        except ModelNotFoundError:
            shown = model_id if model_id is not None else "(default)"
            raise LightRequestError(
                404, "model_not_found",
                f"Model {shown!r} is not available for modality {modality!r}",
            ) from None
        mid = getattr(backend, "model_id", model_id)
        return LightModel(
            model_id=mid,
            capabilities=self._registry.capabilities(modality, mid),
            backend=backend,
        )

    async def forward(self, request: Request, model_id: str) -> Response:
        raise RuntimeError("a worker serves light routes in process; nothing to forward to")


async def resolve_light(
    catalog: LightCatalog, modality: str, model_id: str | None,
) -> LightModel | JSONResponse:
    """``catalog.resolve`` off the event loop; an error becomes its envelope."""
    try:
        return await asyncio.to_thread(catalog.resolve, modality, model_id)
    except LightRequestError as exc:
        return error_response(exc.status, exc.code, exc.message)
