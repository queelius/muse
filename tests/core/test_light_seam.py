"""Light routes: model-free routes a modality package exports beside its
router, mounted in workers (over the registry) and the gateway (over the
catalog), discovered by `light_hooks`."""
import textwrap
from pathlib import Path

import pytest
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient

from muse.core.discovery import _valid_discovery_route, light_hooks
from muse.core.errors import validation_error_response
from muse.core.light import LightModel, LightRequestError, RegistryLightCatalog
from muse.core.registry import ModalityRegistry
from muse.core.server import create_app

TOY = '''
from fastapi import APIRouter, Request

MODALITY = "toy/x"

DISCOVERY = {
    "summary": "toy", "use_for": [], "not_for": [],
    "routes": [{
        "method": "GET", "path": "/v1/toy/ping", "request_content_type": None,
        "response_content_type": "application/json", "compat": None,
        "mcp_tool": None, "file_fields": [], "media_fields": [],
        "example": {"model": "<model>"}, "light": True,
    }],
}


def build_router(registry):
    return APIRouter()


def build_light_router(catalog):
    from muse.core.light import LightModel, resolve_light

    router = APIRouter()

    @router.get("/v1/toy/ping")
    async def ping(model: str | None = None):
        m = await resolve_light(catalog, "toy/x", model)
        if not isinstance(m, LightModel):
            return m
        return {"model": m.model_id, "caps": dict(m.capabilities),
                "has_backend": m.backend is not None}

    return router


def model_options(model_id, capabilities):
    return {"things": {"count": 1}}
'''

PLAIN = '''
MODALITY = "plain/x"
def build_router(registry):
    from fastapi import APIRouter
    return APIRouter()
'''


def _pkg(root: Path, name: str, content: str) -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "__init__.py").write_text(textwrap.dedent(content).lstrip())
    return root


class _Fake:
    def __init__(self, model_id):
        self.model_id = model_id


def _registry():
    reg = ModalityRegistry()
    reg.register("toy/x", _Fake("t1"), manifest={
        "model_id": "t1", "modality": "toy/x", "capabilities": {"flavour": "plain"}})
    return reg


def test_light_hooks_found_only_for_packages_that_export_them(tmp_path):
    hooks = light_hooks([_pkg(_pkg(tmp_path, "toy", TOY), "plain", PLAIN)])
    assert set(hooks) == {"toy/x"}
    assert callable(hooks["toy/x"].build_light_router)
    assert hooks["toy/x"].model_options("t1", {}) == {"things": {"count": 1}}


def test_worker_style_mount_serves_light_route(tmp_path):
    reg = _registry()
    hooks = light_hooks([_pkg(tmp_path, "toy", TOY)])["toy/x"]
    app = create_app(registry=reg, routers={
        "toy/x:light": hooks.build_light_router(RegistryLightCatalog(reg))})
    c = TestClient(app)
    assert c.get("/v1/toy/ping").json() == {
        "model": "t1", "caps": {"flavour": "plain"}, "has_backend": True}
    err = c.get("/v1/toy/ping?model=nope")
    assert err.status_code == 404
    assert err.json()["error"]["code"] == "model_not_found"


def test_registry_catalog_resolve_and_forward():
    cat = RegistryLightCatalog(_registry())
    m = cat.resolve("toy/x", None)
    assert isinstance(m, LightModel) and m.model_id == "t1"
    with pytest.raises(LightRequestError) as e:
        cat.resolve("toy/x", "nope")
    assert (e.value.status, e.value.code) == (404, "model_not_found")


def _route(**over):
    base = {
        "method": "PATCH", "path": "/v1/x/{name}",
        "request_content_type": "application/json",
        "response_content_type": "application/json", "compat": None,
        "mcp_tool": "muse_x", "file_fields": [], "media_fields": [],
        "example": {"model": "<model>", "name": "a"}, "light": True,
    }
    return {**base, **over}


def test_validator_accepts_patch_json_and_light_flag():
    assert _valid_discovery_route(_route())
    assert _valid_discovery_route(_route(method="GET", request_content_type=None))


def test_validator_rejects_patch_without_json_and_non_bool_light():
    assert not _valid_discovery_route(_route(request_content_type=None))
    assert not _valid_discovery_route(_route(request_content_type="multipart/form-data"))
    assert not _valid_discovery_route(_route(light="yes"))


def test_validation_error_response_is_the_422_envelope():
    exc = RequestValidationError([{"loc": ("body", "colour"), "msg": "extra"}])
    r = validation_error_response(exc)
    assert r.status_code == 422
    assert b'"code":"invalid_request"' in r.body and b"body.colour: extra" in r.body
