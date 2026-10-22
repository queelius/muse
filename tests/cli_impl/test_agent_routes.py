"""Discovery entry points: /, /.well-known/muse.json, /llms.txt, full OpenAPI."""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from muse.cli_impl import agent_routes
from muse.core.discovery import modality_discovery

from tests.cli_impl.test_gateway_capabilities import (  # noqa: F401
    _catalog, director_spy, gateway_client, manifest_calls, revalidate_calls, state,
)


def test_openapi_lists_every_declared_route(gateway_client):
    paths = gateway_client.get("/openapi.json").json()["paths"]
    for d in modality_discovery().values():
        for r in d["routes"]:
            assert r["method"].lower() in paths[r["path"]]
    assert "/{full_path}" not in paths


def test_speech_request_schema_present(gateway_client):
    op = gateway_client.get("/openapi.json").json()["paths"]["/v1/audio/speech"]["post"]
    assert "requestBody" in op
    assert op["description"] == modality_discovery()["audio/speech"]["summary"]


def test_openapi_is_cached(gateway_client):
    gateway_client.get("/openapi.json")
    app = gateway_client.app
    first = app.openapi_schema
    assert first is not None and app.openapi() is first


def test_root_and_well_known_point_at_capabilities(gateway_client):
    for path in ("/", "/.well-known/muse.json"):
        body = gateway_client.get(path).json()
        assert body == agent_routes.POINTER
        assert body["capabilities"] == "/v1/capabilities"


def test_llms_txt_is_markdown(gateway_client):
    r = gateway_client.get("/llms.txt")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/markdown")
    assert r.text.startswith("#")


def test_broken_router_is_skipped(monkeypatch, gateway_client):
    real = agent_routes.discover_modalities

    def broken(dirs):
        out = dict(real(dirs))

        def boom(_registry):
            raise RuntimeError("nope")

        out["audio/speech"] = boom
        return out

    monkeypatch.setattr(agent_routes, "discover_modalities", broken)
    gateway_client.app.openapi_schema = None
    paths = gateway_client.get("/openapi.json").json()["paths"]
    assert "/v1/audio/speech" not in paths
    assert "/v1/embeddings" in paths


def test_component_name_collision_is_renamed(monkeypatch):
    from pydantic import create_model

    def make(field_type):
        Req = create_model("Req", x=(field_type, ...))

        def build(_registry):
            r = APIRouter()

            def f(body: Req):
                return {}

            f.__annotations__["body"] = Req
            r.add_api_route("/v1/zz/" + field_type.__name__, f, methods=["POST"])
            return r
        return build

    monkeypatch.setattr(
        agent_routes, "discover_modalities",
        lambda dirs: {"zz/one": make(int), "zz/two": make(str)},
    )
    from fastapi import FastAPI

    merged = agent_routes.merge_openapi(FastAPI().openapi())
    schemas = merged["components"]["schemas"]
    assert "Req" in schemas and "ZzTwoReq" in schemas
    assert schemas["Req"] != schemas["ZzTwoReq"]
    ref = merged["paths"]["/v1/zz/str"]["post"]["requestBody"]["content"][
        "application/json"]["schema"]["$ref"]
    assert ref.endswith("/ZzTwoReq")
