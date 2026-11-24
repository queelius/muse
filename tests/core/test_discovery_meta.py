"""Every mounted modality route is declared in DISCOVERY, and every
declaration matches the real route (methods, paths, request fields)."""
from __future__ import annotations

import json

import pytest

from muse.core.discovery import (
    discover_modalities,
    light_hooks,
    modality_dirs,
    modality_discovery,
)
from muse.core.light import RegistryLightCatalog
from muse.core.registry import ModalityRegistry

ROUTERS = discover_modalities(modality_dirs())
LIGHT = {t: h.build_light_router for t, h in light_hooks().items()
         if h.build_light_router is not None}
DECL = modality_discovery()


def _routes_of(router):
    """The router's documented routes (a hidden route, such as the voice
    router's not-found catch-all, is not part of the contract)."""
    from fastapi.routing import APIRoute

    return [r for r in router.routes if isinstance(r, APIRoute) and r.include_in_schema]


def _light_routes(tag):
    if tag not in LIGHT:
        return []
    return _routes_of(LIGHT[tag](RegistryLightCatalog(ModalityRegistry())))


def _api_routes(tag):
    """Every route a worker mounts for ``tag``: its router plus its light router."""
    return _routes_of(ROUTERS[tag](ModalityRegistry())) + _light_routes(tag)


def test_every_modality_declares_discovery():
    assert set(ROUTERS) == set(DECL)


@pytest.mark.parametrize("tag", sorted(ROUTERS))
def test_every_mounted_route_is_declared(tag):
    mounted = [(m, r.path) for r in _api_routes(tag) for m in r.methods]
    declared = {(r["method"], r["path"]) for r in DECL[tag]["routes"]}
    assert len(mounted) == len(set(mounted)), "router and light router overlap"
    assert set(mounted) == declared


@pytest.mark.parametrize("tag", sorted(ROUTERS))
def test_light_flag_matches_light_router(tag):
    light = {(m, r.path) for r in _light_routes(tag) for m in r.methods}
    flagged = {(r["method"], r["path"]) for r in DECL[tag]["routes"] if r.get("light")}
    assert flagged == light


def _path_params(path: str) -> set[str]:
    import re

    return set(re.findall(r"{([^}]+)}", path))


def test_mcp_tool_names_unique_across_modalities():
    names = [r["mcp_tool"] for d in DECL.values() for r in d["routes"]
             if r["mcp_tool"] is not None]
    assert len(names) == len(set(names))
    assert all(n.startswith("muse_") for n in names)


def test_existing_mcp_tool_names_preserved():
    names = {r["mcp_tool"] for d in DECL.values() for r in d["routes"]}
    assert {
        "muse_chat", "muse_summarize", "muse_rerank", "muse_classify",
        "muse_embed_text", "muse_translate", "muse_decide", "muse_embed_image",
        "muse_generate_image", "muse_edit_image", "muse_vary_image",
        "muse_upscale_image", "muse_segment_image", "muse_vectorize_image",
        "muse_generate_animation", "muse_generate_video", "muse_generate_music",
        "muse_generate_sfx", "muse_embed_audio", "muse_speak", "muse_transcribe",
        "muse_ocr", "muse_estimate_depth", "muse_detect_keypoints",
        "muse_detect_objects", "muse_classify_audio", "muse_generate_3d",
        "muse_generate_3d_from_image", "muse_moderate", "muse_list_languages",
    } <= names


def _example_for(route_decl):
    return {k: ("m" if v == "<model>" else v) for k, v in route_decl["example"].items()}


def _validate_json_body(decl, route, example):
    if route.body_field is not None:
        route.body_field.field_info.annotation.model_validate(example)
    elif decl["path"] in ("/v1/translate", "/translate"):
        from muse.modalities.text_translation.routes import _TranslateRequest

        _TranslateRequest(**example)
    elif decl["path"] == "/v1/systemone":
        from muse.modalities.decision_evaluation.validation import parse_request

        parse_request(json.dumps(example).encode())
    elif decl["path"] == "/v1/systemone/batch":
        from muse.modalities.decision_evaluation.validation import parse_batch_request

        parse_batch_request(json.dumps(example).encode())
    else:  # pragma: no cover
        pytest.fail(f"no validator for JSON route {decl['path']}")


@pytest.mark.parametrize("tag", sorted(ROUTERS))
def test_examples_match_route_request_models(tag):
    """Each example's keys split into path parameters, query parameters and
    the body; every path parameter has an example value, and the body part
    validates against the route's request model."""
    routes = {(m, r.path): r for r in _api_routes(tag) for m in r.methods}
    for decl in DECL[tag]["routes"]:
        route = routes[(decl["method"], decl["path"])]
        example = _example_for(decl)
        path_keys = _path_params(decl["path"])
        assert path_keys <= set(example), f"{decl['path']}: example lacks a path parameter"
        query_keys = {p.alias for p in route.dependant.query_params}
        body = {k: v for k, v in example.items()
                if k not in path_keys and k not in query_keys}
        ctype = decl["request_content_type"]
        if ctype == "application/json":
            _validate_json_body(decl, route, body)
        elif ctype == "multipart/form-data":
            names = {p.alias for p in route.dependant.body_params}
            assert set(body) | set(decl["file_fields"]) <= names
            required = {p.alias for p in route.dependant.body_params if p.field_info.is_required()}
            assert required <= set(body) | set(decl["file_fields"])
        else:
            assert ctype is None and decl["method"] in ("GET", "DELETE")
            assert body == {}, f"{decl['path']}: unknown example keys {sorted(body)}"


def test_text_media_fields_only_on_json_routes_and_svg_declared():
    for d in DECL.values():
        for r in d["routes"]:
            if "text_media_fields" in r:
                assert r["response_content_type"] == "application/json", r["path"]
                assert not set(r["text_media_fields"]) & set(r["media_fields"])
    [vec] = DECL["image/vectorization"]["routes"]
    assert vec["text_media_fields"] == {"svg": "image/svg+xml"}
