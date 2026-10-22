"""Every mounted modality route is declared in DISCOVERY, and every
declaration matches the real route (methods, paths, request fields)."""
from __future__ import annotations

import json

import pytest

from muse.core.discovery import discover_modalities, modality_dirs, modality_discovery
from muse.core.registry import ModalityRegistry

ROUTERS = discover_modalities(modality_dirs())
DECL = modality_discovery()


def _api_routes(tag):
    from fastapi.routing import APIRoute

    router = ROUTERS[tag](ModalityRegistry())
    return [r for r in router.routes if isinstance(r, APIRoute)]


def test_every_modality_declares_discovery():
    assert set(ROUTERS) == set(DECL)


@pytest.mark.parametrize("tag", sorted(ROUTERS))
def test_every_mounted_route_is_declared(tag):
    mounted = {(m, r.path) for r in _api_routes(tag) for m in r.methods}
    declared = {(r["method"], r["path"]) for r in DECL[tag]["routes"]}
    assert mounted == declared


def test_mcp_tool_names_unique_across_modalities():
    names = [r["mcp_tool"] for d in DECL.values() for r in d["routes"]]
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
    else:  # pragma: no cover
        pytest.fail(f"no validator for JSON route {decl['path']}")


@pytest.mark.parametrize("tag", sorted(ROUTERS))
def test_examples_match_route_request_models(tag):
    routes = {(m, r.path): r for r in _api_routes(tag) for m in r.methods}
    for decl in DECL[tag]["routes"]:
        route = routes[(decl["method"], decl["path"])]
        example = _example_for(decl)
        ctype = decl["request_content_type"]
        if ctype == "application/json":
            _validate_json_body(decl, route, example)
        elif ctype == "multipart/form-data":
            names = {p.alias for p in route.dependant.body_params}
            assert set(example) | set(decl["file_fields"]) <= names
            required = {p.alias for p in route.dependant.body_params if p.field_info.is_required()}
            assert required <= set(example) | set(decl["file_fields"])
        else:
            assert ctype is None and decl["method"] == "GET"
            names = {p.alias for p in route.dependant.query_params}
            assert set(example) <= names
