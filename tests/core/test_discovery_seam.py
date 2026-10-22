"""The DISCOVERY seam: AST-read, validated per-modality agent descriptions."""
from __future__ import annotations

import logging

import pytest

from muse.core.discovery import modality_discovery
from muse.core.registry import ModalityRegistry

ROUTE = (
    '{"method": "POST", "path": "/v1/x", "request_content_type": "application/json",'
    ' "response_content_type": "application/json", "compat": None, "mcp_tool": "muse_x",'
    ' "file_fields": [], "media_fields": [], "example": {"model": "<model>"}}'
)


def _pkg(tmp_path, body):
    pkg = tmp_path / "audio_x"
    pkg.mkdir()
    (pkg / "__init__.py").write_text('MODALITY = "audio/x"\n' + body)
    return tmp_path


def test_modality_discovery_reads_literal(tmp_path):
    _pkg(tmp_path, f'DISCOVERY = {{"summary": "s", "use_for": ["a"], "not_for": [], "routes": [{ROUTE}]}}\n')
    assert modality_discovery([tmp_path])["audio/x"]["routes"][0]["mcp_tool"] == "muse_x"


def test_get_route_may_have_no_request_content_type(tmp_path):
    route = ROUTE.replace('"method": "POST"', '"method": "GET"').replace(
        '"request_content_type": "application/json"', '"request_content_type": None')
    _pkg(tmp_path, f'DISCOVERY = {{"summary": "s", "use_for": [], "not_for": [], "routes": [{route}]}}\n')
    assert "audio/x" in modality_discovery([tmp_path])


@pytest.mark.parametrize("body", [
    'DISCOVERY = {"summary": "s", "use_for": [], "not_for": []}\n',
    f'DISCOVERY = {{"summary": 1, "use_for": [], "not_for": [], "routes": [{ROUTE}]}}\n',
    f'DISCOVERY = {{"summary": "s", "use_for": [], "not_for": [], "routes": [{ROUTE[:-1]}, "extra": 1}}]}}\n',
    f'DISCOVERY = {{"summary": "s", "use_for": [], "not_for": [], "routes": [{ROUTE.replace("POST", "PUT")}]}}\n',
    'DISCOVERY = {"summary": "s", "use_for": [], "not_for": [], "routes": []}\n',
    'DISCOVERY = "nope"\n',
])
def test_malformed_discovery_is_skipped(tmp_path, caplog, body):
    _pkg(tmp_path, body)
    with caplog.at_level(logging.WARNING):
        assert "audio/x" not in modality_discovery([tmp_path])
    assert "DISCOVERY" in caplog.text
    assert "audio/x" in caplog.text


def test_pipeline_modalities_declare_discovery():
    d = modality_discovery()
    for tag, tool in [("audio/speech", "muse_speak"), ("audio/alignment", "muse_align"),
                      ("audio/transcription", "muse_transcribe"),
                      ("audio/transcription", "muse_translate_audio"),
                      ("audio/quality", "muse_assess_audio_quality")]:
        assert tool in {r["mcp_tool"] for r in d[tag]["routes"]}
