"""Tests for the gateway's MODEL_OPTIONAL_PATHS default-model seam.

A modality package may export MODEL_OPTIONAL_PATHS (text_translation
does: /v1/translate, /translate, /languages). When a request to one of
those paths omits `model`, the gateway resolves the first ENABLED
catalog model of the matching modality instead of 400ing model_required.
No enabled model of that modality -> 503 no_default_model. Any other
path with no model keeps today's 400 model_required, byte-identical.

Uses the legacy static-routes gateway (build_gateway(routes=...)): the
model_id-is-None check runs BEFORE the director-vs-legacy branch, so
this path is testable without spinning up a LoadDirector.
"""
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from muse.cli_impl.gateway import WorkerRoute, build_gateway
from muse.core.catalog import CatalogEntry


def _entry(model_id: str, modality: str) -> CatalogEntry:
    return CatalogEntry(
        model_id=model_id,
        modality=modality,
        backend_path="fake.module:FakeModel",
        hf_repo="fake/repo",
    )


class TestDefaultModelSeam:
    def test_translate_with_no_model_resolves_first_enabled_match(self):
        routes = [WorkerRoute(model_id="m2m100-418m", worker_url="http://127.0.0.1:9010")]
        app = build_gateway(routes)
        client = TestClient(app)

        known = {"m2m100-418m": _entry("m2m100-418m", "text/translation")}
        fake_response = AsyncMock(
            return_value=JSONResponse({"translatedText": "hola"}),
        )
        with patch("muse.cli_impl.gateway.known_models", return_value=known), \
             patch("muse.cli_impl.gateway.is_enabled", return_value=True), \
             patch("muse.cli_impl.gateway._forward", new=fake_response) as mock_forward:
            r = client.post(
                "/translate", json={"q": "hello", "source": "en", "target": "es"},
            )

        assert r.status_code == 200
        assert mock_forward.await_count == 1
        target_url = mock_forward.await_args[0][1]
        assert target_url == "http://127.0.0.1:9010/translate"

    def test_translate_with_no_model_and_no_enabled_model_returns_503(self):
        routes: list[WorkerRoute] = []
        app = build_gateway(routes)
        client = TestClient(app)

        with patch("muse.cli_impl.gateway.known_models", return_value={}), \
             patch("muse.cli_impl.gateway.is_enabled", return_value=False):
            r = client.post(
                "/translate", json={"q": "hello", "source": "en", "target": "es"},
            )

        assert r.status_code == 503
        assert r.json()["error"]["code"] == "no_default_model"

    def test_translate_with_no_model_skips_disabled_model(self):
        routes: list[WorkerRoute] = []
        app = build_gateway(routes)
        client = TestClient(app)

        known = {"m2m100-418m": _entry("m2m100-418m", "text/translation")}
        with patch("muse.cli_impl.gateway.known_models", return_value=known), \
             patch("muse.cli_impl.gateway.is_enabled", return_value=False):
            r = client.post(
                "/translate", json={"q": "hello", "source": "en", "target": "es"},
            )

        assert r.status_code == 503
        assert r.json()["error"]["code"] == "no_default_model"

    def test_chat_completions_without_model_still_400s_model_required(self):
        routes: list[WorkerRoute] = []
        app = build_gateway(routes)
        client = TestClient(app)

        r = client.post("/v1/chat/completions", json={"messages": []})

        assert r.status_code == 400
        assert r.json()["error"]["code"] == "model_required"

    def test_languages_with_no_model_also_resolves_default(self):
        """GET /languages is also in MODEL_OPTIONAL_PATHS."""
        routes = [WorkerRoute(model_id="m2m100-418m", worker_url="http://127.0.0.1:9011")]
        app = build_gateway(routes)
        client = TestClient(app)

        known = {"m2m100-418m": _entry("m2m100-418m", "text/translation")}
        fake_response = AsyncMock(return_value=JSONResponse([]))
        with patch("muse.cli_impl.gateway.known_models", return_value=known), \
             patch("muse.cli_impl.gateway.is_enabled", return_value=True), \
             patch("muse.cli_impl.gateway._forward", new=fake_response) as mock_forward:
            r = client.get("/languages")

        assert r.status_code == 200
        target_url = mock_forward.await_args[0][1]
        assert target_url == "http://127.0.0.1:9011/languages"


class TestDefaultModelSeamMultipart:
    """The seam reads `model` from multipart forms too (pixelize, autotile):
    curl -F image=@x.png with no -F model resolves the modality's default and
    the multipart body reaches the worker byte for byte."""

    @pytest.fixture(autouse=True)
    def _real_paths(self):
        from muse.core.discovery import _reset_model_optional_paths_cache

        _reset_model_optional_paths_cache()
        yield
        _reset_model_optional_paths_cache()

    @pytest.mark.parametrize("path", ["/v1/images/pixelize", "/v1/images/autotile"])
    def test_multipart_without_model_resolves_default(self, path):
        routes = [WorkerRoute(model_id="pixel-tools", worker_url="http://127.0.0.1:9020")]
        app = build_gateway(routes)
        client = TestClient(app)
        known = {"pixel-tools": _entry("pixel-tools", "image/pixelart")}
        bodies = []

        async def fake_forward(request, target_url, timeout):
            bodies.append(await request.body())
            return JSONResponse({"ok": True})

        with patch("muse.cli_impl.gateway.known_models", return_value=known), \
             patch("muse.cli_impl.gateway.is_enabled", return_value=True), \
             patch("muse.cli_impl.gateway._forward", new=AsyncMock(
                 side_effect=fake_forward)) as mock_forward:
            r = client.post(path, files={"image": ("x.png", b"\x89PNG fake", "image/png")},
                            data={"palette": "db32"})

        assert r.status_code == 200, r.text
        assert mock_forward.await_args[0][1] == f"http://127.0.0.1:9020{path}"
        assert b"\x89PNG fake" in bodies[0] and b'name="palette"' in bodies[0]

    def test_multipart_without_model_and_no_enabled_model_returns_503(self):
        app = build_gateway([])
        client = TestClient(app)
        with patch("muse.cli_impl.gateway.known_models", return_value={}), \
             patch("muse.cli_impl.gateway.is_enabled", return_value=False):
            r = client.post("/v1/images/pixelize",
                            files={"image": ("x.png", b"data", "image/png")})
        assert r.status_code == 503
        assert r.json()["error"]["code"] == "no_default_model"
