"""Tests for the gateway proxy FastAPI app."""
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from muse.cli_impl.gateway import (
    extract_model_from_request,
    build_gateway,
    WorkerRoute,
)


class TestExtractModel:
    @pytest.mark.asyncio
    async def test_extracts_model_from_json_body(self):
        """POST with JSON body: model is body['model']."""
        request = MagicMock()
        request.method = "POST"
        request.headers = {"content-type": "application/json"}
        request.body = AsyncMock(return_value=b'{"input":"hi","model":"soprano-80m"}')
        model = await extract_model_from_request(request)
        assert model == "soprano-80m"

    @pytest.mark.asyncio
    async def test_returns_none_when_body_has_no_model(self):
        request = MagicMock()
        request.method = "POST"
        request.headers = {"content-type": "application/json"}
        request.body = AsyncMock(return_value=b'{"input":"hi"}')
        model = await extract_model_from_request(request)
        assert model is None

    @pytest.mark.asyncio
    async def test_extracts_model_from_query_on_get(self):
        request = MagicMock()
        request.method = "GET"
        request.query_params = {"model": "kokoro-82m"}
        model = await extract_model_from_request(request)
        assert model == "kokoro-82m"

    @pytest.mark.asyncio
    async def test_returns_none_when_get_has_no_query_model(self):
        request = MagicMock()
        request.method = "GET"
        request.query_params = {}
        model = await extract_model_from_request(request)
        assert model is None

    @pytest.mark.asyncio
    async def test_returns_none_when_body_is_invalid_json(self):
        request = MagicMock()
        request.method = "POST"
        request.headers = {"content-type": "application/json"}
        request.body = AsyncMock(return_value=b'not json at all')
        model = await extract_model_from_request(request)
        assert model is None

    @pytest.mark.asyncio
    async def test_returns_none_when_content_type_not_json(self):
        request = MagicMock()
        request.method = "POST"
        request.headers = {"content-type": "multipart/form-data"}
        model = await extract_model_from_request(request)
        assert model is None


class TestWorkerRoute:
    def test_worker_route_stores_model_and_url(self):
        r = WorkerRoute(model_id="soprano-80m", worker_url="http://127.0.0.1:9001")
        assert r.model_id == "soprano-80m"
        assert r.worker_url == "http://127.0.0.1:9001"


class TestBuildGateway:
    def test_returns_fastapi_app(self):
        from fastapi import FastAPI
        app = build_gateway([])
        assert isinstance(app, FastAPI)

    def test_gateway_info_endpoint_exposes_routes(self):
        from fastapi.testclient import TestClient
        routes = [
            WorkerRoute("soprano-80m", "http://127.0.0.1:9001"),
            WorkerRoute("sd-turbo", "http://127.0.0.1:9002"),
        ]
        app = build_gateway(routes)
        client = TestClient(app)
        r = client.get("/_gateway-info")
        assert r.status_code == 200
        data = r.json()
        model_ids = {entry["model_id"] for entry in data["routes"]}
        assert model_ids == {"soprano-80m", "sd-turbo"}


class TestProxy:
    def test_proxy_forwards_post_to_matching_worker(self):
        routes = [WorkerRoute(model_id="soprano-80m", worker_url="http://127.0.0.1:9001")]
        app = build_gateway(routes)
        client = TestClient(app)

        with patch("muse.cli_impl.gateway.httpx.AsyncClient") as mock_client_cls:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.content = b'{"ok": true}'
            mock_response.headers = {"content-type": "application/json"}
            async_mock_request = AsyncMock(return_value=mock_response)
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.request = async_mock_request
            mock_client_cls.return_value = mock_client

            r = client.post("/v1/audio/speech", json={
                "input": "hi", "model": "soprano-80m",
            })

        assert r.status_code == 200
        # The AsyncClient.request call should have targeted the worker url
        call_kwargs = async_mock_request.call_args.kwargs
        call_args = async_mock_request.call_args.args
        target_url = call_args[1] if len(call_args) > 1 else call_kwargs.get("url")
        assert target_url == "http://127.0.0.1:9001/v1/audio/speech"

    def test_proxy_returns_404_openai_envelope_for_unknown_model(self):
        routes = [WorkerRoute(model_id="soprano-80m", worker_url="http://127.0.0.1:9001")]
        app = build_gateway(routes)
        client = TestClient(app)

        r = client.post("/v1/audio/speech", json={
            "input": "hi", "model": "does-not-exist",
        })
        assert r.status_code == 404
        body = r.json()
        assert "error" in body
        assert "detail" not in body
        assert body["error"]["code"] == "model_not_found"
        assert "does-not-exist" in body["error"]["message"]

    def test_proxy_returns_400_when_model_not_specified(self):
        """POST without a model field: 400 (client must provide routing info)."""
        routes = [WorkerRoute(model_id="soprano-80m", worker_url="http://127.0.0.1:9001")]
        app = build_gateway(routes)
        client = TestClient(app)

        r = client.post("/v1/audio/speech", json={"input": "hi"})
        assert r.status_code == 400
        body = r.json()
        assert "error" in body
        assert body["error"]["code"] == "model_required"
