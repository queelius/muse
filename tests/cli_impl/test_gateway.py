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
    async def test_returns_none_for_unknown_content_type(self):
        """text/plain (or any non-JSON, non-multipart) returns None."""
        request = MagicMock()
        request.method = "POST"
        request.headers = {"content-type": "text/plain"}
        model = await extract_model_from_request(request)
        assert model is None

    @pytest.mark.asyncio
    async def test_extracts_model_from_multipart_form_body(self):
        """POST with multipart/form-data: model is form['model'].

        OpenAI's audio.transcriptions / audio.translations / images.edits
        / images.variations endpoints all use multipart and put the model
        in a form field, so the gateway must support extraction here.
        """
        # Use a real Starlette Request because request.form() needs the
        # full receive-channel machinery; MagicMock doesn't carry that.
        from starlette.requests import Request

        body = (
            b"--boundary\r\n"
            b'Content-Disposition: form-data; name="model"\r\n\r\n'
            b"whisper-tiny\r\n"
            b"--boundary\r\n"
            b'Content-Disposition: form-data; name="file"; filename="a.wav"\r\n'
            b"Content-Type: audio/wav\r\n\r\n"
            b"FAKEWAVBYTES\r\n"
            b"--boundary--\r\n"
        )
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/audio/transcriptions",
            "headers": [
                (b"content-type", b"multipart/form-data; boundary=boundary"),
                (b"content-length", str(len(body)).encode()),
            ],
            "query_string": b"",
        }
        sent = False
        async def receive():
            nonlocal sent
            if sent:
                return {"type": "http.disconnect"}
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}

        request = Request(scope, receive=receive)
        model = await extract_model_from_request(request)
        assert model == "whisper-tiny"

    @pytest.mark.asyncio
    async def test_returns_none_when_multipart_body_has_no_model_field(self):
        """A multipart body without a `model` form field returns None."""
        from starlette.requests import Request

        body = (
            b"--boundary\r\n"
            b'Content-Disposition: form-data; name="file"; filename="a.wav"\r\n'
            b"Content-Type: audio/wav\r\n\r\n"
            b"FAKEWAVBYTES\r\n"
            b"--boundary--\r\n"
        )
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/audio/transcriptions",
            "headers": [
                (b"content-type", b"multipart/form-data; boundary=boundary"),
                (b"content-length", str(len(body)).encode()),
            ],
            "query_string": b"",
        }
        sent = False
        async def receive():
            nonlocal sent
            if sent:
                return {"type": "http.disconnect"}
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}

        request = Request(scope, receive=receive)
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
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.aclose = AsyncMock()

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {"content-type": "application/json"}
            mock_response.aread = AsyncMock(return_value=b'{"ok": true}')

            stream_ctx = MagicMock()
            stream_ctx.__aenter__ = AsyncMock(return_value=mock_response)
            stream_ctx.__aexit__ = AsyncMock(return_value=None)
            mock_client.stream = MagicMock(return_value=stream_ctx)

            mock_client_cls.return_value = mock_client

            r = client.post("/v1/audio/speech", json={
                "input": "hi", "model": "soprano-80m",
            })

        assert r.status_code == 200
        # The stream() call should have targeted the worker url
        call_kwargs = mock_client.stream.call_args.kwargs
        call_args = mock_client.stream.call_args.args
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

    def test_proxy_forwards_multipart_to_matching_worker(self):
        """Regression: extracting model from a multipart body must NOT
        consume the receive stream. Without `await request.body()`
        before `await request.form()`, _forward's later body() raises
        RuntimeError("Stream consumed") and the request fails as 500.

        Saw this live on v0.13.1 against /v1/audio/transcriptions.
        """
        routes = [WorkerRoute(model_id="whisper-tiny", worker_url="http://127.0.0.1:9099")]
        app = build_gateway(routes)
        client = TestClient(app)

        captured_body = {}

        with patch("muse.cli_impl.gateway.httpx.AsyncClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.aclose = AsyncMock()

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {"content-type": "application/json"}
            mock_response.aread = AsyncMock(return_value=b'{"text":"hello"}')

            stream_ctx = MagicMock()
            stream_ctx.__aenter__ = AsyncMock(return_value=mock_response)
            stream_ctx.__aexit__ = AsyncMock(return_value=None)

            def _capture_stream(method, url, **kwargs):
                captured_body["body"] = kwargs.get("content")
                captured_body["url"] = url
                return stream_ctx

            mock_client.stream = MagicMock(side_effect=_capture_stream)
            mock_client_cls.return_value = mock_client

            r = client.post(
                "/v1/audio/transcriptions",
                files={"file": ("a.wav", b"FAKEWAV", "audio/wav")},
                data={"model": "whisper-tiny"},
            )

        # Must not be 500 or 400 - the multipart body must have been
        # parsed for routing AND forwarded with its bytes intact.
        assert r.status_code == 200, f"got {r.status_code}: {r.text}"
        assert captured_body["url"] == "http://127.0.0.1:9099/v1/audio/transcriptions"
        # The forwarded body must contain the multipart payload, not be empty
        forwarded = captured_body["body"]
        assert forwarded, "forwarded body is empty (stream was consumed before forward)"
        assert b"whisper-tiny" in forwarded
        assert b"FAKEWAV" in forwarded


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


class TestAggregation:
    def test_v1_models_aggregates_across_workers(self):
        routes = [
            WorkerRoute("soprano-80m", "http://127.0.0.1:9001"),
            WorkerRoute("sd-turbo", "http://127.0.0.1:9002"),
        ]
        app = build_gateway(routes)
        client = TestClient(app)

        with patch("muse.cli_impl.gateway.httpx.AsyncClient") as mock_client_cls:
            def make_resp(data):
                r = MagicMock()
                r.status_code = 200
                r.json.return_value = {"object": "list", "data": data}
                return r

            responses_by_url = {
                "http://127.0.0.1:9001/v1/models": make_resp([
                    {"id": "soprano-80m", "modality": "audio/speech", "object": "model"},
                ]),
                "http://127.0.0.1:9002/v1/models": make_resp([
                    {"id": "sd-turbo", "modality": "image/generation", "object": "model"},
                ]),
            }

            async def fake_get(url, **kwargs):
                return responses_by_url[url]

            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.get = fake_get
            mock_client_cls.return_value = mock_client

            r = client.get("/v1/models")
        assert r.status_code == 200
        data = r.json()["data"]
        ids = {m["id"] for m in data}
        assert ids == {"soprano-80m", "sd-turbo"}

    def test_v1_models_skips_unreachable_workers(self):
        """If a worker is down, its models are omitted (not a 500)."""
        routes = [
            WorkerRoute("soprano-80m", "http://127.0.0.1:9001"),
            WorkerRoute("sd-turbo", "http://127.0.0.1:9999"),  # down
        ]
        app = build_gateway(routes)
        client = TestClient(app)

        with patch("muse.cli_impl.gateway.httpx.AsyncClient") as mock_client_cls:
            r_ok = MagicMock(status_code=200)
            r_ok.json.return_value = {"object": "list", "data": [
                {"id": "soprano-80m", "modality": "audio/speech", "object": "model"},
            ]}

            async def fake_get(url, **kwargs):
                if "9001" in url:
                    return r_ok
                raise httpx.ConnectError("connection refused", request=None)

            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.get = fake_get
            mock_client_cls.return_value = mock_client

            r = client.get("/v1/models")
        assert r.status_code == 200
        ids = {m["id"] for m in r.json()["data"]}
        assert ids == {"soprano-80m"}

    def test_health_aggregates_worker_status(self):
        routes = [
            WorkerRoute("soprano-80m", "http://127.0.0.1:9001"),
            WorkerRoute("sd-turbo", "http://127.0.0.1:9002"),
        ]
        app = build_gateway(routes)
        client = TestClient(app)

        with patch("muse.cli_impl.gateway.httpx.AsyncClient") as mock_client_cls:
            def make_resp(payload):
                r = MagicMock(status_code=200)
                r.json.return_value = payload
                return r

            responses = {
                "http://127.0.0.1:9001/health": make_resp({
                    "status": "ok", "modalities": ["audio/speech"], "models": ["soprano-80m"],
                }),
                "http://127.0.0.1:9002/health": make_resp({
                    "status": "ok", "modalities": ["image/generation"], "models": ["sd-turbo"],
                }),
            }

            async def fake_get(url, **kwargs):
                return responses[url]

            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.get = fake_get
            mock_client_cls.return_value = mock_client

            r = client.get("/health")
        body = r.json()
        assert body["status"] == "ok"
        assert set(body["modalities"]) == {"audio/speech", "image/generation"}
        assert set(body["models"]) == {"soprano-80m", "sd-turbo"}

    def test_health_degraded_when_any_worker_down(self):
        routes = [
            WorkerRoute("soprano-80m", "http://127.0.0.1:9001"),
            WorkerRoute("sd-turbo", "http://127.0.0.1:9002"),
        ]
        app = build_gateway(routes)
        client = TestClient(app)

        with patch("muse.cli_impl.gateway.httpx.AsyncClient") as mock_client_cls:
            r_ok = MagicMock(status_code=200)
            r_ok.json.return_value = {
                "status": "ok", "modalities": ["audio/speech"], "models": ["soprano-80m"],
            }

            async def fake_get(url, **kwargs):
                if "9001" in url:
                    return r_ok
                raise httpx.ConnectError("down", request=None)

            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.get = fake_get
            mock_client_cls.return_value = mock_client

            r = client.get("/health")
        body = r.json()
        assert body["status"] == "degraded"
        assert "sd-turbo" not in body["models"]


class TestStreaming:
    def test_sse_stream_is_relayed_chunk_by_chunk(self):
        """A `stream: true` response (text/event-stream) must pass through."""
        routes = [WorkerRoute("soprano-80m", "http://127.0.0.1:9001")]
        app = build_gateway(routes)
        client = TestClient(app)

        chunks = [b"data: chunk1\n\n", b"data: chunk2\n\n", b"event: done\ndata: \n\n"]

        with patch("muse.cli_impl.gateway.httpx.AsyncClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.aclose = AsyncMock()

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {"content-type": "text/event-stream"}

            async def aiter_raw():
                for c in chunks:
                    yield c
            mock_response.aiter_raw = aiter_raw
            mock_response.aclose = AsyncMock()
            mock_response.aread = AsyncMock(return_value=b"".join(chunks))

            # stream() is an async context manager
            stream_ctx = MagicMock()
            stream_ctx.__aenter__ = AsyncMock(return_value=mock_response)
            stream_ctx.__aexit__ = AsyncMock(return_value=None)
            mock_client.stream = MagicMock(return_value=stream_ctx)

            mock_client_cls.return_value = mock_client

            r = client.post("/v1/audio/speech", json={
                "input": "hi", "model": "soprano-80m", "stream": True,
            })

        assert r.status_code == 200
        assert "text/event-stream" in r.headers.get("content-type", "")
        # All chunks received in order
        assert b"data: chunk1" in r.content
        assert b"data: chunk2" in r.content
        assert b"event: done" in r.content


class TestAdminMount:
    """Verify /v1/admin/* lands on the admin router with auth enforced."""

    def test_admin_path_without_token_returns_503(self, monkeypatch):
        from muse.admin.auth import ADMIN_TOKEN_ENV
        monkeypatch.delenv(ADMIN_TOKEN_ENV, raising=False)
        app = build_gateway([])
        client = TestClient(app, raise_server_exceptions=False)
        r = client.get("/v1/admin/workers")
        assert r.status_code == 503
        assert r.json()["detail"]["error"]["code"] == "admin_disabled"

    def test_admin_path_with_token_passes_auth(self, monkeypatch):
        from muse.admin.auth import ADMIN_TOKEN_ENV
        from muse.cli_impl.supervisor import (
            SupervisorState,
            clear_supervisor_state,
            set_supervisor_state,
        )
        monkeypatch.setenv(ADMIN_TOKEN_ENV, "tok")
        clear_supervisor_state()
        set_supervisor_state(SupervisorState(workers=[], device="cpu"))
        try:
            app = build_gateway([])
            client = TestClient(app, raise_server_exceptions=False)
            r = client.get(
                "/v1/admin/workers",
                headers={"Authorization": "Bearer tok"},
            )
            assert r.status_code == 200
            assert r.json() == {"workers": []}
        finally:
            clear_supervisor_state()

    def test_admin_path_with_wrong_token_returns_403(self, monkeypatch):
        from muse.admin.auth import ADMIN_TOKEN_ENV
        monkeypatch.setenv(ADMIN_TOKEN_ENV, "tok")
        app = build_gateway([])
        client = TestClient(app, raise_server_exceptions=False)
        r = client.get(
            "/v1/admin/workers",
            headers={"Authorization": "Bearer wrong"},
        )
        assert r.status_code == 403

    def test_inference_proxy_still_works_after_admin_mount(self):
        """Regression: /v1/* with body['model'] still hits the proxy."""
        # Use the proxy with no actual workers; it should return 404
        # model_not_found, NOT some admin-route shadow.
        app = build_gateway([])
        client = TestClient(app, raise_server_exceptions=False)
        r = client.post(
            "/v1/audio/speech",
            json={"input": "hi", "model": "ghost"},
        )
        assert r.status_code == 404
        body = r.json()
        # The proxy uses {"error": {...}} (not detail)
        assert body["error"]["code"] == "model_not_found"
