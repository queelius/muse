"""MCPServer: wrap mcp.server.lowlevel.Server with muse's tools.

The tool list is dynamic: ``muse_describe``, one tool generated per route of
the server's own ``/v1/capabilities`` + ``/openapi.json``
(``muse.mcp.generated``), and the fixed admin tools, selected by
``filter_kind``. The discovery documents are cached for 30 s. While the
server is unreachable only ``muse_describe`` (which reports the error) and
the admin tools are listed, never stale generated tools.

The server advertises ``tools.listChanged``. A session that has listed
tools is sent ``notifications/tools/list_changed`` once whenever the
generated set changes (a model or modality added or removed, or the muse
server going down -> up): checked after every tool call and resource read
in that session, and every ``CACHE_SECONDS`` by a background watcher in
stdio mode.

Resources: ``muse://capabilities`` and ``muse://models/{id}`` (JSON).

Importing this module lazily probes the ``mcp`` Python package; without it
the symbols still exist and construction raises a RuntimeError pointing at
``pip install 'museq[mcp]'``.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
import time
import urllib.parse
import weakref
from typing import Any

import httpx

from muse.admin.client import AdminClientError
from muse.mcp.client import MuseClient

log = logging.getLogger("muse.mcp")

_MAX_ERROR_CHARS = 300
CACHE_SECONDS = 30.0
_FAILED_CACHE_SECONDS = 5.0
_FILTERS = ("all", "admin", "inference")
CAPABILITIES_URI = "muse://capabilities"
MODEL_URI_PREFIX = "muse://models/"

try:
    import mcp.types as _mcp_types  # noqa: F401
    from mcp.server.lowlevel import NotificationOptions
    from mcp.server.lowlevel import Server as _SDKServer
    from mcp.server.lowlevel.helper_types import ReadResourceContents

    class _LowServer(_SDKServer):
        """Advertise ``tools.listChanged`` by default.

        The stdio path, the HTTP session manager and the SDK's in-memory
        test transport all call ``create_initialization_options()`` with no
        arguments, so the default is where the capability must live.
        """

        def create_initialization_options(self, notification_options=None,
                                          experimental_capabilities=None):
            return super().create_initialization_options(
                notification_options or NotificationOptions(tools_changed=True),
                experimental_capabilities,
            )

    from muse.mcp.generated import (
        DESCRIBE,
        call_describe,
        call_generated,
        describe_tool,
        generate_tools,
    )
    _MCP_AVAILABLE = True
except ImportError:  # pragma: no cover  (covered indirectly via skip)
    _MCP_AVAILABLE = False
    _mcp_types = None  # type: ignore[assignment]
    _LowServer = None  # type: ignore[assignment]


def _require_mcp() -> None:
    if not _MCP_AVAILABLE:
        raise RuntimeError(
            "the `mcp` Python package is required for muse mcp; "
            "install via: pip install 'museq[mcp]'"
        )


def _upstream_error(response: httpx.Response) -> str:
    """A short description of a muse HTTP error: the status plus the
    server's OpenAI-shape error code and (bounded) message."""
    try:
        error = response.json().get("error") or {}
        code, message = str(error.get("code", "")), str(error.get("message", ""))
    except Exception:  # noqa: BLE001
        code, message = "", ""
    text = f"muse returned HTTP {response.status_code}"
    if code:
        text += f" ({code})"
    if message:
        text += f": {message[:_MAX_ERROR_CHARS]}"
    return text


def _error_block(payload: dict) -> dict:
    return {"type": "text", "text": json.dumps(payload, indent=2), "error": True}


class MCPServer:
    """The mcp SDK Server with muse's dynamic tools and resources.

    Tool handlers are synchronous (blocking httpx); the SDK callbacks run
    them off the event loop.
    """

    def __init__(
        self,
        *,
        client: MuseClient,
        filter_kind: str = "all",
    ) -> None:
        _require_mcp()
        if filter_kind not in _FILTERS:
            raise ValueError(
                f"unknown filter_kind {filter_kind!r}; expected "
                f"'all' | 'admin' | 'inference'"
            )
        from muse.mcp.tools import ADMIN_TOOLS

        self.client = client
        self.filter_kind = filter_kind
        self._admin = list(ADMIN_TOOLS) if filter_kind in ("admin", "all") else []
        self._admin_handlers = {e.tool.name: e.handler for e in self._admin}
        self._inference = filter_kind in ("inference", "all")
        self._lock = threading.Lock()
        self._fetched_at: float | None = None
        self._fetch_error: str | None = None
        self._caps: dict = {}
        self._generated: dict = {}
        # Session -> signature of the tool list it last saw (weak: a closed
        # session drops out). Touched on the event loop only.
        self._sessions: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
        self._server = _LowServer("muse")
        self._wire_handlers()

    # ---- discovery cache ----

    def _refresh(self) -> None:
        """Refetch the discovery documents when the cache is stale."""
        with self._lock:
            ttl = _FAILED_CACHE_SECONDS if self._fetch_error else CACHE_SECONDS
            now = time.monotonic()
            if self._fetched_at is not None and now - self._fetched_at < ttl:
                return
            try:
                caps = self.client.capabilities()
                gens = generate_tools(caps, self.client.openapi())
                self._caps = caps
                self._generated = {g.name: g for g in gens
                                   if g.name not in self._admin_handlers}
                self._fetch_error = None
            except Exception as e:  # noqa: BLE001
                log.warning("could not read muse discovery documents: %s", e)
                self._caps, self._generated = {}, {}
                self._fetch_error = str(e)
            self._fetched_at = time.monotonic()

    @property
    def tools(self) -> list[Any]:
        return self._tool_list(refresh=True)

    def _tool_list(self, *, refresh: bool) -> list[Any]:
        out: list[Any] = []
        if self._inference:
            if refresh:
                self._refresh()
            out.append(describe_tool())
            out.extend(g.tool for g in self._generated.values())
        out.extend(e.tool for e in self._admin)
        return out

    # ---- tools/list_changed ----

    @staticmethod
    def _signature(tools: list[Any]) -> str:
        dumped = [t.model_dump(mode="json", exclude_none=True) for t in tools]
        return hashlib.sha256(
            json.dumps(dumped, sort_keys=True).encode("utf-8")).hexdigest()

    def _current_session(self) -> Any | None:
        try:
            return self._server.request_context.session
        except LookupError:  # called outside a request (direct handler use)
            return None

    async def _notify_changed(self, session: Any | None = None, *,
                              refresh: bool = True) -> None:
        """Send ``tools/list_changed`` to ``session`` (else every session that
        has listed tools) when its last-seen list differs from the current
        one. With ``refresh`` the discovery cache (TTL-bound) is refreshed off
        the loop first; without it only the cached documents are compared."""
        if not self._inference:
            return  # the admin-only set is fixed
        targets = [session] if session is not None else list(self._sessions)
        targets = [t for t in targets if t in self._sessions]
        if not targets:
            return
        sig = self._signature(await asyncio.to_thread(
            lambda: self._tool_list(refresh=refresh)))
        for sess in targets:
            if self._sessions.get(sess) == sig:
                continue
            self._sessions[sess] = sig  # once per change; re-listing confirms it
            try:
                await sess.send_tool_list_changed()
            except Exception as e:  # noqa: BLE001 - a closed session is not fatal
                log.debug("tools/list_changed not delivered: %s", e)

    async def _after_request(self) -> None:
        # Compare against the cache only: a refresh here would add up to two
        # discovery timeouts to every tool call when muse hangs. The tool call
        # itself or the background watcher keeps the cache current.
        try:
            await self._notify_changed(self._current_session(), refresh=False)
        except Exception as e:  # noqa: BLE001 - never fail the request for this
            log.debug("tools/list_changed check failed: %s", e)

    async def _watch_tools(self, interval: float | None = None) -> None:
        """Background push for long-lived (stdio) sessions."""
        while True:
            await asyncio.sleep(CACHE_SECONDS if interval is None else interval)
            try:
                await self._notify_changed()
            except Exception as e:  # noqa: BLE001
                log.debug("tools/list_changed watch failed: %s", e)

    # ---- SDK wiring ----

    def _wire_handlers(self) -> None:
        srv = self._server
        types = _mcp_types

        @srv.list_tools()
        async def _list_tools():
            tools = await asyncio.to_thread(lambda: self.tools)
            session = self._current_session()
            if session is not None:
                self._sessions[session] = self._signature(tools)
            return tools

        @srv.call_tool()
        async def _call_tool(name: str, arguments: dict | None):
            # Off-loop: handlers make blocking httpx calls (up to 600 s for
            # generation); inline they would stall every concurrent call and
            # SSE frame in HTTP mode.
            blocks = await asyncio.to_thread(self.call_handler, name, arguments)
            await self._after_request()
            return _to_result(blocks)

        @srv.list_resources()
        async def _list_resources():
            await asyncio.to_thread(self._refresh)
            out = [types.Resource(uri=CAPABILITIES_URI, name="capabilities",
                                  mimeType="application/json",
                                  description="The muse capabilities document.")]
            for m in self._caps.get("models") or []:
                mid = str(m.get("id"))
                out.append(types.Resource(
                    uri=MODEL_URI_PREFIX + urllib.parse.quote(mid, safe=""),
                    name=mid, mimeType="application/json",
                    description=m.get("summary") or m.get("modality")))
            await self._after_request()
            return out

        @srv.list_resource_templates()
        async def _list_templates():
            return [types.ResourceTemplate(
                uriTemplate=MODEL_URI_PREFIX + "{id}", name="model",
                mimeType="application/json",
                description="The capabilities document for one model id.")]

        @srv.read_resource()
        async def _read_resource(uri):
            text = str(uri)
            if text == CAPABILITIES_URI:
                doc = await asyncio.to_thread(self.client.capabilities)
            elif text.startswith(MODEL_URI_PREFIX):
                mid = urllib.parse.unquote(text[len(MODEL_URI_PREFIX):])
                doc = await asyncio.to_thread(
                    lambda: self.client.capabilities(model=mid))
            else:
                raise ValueError(f"unknown resource {text!r}")
            await self._after_request()
            return [ReadResourceContents(content=json.dumps(doc),
                                         mime_type="application/json")]

    def call_handler(self, name: str, args: dict | None = None) -> list[dict]:
        """Run tool ``name``; the one place tool failures become error blocks.

        An ``AdminClientError`` becomes an ``admin_api_error`` block; an HTTP
        error status from muse reports the status plus muse's error code and
        message; anything else reports ``str(exc)``. Error blocks carry
        ``"error": True``. Returns plain content-block dicts, so tests drive
        tools here without the async transport.
        """
        args = args or {}
        try:
            if name in self._admin_handlers:
                return self._admin_handlers[name](self.client, args)
            if self._inference and name == DESCRIBE:
                self._refresh()  # a discovery call also refreshes the tool cache
                return call_describe(self.client, args)
            gen = self._generated.get(name) if self._inference else None
            if gen is None and self._inference:
                self._refresh()
                gen = self._generated.get(name)
            if gen is None:
                return [_error_block({"error": f"unknown tool {name!r}"})]
            return call_generated(self.client, gen, args)
        except AdminClientError as e:
            return [_error_block({"error": {
                "code": e.code, "message": e.message,
                "status": e.status, "type": "admin_api_error",
            }})]
        except Exception as e:  # noqa: BLE001
            log.exception("tool %s failed", name)
            message = (
                _upstream_error(e.response)
                if isinstance(e, httpx.HTTPStatusError) else str(e)
            )
            return [_error_block({"error": message, "tool": name})]

    async def run_stdio(self) -> None:
        """Drive the MCP server over stdio (default for desktop apps)."""
        from mcp.server.stdio import stdio_server

        async with stdio_server() as (read, write):
            init_opts = self._server.create_initialization_options()
            watcher = (asyncio.ensure_future(self._watch_tools())
                       if self._inference else None)
            try:
                await self._server.run(read, write, init_opts)
            finally:
                if watcher is not None:
                    watcher.cancel()

    def build_http_app(self, *, admin_token: str | None = None) -> Any:
        """Build (but don't run) the Starlette app for HTTP+SSE mode.

        Factored out so tests can mount the app on a TestClient without
        actually launching uvicorn. The session manager runs as a
        lifespan-bound background task; tests that exercise the full
        wire path can drive the app via httpx.AsyncClient.

        Auth: when ``admin_token`` is provided (or ``MUSE_ADMIN_TOKEN`` is
        set), every request to ``/mcp`` must carry
        ``Authorization: Bearer <token>``.  Requests without a valid token
        receive 401. When no token is configured the server starts with a
        prominent WARNING log so operators know the endpoint is open.
        """
        import contextlib
        import secrets as _secrets

        try:
            from mcp.server.streamable_http_manager import (
                StreamableHTTPSessionManager,
            )
        except ImportError as exc:
            raise RuntimeError(
                "MCP HTTP transport requires mcp>=1.19,<2; reinstall "
                "the mcp extra with: pip install 'museq[mcp]'"
            ) from exc
        from starlette.applications import Starlette
        from starlette.responses import Response
        from starlette.routing import Mount

        from muse.core import config

        token = (admin_token or config.get("admin.token") or "").strip()

        manager = StreamableHTTPSessionManager(app=self._server, stateless=True)

        if not token:
            log.warning(
                "MCP HTTP transport is running WITHOUT authentication. "
                "Any client that can reach this port can call all MCP tools. "
                "Set MUSE_ADMIN_TOKEN (or pass --admin-token) and restart, "
                "or bind to localhost and place the server behind an "
                "authenticated reverse proxy."
            )

        async def handle_streamable_http(scope, receive, send):
            if token:
                # Minimal bearer-token middleware: read the Authorization header
                # directly from the ASGI scope without pulling in all of FastAPI.
                headers = dict(scope.get("headers", []))
                auth = headers.get(b"authorization", b"").decode("latin-1")
                if not auth.startswith("Bearer "):
                    resp = Response(
                        content='{"error":{"code":"missing_token","message":"Authorization: Bearer <token> required"}}',
                        status_code=401,
                        media_type="application/json",
                    )
                    await resp(scope, receive, send)
                    return
                presented = auth[len("Bearer "):]
                if not _secrets.compare_digest(
                    presented.encode("utf-8"), token.encode("utf-8"),
                ):
                    resp = Response(
                        content='{"error":{"code":"invalid_token","message":"Bad MCP token"}}',
                        status_code=403,
                        media_type="application/json",
                    )
                    await resp(scope, receive, send)
                    return
            await manager.handle_request(scope, receive, send)

        @contextlib.asynccontextmanager
        async def lifespan(_app):
            async with manager.run():
                yield

        return Starlette(
            routes=[Mount("/mcp", app=handle_streamable_http)],
            lifespan=lifespan,
        )

    async def run_http(
        self,
        host: str = "127.0.0.1",
        port: int = 8088,
        *,
        admin_token: str | None = None,
    ) -> None:
        """Drive the MCP server over HTTP+SSE via uvicorn."""
        import uvicorn

        from muse.cli_impl.serve_util import shutdown_grace_seconds

        app = self.build_http_app(admin_token=admin_token)
        # Bound graceful shutdown so Ctrl-C releases the port even with a
        # lingering SSE connection, matching `muse serve` / `muse _worker`.
        # uvicorn's default (None) waits forever, orphaning the bound port.
        config = uvicorn.Config(
            app, host=host, port=port, log_level="info",
            timeout_graceful_shutdown=shutdown_grace_seconds(),
        )
        await uvicorn.Server(config).serve()


def _to_result(blocks: list[dict]) -> Any:
    """Plain-dict blocks to an ``mcp.types.CallToolResult``.

    The pseudo-block ``{"type": "structured", "data": ...}`` becomes
    ``structuredContent``; a block marked ``"error": True`` sets ``isError``.
    Returning the result object bypasses the SDK's output validation, so an
    error result needs no structured content.
    """
    t = _mcp_types
    content: list[Any] = []
    structured = None
    is_error = False
    for b in blocks:
        kind = b.get("type")
        is_error = is_error or bool(b.get("error"))
        if kind == "structured":
            structured = b.get("data")
        elif kind == "text":
            content.append(t.TextContent(type="text", text=b.get("text", "")))
        elif kind == "image":
            content.append(t.ImageContent(type="image", data=b.get("data", ""),
                                          mimeType=b.get("mimeType", "image/png")))
        elif kind == "resource_link":
            content.append(t.ResourceLink(
                type="resource_link", uri=b["uri"], name=b.get("name", b["uri"]),
                mimeType=b.get("mimeType"), size=b.get("size")))
        else:
            content.append(t.TextContent(type="text", text=json.dumps(b)))
    return t.CallToolResult(content=content, structuredContent=structured,
                            isError=is_error)
