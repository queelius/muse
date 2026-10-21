"""MCPServer: wrap mcp.server.lowlevel.Server with muse-specific tools.

Importing this module lazily probes the ``mcp`` Python package. If it's
not installed, the symbols still exist at module level (so the rest of
muse can ``from muse.mcp import MCPServer`` without importing the SDK
unless instantiated). Construction or ``build_tools`` calls then raise
a clear RuntimeError pointing at ``pip install museq[server]``.

Stdio mode (default) is for desktop LLM apps that spawn the MCP server
as a child process. HTTP+SSE mode is for remote / web embedders.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import httpx

from muse.admin.client import AdminClientError
from muse.mcp.client import MuseClient

log = logging.getLogger("muse.mcp")

_MAX_ERROR_CHARS = 300

try:
    import mcp.types as _mcp_types  # noqa: F401
    from mcp.server.lowlevel import Server as _LowServer  # noqa: F401
    _MCP_AVAILABLE = True
except ImportError:  # pragma: no cover  (covered indirectly via skip)
    _MCP_AVAILABLE = False
    _mcp_types = None  # type: ignore[assignment]
    _LowServer = None  # type: ignore[assignment]


def _require_mcp() -> None:
    if not _MCP_AVAILABLE:
        raise RuntimeError(
            "the `mcp` Python package is required for muse mcp; "
            "install via: pip install 'museq[server]'"
        )


def _entries(filter_kind: str) -> list[Any]:
    """The ``ToolEntry`` rows exposed under ``filter_kind``."""
    _require_mcp()
    from muse.mcp.tools import ADMIN_TOOLS, INFERENCE_TOOLS

    if filter_kind not in ("all", "admin", "inference"):
        raise ValueError(
            f"unknown filter_kind {filter_kind!r}; expected "
            f"'all' | 'admin' | 'inference'"
        )
    out: list[Any] = []
    if filter_kind in ("admin", "all"):
        out.extend(ADMIN_TOOLS)
    if filter_kind in ("inference", "all"):
        out.extend(INFERENCE_TOOLS)
    return out


def build_tools(filter_kind: str = "all") -> list[Any]:
    """Build the list of registered MCP Tool objects.

    ``filter_kind``: ``'admin'`` | ``'inference'`` | ``'all'`` (default).
    """
    return [entry.tool for entry in _entries(filter_kind)]


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


class MCPServer:
    """Wrap the mcp SDK Server with muse-specific list_tools / call_tool.

    The server stays stateless; each ``call_tool`` invocation builds a
    fresh HTTP request via the injected ``MuseClient``. Filter mode
    pre-selects which tools are exposed at boot; the server never sees
    tools outside the selected subset.
    """

    def __init__(
        self,
        *,
        client: MuseClient,
        filter_kind: str = "all",
    ) -> None:
        _require_mcp()
        self.client = client
        self.filter_kind = filter_kind
        self._server = _LowServer("muse")
        entries = _entries(filter_kind)
        self._tools = [entry.tool for entry in entries]
        self._handlers = {entry.tool.name: entry.handler for entry in entries}
        self._wire_handlers()

    def _wire_handlers(self) -> None:
        srv = self._server
        tools_snapshot = list(self._tools)

        @srv.list_tools()
        async def _list_tools():
            return tools_snapshot

        @srv.call_tool()
        async def _call_tool(name: str, arguments: dict | None):
            # Handlers are synchronous and make blocking httpx calls
            # (up to 600s for generation). In HTTP+SSE mode there is a
            # single event loop; running the handler inline would stall
            # every concurrent tool call and SSE frame for the duration
            # of one slow generation. Dispatch off-loop so the loop
            # stays responsive. MuseClient is stateless per call, so
            # this is thread-safe.
            blocks = await asyncio.to_thread(self.call_handler, name, arguments)
            return _to_content(blocks)

    @property
    def tools(self) -> list[Any]:
        return list(self._tools)

    @property
    def handlers(self) -> dict[str, Any]:
        return dict(self._handlers)

    def call_handler(self, name: str, args: dict | None = None) -> list[dict]:
        """Invoke the registered handler for ``name``; the one place tool
        failures become error blocks.

        An ``AdminClientError`` becomes an ``admin_api_error`` block; an
        HTTP error status from muse reports the status plus muse's error
        code and message; anything else reports ``str(exc)``. Returns plain
        content-block dicts (pre-conversion to SDK types), so tests drive
        tools through here without the async transport.
        """
        handler = self._handlers.get(name)
        if handler is None:
            return [{
                "type": "text",
                "text": json.dumps({"error": f"unknown tool {name!r}"}),
            }]
        try:
            return handler(self.client, args or {})
        except AdminClientError as e:
            return [{"type": "text", "text": json.dumps({
                "error": {
                    "code": e.code,
                    "message": e.message,
                    "status": e.status,
                    "type": "admin_api_error",
                },
            }, indent=2)}]
        except Exception as e:  # noqa: BLE001
            log.exception("tool %s failed", name)
            message = (
                _upstream_error(e.response)
                if isinstance(e, httpx.HTTPStatusError) else str(e)
            )
            return [{
                "type": "text",
                "text": json.dumps({"error": message, "tool": name}),
            }]

    async def run_stdio(self) -> None:
        """Drive the MCP server over stdio (default for desktop apps)."""
        from mcp.server.stdio import stdio_server

        async with stdio_server() as (read, write):
            init_opts = self._server.create_initialization_options()
            await self._server.run(read, write, init_opts)

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
                "MCP HTTP transport requires mcp>=1.8.0,<2; reinstall "
                "the server extra with: pip install 'museq[server]'"
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


def _to_content(blocks: list[dict]) -> list[Any]:
    """Convert plain-dict content blocks to mcp types instances.

    Test code can compare against the dict shapes; runtime code returns
    SDK types via this shim. Keeps the rest of the codebase free of
    per-call SDK imports.
    """
    out: list[Any] = []
    for b in blocks:
        kind = b.get("type")
        if kind == "text":
            out.append(_mcp_types.TextContent(
                type="text", text=b.get("text", ""),
            ))
        elif kind == "image":
            out.append(_mcp_types.ImageContent(
                type="image",
                data=b.get("data", ""),
                mimeType=b.get("mimeType", "image/png"),
            ))
        elif kind == "audio":
            out.append(_mcp_types.AudioContent(
                type="audio",
                data=b.get("data", ""),
                mimeType=b.get("mimeType", "audio/wav"),
            ))
        else:
            out.append(_mcp_types.TextContent(
                type="text", text=json.dumps(b),
            ))
    return out
