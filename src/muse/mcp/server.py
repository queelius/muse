"""MCPServer: wrap mcp.server.lowlevel.Server with muse-specific tools.

Importing this module lazily probes the ``mcp`` Python package. If it's
not installed, the symbols still exist at module level (so the rest of
muse can ``from muse.mcp import MCPServer`` without importing the SDK
unless instantiated). Construction or ``build_tools`` calls then raise
a clear RuntimeError pointing at ``pip install muse[server]``.

Stdio mode (default) is for desktop LLM apps that spawn the MCP server
as a child process. HTTP+SSE mode is for remote / web embedders.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from muse.mcp.client import MuseClient

log = logging.getLogger("muse.mcp")

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
            "install via: pip install 'muse[server]'"
        )


def build_tools(filter_kind: str = "all") -> list[Any]:
    """Build the list of registered MCP Tool objects.

    ``filter_kind``: ``'admin'`` | ``'inference'`` | ``'all'``. Default
    ``'all'`` returns 11 + 18 = 29 tools.
    """
    _require_mcp()
    from muse.mcp.tools import ADMIN_TOOLS, INFERENCE_TOOLS

    if filter_kind not in ("all", "admin", "inference"):
        raise ValueError(
            f"unknown filter_kind {filter_kind!r}; expected "
            f"'all' | 'admin' | 'inference'"
        )
    out: list[Any] = []
    if filter_kind in ("admin", "all"):
        out.extend(t.tool for t in ADMIN_TOOLS)
    if filter_kind in ("inference", "all"):
        out.extend(t.tool for t in INFERENCE_TOOLS)
    return out


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
        self._tools = build_tools(filter_kind)
        self._handlers = self._build_handler_map()
        self._wire_handlers()

    def _build_handler_map(self) -> dict[str, Any]:
        from muse.mcp.tools import ADMIN_TOOLS, INFERENCE_TOOLS

        out: dict[str, Any] = {}
        if self.filter_kind in ("admin", "all"):
            for t in ADMIN_TOOLS:
                out[t.tool.name] = t.handler
        if self.filter_kind in ("inference", "all"):
            for t in INFERENCE_TOOLS:
                out[t.tool.name] = t.handler
        return out

    def _wire_handlers(self) -> None:
        srv = self._server
        tools_snapshot = list(self._tools)
        handlers_snapshot = dict(self._handlers)
        client = self.client

        @srv.list_tools()
        async def _list_tools():
            return tools_snapshot

        @srv.call_tool()
        async def _call_tool(name: str, arguments: dict | None):
            args = arguments or {}
            handler = handlers_snapshot.get(name)
            if handler is None:
                return _to_content([{
                    "type": "text",
                    "text": json.dumps(
                        {"error": f"unknown tool {name!r}"},
                    ),
                }])
            try:
                blocks = handler(client, args)
            except Exception as e:  # noqa: BLE001
                log.exception("tool %s failed", name)
                blocks = [{
                    "type": "text",
                    "text": json.dumps(
                        {"error": str(e), "tool": name},
                    ),
                }]
            return _to_content(blocks)

    @property
    def tools(self) -> list[Any]:
        return list(self._tools)

    @property
    def handlers(self) -> dict[str, Any]:
        return dict(self._handlers)

    def call_handler(self, name: str, args: dict | None = None) -> list[dict]:
        """Synchronously invoke the registered handler for ``name``.

        Tests use this to drive tool calls without going through the
        async stdio / HTTP transport. Returns the raw list of content
        block dicts (pre-conversion to SDK types).
        """
        handler = self._handlers.get(name)
        if handler is None:
            return [{
                "type": "text",
                "text": json.dumps({"error": f"unknown tool {name!r}"}),
            }]
        try:
            return handler(self.client, args or {})
        except Exception as e:  # noqa: BLE001
            log.exception("tool %s failed", name)
            return [{
                "type": "text",
                "text": json.dumps({"error": str(e), "tool": name}),
            }]

    async def run_stdio(self) -> None:
        """Drive the MCP server over stdio (default for desktop apps)."""
        from mcp.server.stdio import stdio_server

        async with stdio_server() as (read, write):
            init_opts = self._server.create_initialization_options()
            await self._server.run(read, write, init_opts)

    async def run_http(
        self,
        host: str = "127.0.0.1",
        port: int = 8088,
    ) -> None:
        """Drive the MCP server over HTTP+SSE (lands in Task B)."""
        raise NotImplementedError("HTTP+SSE mode lands in Task B")


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
