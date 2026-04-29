"""CLI entry for ``muse mcp``.

Probes the muse server at startup, builds an MCPServer, runs in stdio
(default) or HTTP+SSE mode. The argparse handler in ``muse.cli`` routes
here.

Heavy imports live inside the function so ``muse --help`` stays cheap.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys

log = logging.getLogger("muse.mcp.cli")


def run_mcp_server(
    *,
    http: bool,
    port: int,
    server_url: str,
    admin_token: str | None,
    filter_kind: str,
) -> int:
    """Synchronous entry used by the argparse handler.

    Returns the process exit code; 0 on clean shutdown.
    """
    try:
        from muse.mcp.client import MuseClient
        from muse.mcp.server import MCPServer
    except ImportError as e:
        print(
            f"error: muse mcp requires the mcp Python package "
            f"(install via `pip install 'muse[server]'`). Original: {e}",
            file=sys.stderr,
        )
        return 2

    # Probe muse server reachability. Failure is a warning, not fatal:
    # muse may come up later, and the MCP client should not refuse to
    # register tools just because the gateway isn't listening yet.
    client = MuseClient(server_url=server_url, admin_token=admin_token)
    try:
        client.health()
    except Exception as e:  # noqa: BLE001
        print(
            f"warning: could not reach muse server at {server_url}: {e}",
            file=sys.stderr,
        )

    if filter_kind == "admin" and not (
        admin_token or os.environ.get("MUSE_ADMIN_TOKEN")
    ):
        print(
            "warning: --filter admin without MUSE_ADMIN_TOKEN; "
            "admin tools will return admin_disabled per call",
            file=sys.stderr,
        )

    try:
        server = MCPServer(client=client, filter_kind=filter_kind)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    try:
        if http:
            print(
                f"muse mcp HTTP+SSE on http://127.0.0.1:{port}/mcp "
                f"({len(server.tools)} tools, filter={filter_kind})",
                file=sys.stderr,
            )
            asyncio.run(server.run_http(host="127.0.0.1", port=port))
        else:
            # Stdio mode: avoid printing to stdout (it's the wire).
            print(
                f"muse mcp stdio ({len(server.tools)} tools, filter={filter_kind})",
                file=sys.stderr,
            )
            asyncio.run(server.run_stdio())
    except KeyboardInterrupt:
        return 0
    return 0
