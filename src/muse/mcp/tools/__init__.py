"""The fixed admin tool registry, ``ADMIN_TOOLS``.

Each ``ToolEntry`` pairs a tool definition (``tool``) with a synchronous
handler (``handler(client, args) -> list[dict content blocks]``). The
inference tools are not listed here: ``muse.mcp.generated`` builds them from
the server's own discovery documents.

The plain-dict content-block shape lets the registry stay testable
without the mcp SDK installed; ``MCPServer`` converts to SDK types at the
call boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class ToolEntry:
    """A registered tool plus its handler.

    Attributes:
        tool: an ``mcp.types.Tool`` instance.
        handler: ``(client, args) -> list[dict]`` of content blocks.
    """

    tool: Any
    handler: Callable[..., list[dict]]


ADMIN_TOOLS: list[ToolEntry] = []


def _eager_import_admin() -> None:
    """Populate ``ADMIN_TOOLS``; a missing mcp SDK leaves it empty so
    ``muse --help`` and other non-MCP paths still import this package."""
    try:
        import mcp.types  # noqa: F401
    except ImportError:
        return
    from muse.mcp.tools import admin  # noqa: F401


_eager_import_admin()
