"""Tool registry: ADMIN_TOOLS + INFERENCE_TOOLS.

Each ``ToolEntry`` pairs a tool definition (``tool``) with a synchronous
handler (``handler(client, args) -> list[dict content blocks]``). Tasks
C-G populate these lists by importing their respective ``inference_*``
or ``admin`` modules at import time.

The plain-dict content-block shape lets the registry stay testable
without the mcp SDK installed; ``MCPServer._to_content`` converts to
SDK types at the call boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class ToolEntry:
    """A registered tool plus its handler.

    Attributes:
        tool: an ``mcp.types.Tool`` instance (built via the SDK at
            module import time when the SDK is available; tests skip
            the modules wholesale otherwise).
        handler: ``(client, args) -> list[dict]`` of content blocks.
            Each block has a ``"type"`` key (``"text"``, ``"image"``,
            ``"audio"``) plus the modality-specific fields.
    """

    tool: Any
    handler: Callable[..., list[dict]]


ADMIN_TOOLS: list[ToolEntry] = []
INFERENCE_TOOLS: list[ToolEntry] = []


def _eager_import_subpackages() -> None:
    """Import every tool subpackage so they register their entries.

    Called at the bottom of this module so the registries are populated
    by the time MCPServer reads them. Each submodule's import side
    effect is to ``.append(...)`` on the lists above.

    If the mcp SDK isn't installed, the submodules raise ImportError on
    their ``from mcp.types import Tool`` line; the import is wrapped so
    a missing SDK doesn't break ``muse --help`` or other CLI paths that
    don't actually need MCP.
    """
    try:
        import mcp.types  # noqa: F401
    except ImportError:
        return
    # Order doesn't matter; each submodule just appends to its list.
    from muse.mcp.tools import admin  # noqa: F401
    from muse.mcp.tools import inference_text  # noqa: F401
    from muse.mcp.tools import inference_image  # noqa: F401
    from muse.mcp.tools import inference_audio  # noqa: F401
    from muse.mcp.tools import inference_video  # noqa: F401


_eager_import_subpackages()
