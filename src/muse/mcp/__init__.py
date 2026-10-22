"""muse mcp: MCP server bridging a muse server to LLM clients.

`muse mcp` exposes tools generated from the server's own discovery
documents (`/v1/capabilities`, `/openapi.json`), `muse_describe`, and the
fixed admin tools over the Model Context Protocol. Stdio mode is the
default; HTTP+SSE mode is available for remote / web embedders.

Install the light client with `pip install 'museq[mcp]'`.
"""
from muse.mcp.server import MCPServer

__all__ = ["MCPServer"]
