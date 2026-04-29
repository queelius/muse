# muse mcp: MCP server exposing muse to LLM clients (v0.29.0)

**Date:** 2026-04-28
**Driver:** ship `muse mcp`, an MCP (Model Context Protocol) server that
bridges LLM clients (Claude Desktop, Cursor, etc.) to muse's HTTP API
plus the v0.28.0 admin REST surface. Closes task #149.

This is the last task in the v0.x backlog: after it lands muse will have
fifteen modalities, an admin REST API, and an MCP server. The first two
are wire surfaces for programmatic clients; this is the surface for LLM
clients that natively speak MCP.

## Goal

1. Add a `muse mcp` CLI subcommand that starts an MCP server in stdio
   mode (default, for desktop apps) or HTTP+SSE mode (for remote/web
   embedders).
2. Expose 29 tools split into two groups:
   - **ADMIN (11):** wraps `/v1/admin/*` via `muse.admin.client.AdminClient`.
     Auth via `MUSE_ADMIN_TOKEN`.
   - **INFERENCE (18):** wraps the OpenAI-compat HTTP routes for every
     modality (`/v1/chat/completions`, `/v1/audio/speech`, ...). No auth
     required (matches the inference-route policy).
3. Tool descriptions are written for the LLM as the consumer: each tool
   has a 1-2 sentence description that explains when to use it, and
   each input field has a `description` line. The LLM picks the right
   tool from these descriptions.
4. Optional `--filter admin` / `--filter inference` lets ops operators
   pin to a smaller surface (control panels need only admin; chat
   assistants need only inference).
5. Optional `--server URL` connects the MCP server to a remote muse
   gateway. Default is `http://localhost:8000`.
6. Binary inputs accept `_b64` (base64), `_url` (data: or http URL),
   or `_path` (local file). Binary outputs return `b64` plus an
   optional MCP `ImageContent` / `AudioContent` block so clients that
   render media inline see it without a follow-up tool call.
7. Long-running admin operations (pull, probe, enable) return a `job_id`
   and the LLM polls `muse_get_jobs`. Tool description tells the LLM
   the polling pattern. v1 does not block-wait on the server side; the
   LLM stays in control of cadence.

## Non-goals

- **Bidirectional streaming for inference.** Chat tool returns the full
  completion. SSE streaming through MCP is filed as v0.30.x work.
- **WebSocket-based long-running ops.** Pull/probe progress is
  observable via repeated calls to `muse_get_jobs` (matches the admin
  API's polling pattern).
- **Multi-user auth.** A single `MUSE_ADMIN_TOKEN` gates admin
  operations. The MCP server is single-tenant; production deployments
  fronting it with their own auth layer are out of scope.
- **MCP resources / prompts.** v1 ships tools only. Resources (model
  catalog as an MCP resource) and prompts (capability advertisements)
  are filed for v0.30.x.
- **Multi-server federation.** One MCP server connects to one muse
  server. Federating across hosts is filed for v0.30.x.
- **Embedded muse server.** `muse mcp` always talks HTTP to a separate
  `muse serve`. There is no in-process mode that boots the gateway
  inside the MCP process. (Future work; not blocking.)
- **Per-tool rate limiting / quotas.** Out of scope.
- **Tool composition / pipelines.** The LLM composes; the MCP server
  exposes flat tools.

## Architecture

```
Claude Desktop / Cursor / other MCP client
    |
    | stdio (JSON-RPC framed) OR HTTP+SSE
    v
muse mcp (this package)
    |
    +-- MCPServer  (mcp.server.lowlevel.Server)
    |     |
    |     +-- list_tools handler -> tool registry
    |     +-- call_tool handler  -> dispatch by name
    |
    +-- MuseClient (HTTP wrapper)
    |     |
    |     +-- AdminClient    -> /v1/admin/*  (Bearer token)
    |     +-- modality clients (one per modality) -> /v1/...
    |
    v
muse serve (the gateway, port 8000)
```

The MCP server is stateless. Each `call_tool` invocation builds a fresh
HTTP request via the relevant client, decodes the response, and returns
either MCP `TextContent` (JSON dumps) or media content blocks
(ImageContent / AudioContent) for binary outputs.

## CLI

```
muse mcp                                       stdio mode (default)
muse mcp --http --port 8088                    HTTP+SSE mode
muse mcp --server http://192.168.0.225:8000    point at a remote gateway
muse mcp --admin-token TOKEN                   override env var
muse mcp --filter admin                        only the 11 admin tools
muse mcp --filter inference                    only the 18 inference tools
```

`--filter` is a single-value choice with values `admin`, `inference`,
or `all` (default).

The CLI subcommand lives in `src/muse/cli.py` (argparse) and dispatches
to `src/muse/cli_impl/mcp_server.py:run_mcp_server`. The entry point
is async; argparse calls `asyncio.run` to drive it.

## Tool taxonomy (29 tools)

### Group ADMIN (11 tools, wraps AdminClient)

| Tool name | Args (required, optional) | Backing call |
|---|---|---|
| `muse_list_models` | (), filter_modality?, filter_status? | `GET /v1/models` filtered + catalog metadata |
| `muse_get_model_info` | model_id | `GET /v1/admin/models/{id}/status` (live worker fields) + catalog |
| `muse_search_models` | query, modality?, max_size_gb?, limit? | resolver search backend |
| `muse_pull_model` | identifier | `POST /v1/admin/models/_/pull` |
| `muse_remove_model` | model_id, purge? | `DELETE /v1/admin/models/{id}?purge=...` |
| `muse_enable_model` | model_id | `POST /v1/admin/models/{id}/enable` |
| `muse_disable_model` | model_id | `POST /v1/admin/models/{id}/disable` |
| `muse_probe_model` | model_id, no_inference?, device? | `POST /v1/admin/models/{id}/probe` |
| `muse_get_memory_status` | () | `GET /v1/admin/memory` |
| `muse_get_workers` | () | `GET /v1/admin/workers` |
| `muse_get_jobs` | job_id? | `GET /v1/admin/jobs/{id}` if id else `GET /v1/admin/jobs` |

`muse_list_models` uses the unauthenticated `GET /v1/models` plus a
secondary catalog scrape (no token) so it works in the
inference-only filter mode. The other ten admin tools require the
admin token; calling them without a configured token returns a
clear `admin_disabled` error to the LLM.

### Group INFERENCE (18 tools, wraps modality HTTP routes)

| Tool name | Backing route | Modality |
|---|---|---|
| `muse_chat` | POST /v1/chat/completions | chat/completion |
| `muse_summarize` | POST /v1/summarize | text/summarization |
| `muse_rerank` | POST /v1/rerank | text/rerank |
| `muse_classify` | POST /v1/moderations | text/classification |
| `muse_embed_text` | POST /v1/embeddings | embedding/text |
| `muse_embed_image` | POST /v1/images/embeddings | image/embedding |
| `muse_embed_audio` | POST /v1/audio/embeddings | audio/embedding (multipart) |
| `muse_generate_image` | POST /v1/images/generations | image/generation |
| `muse_edit_image` | POST /v1/images/edits | image/generation (multipart) |
| `muse_vary_image` | POST /v1/images/variations | image/generation (multipart) |
| `muse_upscale_image` | POST /v1/images/upscale | image/upscale (multipart) |
| `muse_segment_image` | POST /v1/images/segment | image/segmentation (multipart) |
| `muse_generate_animation` | POST /v1/images/animations | image/animation |
| `muse_generate_video` | POST /v1/video/generations | video/generation |
| `muse_generate_music` | POST /v1/audio/music | audio/generation |
| `muse_generate_sfx` | POST /v1/audio/sfx | audio/generation |
| `muse_speak` | POST /v1/audio/speech | audio/speech |
| `muse_transcribe` | POST /v1/audio/transcriptions | audio/transcription (multipart) |

Total: 11 + 18 = 29 tools.

### Tool descriptions

Each tool's `description` field is written for the LLM. Example:

```python
Tool(
    name="muse_generate_image",
    description=(
        "Generate one or more images from a text prompt using a "
        "diffusion model. Use this when the user asks for a new image "
        "to be created from scratch. Returns an MCP ImageContent block "
        "for each generated image plus a JSON summary."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "description": "Text description of the image"},
            "model": {"type": "string", "description": "Optional model id; omit for default"},
            "n": {"type": "integer", "default": 1, "minimum": 1, "maximum": 4},
            "size": {"type": "string", "default": "1024x1024", "pattern": "^[0-9]+x[0-9]+$"},
            "negative_prompt": {"type": "string"},
            "steps": {"type": "integer", "minimum": 1, "maximum": 100},
            "guidance": {"type": "number", "minimum": 0, "maximum": 20},
            "seed": {"type": "integer"},
        },
        "required": ["prompt"],
    },
)
```

## Binary I/O conventions

### Inputs

For tools that accept binary data (edit_image, vary_image, segment_image,
upscale_image, transcribe, embed_image, embed_audio), the input schema
accepts three alternative fields:

- `image_b64`: base64-encoded bytes (most natural for LLM-generated
  content carried through the conversation context)
- `image_url`: a `data:` URL or `http(s)://` URL
- `image_path`: a local filesystem path (only meaningful when the MCP
  server runs on the same host as the file; documented as such)

Exactly one of these three must be present. The server resolves to raw
bytes once and forwards them as multipart to muse. Same convention for
`audio_*` and `mask_*` fields.

### Outputs

For tools that return binary data (generate_image, edit_image,
vary_image, upscale_image, generate_animation, generate_video,
generate_music, generate_sfx, speak), the response is a list of MCP
content blocks:

```python
[
    ImageContent(type="image", data="<b64>", mimeType="image/png"),
    TextContent(type="text", text=json.dumps({"model": "sd-turbo", "n": 1, "size": "1024x1024"})),
]
```

The summary block carries metadata; the media blocks are the artifacts
the LLM client can render inline. For modalities the SDK doesn't model
(video), the bytes return as base64 inside the JSON summary block:

```python
[
    TextContent(type="text", text=json.dumps({
        "model": "wan2_1_t2v_1_3b",
        "format": "mp4",
        "video_b64": "<b64>",
    })),
]
```

The MCP SDK ships `AudioContent` for audio bytes; `speak`,
`generate_music`, and `generate_sfx` use it. `transcribe` returns a
`TextContent` with the transcription text.

## Auth model

Two layers:

1. **MCP-level auth.** v1 has none. The MCP server trusts whoever
   connects to it. Stdio mode is implicitly local (the parent process
   spawns the server); HTTP mode binds 127.0.0.1 by default.
2. **Admin-tool auth.** The MCP server reads `MUSE_ADMIN_TOKEN` at
   startup (or from `--admin-token`) and passes it as the bearer token
   for admin tool calls. With no token configured, admin tools return
   a clear `admin_disabled` error to the LLM (translated from the
   `503 admin_disabled` muse returns). The LLM gets a structured
   message it can relay to the user: "the muse admin token is not
   configured; ask the user to set MUSE_ADMIN_TOKEN."

Filter mode interaction:
- `--filter admin` requires a token; without one the server logs a
  warning at startup but still runs (admin tools then return
  `admin_disabled` per call).
- `--filter inference` doesn't need a token. Admin tools are not
  registered, so the LLM cannot call them.

## Long-running operations

Three admin operations are async on the muse side: `pull`, `probe`,
`enable`. Each returns a `job_id` from the admin API.

The MCP `muse_pull_model`, `muse_probe_model`, and `muse_enable_model`
tools return immediately with the job_id and a hint to poll:

```json
{
  "job_id": "abc123",
  "status": "pending",
  "poll_with": "muse_get_jobs(job_id='abc123')",
  "estimated_time": "30s to 5min depending on model size"
}
```

The LLM is in charge of polling cadence. `muse_get_jobs` accepts an
optional `job_id` to poll a specific job, or returns all recent jobs
when called without arguments. The response includes `state`, `result`,
`error`, and recent `log_lines`.

Server-side blocking is rejected because:
- LLM clients don't deal well with tool calls that block for minutes.
- The polling pattern is the same one a Python user using AdminClient
  follows (`AdminClient.wait` is a convenience over `AdminClient.job`).
- It keeps the MCP server stateless. No background threads, no
  in-flight bookkeeping.

## Filter mode

A single CLI flag, three values: `admin`, `inference`, `all` (default).

```python
def build_tools(filter_kind: str) -> list[Tool]:
    tools = []
    if filter_kind in ("admin", "all"):
        tools.extend(ADMIN_TOOLS)
    if filter_kind in ("inference", "all"):
        tools.extend(INFERENCE_TOOLS)
    return tools
```

The LLM only sees tools the operator wants exposed. Use cases:
- Control-panel agent: `--filter admin` (only catalog and worker ops).
- Production chat assistant: `--filter inference` (no model-management).
- Power-user setup: omit `--filter` (default `all`, all 29 tools).

## Module structure

```
src/muse/mcp/
  __init__.py                  re-exports MCPServer + entry point
  server.py                    MCPServer class wrapping mcp.server.Server
  client.py                    MuseClient (httpx-based; aggregates AdminClient + modality clients)
  binary_io.py                 input resolution (b64 | url | path -> bytes); output content-block packing
  tools/
    __init__.py                ALL_TOOLS = ADMIN_TOOLS + INFERENCE_TOOLS
    admin.py                   11 admin tools: schema + handlers
    inference_text.py          chat, summarize, rerank, classify, embed_text (5)
    inference_image.py         generate, edit, vary, upscale, segment, animation, embed_image (7)
    inference_audio.py         speak, transcribe, music, sfx, embed_audio (5)
    inference_video.py         generate_video (1)
src/muse/cli_impl/mcp_server.py   CLI entry: run_mcp_server(http, port, server_url, admin_token, filter_kind)
```

Plus argparse changes in `src/muse/cli.py`.

## MuseClient

`muse.mcp.client.MuseClient` is a thin aggregator over the per-modality
HTTP clients. It instantiates one of each on demand and shares one
`base_url` across all of them:

```python
class MuseClient:
    def __init__(self, server_url: str, admin_token: str | None = None, timeout: float = 600.0):
        self.server_url = server_url.rstrip("/")
        self.admin_token = admin_token
        self.timeout = timeout
        self.admin = AdminClient(base_url=self.server_url, token=self.admin_token, timeout=30.0)

    def health(self) -> dict: ...           # GET /health
    def list_models(self) -> dict: ...      # GET /v1/models

    # Modality routes (returns parsed JSON envelope for non-binary,
    # raw bytes for binary):
    def chat(self, **kwargs) -> dict: ...
    def summarize(self, **kwargs) -> dict: ...
    def rerank(self, **kwargs) -> dict: ...
    def classify(self, **kwargs) -> dict: ...
    def embed_text(self, **kwargs) -> dict: ...
    def embed_image(self, **kwargs) -> dict: ...
    def embed_audio(self, **kwargs) -> dict: ...
    def generate_image(self, **kwargs) -> dict: ...   # returns {"data": [{"b64_json": "..."}, ...]}
    def edit_image(self, **kwargs) -> dict: ...
    def vary_image(self, **kwargs) -> dict: ...
    def upscale_image(self, **kwargs) -> dict: ...
    def segment_image(self, **kwargs) -> dict: ...
    def generate_animation(self, **kwargs) -> dict: ...
    def generate_video(self, **kwargs) -> dict: ...
    def generate_music(self, **kwargs) -> bytes: ...
    def generate_sfx(self, **kwargs) -> bytes: ...
    def speak(self, **kwargs) -> bytes: ...
    def transcribe(self, **kwargs) -> dict: ...
```

The client does not retain any per-modality client instances; it builds
the right HTTP request with `httpx` directly. This avoids importing
fifteen modality client modules at startup (each pulls in `requests`
or `httpx`, fine, but also their codec module, which sometimes pulls
numpy). Direct httpx keeps the MCP boot path cheap.

## Tests

`tests/mcp/` mirrors the source tree:
```
tests/mcp/
  test_server.py             server lifecycle, list_tools, call_tool dispatch, filter mode, auth passthrough
  test_client.py             MuseClient with mocked httpx
  test_binary_io.py          b64/url/path input resolution; output content-block packing
  test_tools_admin.py        each of the 11 admin tools with mocked AdminClient
  test_tools_inference_text.py    each of the 5 text inference tools
  test_tools_inference_image.py   each of the 7 image inference tools
  test_tools_inference_audio.py   each of the 5 audio inference tools
  test_tools_inference_video.py   the 1 video inference tool
  test_stdio.py              stdio framing via fake stdin/stdout pair
  test_http_sse.py           HTTP+SSE mode via FastAPI TestClient
```

Skip pattern: every test file imports `mcp` at module-level inside a
try/except and `pytest.skip`s the whole module if the SDK isn't there.
The rest of the muse test suite is unaffected.

## Resilience considerations

1. **Reachability at startup.** The CLI probes `GET /health` once; on
   failure it warns to stderr and starts anyway. Muse may come up
   later, and we don't want to refuse to register tools just because
   the gateway isn't listening yet.

2. **Tool count budget.** 29 tools is a lot. Filter mode lets users
   drop to 11 or 18. Tool descriptions are short to keep the system
   prompt LLM-clients build under the size limits typical clients
   impose.

3. **Tool name collisions.** All tool names are prefixed `muse_`. This
   guarantees no clash with other MCP servers a client may run in
   parallel.

4. **Schema validation.** The MCP SDK validates `inputSchema` against
   the call args before dispatching. Bad LLM-generated args fail with
   a structured error the LLM can correct from. We rely on this for
   defense rather than re-validating in handlers.

5. **Binary input size.** Base64 inputs are size-bounded by the
   client's tool-call payload limit (typically 1-10 MB depending on
   the client). For large files (a 30-min audio file for transcription),
   the LLM should use `audio_path` to point at a local file. This is
   documented in the tool description.

6. **Token leakage.** The token is set on `AdminClient` at startup and
   never echoed in tool descriptions or response payloads. Tests
   assert that with `MUSE_ADMIN_TOKEN=secret-test-token` set, no
   captured tool-call output contains the literal "secret-test-token".

7. **httpx timeouts.** Default 600s for inference tools (long enough
   for video gen on slow boxes), 30s for admin tools (admin ops are
   fast or async). Per-tool override is filed for v0.30.x.

## Migration

Zero migration cost. New CLI subcommand; no existing behavior changes.
The `mcp` package is added to `muse[server]` extras so a fresh
`pip install muse[server]` gets it. Users who already have muse
installed get MCP support after a `pip install -U muse[server]`.

## Why this matters

Muse already has 15 modalities and 11 admin operations. Without an
MCP server, an LLM client (Claude Desktop, Cursor, ...) has no way to
call them: the OpenAI-compat interface needs structured tool definitions
the LLM can pick from, not raw HTTP routes.

`muse mcp` closes that gap. With it:
- A user in Claude Desktop says "generate an image of a cat, upscale
  it 4x, then describe what you see." Claude picks `muse_generate_image`,
  feeds the result into `muse_upscale_image`, and feeds the upscaled
  image back to itself for the description.
- A power user runs `muse mcp --filter admin` in a separate Claude
  agent and uses natural language to manage their model catalog: "pull
  the new Qwen3.5-9B GGUF, then disable the old 4B."
- Cursor uses muse for embeddings during code-search via
  `muse_embed_text`.

This makes muse the first-party way to wire any number of LLM clients
to any number of modalities, with no glue code per client.

## Out of scope (filed for v0.30.x and later)

- SSE streaming for chat completions through MCP.
- WebSocket-based long-running ops with progress.
- MCP resources (model catalog as a resource) and prompts (capability
  advertisements).
- Multi-server federation.
- Embedded muse server (boot the gateway in-process under MCP).
- Per-tool rate limiting / quotas.
- Tool composition pipelines.
- OAuth-style auth between MCP and muse.
- Structured outputs for tools (`outputSchema` field).
