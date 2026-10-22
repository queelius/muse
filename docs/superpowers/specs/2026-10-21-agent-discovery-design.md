# Agent discovery and model provenance

Date: 2026-10-21
Status: draft for owner review

## Why

The owner is rebuilding the YouTube pipeline (`~/github/youtube-channel`,
engine in `engine/companion/`) so that a frontier coding agent (Claude Code,
Codex, or another harness) does more of the work, and so the videos can use
better generative models. Muse is the pipeline's speech, alignment, audio
quality, music and (in principle) image server.

Today an agent learns what muse offers only from prose: the pipeline's
`docs/muse-contract.md`, which is hand-written from the client code and lists
fixed model ids, the Kokoro voices in use, and the host `192.168.0.204:8000`.
Nothing tells the agent that muse also has forced alignment details, vision
chat, OCR, detection, vectorization, or that a better model was just added.
The pipeline's own open questions for muse ask for exactly this:

1. Can responses report a model revision (weights hash or version) so frozen
   speech records can pin it?
2. Is there a capabilities endpoint listing models, voices, and modalities, so
   the pipeline can fail fast before a long batch?

This project answers those two. It is the first of four muse projects agreed
with the owner, in this order: (1) discovery and provenance, (2) images (a
modern model with 16:9 sizes and multi-reference edits), (3) narration
(expressive multi-speaker TTS, voice registry, pronunciation, synthesis
timings, stronger ASR checks), (4) a vision critic for rendered frames.
Discovery comes first so the agent can find each later addition without
pipeline rewiring.

## Who uses it, and how

Both of these, from one source of facts:

- **Live use mid-session (mostly MCP).** The agent asks what muse can do right
  now and calls models as tools. It needs current state (what is in memory,
  what is queued, what a switch costs) and tool descriptions that say what each
  model is good at.
- **Writing pipeline code (mostly HTTP).** The agent reads the server once and
  writes code that runs later, possibly against another muse box. It needs
  stable route contracts, OpenAI-compatible shapes where they exist, copyable
  examples, and capability flags the code can check at run time instead of
  hardcoding model ids.

Often a human is in the loop. The agent makes the subtle choices; muse supplies
honest, current facts.

## Goals

1. One machine-readable description of what a muse server hosts: modalities,
   routes, models, what each is good for, how to call it, and its current cost.
2. Reachable from only the server URL over HTTP, and through a thin local MCP
   program that is configured with just that URL, with the same content on
   both.
3. A complete OpenAPI document that includes every inference route.
4. Every inference response says which weights produced it.
5. Works the same through the federation coordinator, across nodes.

## Non-goals

- Muse does not rank candidates, plan pipelines, or recommend a model for a
  task. It states facts; the agent decides.
- No predicted latency model. Muse reports measured facts (last cold-load time,
  memory, queue depth) and the agent judges.
- No A2A agent card. Muse is a tool server, not an agent.
- No new inference features here (synthesis word timings for Kokoro, new
  models, pronunciation lexicons). Those belong to the later projects; this
  project only makes whatever exists discoverable.
- No changes to the YouTube pipeline repos. They are read-only to muse work;
  the owner updates the pipeline separately.

## Anti-goals (would make this a failure even if it works)

- HTTP and MCP disagreeing about what a model can do.
- A hand-maintained registry that drifts from the code: route lists, flags and
  state must be derived from what muse actually mounts and loads.
- Breaking existing OpenAI-SDK clients of `/v1/models` or any inference route.
- A discovery payload so large or noisy (internal knobs like `gguf_file`,
  `n_threads`, `allow_patterns`, `torch_compile`) that an agent wastes context
  or mistakes an internal knob for a feature.
- Leaking the admin token or admin-only data through a public endpoint.

## What gets built

### 1. Per-modality and per-model descriptions (the one source)

- Each modality package exports a literal `DISCOVERY` dict (AST-read, like
  `MODEL_OPTIONAL_PATHS` and `CONCURRENCY_CEILING`, so the gateway reads it
  without importing heavy code): a one-paragraph `summary`, `use_for` (short
  list), `not_for` (short list), and per-route entries
  `{method, path, request_content_type, response_content_type, compat, example}`.
  `compat` names the standard the route follows (`"openai:audio.speech"`,
  `"cohere:rerank"`, `"libretranslate"`, or `null` for muse-native).
  `example` is a minimal working request (curl-style or JSON body) using a
  placeholder model id.
- Model manifests (bundled `MANIFEST`) and curated entries gain optional
  agent-facing fields: `summary` (one line), `strengths`, `limits` (short
  lists), and `quality_tier` (`"best" | "good" | "draft"`). The tier is an
  absolute judgment within the modality across all bundled and curated models,
  so it is plain data and needs no per-server ranking or federation
  reconciliation. These are hand-written: the builder drafts them for every
  bundled script and curated entry, from model cards and the 2026 roundups
  already gathered, and lists them in one table in the PR for the owner to
  review before merge. Resolver-pulled models without them fall back to
  `description` and no tier.
- Which capability keys are public: discovery shows feature flags and limits an
  agent can act on (`supports_*`, `voices`, `max_*`, `min_*`, `default_*`,
  `sample_rate`, `dimensions`, `supported_languages`, `context_length`, ...)
  and hides runtime knobs. Builder's call how to express this (an allowlist of
  key patterns, or a hidden-keys list); it must be one list in one place, with
  a test that fails when a new capability key appears that is in neither.

### 2. `GET /v1/capabilities` (HTTP)

Public, same auth as `/v1/models` (none). Served by the gateway; a single
worker serves the same shape for its own models. Content:

- `server`: muse version, node name, and links to `/openapi.json`,
  `/v1/models`, `/llms.txt`, and a one-line note on installing the local MCP
  program (`muse[mcp]`).
- `modalities`: for each mounted modality, its `DISCOVERY` content plus the
  ids of models serving it.
- `models`: for each enabled model:
  - identity: `id`, `modality`, `hf_repo`, `revision`, `license`;
  - description: `summary`, `strengths`, `limits`, `quality_tier`;
  - public capabilities (filtered as above), including `voices`;
  - state and cost: `loaded`, `memory_gb` (the same sizing-ladder number the
    director uses), `last_cold_load_seconds` (from telemetry; `null` when
    unknown), `queue_depth` (`null` when no gate is bound), `unservable_reason`.
- Query filters: `?modality=audio/speech` and `?model=<id>`, so an agent can
  fetch a small slice.

`/v1/models` stays as it is for OpenAI-SDK compatibility, except that it gains
`revision` per entry (an additive field).

### 3. Model revision on every response

- Revision source for the primary artifact, in order: the catalog's recorded `revision`; else the HF
  snapshot commit recoverable from the model's `local_dir`
  (`.../snapshots/<40-hex>`); else the manifest's pinned `revision`; else
  `null`. Old pulls (whisper-base, the aligner, utmos and audiobox on the dev
  host have no recorded revision) must still resolve through the snapshot path.
- Models built from several artifacts (a LoRA on a base, a separate
  `code_revision`, more than one repo in `artifact_provenance`): the
  capabilities document lists every component as `{role, repo, revision}`,
  and responses carry a third header `X-Muse-Model-Artifacts`, a short sha256
  digest over the sorted component list, so a frozen record can pin the whole
  combination. Single-artifact models omit that header.
- The gateway adds `X-Muse-Model` and `X-Muse-Model-Revision` response headers
  to every forwarded inference response, buffered or streamed, because many
  responses are binary (WAV, PNG, MP4) and have no body field to carry it. No
  per-modality code changes for this.
- JSON responses that already follow OpenAI chat shape also set
  `system_fingerprint` to `<model-id>@<revision12>` (OpenAI's own field for
  "which backend produced this"). Other JSON bodies are not changed.
- `revision` also appears in `/v1/capabilities` and `/v1/models`.

### 4. Complete OpenAPI

The gateway's `/openapi.json` includes every modality route with its request
schema, by building each modality router against an empty registry and merging
its paths (routers import only fastapi, pydantic, numpy and sse_starlette at
module top, so this stays cheap; `model_3d_generation/routes.py` imports PIL,
which `muse[server]` does not declare, so that import becomes lazy or the merge
tolerates a router that fails to build). Route operations carry the `DISCOVERY`
summary as their description. The catch-all proxy stays hidden from the
schema. Admin and telemetry paths stay listed as today.

### 5. Entry points an agent can find from the URL alone

- `GET /` on the gateway returns a short JSON pointer document (or redirects;
  builder's call) naming `/v1/capabilities`, `/openapi.json` and `/llms.txt`.
- `GET /llms.txt`: a short generated markdown page (from the same source) that
  says what the server is, lists modalities with one line each and their
  routes, and points at `/v1/capabilities` and `/openapi.json`. Cheap, and some
  agent harnesses read it; it is not a substitute for the JSON.
- `/.well-known/muse.json`: the same pointer document as `GET /`. (Builder may
  drop this if it adds nothing over `/`.)

### 6. MCP: a thin local program generated from the server

The common case is Claude Code on one machine using a muse server on another.
Nearly every useful call involves a file on the agent's machine (narration
takes to align or check, frames to critique, reference images to edit), and
media outputs are too large for the agent's context (Claude Code caps MCP
output at about 25k tokens by default; base64 images have blown past it). So
MCP stays a separate local program and is not served from the muse port.

- **Shape.** `muse mcp` remains an HTTP client of one server (`MUSE_SERVER`,
  a single node or a federation coordinator). It is rewritten to be small: at
  startup and on `list_tools` it reads the server's `/v1/capabilities` and
  `/openapi.json` (cached; refreshed when the server's document changes) and
  generates one tool per inference route from the route's request schema and
  `DISCOVERY` entry. New modalities and models therefore appear with no MCP
  code changes, and MCP cannot disagree with HTTP. The hand-written per-route
  tool modules (`mcp/tools/inference_*.py`) are replaced. Rough target: a few
  hundred lines for the generator plus file I/O.
- **Tool names.** Each route's `DISCOVERY` entry names its MCP tool
  (`mcp_tool: "muse_speak"`), so today's names (`muse_speak`,
  `muse_transcribe`, `muse_generate_image`, ...) stay stable. Routes that had
  no tool (alignment, audio quality, OCR, depth, keypoints, detection,
  vectorization, 3D, classification) get one automatically.
- **`muse_describe`.** One extra tool returning the capabilities document with
  the `modality` / `model` filters. Its description tells the agent to call it
  before choosing models. Also exposed as MCP resources `muse://capabilities`
  and `muse://models/{id}`.
- **File inputs.** Upload fields (multipart `file`, `image`, `mask`, and image
  URL fields in JSON bodies) become `<name>_path` arguments that read a local
  file and upload it; `<name>_url` stays for http(s) and data URLs. Base64
  inputs are dropped from generated tools. Local reads are confined to
  `fetch.mcp_allowed_path_prefixes` (default: the MCP process's working
  directory when unset; builder confirms the current default and keeps it at
  least that strict).
- **Outputs go to files.** Every media output (audio, image, video, 3D, SVG,
  raw depth) is written to the agent's `out_path`, else to a default output
  directory (`mcp.output_dir`, default `./muse-output/`) under a
  content-hash name. The tool result is a short text summary plus the same
  facts in `structuredContent` (declared via `outputSchema`) and a
  `resource_link` to the file: path, sha256, MIME type, bytes, duration or
  pixel size, model, revision, and the artifacts digest when present. That is
  the shape a provenance record or a content-addressed media store needs.
  JSON-only responses (transcripts, alignments, scores, embeddings) are
  returned as `structuredContent` with a text summary; very large ones
  (embeddings, long word lists) are also written to a file and summarized.
- **Image preview.** Image-producing tools take `preview: true` to also
  return one downscaled inline image (long side 512 px), so the agent can look
  at a result without loading the full file. Off by default. Audio is never
  returned inline.
- **Admin tools.** The existing small fixed set of admin tools stays as it is
  (wrapping `AdminClient`), under `--filter admin|all`. They are not
  generated.
- **When the server is unreachable.** `list_tools` returns only
  `muse_describe` (which reports the connection error) and the admin tools;
  no stale generated tools.
- **Install.** A new extra `muse[mcp]` with only `httpx` and `mcp`, and
  `muse mcp` (stdio) must import nothing from `muse[server]`. Setup for
  Claude Code is one line, documented in `docs/AGENT_DISCOVERY.md`:
  `claude mcp add muse -e MUSE_SERVER=http://host:8000 -- uvx --from "muse[mcp]" muse mcp`.
  The `--http` transport of `muse mcp` is kept but documented as the
  less-preferred option, since it cannot read the agent's files.

### 7. Federation

The coordinator serves `/v1/capabilities` by merging node documents: models
unioned by id with a `nodes` list per model (each node's `loaded`, queue depth
and revision), modalities unioned. If two nodes report different revisions
for one model id, both are shown; the coordinator does not pick. Revision
headers pass through unchanged from the node that served the request.
The coordinator also serves `/` and `/llms.txt`, generated from the merged
document. Its `/openapi.json` is the schema of one reachable node (nodes
normally run the same muse version), documented as such; merging schemas
across versions is deferred.

### 8. Docs

`docs/AGENT_DISCOVERY.md` explains the surface for agent and pipeline authors,
with a "first three calls" example for HTTP and for MCP. `CLAUDE.md` gets a
short section.

## Constraints

- Follows existing seams: AST-read literal exports from modality packages,
  `build_model_entry` for model entries, the sizing ladder and director state
  for cost fields, the deferred-imports rule (`muse --help` must not import
  fastapi).
- Discovery reads must not block the gateway event loop or trigger model loads.
- Public endpoints never include admin token material, job records, or worker
  pids.
- Additive only for existing clients.

## Left to the builder

- Exact JSON field names beyond those named above, and envelope details.
- How `DISCOVERY` examples are validated (a test that each example parses
  against the route's request model is strongly suggested).
- Caching of the capabilities document and how it invalidates (catalog mtime
  plus director state is the obvious key).
- Whether the per-model descriptive fields live in curated.yaml, manifests, or
  both, as long as there is one source per model.
- Order of implementation inside this project.

## Calls made by the author (owner to check)

- Hand-written `quality_tier` within modality, not benchmark numbers.
- Revision carried in response headers for all responses, plus
  `system_fingerprint` for chat; no other body changes.
- `/v1/capabilities` is public like `/v1/models`, including queue depth.
- `llms.txt` included as a cheap extra.
- MCP is a thin local program generated from the server's description, not
  served from the muse port; outputs go to files with an optional image
  preview; base64 inputs dropped from generated tools.
- Existing MCP tool names kept via a per-route `mcp_tool` name.
- No A2A agent card.
