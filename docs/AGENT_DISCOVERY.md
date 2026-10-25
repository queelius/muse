# Agent discovery and model provenance

A muse server describes itself, from only its URL, so an AI agent (or a
pipeline an agent is writing) never has to learn muse from prose. Every
inference response also says which weights produced it.

Design: `docs/superpowers/specs/2026-10-21-agent-discovery-design.md`.

## Endpoints

All are public (no token), read-only, and never trigger a model load.

| Endpoint | Returns |
|---|---|
| `GET /` | Pointer document: links to `/v1/capabilities`, `/openapi.json`, `/llms.txt`, `/v1/models`. Not proxied to a worker. |
| `GET /v1/capabilities` | The main document: `server` (version, node, links, MCP install line), `modalities[]` (summary, `use_for`, `not_for`, routes with request/response content types, an example, the MCP tool name, file fields), and `models[]` (id, modality, license, `revision`, `components`, one-line `summary`, `strengths`, `limits`, `quality_tier` of best/good/draft, filtered capability flags, memory estimate, and live state: loaded, last cold-load seconds, queue depth, `unservable_reason`). Only enabled models are listed. |
| `GET /v1/capabilities?modality=audio/speech` | The same document narrowed to one modality. `?model=<id>` narrows to one model. An unknown value is a 404. |
| `GET /openapi.json` | Complete OpenAPI for every inference route (alignments, vectorize, systemone and the rest), generated from the same declarations. |
| `GET /llms.txt` | A short markdown page rendered from the capabilities document, for agents that read `llms.txt`. |
| `GET /.well-known/muse.json` | Same pointer document as `/`, at a well-known path. |
| `GET /v1/models` | Unchanged OpenAI-shape list, plus a `revision` field per model. |

`quality_tier` is absolute within a modality across bundled and curated
models, so an agent can pick the best model for a task without a table of ids.
Model notes live next to each model (bundled script MANIFEST or curated
entry); a model without notes still appears, with only the facts.

## First three HTTP calls

```bash
# 1. What is this server?
curl -s http://host:8000/

# 2. What speech models and voices exist, and how do I call them?
curl -s 'http://host:8000/v1/capabilities?modality=audio/speech' | jq .

# 3. Call one; read the revision from the response headers.
curl -sD- -o out.wav http://host:8000/v1/audio/speech \
  -H 'content-type: application/json' \
  -d '{"model":"kokoro-82m","input":"hello","response_format":"wav"}' | grep -i x-muse
```

Write pipeline code against capability flags (`supports_*`, `supported_scales`
and so on) from step 2 rather than hardcoding model ids; the flags are the
stable contract and the ids are the choice.

## Provenance headers

Every inference response carries:

| Header | Meaning |
|---|---|
| `X-Muse-Model` | Catalog model id that served the request. |
| `X-Muse-Model-Revision` | 40-hex Hugging Face commit of the weights. Omitted when unknown. |
| `X-Muse-Model-Artifacts` | Short digest over all components (weights plus, for example, a LoRA base) when a model has more than one. Omitted for single-component models. |

Chat completions also set `system_fingerprint` to `<model_id>@<revision[:12]>`
(omitted when the revision is unknown). The revision is recovered from the
HF snapshot path or the persisted pin, and is cached against the catalog
mtime. `GET /v1/capabilities?model=<id>` returns the same value as
`models[0].revision`, so a pipeline can check it before a batch.

Recording provenance in a pipeline: store `{model, revision, artifacts}` from
the three headers next to every generated artifact (the audio file, the
alignment, the image). Frozen records then say exactly which weights made
them, and a later run can compare the current revision to detect drift before
reusing a cached result. With MCP, the same triple is returned in each
tool result (below).

## First three MCP calls

1. Register the local MCP program (only the server URL is needed):

   ```bash
   claude mcp add muse -e MUSE_SERVER=http://host:8000 -- uvx --from "museq[mcp]" muse mcp
   ```

2. Call `muse_describe`: returns the capabilities document for the live
   server. Narrowed by `modality` or `model`, it comes back inline. Unfiltered
   and over 20,000 characters (a full server), the full document is written
   to a file and the result is an inline index: per modality its summary,
   tool names and model ids; per model its id, modality, license,
   quality_tier, summary and loaded state.
3. Call `muse_speak` with `model`, `input` and `out_path`. The result is
   `{path, sha256, model, revision}` for the written file. Pass
   `preview: true` on image tools to also get one inline 512 px PNG.

`muse mcp` is a thin local HTTP client. It reads `/openapi.json` and
`/v1/capabilities` (cached 30 s) and generates one tool per inference route,
so tools track the server: a new route or model appears without a client
release. It lists `muse_describe`, the generated tools, and the 11 admin
tools. If the server is unreachable, only `muse_describe` and the admin tools
are listed (never stale generated tools). The server advertises
`tools.listChanged` and notifies a session once when the tool set changes
(including the server coming up after the client connected), so clients
re-list. A `model` argument is a string whose description lists the serving
models, not an enum, so a model enabled mid-session is accepted. Behaviour:

- Media outputs are written to files and returned as path, sha256, model and
  revision, never inline bytes. `out_path` names the file; otherwise a
  content-named file goes in `mcp.output_dir` (`MUSE_MCP_OUTPUT_DIR`,
  default `./muse-output`, relative to the MCP process cwd; when the cwd is
  `/`, as Claude Desktop launches it, `~/muse-output`). SVG from
  `muse_vectorize_image` is written as an `.svg` file too.
- File inputs take a local `<name>_path`. With
  `fetch.mcp_allowed_path_prefixes` (`MUSE_MCP_ALLOWED_PATH_PREFIXES`) empty,
  paths under the current directory or `mcp.output_dir` are allowed; a cwd of
  `/` is never allowed implicitly. A set value replaces the current-directory
  default with its colon-separated directories; `mcp.output_dir` stays
  allowed for reads and writes either way, so one tool's output file can be
  another tool's input (speak, then align). Multipart `*_url` inputs are fetched by the MCP process through muse's
  SSRF guard; JSON `*_url` inputs are passed to the server, whose SSRF guard
  fetches them. Base64 inputs are not supported.
- Speech with `timestamps: true` (Kokoro only) returns one timing per written
  word in `response.words`; `response.audio.b64_json` becomes the audio file
  path. Models without native timings refuse the request. `pronunciations`
  carries respellings or supported IPA separately from the written input.
- Speech headers surface as `response.voice`, `response.speed`,
  `response.audio_sha256`, and, when supplied, `response.pronunciations_digest`.
- `X-Alignment` from `audio/speech align=true` contains Soprano's paragraph
  timings and is surfaced as `response.alignment`.
- Tool descriptions come from the capabilities document, so they carry the
  model notes.

## Federation

`muse federate` serves the merged view of its nodes: `/v1/capabilities`
(models from every reachable node, each annotated with the nodes that carry
it), `/`, `/llms.txt`, and the first reachable node's `/openapi.json`.
The coordinator's own `/docs` and `/redoc` are disabled. Inference routes
forward as before, and the provenance headers pass through from the node that
served the request. See the federation section of `CLAUDE.md`.
