# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Muse is a multi-modality generation server and client. It currently supports
six modalities:

- **audio/speech**: text-to-speech via `/v1/audio/speech` (Soprano, Kokoro, Bark)
- **audio/transcription**: speech-to-text via `/v1/audio/transcriptions` and `/v1/audio/translations` (Systran faster-whisper family; any CT2 Whisper on HF)
- **chat/completion**: text-to-text LLMs via `/v1/chat/completions` (OpenAI-compatible incl. tools + streaming; powered by llama-cpp-python; any GGUF on HF via the resolver)
- **embedding/text**: text-to-vector via `/v1/embeddings` (MiniLM, Qwen3-Embedding, NV-Embed-v2; any sentence-transformers HF repo via the resolver)
- **image/generation**: text-to-image via `/v1/images/generations` (SD-Turbo, SDXL-Turbo, FLUX.1-schnell, any diffusers HF repo)
- **text/classification**: text moderation/classification via `/v1/moderations` (any HuggingFace text-classification model)

Modality tags are MIME-style (`audio/speech`, not `audio.speech`). The HTTP
path hierarchy still mirrors OpenAI (`/v1/audio/speech`,
`/v1/chat/completions`, `/v1/embeddings`, `/v1/images/generations`) for
client compatibility.

The package is organized around three plugin surfaces:

- `src/muse/modalities/<mime_name>/`: self-contained wire contract
  (protocol + routes + codec + client). Each modality package exports
  `MODALITY: str` + `build_router: Callable[[registry], APIRouter]`.
  Discovered at runtime by `discover_modalities`.
- `src/muse/models/*.py`: flat directory of drop-in model scripts.
  Each `.py` file declares `MANIFEST: dict` + a `Model` class.
  Discovered at runtime by `discover_models`. Best for one-off models
  with custom code (NV-Embed, Soprano).
- `muse.core.resolvers.*`: URI-addressable model sources. `muse pull
  hf://Qwen/Qwen3-8B-GGUF@q4_k_m` synthesizes a manifest, persists it
  in `catalog.json`, and routes requests through a generic runtime class
  (`LlamaCppModel` for GGUF, `SentenceTransformerModel` for ST embedders).
  Best for uniform model classes where one runtime serves many models.
  See `docs/RESOLVERS.md` and `docs/CHAT_COMPLETION.md`.

A modality-agnostic core (`muse.core`) holds the registry, discovery,
resolver dispatch, HF downloader, per-venv pip install, and FastAPI
app factory.

## Architecture

```
HTTP API (/v1/audio/speech, /v1/images/generations, /v1/models, /health)
    |
    v
muse.core.server   (FastAPI factory, mounts per-modality routers)
    |
    v
muse.core.registry (ModalityRegistry: {modality: {model_id: Model}})
    |
    v
Modality backends implementing modality-specific protocols
```

### Key modules

- `muse.core.discovery`: scans directories and returns `{model_id:
  DiscoveredModel}` (for model scripts) and `{mime_tag: build_router}`
  (for modality packages). First-found-wins on collisions; script
  errors are logged, never raised. Bundled scripts in the installed
  `muse/models/` tree get their canonical Python import name
  (`muse.models.<stem>`); external scripts get a mangled private name
  to avoid sys.modules collisions.
- `muse.core.catalog.known_models()`: discovery-driven, cached on first
  call. Projects each script's MANIFEST onto the `CatalogEntry` shape
  the rest of muse consumes (backend_path is synthesized from the
  Model class's `__module__:__name__`). Merges two sources: discovered
  bundled scripts PLUS catalog.json entries that carry a persisted
  `manifest` field (resolver-pulled models). Bundled wins on collision.
  `pull()` dispatches by identifier shape: curated alias > `://` URI
  > bare id. `get_manifest(model_id)` prefers the persisted manifest
  for resolver-pulled models, falls back to the script module's
  MANIFEST. Catalog state lives at `~/.muse/catalog.json` (or
  `MUSE_CATALOG_DIR` env override); writes are atomic (write-then-rename).
- `muse.core.resolvers` + `muse.core.resolvers_hf`: URI -> `ResolvedModel`
  dispatch for `muse pull hf://...`. `HFResolver` sniffs each HF repo
  (`.gguf` siblings -> `chat/completion` / LlamaCppModel; sentence-
  transformers tag -> `embedding/text` / SentenceTransformerModel).
  GGUF `@variant` is required; no magic default. Search implemented for
  both modalities with per-variant deduping (sharded GGUFs don't emit
  one row per shard). Registers `hf://` scheme on import.
- `muse.core.curated`: loads `src/muse/curated.yaml` (hand-edited
  recommendations list). `find_curated(id)` / `expand_curated_pull(id)`.
  Curated entries either alias a bundled script (`bundled: true`) or
  point at a URI; the curated id is preserved as the catalog key even
  when the URI would synthesize a different one, so newbie-friendly ids
  like `qwen3.5-4b-q4` survive end-to-end.
- `muse.core.chat_formats`: loads `src/muse/chat_formats.yaml` (hand-
  edited map from HF repo substring to llama-cpp-python `chat_format`
  string + `supports_tools` flag). Consulted by the HF resolver when
  synthesizing GGUF manifests. First-match-wins; case-insensitive
  substring on `hf_repo`. More-specific patterns must come first.
- `muse.core.registry.ModalityRegistry`: keyed by `(modality, model_id)`.
  First registered model per modality is the default for that modality.
  `register(modality, model, manifest=...)` stores the MANIFEST verbatim;
  `/v1/models` splats `manifest.capabilities` + top-level description
  /license/hf_repo into each entry. No shared protocol base across
  modalities.
- `muse.core.server.create_app(registry, routers)`: builds the FastAPI app
  with shared `/health` and `/v1/models`, mounts per-modality routers, and
  registers the `ModelNotFoundError` exception handler so 404s use the
  OpenAI-style `{"error":{...}}` envelope instead of FastAPI's `{"detail":...}`.
- `muse.core.venv`: venv creation (`create_venv`, `install_into_venv`, `find_free_port`). Each `muse pull` creates `~/.muse/venvs/<model-id>/`; catalog records the `python_path`.
- `muse.cli_impl.worker`: single-worker mode (runs one uvicorn in one venv). Invoked via `muse _worker` (hidden subcommand).
- `muse.cli_impl.gateway`: FastAPI proxy app. Routes by `model` field in request body/query; aggregates `/v1/models` and `/health` across workers.
- `muse.cli_impl.supervisor`: orchestrates workers + gateway. `plan_workers` groups catalog by venv; `spawn_worker` + `wait_for_ready` manage subprocess lifecycle; `run_supervisor` is the entrypoint `muse serve` delegates to.
- `muse.cli_impl.search`: `run_search()` for `muse search`. Thin wrapper over `resolvers.search()` plus log-level quieting so httpx's per-request debug lines don't interleave with the table output.

### Modality conventions

Each modality subpackage (`src/muse/modalities/<mime_name>/`) contains:
- `__init__.py`: exports `MODALITY: str` (MIME-style tag like `"audio/speech"`) and `build_router: Callable[[ModalityRegistry], APIRouter]`. These two are what `discover_modalities` scans for.
- `protocol.py`: Protocol + Result dataclass(es) for this modality
- `routes.py`: defines `build_router(registry) -> APIRouter`
- `client.py`: HTTP client for this modality's endpoints
- `codec.py`: modality-specific encoding (wav/opus for audio; png/jpeg for images; base64 float32 for embeddings; SSE+OpenAI chunk shape for chat)
- `runtimes/` (optional): *generic* runtime classes that serve many models from one implementation. `chat_completion/runtimes/llama_cpp.py:LlamaCppModel` wraps any GGUF; `embedding_text/runtimes/sentence_transformers.py:SentenceTransformerModel` wraps any sentence-transformers repo. Runtime class paths are referenced by resolver-synthesized manifests.
- `backends/` (optional): *private helpers* used by this modality's own model scripts. NOT a plugin surface. Only `audio_speech/backends/` exists (`base.py` with `voices_dir` + `BaseModel`; `transformers.py` with the Narro engine Soprano delegates to).
- `audio_transcription/` is muse's first modality with multipart/form-data uploads (OpenAI Whisper wire shape). `routes.py` handles UploadFile + Form fields inline. If a second multipart modality lands (images/edits, audio-conditioned audio/generation), factor out to `muse.modalities._common.uploads`.
- `text_classification/` is muse's first modality whose internal MIME tag (`text/classification`) is broader than its primary URL route (`/v1/moderations`). The wire path is OpenAI-specific; the modality tag is broad enough to host future routes (`/v1/text/classifications` for sentiment/intent) sharing the same runtime + dataclasses without a new modality package.

Three distinct concepts worth keeping straight:

| Surface | Who writes it | Purpose |
|---|---|---|
| `muse/models/*.py` | bundled muse + users | public model scripts, one per model, discoverable |
| `modalities/*/runtimes/*.py` | muse internal | generic runtimes, one class serves many models (GGUF, ST) |
| `modalities/*/backends/*.py` | muse internal | private helpers shared inside a modality |

Each model script (`src/muse/models/<id>.py`) contains:
- Top-level `MANIFEST: dict` with required keys `model_id`, `modality`, `hf_repo` and optional `description`, `license`, `pip_extras`, `system_packages`, `capabilities`. Anything else passes through.
- A class named exactly `Model` (tests alias it: `from muse.models.kokoro_82m import Model as KokoroModel`).

Each `Model` class:
- Satisfies the modality's Protocol structurally (no base class required).
- Accepts `hf_repo=`, `local_dir=`, `device=`, `**_` in its constructor (the catalog loader calls with those kwargs; `**_` absorbs future additions).
- Prefers `local_dir` over `hf_repo` when loading weights.
- Defers heavy imports (torch, transformers, diffusers, kokoro, llama_cpp) to inside `__init__` (via an `_ensure_deps()` helper that lazy-imports into module-level sentinels). Tests patch the module-level names directly; the `_ensure_deps` check `if X is None` short-circuits when tests have pre-populated mocks. `muse --help` and `muse pull` must work without any ML deps installed.

### Capability precedence

For resolver-pulled GGUFs, the `chat_format` and `supports_tools` fields
in the catalog's persisted manifest come from layered lookups:

1. MANIFEST `capabilities.chat_format` (user-set explicitly) -- highest
2. `src/muse/chat_formats.yaml` pattern match on `hf_repo` at resolve time
3. None -- falls through to llama-cpp-python's GGUF-metadata autodetection

At `load_backend` time, `manifest.capabilities` is merged into the
runtime constructor's kwargs (caller kwargs win). This lets generic
runtimes like `LlamaCppModel` receive `gguf_file`, `chat_template`,
`context_length`, etc. without the worker layer knowing those keys
exist. The generic runtime also gets `model_id` injected, since one
class serves many models.

### No shared supertype across modalities

`AudioResult` and `ImageResult` do NOT share a common base. Streaming semantics
differ (audio chunks are time-ordered and playable immediately; diffusion steps
are progressive refinement of one frame). A `GenerationModel` abstract base
would be a leaky abstraction. Instead, `ModalityRegistry` treats models as
`Any`, and each modality's router + codec knows its own types.

## Process model

`muse serve` is a **supervisor**, not a single process:

```
User request
    |
    v
muse serve (supervisor, port 8000)
  ├── gateway FastAPI app (in-process)
  │    routes by request body `model` field
  │
  └── subprocess per venv group:
       ├── worker (port 9001, venv-A) hosts soprano-80m, kokoro-82m
       ├── worker (port 9002, venv-B) hosts bark-small
       └── worker (port 9003, venv-C) hosts sd-turbo
```

Each pulled model gets its own venv at `~/.muse/venvs/<model-id>/`
with exactly the pip_extras it declares. Workers run the existing
`muse.cli_impl.worker.run_worker` logic via `muse _worker`
(hidden subcommand). The supervisor spawns them with each venv's
Python interpreter, polls `/health` until ready, then runs the gateway.

The gateway extracts `model` from the request body (POST) or query
(GET), looks up which worker hosts it, and forwards the request,
streaming SSE through without buffering. `/v1/models` and `/health`
are aggregated across all workers via parallel httpx calls.

This gives you dep isolation (transformers 4.46 for parler-tts
coexists with transformers 5.x for newer models), crash isolation
(a segfault in one worker does not kill the rest), and a uniform
HTTP surface (clients hit one port, do not care about internal venvs).

The supervisor also runs an auto-restart monitor thread. Every 5
seconds it polls each worker's /health and checks for process death
via Popen.poll. After 3 consecutive failures (or immediate process
exit), the monitor terminates the existing process and respawns it
with exponential backoff (1s, 2s, 4s, ..., capped at 30s). After 10
unsuccessful restart attempts the worker is marked dead; /health
reports "degraded" and /v1/models skips its entries.

Use `muse models disable <id>` to mark a pulled model as inactive
(supervisor skips it at plan_workers time, freeing its venv's memory
budget). `muse models enable <id>` re-enables it. Neither command
restarts the server; the change takes effect next `muse serve`.

## Development commands

```bash
# Install (dev)
pip install -e ".[dev,server,audio,images]"

# Fast lane: unit tests only (excludes slow e2e + integration)
pytest tests/ -q -m "not slow"

# Full lane: fast + slow e2e supervisor test (in-process)
pytest tests/ -q

# Integration tests (opt-in, hit a real muse server)
MUSE_REMOTE_SERVER=http://192.168.0.225:8000 pytest tests/integration/
# Override which chat model the integration suite targets:
MUSE_REMOTE_SERVER=http://192.168.0.225:8000 \
    MUSE_CHAT_MODEL_ID=qwen3.5-9b-q4 pytest tests/integration/

# One modality contract, or one bundled model
pytest tests/modalities/chat_completion/
pytest tests/models/test_kokoro_82m.py

# Single test by name
pytest tests/core/test_resolvers.py::test_register_and_get_resolver -v

# Coverage
pytest tests/ -m "not slow" --cov=muse

# CLI: admin surface (no per-modality verbs; generation is HTTP)
muse --help                                    # top-level help
muse models list                               # bundled + curated + pulled
muse models list --available --modality chat/completion
muse search qwen3 --modality chat/completion --max-size-gb 10
muse pull qwen3.5-4b-q4                        # curated alias
muse pull hf://unsloth/Qwen3.5-9B-GGUF@q4_k_m  # resolver URI
muse pull kokoro-82m                           # bundled bare id
muse models info <id>
muse models enable <id> / disable <id>
muse models remove <id>
muse serve --device cuda

# Python clients (HTTP)
python - <<'PY'
from muse.modalities.audio_speech import SpeechClient
from muse.modalities.image_generation import GenerationsClient
from muse.modalities.embedding_text import EmbeddingsClient
from muse.modalities.chat_completion import ChatClient
SpeechClient().infer("hello")                         # WAV bytes; MUSE_SERVER env sets base URL
GenerationsClient().generate("a cat")                 # list[bytes] of PNGs
EmbeddingsClient().embed(["alpha", "beta"])           # list[list[float]]
ChatClient().chat(model="qwen3.5-4b-q4", messages=[{"role": "user", "content": "hi"}])
# or via OpenAI SDK (muse is wire-compatible):
#   OpenAI(base_url="http://localhost:8000/v1", api_key="not-used")
PY
```

## Project-specific conventions

- **Deferred imports:** `src/muse/__init__.py` and `src/muse/cli.py` MUST NOT
  import heavy libs (torch, diffusers, transformers). Each backend imports
  its heavy deps at module top-level inside a try/except so import of the
  backend module succeeds even without the deps. Tests mock at the module
  path where the library is imported. `muse --help` and `muse pull` work
  without any ML deps installed; pulling a model installs them on demand.
- **FakeModel-pattern tests:** Server and router tests use plain classes that
  satisfy the modality protocol, no real weights. Backend tests also mock
  heavy libs (see `tests/models/test_sd_turbo.py`).
- **Registry is a singleton at module level** (`muse.core.registry.registry`),
  but tests create their own `ModalityRegistry()` instances to avoid coupling.
- **Audio is float32 in `[-1, 1]`** at the protocol boundary; codec converts
  to int16 PCM at output. Scaling uses `* 32768` + `np.clip` to reach full
  int16 range `[-32768, 32767]`.
- **Images are `Any`** at the protocol boundary; codec normalizes PIL / numpy /
  torch to PIL before encoding.
- **OpenAI error envelopes:** Use `raise ModelNotFoundError(model_id, modality)`
  from `muse.core.errors`, not `HTTPException(detail=...)`. The former gives
  `{"error":{"code","message","type"}}`; the latter gives `{"detail":...}`.
- **Streaming uses producer thread + `asyncio.Queue`**, not `list(generator)`.
  Synthesis chunks must dispatch as they're produced, not after full generation.
- **Env vars:** `MUSE_SERVER` (client base URL), `MUSE_CATALOG_DIR` (catalog
  location, defaults `~/.muse/`), `MUSE_HOME` (voices dir base).
- **Auto-restart is always on.** No --no-autorestart flag in this iteration. Workers that can't stay up through 10 restart attempts are marked dead; manual restart via `Ctrl+C` + `muse serve` is required to reset the counter.
- **Enable/disable is catalog state**, not runtime state. `muse serve` reads the catalog at startup. Changing a model's enabled bit while the server is running has no effect until the next restart.
- **Tool-use asymmetry (known landmine).** llama-cpp-python's `chatml-function-calling` handler parses tool calls *out* of a model's response into structured `tool_calls`, but does NOT format tool *result* messages (role=`tool`) back to the model in a way Qwen's chat template always recognizes. The muse-side contract is correct (verified by `tests/modalities/chat_completion/test_routes_messages_passthrough.py`); the asymmetry is upstream. Larger models (Qwen3.5-9B+) tolerate it in context; smaller models (Qwen3.5-4B) often ignore the tool result and give a generic "I don't have access to tools" reply. Tracked by `tests/integration/test_remote_tools.py::test_observe_tool_result_content_influences_next_response` (xfail-style watchdog). Upstream: [abetlen/llama-cpp-python#2063](https://github.com/abetlen/llama-cpp-python/issues/2063).
- **The `model` field in chat responses is the catalog id**, not the GGUF filesystem path. `LlamaCppModel._dict_to_chat_result` and `_dict_to_chat_chunk` override `response["model"]` with the muse catalog id (not the `resp.get("model") or fallback` pattern that lets llama-cpp's internal `model_path` win). Applies to both non-streaming responses and every streaming chunk.

## Adding a new model (the common case)

Three paths, in order of least-to-most effort:

**1. Curated alias (easiest; a muse-blessed shortcut).** Edit
`src/muse/curated.yaml` to add a friendly id that points at an HF URI
or a bundled script. Users then `muse pull <id>` (no `hf://` prefix
needed). The curated id is preserved as the catalog key even when the
URI would synthesize a different one. See existing entries for shape.

**2. Resolver URI (good for GGUF + sentence-transformers).** No script
needed; let the HF resolver synthesize a manifest:

```bash
muse pull hf://unsloth/Qwen3.5-9B-GGUF@q4_k_m
muse pull hf://sentence-transformers/all-MiniLM-L6-v2
```

The HF resolver sniffs the repo (`.gguf` -> chat/completion via
`LlamaCppModel`; sentence-transformers tag -> embedding/text via
`SentenceTransformerModel`), persists a synthesized manifest in
`~/.muse/catalog.json`. For chat/completion, the resolver also
consults `src/muse/chat_formats.yaml` for the right llama-cpp
`chat_format` string. `muse search <query> --modality chat/completion
--max-size-gb N` helps discover candidates. See `docs/RESOLVERS.md`.

**3. Script path (for one-offs with custom code).** Models that need
non-uniform behavior (NV-Embed's custom `encode` method, Soprano's
Narro engine) get a hand-written script:

1. Write a `.py` file with a `MANIFEST` dict + `Model` class (see
   `docs/MODEL_SCRIPTS.md` for the full schema).
2. Drop it in `~/.muse/models/` or any dir pointed to by `$MUSE_MODELS_DIR`.
3. `muse pull <model_id>` to install deps + download weights.
4. `muse serve` picks it up.

Bundled model scripts live in `src/muse/models/<id>.py`. Adding a
bundled model requires no catalog edits, no registry changes, and no
worker changes: discovery just finds it.

Collision precedence: bundled scripts > resolver-pulled (persisted
manifest). A user pulling `hf://malicious/fake` that claims an
existing bundled id gets shadowed by the bundled script.

## Adding a new modality (rare)

Modalities define wire contracts; most users should NOT need to add one.
If you do:

1. Create `src/muse/modalities/<mime_name>/` (e.g. `audio_transcriptions/`
   for MODALITY `"audio/transcription"`). Use underscores in the dir
   name; the MIME tag has the slash.
2. Write `protocol.py` (Protocol + Result dataclass), `routes.py`
   (with `build_router(registry) -> APIRouter`), `client.py` (HTTP
   client), and `codec.py` (encoding for this modality's output).
3. Export from `__init__.py`: `MODALITY = "audio/transcription"` (the
   MIME string) and `build_router` (the router factory). Also re-export
   the Protocol + Result for user imports.
4. (HF support) write `hf.py` exporting `HF_PLUGIN: dict` (sniff/
   resolve/search + metadata). See `docs/HF_PLUGINS.md` for the
   contract and authoring rules. Loaded via single-file import,
   so no relative imports.
5. Add bundled model scripts under `src/muse/models/` (or rely on
   the resolver alone for uniform-shape modalities).
6. Add tests under `tests/modalities/<mime_name>/` (route + plugin)
   and `tests/models/test_<new_model>.py`.

No edits to `worker.py`, `catalog.py`, `registry.py`, `server.py`,
or `resolvers_hf.py` are needed: discovery handles the wiring.

No gateway changes are needed either: the gateway routes by the
`model` field in the request body and forwards to whichever worker
loaded that model. New modalities are transparent to the proxy layer.

External escape hatch: dropping a modality subpackage into
`$MUSE_MODALITIES_DIR/<mime_name>/` registers it without forking muse.
Intended for experimentation, not routine extension.

## Test organization

```
tests/
├── core/                 # resolvers, catalog, curated, chat_formats, discovery, registry, server
├── modalities/<name>/    # protocol + codec + routes + client for each modality
├── models/               # one test per bundled model script (fully mocked)
├── cli_impl/             # worker, search, supervisor (in-process e2e marked @pytest.mark.slow)
├── integration/          # opt-in; hits a real muse server via OpenAI SDK
└── test_cli.py           # subprocess-level CLI smoke
```

Fast lane is `-m "not slow"`. The one `@pytest.mark.slow` test in
`cli_impl/test_e2e_supervisor.py` spawns a real supervisor subprocess
(in-process, no network). The `tests/integration/` suite is separately
opt-in via `MUSE_REMOTE_SERVER` env var; fixtures auto-skip when the
server isn't reachable or the required model isn't loaded. The
`MUSE_CHAT_MODEL_ID` env var (default `qwen3.5-4b-q4`) lets the same
integration suite run against any chat model on the target server.

Test naming for integration tests:
- `test_protocol_*`: hard claims muse should always satisfy. Failure is a regression.
- `test_observe_*`: records what a particular model actually did. Usually xfail-style; useful as a watchdog (a passing xfail shows up as XPASS, signaling "upstream fixed something; promote to a hard assertion").
