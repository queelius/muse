# CLAUDE.md

Guidance for Claude Code when working on Muse.

## Project overview

Muse is a multi-modality generation server and client. It currently supports
four modalities:

- **audio/speech**: text-to-speech via `/v1/audio/speech` (Soprano, Kokoro, Bark)
- **image/generation**: text-to-image via `/v1/images/generations` (SD-Turbo)
- **embedding/text**: text-to-vector via `/v1/embeddings` (MiniLM, Qwen3-Embedding, NV-Embed-v2; any sentence-transformers HF repo via the resolver)
- **chat/completion**: text-to-text LLMs via `/v1/chat/completions` (OpenAI-compatible incl. tools + streaming; powered by llama-cpp-python; any GGUF on HF via the resolver)

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
  Model class's `__module__:__name__`). `pull()` creates a per-model
  venv, installs muse editable + `[server]` extras + MANIFEST's
  `pip_extras`, warns on missing `system_packages`, and downloads
  weights from HF. Catalog state lives at `~/.muse/catalog.json` (or
  `MUSE_CATALOG_DIR` env override); writes are atomic (write-then-rename).
  `get_manifest(model_id)` returns the raw MANIFEST dict for a known
  model (used by the worker to forward it to the registry).
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

### Modality conventions

Each modality subpackage contains:
- `__init__.py`: exports `MODALITY: str` (MIME-style tag like `"audio/speech"`) and `build_router: Callable[[ModalityRegistry], APIRouter]`. These two are what `discover_modalities` scans for.
- `protocol.py`: Protocol + Result dataclass(es) for this modality
- `routes.py`: defines `build_router(registry) -> APIRouter`
- `client.py`: HTTP client for this modality's endpoints
- `codec.py`: modality-specific encoding (wav/opus for audio; png/jpeg for images; base64 float32 for embeddings)
- `backends/` (optional): private helpers used by model scripts; NOT a plugin surface. `audio_speech/backends/` currently holds `base.py` (shared voices_dir + BaseModel) and `transformers.py` (the Narro engine Soprano delegates to). Public backends live in `muse/models/`.

Each model script (`src/muse/models/<id>.py`) contains:
- Top-level `MANIFEST: dict` with required keys `model_id`, `modality`, `hf_repo` and optional `description`, `license`, `pip_extras`, `system_packages`, `capabilities`. Anything else passes through.
- A class named exactly `Model` (tests alias it: `from muse.models.kokoro_82m import Model as KokoroModel`).

Each `Model` class:
- Satisfies the modality's Protocol structurally (no base class required).
- Accepts `hf_repo=`, `local_dir=`, `device=`, `**_` in its constructor (the catalog loader calls with those kwargs; `**_` absorbs future additions).
- Prefers `local_dir` over `hf_repo` when loading weights.
- Defers heavy imports (torch, transformers, diffusers, kokoro) to inside `__init__` or function bodies behind a try/except so `muse --help` stays instant.

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

# Run all tests
pytest tests/

# Run tests for one modality contract
pytest tests/modalities/audio_speech/
pytest tests/modalities/embedding_text/
pytest tests/modalities/image_generation/

# Run tests for one bundled model
pytest tests/models/test_kokoro_82m.py

# Coverage
pytest tests/ --cov=muse

# Start server
muse serve --device cuda

# Generation is over HTTP (Python client, curl, or future muse mcp).
# There are deliberately no `muse speak` / `muse imagine` subcommands.
# The CLI is admin-only (serve / pull / models) so new modalities land
# without CLI churn.
python - <<'PY'
from muse.modalities.audio_speech import SpeechClient
from muse.modalities.image_generation import GenerationsClient
from muse.modalities.embedding_text import EmbeddingsClient
SpeechClient().infer("hello")           # WAV bytes; MUSE_SERVER env sets base URL
GenerationsClient().generate("a cat")   # list[bytes] of PNGs
EmbeddingsClient().embed(["alpha", "beta"])   # list[list[float]]
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

## Adding a new model (the common case)

Two paths:

**Resolver path (recommended for uniform model classes).** If the model
is a GGUF on HuggingFace or a sentence-transformers repo, no script
needed. Just pull by URI:

```bash
muse pull hf://Qwen/Qwen3-8B-GGUF@q4_k_m
muse pull hf://Qwen/Qwen3-Embedding-0.6B
```

The HF resolver synthesizes a manifest, persists it in
`~/.muse/catalog.json`, and routes inference through the matching
generic runtime (`LlamaCppModel` / `SentenceTransformerModel`). Use
`muse search <query> --modality chat/completion --max-size-gb N` to
discover candidates. See `docs/RESOLVERS.md`.

**Script path (for one-offs with custom code).** Models that need
non-uniform behavior (NV-Embed's custom encode method, Soprano's Narro
engine) get a hand-written script:

1. Write a `.py` file with a `MANIFEST` dict + `Model` class (see
   `docs/MODEL_SCRIPTS.md` for the full schema).
2. Drop it in `~/.muse/models/` or any dir pointed to by `$MUSE_MODELS_DIR`.
3. `muse pull <model_id>` to install deps + download weights.
4. `muse serve` picks it up.

Bundled model scripts live in `src/muse/models/<id>.py`. Adding a
bundled model requires no catalog edits, no registry changes, and no
worker changes: discovery just finds it.

Bundled scripts always win on `model_id` collision with resolver-pulled
entries.

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
4. Add bundled model scripts under `src/muse/models/`.
5. Add tests under `tests/modalities/<mime_name>/` and
   `tests/models/test_<new_model>.py`.

No edits to `worker.py`, `catalog.py`, `registry.py`, or `server.py`
are needed: discovery handles the wiring.

No gateway changes are needed either: the gateway routes by the
`model` field in the request body and forwards to whichever worker
loaded that model. New modalities are transparent to the proxy layer.

External escape hatch: dropping a modality subpackage into
`$MUSE_MODALITIES_DIR/<mime_name>/` registers it without forking muse.
Intended for experimentation, not routine extension.
