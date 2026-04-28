# Muse

Model-agnostic multi-modality generation server. OpenAI-compatible HTTP is the canonical interface:
- text-to-speech on `/v1/audio/speech`
- speech-to-text on `/v1/audio/transcriptions` and `/v1/audio/translations`
- text-to-image on `/v1/images/generations`
- text-to-animation on `/v1/images/animations`
- text-to-vector on `/v1/embeddings`
- text-to-text (LLM, tool calls, streaming) on `/v1/chat/completions`
- text moderation/classification on `/v1/moderations`
- text rerank (Cohere-compat) on `/v1/rerank`

Modality tags are MIME-style (`audio/speech`, `audio/transcription`, `chat/completion`, `embedding/text`, `image/animation`, `image/generation`, `text/classification`, `text/rerank`).

Three ways to add a model, in order of how often you'll reach for them:

1. **Pull a GGUF or sentence-transformers model from HuggingFace by URI.** No script, no edits:
   ```bash
   muse search qwen3 --modality chat/completion --max-size-gb 10
   muse pull hf://Qwen/Qwen3-8B-GGUF@q4_k_m
   ```
2. **Drop a `.py` script into `~/.muse/models/`** for a one-off model with custom code (see `docs/MODEL_SCRIPTS.md`).
3. **Add a whole new modality** (rare) by dropping a subpackage into
   `src/muse/modalities/` or `$MUSE_MODALITIES_DIR`. The subpackage
   exports `MODALITY` + `build_router` and discovery picks it up.
   Optional: drop a `hf.py` next to `__init__.py` exporting an
   `HF_PLUGIN` dict; muse's HF resolver picks it up the same way and
   `muse search`/`muse pull hf://...` work for the new modality.

All three surfaces are discovered at runtime; there is no hardcoded catalog, no allowlist, and no registration calls.

The CLI is deliberately admin-only (`serve`, `pull`, `search`, `models`). Generation is reached via the HTTP API, consumed by Python clients, `curl`, or future wrappers like `muse mcp`.

## Install

```bash
pip install -e ".[server,audio,images]"
```

Optional extras:
- `audio`: PyTorch + transformers for TTS backends
- `audio-kokoro`: Kokoro TTS (needs system `espeak-ng`)
- `images`: diffusers + Pillow for SD-Turbo and future image backends
- `server`: FastAPI + uvicorn + sse-starlette (only needed on the serving host)
- `dev`: pytest + coverage tools

## Quick start

```bash
# Pull bundled models by id (creates a dedicated venv + installs deps + downloads weights)
muse pull soprano-80m
muse pull sd-turbo

# Or pull anything resolvable from HuggingFace by URI
muse pull hf://Qwen/Qwen3-8B-GGUF@q4_k_m
muse pull hf://sentence-transformers/all-MiniLM-L6-v2

# Admin: list what's in the catalog
muse models list

# Start the server (loads pulled models; serves OpenAI-compatible endpoints)
muse serve --host 0.0.0.0 --port 8000
```

From any client, generation is an HTTP call:

```bash
# Text-to-speech
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input":"Hello world","model":"soprano-80m"}' \
  --output hello.wav

# Embeddings (accepts single string or list)
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input":"hello world","model":"all-minilm-l6-v2"}'

# Chat (OpenAI-compatible incl. tools and streaming)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-8b-gguf-q4-k-m","messages":[{"role":"user","content":"Capital of France?"}]}'

# Rerank (Cohere-compat); pulls bge-reranker-v2-m3 by default
curl -X POST http://localhost:8000/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "what is muse?",
    "documents": [
      "muse is an audio server",
      "muse is a multi-modality generation server",
      "muse is the goddess of inspiration"
    ],
    "model": "bge-reranker-v2-m3",
    "top_n": 2,
    "return_documents": true
  }'
```

```python
from muse.modalities.audio_speech import SpeechClient
from muse.modalities.image_generation import GenerationsClient
from muse.modalities.embedding_text import EmbeddingsClient
from muse.modalities.chat_completion import ChatClient

# MUSE_SERVER env var sets the base URL for remote use; default http://localhost:8000
wav_bytes = SpeechClient().infer("Hello world")
pngs = GenerationsClient().generate("a cat on mars, cinematic", n=1)
vectors = EmbeddingsClient().embed(["alpha", "beta"])   # list[list[float]]
chat = ChatClient().chat(
    model="qwen3-8b-gguf-q4-k-m",
    messages=[{"role": "user", "content": "Capital of France?"}],
)
```

The OpenAI Python SDK works against muse with no modifications:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-used")
client.chat.completions.create(model="qwen3-8b-gguf-q4-k-m", messages=[...])
```

`muse serve` auto-restarts crashed worker processes with exponential backoff.
Individual model failures don't take down the server or other modalities.

## CLI (admin-only)

| Command | Description |
|---|---|
| `muse serve` | start the HTTP server |
| `muse pull <model-id-or-uri>` | download weights + install deps (accepts bundled id OR resolver URI like `hf://org/repo@variant`) |
| `muse search <query> [--modality M]` | search HuggingFace for pullable GGUF / sentence-transformers models |
| `muse models list [--modality X]` | list known/pulled models |
| `muse models info <model-id>` | show catalog entry |
| `muse models remove <model-id>` | unregister from catalog |
| `muse models enable <model-id>` | mark a pulled model active (load on next serve) |
| `muse models disable <model-id>` | mark a pulled model inactive (skip on next serve) |

No per-modality subcommands (`muse speak`, `muse audio ...`). Those would be hardcoded modality-to-verb mappings that grow with every new modality. Keeping the CLI modality-agnostic means embeddings, transcriptions, and video land without CLI churn.

## HTTP endpoints

| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness + enabled modalities |
| `GET /v1/models` | all registered models, aggregated |
| `POST /v1/audio/speech` | synthesize speech (OpenAI-compatible) |
| `GET /v1/audio/speech/voices` | list voices for a model |
| `POST /v1/audio/transcriptions` | transcribe audio to text (OpenAI-compatible) |
| `POST /v1/audio/translations` | transcribe + translate audio to English (OpenAI-compatible) |
| `POST /v1/images/generations` | generate images (OpenAI-compatible) |
| `POST /v1/embeddings` | text embeddings (OpenAI-compatible) |
| `POST /v1/chat/completions` | chat (OpenAI-compatible incl. tools, structured output, streaming) |
| `POST /v1/moderations` | text moderation/classification (OpenAI-compatible) |

Error shape is uniform: `{"error": {"code", "message", "type"}}` across 404 (model not found) and 422 (validation). Matches OpenAI's envelope so clients written against their API work against muse.

## Architecture

- `muse.core`: modality-agnostic discovery, registry, catalog, venv management, HF downloader, pip auto-install, FastAPI app factory.
- `muse.cli_impl`: `serve` (supervisor), `worker` (single-venv process), `gateway` (HTTP proxy routing by request's `model` field).
- `muse.modalities/`: one subpackage per modality (wire contract: protocol + routes + codec + client).
  - `audio_speech/` (MODALITY `"audio/speech"`)
  - `audio_transcription/` (MODALITY `"audio/transcription"`; multipart/form-data upload, OpenAI Whisper wire shape)
  - `chat_completion/` (MODALITY `"chat/completion"`; includes `runtimes/llama_cpp.py`)
  - `embedding_text/` (MODALITY `"embedding/text"`; includes `runtimes/sentence_transformers.py`)
  - `image_generation/` (MODALITY `"image/generation"`)
  - `text_classification/` (MODALITY `"text/classification"`; OpenAI `/v1/moderations` wire shape)
- `muse.models/`: flat directory of drop-in model scripts, one file per model (MANIFEST + Model class).
  - `soprano_80m.py`, `kokoro_82m.py`, `bark_small.py` (audio/speech)
  - `nv_embed_v2.py` (embedding/text; MiniLM and Qwen3-Embedding are now resolver-pulled via the generic runtime, see `curated.yaml`)
  - `sd_turbo.py` (image/generation)
- `muse.core.resolvers`: URI -> ResolvedModel dispatch for `muse pull hf://...`.
  - `resolvers_hf` registers the `hf://` resolver for HuggingFace GGUF + sentence-transformers repos.

`muse serve` is a supervisor process. It spawns one worker subprocess per venv (each pulled model has its own venv with its own deps) and runs a gateway that proxies by the `model` field. Dep conflicts between models are structurally impossible.

Three ways to extend muse:
1. **Resolver URI**: `muse pull hf://Qwen/Qwen3-8B-GGUF@q4_k_m` for any GGUF or sentence-transformers HF repo. See `docs/RESOLVERS.md`.
2. **Model script**: drop a `.py` into `~/.muse/models/` for one-off models with custom code. See `docs/MODEL_SCRIPTS.md`.
3. **Modality subpackage**: drop into `src/muse/modalities/` or `$MUSE_MODALITIES_DIR` for a whole new modality.

See `CLAUDE.md` for implementation details and contribution guide,
`docs/MODEL_SCRIPTS.md` for writing your own model scripts,
`docs/RESOLVERS.md` for adding a new URI scheme, and
`docs/CHAT_COMPLETION.md` for the chat endpoint specification.

## License

MIT
