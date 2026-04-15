# Muse

Model-agnostic multi-modality generation server. OpenAI-compatible HTTP is the canonical interface: text-to-speech on `/v1/audio/speech`, text-to-image on `/v1/images/generations`, text-to-vector on `/v1/embeddings`, more landing the same way (transcriptions, video). Modality tags are MIME-style (`audio/speech`, `embedding/text`, `image/generation`).

Adding a new model is a drop-in: write one `.py` file with a `MANIFEST` dict and a `Model` class, put it in `~/.muse/models/`, run `muse pull`. Adding a new modality (rarer) is dropping a subpackage under `src/muse/modalities/` (or `$MUSE_MODALITIES_DIR` for the escape hatch). Both surfaces are discovered at runtime; there is no hardcoded catalog, no allowlist, and no registration calls.

The CLI is deliberately admin-only (`serve`, `pull`, `models`). Generation is reached via the HTTP API, consumed by Python clients, `curl`, or future wrappers like `muse mcp`.

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
# Pull a model (creates a dedicated venv + installs its pip deps + downloads HF weights)
muse pull soprano-80m
muse pull sd-turbo

# Admin: list what's in the catalog
muse models list

# Start the server (loads pulled models; serves OpenAI-compatible endpoints)
muse serve --host 0.0.0.0 --port 8000
```

From any client, generation is an HTTP call:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input":"Hello world","model":"soprano-80m"}' \
  --output hello.wav
```

```bash
# Embeddings (accepts single string or list)
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input":"hello world","model":"all-minilm-l6-v2"}'
```

```python
from muse.modalities.audio_speech import SpeechClient
from muse.modalities.image_generation import GenerationsClient
from muse.modalities.embedding_text import EmbeddingsClient

# MUSE_SERVER env var sets the base URL for remote use; default http://localhost:8000
wav_bytes = SpeechClient().infer("Hello world")
pngs = GenerationsClient().generate("a cat on mars, cinematic", n=1)
vectors = EmbeddingsClient().embed(["alpha", "beta"])   # list[list[float]]
```

`muse serve` auto-restarts crashed worker processes with exponential backoff.
Individual model failures don't take down the server or other modalities.

## CLI (admin-only)

| Command | Description |
|---|---|
| `muse serve` | start the HTTP server |
| `muse pull <model-id>` | download weights + install deps |
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
| `POST /v1/images/generations` | generate images (OpenAI-compatible) |
| `POST /v1/embeddings` | text embeddings (OpenAI-compatible) |

Error shape is uniform: `{"error": {"code", "message", "type"}}` across 404 (model not found) and 422 (validation). Matches OpenAI's envelope so clients written against their API work against muse.

## Architecture

- `muse.core`: modality-agnostic discovery, registry, catalog, venv management, HF downloader, pip auto-install, FastAPI app factory.
- `muse.cli_impl`: `serve` (supervisor), `worker` (single-venv process), `gateway` (HTTP proxy routing by request's `model` field).
- `muse.modalities/`: one subpackage per modality (wire contract: protocol + routes + codec + client).
  - `audio_speech/` (MODALITY `"audio/speech"`)
  - `embedding_text/` (MODALITY `"embedding/text"`)
  - `image_generation/` (MODALITY `"image/generation"`)
- `muse.models/`: flat directory of drop-in model scripts, one file per model (MANIFEST + Model class).
  - `soprano_80m.py`, `kokoro_82m.py`, `bark_small.py` (audio/speech)
  - `all_minilm_l6_v2.py`, `qwen3_embedding_0_6b.py`, `nv_embed_v2.py` (embedding/text)
  - `sd_turbo.py` (image/generation)

`muse serve` is a supervisor process. It spawns one worker subprocess per venv (each pulled model has its own venv with its own deps) and runs a gateway that proxies by the `model` field. Dep conflicts between models are structurally impossible.

Users extend muse by dropping a `.py` file into `~/.muse/models/` (see `docs/MODEL_SCRIPTS.md` for the MANIFEST schema and Model class contract). No muse source edits required.

See `CLAUDE.md` for implementation details and contribution guide, and `docs/MODEL_SCRIPTS.md` for writing your own model scripts.

## License

MIT
