# Muse

Model-agnostic multi-modality generation server. OpenAI-compatible HTTP is the canonical interface:
- text-to-speech on `/v1/audio/speech`
- speech-to-text on `/v1/audio/transcriptions` and `/v1/audio/translations`
- text-to-music on `/v1/audio/music` and text-to-sound-effects on `/v1/audio/sfx`
- text-to-image on `/v1/images/generations`, image inpainting on `/v1/images/edits`, image variations on `/v1/images/variations`
- image-to-image super-resolution on `/v1/images/upscale`
- promptable segmentation on `/v1/images/segment`
- text-to-animation on `/v1/images/animations`
- text-to-video on `/v1/video/generations`
- image-to-vector on `/v1/images/embeddings`
- audio-to-vector on `/v1/audio/embeddings`
- text-to-vector on `/v1/embeddings`
- text-to-text (LLM, tool calls, streaming) on `/v1/chat/completions`
- text moderation/classification on `/v1/moderations`
- text rerank (Cohere-compat) on `/v1/rerank`
- text summarization (Cohere-compat) on `/v1/summarize`

Modality tags are MIME-style (`audio/embedding`, `audio/generation`, `audio/speech`, `audio/transcription`, `chat/completion`, `embedding/text`, `image/animation`, `image/embedding`, `image/generation`, `image/segmentation`, `image/upscale`, `text/classification`, `text/rerank`, `text/summarization`, `video/generation`).

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

# Image embeddings (input is data: URL or http(s):// URL; mirrors /v1/embeddings)
IMG_B64=$(base64 -w0 cat.png)
curl -X POST http://localhost:8000/v1/images/embeddings \
  -H "Content-Type: application/json" \
  -d "{\"input\":\"data:image/png;base64,${IMG_B64}\",\"model\":\"dinov2-small\"}"

# Audio embeddings (multipart upload; one or more `file` parts; mirrors /v1/embeddings envelope)
curl -X POST http://localhost:8000/v1/audio/embeddings \
  -F "file=@clip.wav" \
  -F "model=mert-v1-95m"

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

# Summarize (Cohere-compat); pulls bart-large-cnn by default
curl -X POST http://localhost:8000/v1/summarize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "muse is a model-agnostic multi-modality generation server. It hosts text, image, audio, and video models behind a unified HTTP API that mirrors OpenAI where possible.",
    "length": "short",
    "format": "paragraph",
    "model": "bart-large-cnn"
  }'

# Music generation (capability-gated; default model: stable-audio-open-1.0)
curl -X POST http://localhost:8000/v1/audio/music \
  -H "Content-Type: application/json" \
  -d '{"prompt":"ambient piano with light rain","model":"stable-audio-open-1.0","duration":10.0}' \
  --output music.wav

# Sound effects generation (same model, different intent)
curl -X POST http://localhost:8000/v1/audio/sfx \
  -H "Content-Type: application/json" \
  -d '{"prompt":"footsteps on gravel","model":"stable-audio-open-1.0","duration":3.0}' \
  --output footsteps.wav

# Image inpainting (multipart: image + mask + prompt)
# White mask pixels are regenerated; black pixels are kept.
curl -X POST http://localhost:8000/v1/images/edits \
  -F "image=@scene.png" \
  -F "mask=@mask.png" \
  -F "prompt=add a moon to the sky" \
  -F "model=sd-turbo" \
  -F "size=512x512" \
  -F "n=1"

# Image variations (multipart: image only, no prompt)
curl -X POST http://localhost:8000/v1/images/variations \
  -F "image=@scene.png" \
  -F "model=sd-turbo" \
  -F "size=512x512" \
  -F "n=2"

# Image upscale (multipart: 4x super-resolution; SD x4 supports scale=4 only)
curl -s -X POST http://localhost:8000/v1/images/upscale \
  -F "image=@source.png" \
  -F "model=stable-diffusion-x4-upscaler" \
  -F "scale=4" \
  -F "prompt=high detail" \
  | jq -r '.data[0].b64_json' \
  | base64 -d > upscaled.png

# Image segmentation (multipart: SAM-2 promptable masks)
# Mode 1: automatic (sweep grid of point prompts internally)
curl -s -X POST http://localhost:8000/v1/images/segment \
  -F "image=@scene.png" \
  -F "model=sam2-hiera-tiny" \
  -F "mode=auto" \
  -F "max_masks=8"

# Mode 2: foreground click points
curl -s -X POST http://localhost:8000/v1/images/segment \
  -F "image=@scene.png" \
  -F "model=sam2-hiera-tiny" \
  -F "mode=points" \
  -F 'points=[[150, 200]]'

# Mode 3: bounding boxes
curl -s -X POST http://localhost:8000/v1/images/segment \
  -F "image=@scene.png" \
  -F "model=sam2-hiera-tiny" \
  -F "mode=boxes" \
  -F 'boxes=[[50, 60, 250, 240]]' \
  -F "mask_format=rle"

# Video generation (since v0.27.0; GPU-required, 8GB+ VRAM tight)
# Default response_format=mp4; "webm" and "frames_b64" also supported.
curl -s -X POST http://localhost:8000/v1/video/generations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a flag waving in the wind",
    "model": "wan2-1-t2v-1-3b",
    "duration_seconds": 5.0,
    "fps": 5,
    "size": "832x480",
    "steps": 30
  }' \
  | jq -r '.data[0].b64_json' \
  | base64 -d > flag.mp4
```

```python
from muse.modalities.audio_speech import SpeechClient
from muse.modalities.image_generation import (
    GenerationsClient, ImageEditsClient, ImageVariationsClient,
)
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

# Image inpainting and variations (since v0.21.0)
src = open("scene.png", "rb").read()
msk = open("mask.png", "rb").read()
edited = ImageEditsClient().edit(
    "add a moon to the sky", image=src, mask=msk, model="sd-turbo",
)
variants = ImageVariationsClient().vary(image=src, model="sd-turbo", n=2)

# Image upscale (since v0.25.0): 4x super-resolution
from muse.modalities.image_upscale import ImageUpscaleClient
from pathlib import Path
upscaled = ImageUpscaleClient().upscale(
    image=Path("source.png").read_bytes(),
    model="stable-diffusion-x4-upscaler",
    scale=4,
    prompt="razor sharp detail",
)
Path("upscaled.png").write_bytes(upscaled[0])

# Image segmentation (since v0.26.0): SAM-2 promptable masks
from muse.modalities.image_segmentation import ImageSegmentationClient
seg = ImageSegmentationClient()
src_bytes = Path("scene.png").read_bytes()
result_auto = seg.segment(
    image=src_bytes, model="sam2-hiera-tiny", mode="auto", max_masks=8,
)
result_points = seg.segment(
    image=src_bytes, model="sam2-hiera-tiny", mode="points",
    points=[[150, 200]],
)
result_boxes = seg.segment(
    image=src_bytes, model="sam2-hiera-tiny", mode="boxes",
    boxes=[[50, 60, 250, 240]], mask_format="rle",
)
# Each result is a dict {id, model, mode, image_size, masks: [...]}
# masks[i]["mask"] is a base64 PNG (mask_format=png_b64) or
# a {"size": [H, W], "counts": str} dict (mask_format=rle)

# Video generation (since v0.27.0): GPU-required, 8GB+ VRAM tight
# Wan2.1 T2V 1.3B (~3GB at fp16) is the default low-VRAM bundle;
# CogVideoX-2b (~9GB) and LTX-Video (~16GB) are curated additions.
from muse.modalities.video_generation import VideoGenerationClient
vid = VideoGenerationClient()
mp4_bytes = vid.generate(
    "a flag waving in the wind",
    model="wan2-1-t2v-1-3b",
    duration_seconds=5.0,
    fps=5,
    size="832x480",
    steps=30,
)
Path("flag.mp4").write_bytes(mp4_bytes)
```

VRAM caveats for `video/generation`: even Wan 1.3B at fp16 is tight on 8GB cards; 12GB+ recommended for headroom. CogVideoX-2b realistically wants 16GB. LTX-Video needs 16GB+. Mochi-1 (24GB+) and HunyuanVideo (60GB+) are documented but not curated; their dedicated runtimes ship in v1.next.

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
| `POST /v1/images/generations` | generate images (OpenAI-compatible; supports img2img via `image` + `strength`) |
| `POST /v1/images/edits` | inpaint masked regions (OpenAI-compatible; multipart with image+mask+prompt) |
| `POST /v1/images/variations` | generate alternates of one image (OpenAI-compatible; multipart, no prompt) |
| `POST /v1/embeddings` | text embeddings (OpenAI-compatible) |
| `POST /v1/images/embeddings` | image embeddings (OpenAI-shape envelope mirroring /v1/embeddings) |
| `POST /v1/audio/embeddings` | audio embeddings (multipart upload + OpenAI-shape envelope mirroring /v1/embeddings) |
| `POST /v1/chat/completions` | chat (OpenAI-compatible incl. tools, structured output, streaming) |
| `POST /v1/moderations` | text moderation/classification (OpenAI-compatible) |
| `POST /v1/rerank` | text rerank (Cohere-compat) |
| `POST /v1/summarize` | text summarization (Cohere-compat) |
| `POST /v1/audio/music` | music generation (capability-gated; muse-native shape) |
| `POST /v1/audio/sfx` | sound-effect generation (capability-gated; muse-native shape) |
| `POST /v1/video/generations` | text-to-video generation (mp4/webm/frames_b64; GPU-required) |

Error shape is uniform: `{"error": {"code", "message", "type"}}` across 404 (model not found) and 422 (validation). Matches OpenAI's envelope so clients written against their API work against muse.

## Architecture

- `muse.core`: modality-agnostic discovery, registry, catalog, venv management, HF downloader, pip auto-install, FastAPI app factory.
- `muse.cli_impl`: `serve` (supervisor), `worker` (single-venv process), `gateway` (HTTP proxy routing by request's `model` field).
- `muse.modalities/`: one subpackage per modality (wire contract: protocol + routes + codec + client).
  - `audio_embedding/` (MODALITY `"audio/embedding"`; multipart upload + OpenAI-shape envelope; includes `runtimes/transformers_audio.py`)
  - `audio_generation/` (MODALITY `"audio/generation"`; mounts both `/v1/audio/music` and `/v1/audio/sfx` on one MIME tag with per-route capability gates)
  - `audio_speech/` (MODALITY `"audio/speech"`)
  - `audio_transcription/` (MODALITY `"audio/transcription"`; multipart/form-data upload, OpenAI Whisper wire shape)
  - `chat_completion/` (MODALITY `"chat/completion"`; includes `runtimes/llama_cpp.py`)
  - `embedding_text/` (MODALITY `"embedding/text"`; includes `runtimes/sentence_transformers.py`)
  - `image_embedding/` (MODALITY `"image/embedding"`; includes `runtimes/transformers_image.py`)
  - `image_generation/` (MODALITY `"image/generation"`)
  - `text_classification/` (MODALITY `"text/classification"`; OpenAI `/v1/moderations` wire shape)
  - `text_rerank/` (MODALITY `"text/rerank"`; Cohere `/v1/rerank` wire shape)
  - `text_summarization/` (MODALITY `"text/summarization"`; Cohere `/v1/summarize` wire shape)
  - `video_generation/` (MODALITY `"video/generation"`; includes `runtimes/wan_runtime.py` and `runtimes/cogvideox_runtime.py`)
- `muse.models/`: flat directory of drop-in model scripts, one file per model (MANIFEST + Model class).
  - `soprano_80m.py`, `kokoro_82m.py`, `bark_small.py` (audio/speech)
  - `nv_embed_v2.py` (embedding/text; MiniLM and Qwen3-Embedding are now resolver-pulled via the generic runtime, see `curated.yaml`)
  - `sd_turbo.py` (image/generation)
  - `bge_reranker_v2_m3.py` (text/rerank)
  - `stable_audio_open_1_0.py` (audio/generation; Stable Audio Open 1.0, Apache 2.0)
  - `bart_large_cnn.py` (text/summarization; facebook/bart-large-cnn, Apache 2.0, ~400MB CPU-friendly)
  - `dinov2_small.py` (image/embedding; facebook/dinov2-small, Apache 2.0, 88MB, 384-dim CPU-friendly)
  - `mert_v1_95m.py` (audio/embedding; m-a-p/MERT-v1-95M, MIT, 95MB, 768-dim music understanding via mean-pool over time)
  - `wan2_1_t2v_1_3b.py` (video/generation; Wan-AI/Wan2.1-T2V-1.3B, Apache 2.0, ~3GB at fp16, 5s clips at 832x480, GPU-required)
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
