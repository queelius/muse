# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Muse is a multi-modality generation server and client. It currently supports
fifteen modalities:

- **audio/embedding**: audio-to-vector via `/v1/audio/embeddings` (mert-v1-95m bundled; CLAP, MERT, wav2vec family via the resolver; multipart upload + OpenAI-shape envelope mirroring `/v1/embeddings`)
- **audio/generation**: text-to-music + text-to-SFX via `/v1/audio/music` and `/v1/audio/sfx` (Stable Audio Open 1.0; per-model capability gates on `supports_music` / `supports_sfx`)
- **audio/speech**: text-to-speech via `/v1/audio/speech` (Soprano, Kokoro, Bark)
- **audio/transcription**: speech-to-text via `/v1/audio/transcriptions` and `/v1/audio/translations` (Systran faster-whisper family; any CT2 Whisper on HF)
- **chat/completion**: text-to-text LLMs via `/v1/chat/completions` (OpenAI-compatible incl. tools + streaming; powered by llama-cpp-python; any GGUF on HF via the resolver)
- **embedding/text**: text-to-vector via `/v1/embeddings` (MiniLM, Qwen3-Embedding, NV-Embed-v2; any sentence-transformers HF repo via the resolver)
- **image/animation**: text-to-animation via `/v1/images/animations` (AnimateDiff: 16-frame loops, animated WebP/GIF/MP4 output)
- **image/embedding**: image-to-vector via `/v1/images/embeddings` (dinov2-small bundled; CLIP, SigLIP, DINOv2 family via the resolver; OpenAI-shape wire envelope mirroring `/v1/embeddings`)
- **image/generation**: text-to-image and img2img via `/v1/images/generations` (SD-Turbo, SDXL-Turbo, FLUX.1-schnell, any diffusers HF repo)
- **image/segmentation**: promptable segmentation via `/v1/images/segment` (sam2-hiera-tiny bundled; SAM-2 family via the resolver; multipart upload, mode dispatch auto/points/boxes/text gated by capability flags; masks emitted as base64 PNG or COCO RLE)
- **image/upscale**: image-to-image super-resolution via `/v1/images/upscale` (stable-diffusion-x4-upscaler bundled; multipart upload, OpenAI-shape envelope; 4x diffusion-based upscaling; capability-gated on `supported_scales`; env-tunable input cap via `MUSE_UPSCALE_MAX_INPUT_SIDE`)
- **text/classification**: text moderation/classification via `/v1/moderations` (any HuggingFace text-classification model)
- **text/rerank**: cross-encoder rerank via `/v1/rerank` (bge-reranker-v2-m3 bundled; any cross-encoder reranker on HF; Cohere-compat wire shape)
- **text/summarization**: BART/PEGASUS seq2seq summarization via `/v1/summarize` (bart-large-cnn bundled; any summarization-tagged HF repo via the resolver; Cohere-compat wire shape)
- **video/generation**: text-to-video via `/v1/video/generations` (wan2-1-t2v-1-3b bundled; Wan / CogVideoX / LTX-Video families via the resolver; narrative clips up to 30s; mp4/webm/frames_b64 output; GPU-required, 8GB+ VRAM tight, 12GB+ recommended)

Modality tags are MIME-style (`audio/speech`, not `audio.speech`). The HTTP
path hierarchy mirrors the OpenAI shape where possible (`/v1/audio/speech`,
`/v1/chat/completions`, `/v1/embeddings`, `/v1/images/animations`,
`/v1/images/generations`) for client compatibility.

In addition to the per-modality routes, muse also exposes an admin
REST API under `/v1/admin/*` (v0.28.0+) for runtime model control
without restarting `muse serve`: enable, disable, probe, pull, remove,
plus worker introspection and async-job tracking. Closed-by-default
behind `MUSE_ADMIN_TOKEN`. See "Admin REST API" below.

`text/rerank` is muse's first Cohere-compat modality (rather than
OpenAI-compat): OpenAI has no rerank API, and Cohere's `/v1/rerank` is
the de-facto standard that downstream tooling (LangChain, LlamaIndex,
Haystack) expects. Response envelope mirrors Cohere's: `results[]`
with `index` + `relevance_score`, optional `document.text`, plus
`meta.billed_units.search_units` for SDK compatibility.

`text/summarization` is muse's second Cohere-compat modality (after
text/rerank). OpenAI has no summarization API; Cohere's `/v1/summarize`
was the de-facto reference until its 2024 deprecation, and the wire
shape is what summarization tooling expects. Request: `{text, length,
format, model}`. Response: `{id, model, summary, usage, meta}`.
`length` ("short"|"medium"|"long") deterministically maps to
`max_new_tokens` in the runtime: short=80, medium=180, long=400.
`format` ("paragraph"|"bullets") is recorded in `meta.format` and is
metadata-only for non-instruction summarizers like BART-CNN; future
instruction-tuned summarizers can consult it. The bundled
`bart-large-cnn` (Apache 2.0, ~400MB, CPU-friendly) is the default;
the curated `bart-cnn-samsum` (`supports_dialog_summarization=true`)
is dialog-tuned. The HF resolver sniffs any `summarization`-tagged
repo at priority 110 and serves it via `BartSeq2SeqRuntime` over
`transformers.AutoModelForSeq2SeqLM` (BART, PEGASUS, T5).

`image/embedding` is muse's first image-to-vector modality. The wire
envelope at `POST /v1/images/embeddings` mirrors `/v1/embeddings`
exactly (`{object: "list", data, model, usage}`) so OpenAI SDK clients
that already consume embeddings can reuse helper code. Each `input`
entry is a `data:image/...;base64,...` URL or `http(s)://...` URL
pointing at PNG/JPEG/WEBP; image decoding goes through the shared
`decode_image_input` helper from the image_generation modality.
The bundled `dinov2-small` (Apache 2.0, 88MB, 384-dim, CPU-friendly)
is the default; curated additions cover SigLIP2 and CLIP. The HF
resolver sniffs any repo with an image-feature-extraction-class tag
plus a `preprocessor_config.json` sibling at priority 105 (between
embedding/text and image-generation file-pattern) and serves it via
`ImageEmbeddingRuntime` over `transformers.AutoModel` +
`AutoProcessor`. The runtime's `_extract_embeddings` dispatch picks
the right pooling per architecture: CLIP `image_embeds` >
SigLIP/DINOv2 `pooler_output` > DINOv2 base `last_hidden_state[:, 0]`
(CLS token).

`audio/embedding` is muse's first audio-to-vector modality. Wire shape
is multipart-in (one or more `file` parts, mirroring
`/v1/audio/transcriptions`) and `/v1/embeddings`-shaped JSON out
(`{object: "list", data, model, usage}`). Audio decoding goes through
`librosa` (already installed for Whisper) inside the runtime/script,
which resamples on the way in to each model's preferred rate (CLAP
48kHz, MERT 24kHz). The bundled `mert-v1-95m` (MIT, 95MB, 768-dim
music understanding via mean-pool over time, `trust_remote_code=True`
for the custom feature extractor) is the default; the curated
`clap-htsat-fused` adds 512-dim audio + text-aligned embeddings
(BSD-3, supports_text_embeddings_too=True). The HF resolver sniffs
any repo with `feature-extraction` tag plus a name pattern matching
`clap`, `mert`, `audio-encoder`, `wav2vec`, or `audio-embedding` at
priority 105, and serves it via `AudioEmbeddingRuntime` over
`transformers.AutoModel` + `AutoFeatureExtractor` (with
`AutoProcessor` preferred for newer repos). The runtime's
`_extract_embeddings` dispatch picks the right pooling per
architecture: CLAP `audio_embeds` > pooler_output > MERT/wav2vec
`last_hidden_state.mean(dim=1)` (mean-pool over time). Per-file size
cap via `MUSE_AUDIO_EMBEDDINGS_MAX_BYTES` (default 50MB); duration
cap via `MUSE_AUDIO_EMBEDDINGS_MAX_SECONDS` (default 60s; runtime
truncates after decode).

`audio/generation` is muse's first modality with TWO URL routes mounted
on ONE MIME tag. `/v1/audio/music` and `/v1/audio/sfx` share the same
request body, codec, registry surface, and runtime. The only per-route
difference is the manifest capability key consulted: `supports_music`
or `supports_sfx`. When a flag is False (or a future MusicGen-only
model lacks `supports_sfx`), the unsupported route returns 400. The
two-URL split is for legibility: a "footsteps on gravel" prompt sent
to `/v1/audio/music` would silently produce a 30-second loop of
footsteps treated as music; routing the same prompt to `/v1/audio/sfx`
makes the user's intent explicit to operators reading logs and to the
model itself.

The `/v1/images/generations` route also accepts optional `image` (data URL or http(s):// URL) + `strength` (0.0 to 1.0, default 0.5) fields for img2img since v0.17.0. OpenAI SDK clients pass them via `extra_body`:

    client.images.generate(prompt="oil painting", model="sdxl-turbo",
                           extra_body={"image": "data:image/png;base64,...", "strength": 0.6})

Models advertise support via `capabilities.supports_img2img`. Requests for non-supporting models return 400.

The `image/generation` modality also exposes `/v1/images/edits` (inpainting) and `/v1/images/variations` (alternates of one image, no prompt) since v0.21.0. Both are multipart/form-data routes that mount on the same modality. Inpainting takes `image` + `mask` + `prompt` and routes to `backend.inpaint(...)`, which lazy-loads `AutoPipelineForInpainting.from_pipe(self._pipe)` to share VRAM with the loaded t2i pipeline. Variations takes `image` only and routes to `backend.vary(...)`, which delegates to the existing img2img path with empty prompt and high strength (default 0.85). Capability flags `supports_inpainting` and `supports_variations` gate the routes; OpenAI SDK clients use `client.images.edit(image=..., mask=..., prompt=..., model=...)` and `client.images.create_variation(image=..., model=...)` natively.

`image/upscale` (v0.25.0) is muse's super-resolution modality: a separate MIME tag from `image/generation` because the runtime backbone is different (`StableDiffusionUpscalePipeline`, not `AutoPipelineForText2Image`). Wire shape at `POST /v1/images/upscale` is multipart/form-data (mirroring `/v1/images/edits`), with `image` as the source file plus `model`, `scale`, optional `prompt`, `negative_prompt`, `steps`, `guidance`, `seed`, `n`, and `response_format` as Form fields. Output envelope mirrors `/v1/images/generations`: `{created, data: [{b64_json|url, revised_prompt}]}`. The bundled `stable-diffusion-x4-upscaler` (Apache 2.0, ~3GB, fixed 4x) is the default; the HF resolver plugin (priority 105) sniffs other diffusers-shape upscalers (`model_index.json` + `image-to-image` tag + upscaler-name allowlist). The `supported_scales` capability gates the request `scale` parameter (returns 400 for unsupported values; SD x4 supports `[4]` only). An env-tunable input-side cap (`MUSE_UPSCALE_MAX_INPUT_SIDE`, default 1024) prevents runaway VRAM use on oversized inputs; the cap is read per-request, so changes take effect on the next request, not at supervisor restart. GAN-based upscalers (AuraSR, Real-ESRGAN) need separate non-diffusers runtimes and are deferred to v1.next.

`image/segmentation` (v0.26.0) is muse's promptable-segmentation modality. Wire shape at `POST /v1/images/segment` is multipart/form-data: `image` as the source file plus `model`, `mode` (auto/points/boxes/text), `prompt` (text mode), `points` (JSON-encoded `[[x, y], ...]`), `boxes` (JSON-encoded `[[x1, y1, x2, y2], ...]`), `mask_format` (`png_b64` or `rle`), and `max_masks` as Form fields. Output: `{id, model, mode, image_size, masks: [{index, score, mask, bbox, area}]}`. Mode dispatch is capability-gated end-to-end: `supports_automatic`, `supports_point_prompts`, `supports_box_prompts`, `supports_text_prompts`. A request with `mode=text` against a model declaring `supports_text_prompts: False` returns 400 before the runtime is invoked. The bundled `sam2-hiera-tiny` (Apache 2.0, ~40MB, point/box/auto, no text) is the default; curated `sam2-hiera-base-plus` and `sam2-hiera-large` extend the family. The HF resolver plugin (priority 110) sniffs `mask-generation` and `image-segmentation` tags. CLIPSeg is a deferred future: the plugin pattern recognizes it and flips `supports_text_prompts: True`, but the SAM2Runtime backbone needs a CLIPSeg-specific replacement to actually consume the text prompt. The mask format dispatch (`png_b64` for portable / viewable, `rle` for compact / pycocotools-compatible) introduces a precedent: the codec ships pure-Python RLE encode/decode that round-trips internally, with `pycocotools` as an optional faster path that produces output other COCO tooling can decode directly. Axis-order discipline at the wire layer: `image_size` is `[W, H]` (PIL convention); RLE `size` is `[H, W]` (COCO convention); `bbox` is `[x, y, w, h]` (COCO bbox convention).

`video/generation` (v0.27.0) is muse's narrative-clip modality, the heaviest yet. Wire shape at `POST /v1/video/generations` is JSON-only (no multipart) with `prompt` plus optional `model`, `duration_seconds` (0.5 to 30), `fps` (1 to 60), `size` (WxH string), `seed`, `negative_prompt`, `steps`, `guidance`, `response_format` (`mp4` default, `webm`, or `frames_b64`), and `n` (capped at 2 because each video is heavy). Output envelope mirrors `/v1/images/animations`: `{data: [{b64_json}], model, metadata: {frames, fps, duration_seconds, format, size}}`. Two distinct runtimes ship under the same MIME tag: `WanRuntime` (`diffusers.WanPipeline` or `DiffusionPipeline` fallback) and `CogVideoXRuntime` (`diffusers.CogVideoXPipeline`); the HF resolver dispatches per architecture. The bundled `wan2-1-t2v-1-3b` (Apache 2.0, ~3GB at fp16, 5s clips at 832x480) targets 8GB cards but is tight; 12GB+ recommended. Curated additions: `cogvideox-2b` (~6GB at fp16, 6s at 720x480, fits 12GB) and `ltx-video` (~13GB, 30fps at 1216x704, requires 16GB+). The HF plugin (priority 105) sniffs `text-to-video`-tagged repos whose name matches one of `wan`, `cogvideox`, `ltx-video`, `mochi`, or `hunyuan`. LTX/Mochi/Hunyuan currently fall back to `WanRuntime`; their dedicated runtimes ship in v1.next. Distinction from `image/animation`: animation is short looping clips (16 frames @ 8fps = 2s, default `loop=true`, animated WebP), video is narrative clips (5s+, single play, no loop field, mp4). The codec includes vp9 webm with vp8 fallback (when ffmpeg lacks vp9) and an explicit `UnsupportedFormatError` when neither codec is available. All bundled video models declare `device: "cuda"`; CPU inference would take 10 to 30 minutes per clip and isn't a useful default.

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
HTTP API (/v1/audio/speech, /v1/images/generations, /v1/images/segment, /v1/images/upscale, /v1/video/generations, /v1/models, /health)
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
- `audio_transcription/` was muse's first modality with multipart/form-data uploads (OpenAI Whisper wire shape). `routes.py` handles UploadFile + Form fields inline. As of v0.21.0, `image_generation/` is the second multipart consumer (`/v1/images/edits` + `/v1/images/variations`); v0.25.0 adds `image_upscale/` (`/v1/images/upscale`) as the third. All three implement multipart inline; if a fourth multipart modality lands, factor out to `muse.modalities._common.uploads`.
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

When the supervisor is running and `MUSE_ADMIN_TOKEN` is set, the same
`muse models enable/disable` commands route through the admin API
instead, so the running gateway picks up the change live (worker spawn
or unload). The catalog flip + worker mutation are part of the same
operation.

## Admin REST API

`muse.admin/` provides eleven endpoints under `/v1/admin/*` for runtime
model control. Admin is closed-by-default: every request returns 503
`admin_disabled` until `MUSE_ADMIN_TOKEN` is set in the supervisor's
environment, after which `Authorization: Bearer <token>` is required.

Endpoints:

| Endpoint | Sync/async | Effect |
|---|---|---|
| `POST /v1/admin/models/{id}/enable` | async (202+job_id) | spawn or restart-in-place |
| `POST /v1/admin/models/{id}/disable` | sync | unload + catalog flip |
| `POST /v1/admin/models/{id}/probe` | async | run `muse models probe` in venv |
| `POST /v1/admin/models/_/pull` | async | run `muse pull <body.identifier>` |
| `DELETE /v1/admin/models/{id}?purge=bool` | sync | refuses 409 if loaded |
| `GET /v1/admin/models/{id}/status` | sync | merged catalog + worker view |
| `GET /v1/admin/workers` | sync | list workers + pid/uptime/restarts |
| `POST /v1/admin/workers/{port}/restart` | sync | SIGTERM; monitor handles bringup |
| `GET /v1/admin/memory` | sync | psutil + pynvml + per-model breakdown |
| `GET /v1/admin/jobs/{job_id}` | sync | one job; 404 once reaped |
| `GET /v1/admin/jobs` | sync | recent jobs newest-first |

State management:
- `SupervisorState` (in `muse.cli_impl.supervisor`) is a module-level
  singleton holding the live worker list, device flag, and an RLock.
  `run_supervisor` registers it on boot; admin endpoints reach it via
  `get_supervisor_state()`. The auto-restart monitor reads
  `state.workers` directly so admin mutations show up on the next tick.
- `JobStore` (in `muse.admin.jobs`) is an in-memory map of async jobs
  with 10-minute retention (lazy reap on every list call). Each
  enable/pull/probe spawns a daemon thread tracked by the JobStore;
  `get_default_store().shutdown()` joins them on gateway exit.
- One global RLock guards SupervisorState mutations; per-model locks
  are deferred until contention becomes measurable.

Token leakage rules: the configured token is never echoed in any
error message, log line, or job record. `tests/admin/test_e2e_admin_router.py
::TestAuthEnvelope::test_token_never_appears_in_error_body` is a
regression watchdog.

`muse.admin.client.AdminClient` wraps every endpoint. `wait(job_id)`
polls `/jobs/{id}` until done/failed for "fire and block" usage. The
CLI's `muse models enable/disable` falls back to AdminClient when
`MUSE_ADMIN_TOKEN` is set and the supervisor is reachable, otherwise
to the legacy catalog-only mutation with a warning.

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

## Memory accounting

Three sources of truth, in order of fidelity:

1. **`muse models probe <id>`** (most honest). Loads the model in
   isolation in its per-model venv, runs a representative inference
   (per-modality default shape from `PROBE_DEFAULTS`), captures peak
   VRAM/RAM via `torch.cuda.max_memory_allocated()` on GPU or process
   RSS on CPU. Persists per-device measurement to `~/.muse/catalog.json`
   under `measurements.<device>`. Default runs inference;
   `--no-inference` is a faster load-only mode that undersells peak.
2. **`capabilities.memory_gb` annotation** (peak-inference estimate).
   Hand-set per-model from architecture knowledge, conservative.
   Used by `muse models list` until probe measurements exist; shown
   with a `~` prefix.
3. **No data**: `-` in the list. Run probe to populate.

`muse models list` picks the most honest available number per row,
tagged GPU or CPU based on `capabilities.device`. The footer aggregates
GPU and CPU separately across enabled models.

`muse models info <id>` shows annotation and probe measurement
side-by-side, including the inference shape that produced the peak
and the date probed.

**Memory is a function of input shape, not a single number.** Whisper
at 30s audio uses different VRAM than at 5min. SDXL-Turbo at 512^2 uses
~half as much as at 1024^2. AnimateDiff at 8 frames uses roughly half
as much as at 16 frames. The `memory_gb` annotation reflects a
typical-shape peak. Probe measures the actual default shape. Future
versions may add `--shape preset=small|medium|large` sweeps to map the
full curve.

Each modality declares its representative inference via a
`PROBE_DEFAULTS = {"shape": ..., "call": lambda m: ...}` dict in its
`__init__.py`. The probe worker imports the modality at run time and
calls the shape-default lambda against the loaded backend.

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
