# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Muse is a multi-modality generation server and client. It currently supports
twenty-five modalities:

- **audio/alignment**: trusted reference transcript to word timestamps via `/v1/audio/alignments` (Qwen3 ForcedAligner 0.6B curated; multipart audio + text; 11 languages; per-word start/end and uncalibrated timestamp confidence)
- **audio/classification**: audio event / emotion / language classification via `/v1/audio/classifications` (ast-audioset bundled, 527-class multi-label; emotion + speech-commands + language-ID via the resolver; multipart upload + per-input list of `{label, score}` pairs mirroring `/v1/text/classifications`; decodes at most `capabilities.max_duration_seconds`, default 60 s, and ignores the rest)
- **audio/embedding**: audio-to-vector via `/v1/audio/embeddings` (mert-v1-95m bundled; CLAP, MERT, wav2vec family via the resolver; multipart upload + OpenAI-shape envelope mirroring `/v1/embeddings`)
- **audio/generation**: text-to-music + text-to-SFX via `/v1/audio/music` and `/v1/audio/sfx` (Stable Audio Open 1.0 bundled; ACE-Step v1 3.5B for full songs with optional structured lyrics + instrumentals; per-model capability gates on `supports_music` / `supports_sfx`; optional `lyrics` field on `/v1/audio/music`)
- **audio/quality**: single-ended speech naturalness and audio production-quality assessment via `/v1/audio/quality` (UTMOS 1-5 naturalness MOS + Meta Audiobox Aesthetics four-axis 1-10 scoring via curated HF models; multipart upload; named scaled scores with a model-selected `primary_score`)
- **audio/speech**: text-to-speech via `/v1/audio/speech` (Kokoro with `timestamps` / `pronunciations`, Soprano, Bark, Supertonic-3: the last is an ONNX on-device CPU engine, 31 languages, preset voice styles)
- **audio/transcription**: speech-to-text via `/v1/audio/transcriptions` and `/v1/audio/translations` (Qwen3-ASR-1.7B transcripts and Systran faster-whisper family; any CT2 Whisper on HF)
- **chat/completion**: text-to-text LLMs AND vision-language models via `/v1/chat/completions` (OpenAI-compatible incl. tools + streaming + multimodal `messages` content; powered by llama-cpp-python for GGUF chat; `transformers.AutoModelForImageTextToText` for VLMs; SmolVLM, Qwen2-VL, LLaVA via the resolver)
- **decision/evaluation**: typed Choice / Score / Noul decisions via `/v1/systemone` (a text-only subset of TypeSafe's System One wire shape; explicit `model` required; Decision-0.8B Q8 GGUF scored by option-letter logits and Decima-base INT8 ONNX on CPU, both exact-repo and commit-pinned; complete probability distributions, never generated text; see docs/DECISIONS.md)
- **embedding/text**: text-to-vector via `/v1/embeddings` (MiniLM, Qwen3-Embedding, NV-Embed-v2; any sentence-transformers HF repo via the resolver)
- **image/animation**: text-to-animation via `/v1/images/animations` (AnimateDiff: 16-frame loops, animated WebP/GIF/MP4 output)
- **image/embedding**: image-to-vector via `/v1/images/embeddings` (dinov2-small bundled; CLIP, SigLIP, DINOv2 family via the resolver; OpenAI-shape wire envelope mirroring `/v1/embeddings`)
- **image/generation**: text-to-image and img2img via `/v1/images/generations`, plus multi-reference edits on `/v1/images/edits` (FLUX.2 klein 4B curated as `flux2-klein-4b`, klein base 4B as `flux2-klein-base-4b` with the `klein-pixel-walk` 4x4 walk-sheet LoRA, SD-Turbo, SDXL-Turbo, FLUX.1-schnell, any diffusers HF repo; LoRA adapters via lora_adapter/base_model capabilities (pixel-art-xl curated, pre-paired with sdxl-turbo; per-request lora_scale; muse pull --base overrides the declared base))
- **image/cv**: image CV primitives (depth, keypoints, object detection) via `/v1/images/depth`, `/v1/images/keypoints`, `/v1/images/detect` (depth-anything-v2-small + vitpose-base-simple + detr-resnet-50 bundled; ZoeDepth + DPT-large + RT-DETR via the resolver; multipart upload; capability flags `supports_depth` / `supports_keypoints` / `supports_detection` gate per route; depth supports `png16` (16-bit grayscale, default) or `float32` raw bytes via `response_format`)
- **image/ocr**: image-to-text via `/v1/images/ocr` (trocr-base-printed bundled; Nougat / TexTeller / TrOCR-handwritten via the resolver; multipart upload, optional `prompt` for task hints, optional `max_new_tokens`; OpenAI-shape envelope mirroring `/v1/audio/transcriptions`)
- **image/pixelart**: deterministic pixel-art tools via `/v1/images/pixelize` (grid snap, named/auto palettes, hard alpha, cleanup, outline, seamless, shared-palette sheets) and `/v1/images/autotile` (RPG Maker A2 block + blob-47 sheet + Godot 4 `.tres` / Tiled `.tsx`) (pixel-tools bundled; CPU, weightless; multipart upload; JSON or raw PNG / zip; see docs/PIXELART.md)
- **image/segmentation**: promptable segmentation via `/v1/images/segment` (sam2-hiera-tiny bundled; SAM-2 family via the resolver; multipart upload, mode dispatch auto/points/boxes/text gated by capability flags; masks emitted as base64 PNG or muse's `p:` run-length format, not COCO-decodable)
- **image/upscale**: image-to-image super-resolution via `/v1/images/upscale` (stable-diffusion-x4-upscaler bundled; multipart upload, OpenAI-shape envelope; 4x diffusion-based upscaling; capability-gated on `supported_scales`; env-tunable input cap via `MUSE_UPSCALE_MAX_INPUT_SIDE`)
- **image/vectorization**: raster-to-static-SVG via `/v1/images/vectorize` (StarVector-1B im2svg bundled; multipart upload; JSON or raw `image/svg+xml`; pinned, local architecture with no Hub code execution; active SVG content rejected)
- **text/classification**: text moderation via `/v1/moderations` and full label distribution via `/v1/text/classifications` (any HF text-classifier or zero-shot NLI; bundled twitter-roberta sentiment + deberta-v3 zero-shot; capability-gated dispatch on `supports_classification` / `supports_zero_shot`)
- **text/rerank**: cross-encoder rerank via `/v1/rerank` (bge-reranker-v2-m3 bundled; any cross-encoder reranker on HF; Cohere-compat wire shape)
- **text/summarization**: BART/PEGASUS seq2seq summarization via `/v1/summarize` (bart-large-cnn bundled; any summarization-tagged HF repo via the resolver; Cohere-compat wire shape)
- **text/translation**: machine translation via `/v1/translate` and the bare `/translate` alias, plus `GET /languages` (LibreTranslate-compat wire shape; m2m100-418m bundled, ~2GB CPU-runnable; NLLB-200-distilled-600M and Opus-MT en-es/en-de curated via the resolver; the gateway's `MODEL_OPTIONAL_PATHS` default-model seam resolves the model when LibreTranslate clients omit `model` entirely)
- **video/generation**: text-to-video via `/v1/video/generations` (wan2-1-t2v-1-3b bundled with sequential CPU offload by default to fit 8-12GB GPUs; Wan / CogVideoX / LTX-Video families via the resolver; narrative clips up to 30s; mp4/webm/frames_b64 output; GPU-required; `cpu_offload` capability (model|sequential) + `vae_tiling`, globally overridable via `server.video_cpu_offload`)
- **3d/generation**: image-to-3d via `/v1/3d/from-image` and text-to-3d via `/v1/3d/generations` (triposr bundled; TRELLIS, Hunyuan3D-2, Shap-E via the resolver; Shap-E is text-to-3D only via `diffusers.ShapEPipeline`; GLB output as `data:model/gltf-binary;base64,...` URL or b64_json; capability flags `supports_text_to_3d` / `supports_image_to_3d` gate per route; work and GLB encoding off the event loop like video/generation)

Per-modality wire details, runtimes, caps and history are in `docs/MODALITIES.md`
(read the modality's section before changing it). Topic docs: `docs/PIXELART.md`,
`docs/DECISIONS.md`, `docs/CHAT_COMPLETION.md`, `docs/AGENT_DISCOVERY.md`,
`docs/STORAGE.md`, `docs/CONFIG.md`, `docs/RESOLVERS.md`, `docs/HF_PLUGINS.md`,
`docs/MODEL_SCRIPTS.md`; runtime internals in `docs/INTERNALS.md`.

Modality tags are MIME-style (`audio/speech`, not `audio.speech`). The HTTP
path hierarchy mirrors the OpenAI shape where possible (`/v1/audio/speech`,
`/v1/chat/completions`, `/v1/embeddings`, `/v1/images/animations`,
`/v1/images/generations`) for client compatibility.

In addition to the per-modality routes, muse also exposes an admin
REST API under `/v1/admin/*` (v0.28.0+) for runtime model control
without restarting `muse serve`: enable, disable, probe, pull, remove,
plus worker introspection and async-job tracking. Closed-by-default
behind `MUSE_ADMIN_TOKEN`. See "Admin REST API" below.

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
- `muse.core.catalog.known_models()`: discovery-driven, two-tier cache.
  The script-discovery scan (importlib) is cached for the process
  lifetime (new scripts need a restart); the merged result is memoized
  against catalog.json's file identity (path, dev, ino, size, mtime_ns,
  ctime_ns), so catalog writes from ANY
  process (admin pull subprocess, operator CLI pull/remove beside a
  running supervisor) are visible on the next call without a restart.
  Projects each script's MANIFEST onto the `CatalogEntry` shape
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
  GGUF `@variant` is required; no magic default. A split variant
  (`*-00001-of-0000N.gguf`) downloads every shard, records the first as
  `gguf_file` (what llama.cpp opens), and fails to resolve if a shard is
  missing from the repo (`muse.core.gguf.gguf_shard_files`). Search implemented for
  both modalities with per-variant deduping (sharded GGUFs don't emit
  one row per shard). Registers `hf://` scheme on import.
- `muse.core.curated`: loads `src/muse/curated.yaml` (hand-edited
  recommendations list). `find_curated(id)` / `find_curated_by_uri(uri)`.
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
- Weightless bundled models: a bundled script may set `MANIFEST["weights"] = "none"` and omit `hf_repo` / `revision`; it must declare a positive `capabilities.memory_gb` (admission sizing has nothing on disk to measure). `muse pull` then installs the venv only, creates an empty `<catalog>/local/<model_id>` as `local_dir`, and writes a code receipt `{kind: "code", model_id, muse_version, source_sha256}` instead of a weights receipt. `/v1/models` reports `revision: null` and there is no `X-Muse-Model-Revision`; the one `code` component (`muse/<id>`) has `revision: "muse-<running version>"` in `/v1/capabilities`, and `X-Muse-Model-Artifacts` is the first 16 hex of `sha256(f"{source_sha256}:{muse.__version__}")`, so a muse upgrade that changes the library the alias script imports changes the digest. The receipt's `muse_version` is recorded but never enforced at load. Code: `catalog_pull`, `catalog`, `provenance`; tests `tests/core/test_weightless_models.py`. `pixel-tools` is the first.
- `muse.core.securefs`: the one implementation of owned-directory checks, no-follow regular-file opens, advisory locks and atomic fsynced writes; catalog, resource_registry and config call it with their own error class.
- `muse.core.owned_process`: the one implementation of owned subprocess groups (new session per child, PID==PGID validation, WNOWAIT leader pinning so a group is only signalled while its unreaped leader holds the id, never our own group, TERM-then-KILL deadlines, bounded output tails). Used by the installer runner, admin jobs and the supervisor's worker teardown and restart. Where `os.waitid` is missing (macOS on Python 3.12 and older), a leader is pinned by psutil instead: zombie status plus `ppid() == os.getpid()`, read without reaping, so it still holds its id through the final group KILL. A child that can't be pinned (no psutil, or an unvalidated group) falls back to `Popen.poll()` under its owning lock. Only owners reap: `GET /v1/admin/memory` observes with `may_reap=False`, and `tests/core/test_child_reaping_guard.py` fails on any other reaping or signalling call. macOS `killpg` returns EPERM (not ESRCH) when every group member is a zombie (XNU `killpg1` skips zombies), and `getpgid` returns ESRCH for a zombie: `signal_isolated_group(..., pinned=True)` counts ESRCH/EPERM as delivered only when a non-reaping re-observation right after the signal, under the owner's lock, shows the leader is our unreaped exited child (on Linux too, where EPERM can occur if no member accepts the signal; the guarantee stays "every member we may signal was signalled"); unpinned callers fail closed. `validated_isolated_group(..., new_session=True)` proves a pinned zombie's PGID from its `start_new_session` spawn, after a bounded (<= 80 ms) retry of the exit observation. `tests/darwin_process_semantics.py` emulates both macOS rules on Linux (`+darwin` test params; `MUSE_TEST_DARWIN_PROCESS_SEMANTICS=1` suite-wide).
- `muse.core.venv`: venv creation and `venv_transaction` (rollback-safe in-place mutation). Each `muse pull` creates `~/.muse/venvs/<model-id>/`; catalog records the `python_path`. Siblings: `pip_install` (`install_into_venv`, `pip_install_commands`), `python_sources` (pinned git sources), `owned_command` (`run_owned_command`, the installer's owned-process runner), `ports` (`find_free_port`).
- `muse.core.catalog` is the facade (enable/device/GPU-layer state, `load_backend`, provenance checks) over `catalog_store` (catalog.json I/O, schema, read cache, `_CATALOG_WRITE_LOCK`), `catalog_locks` (pull lock, storage-cache lock, resource lease), `catalog_models` (`CatalogEntry`, discovery caches, `known_models`, `get_manifest`), `catalog_pull` (`pull()` and its transactions) and `catalog_remove`. Tests patch names where they are looked up, i.e. in the submodule.
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
- `audio_transcription/` was muse's first modality with multipart/form-data uploads (OpenAI Whisper wire shape). `routes.py` handles UploadFile + Form fields inline. As of v0.21.0, `image_generation/` is the second multipart consumer (`/v1/images/edits` + `/v1/images/variations`); v0.25.0 adds `image_upscale/` (`/v1/images/upscale`) as the third. Shared helpers live in `muse.modalities._common` (`uploads`: bounded upload reads, spooled temp files that are removed only after the backend returns, the fallback decode-error heuristic; `audio_decode`: TorchCodec decoding with typed errors; `audio_codec`: PCM16/WAV/Opus; `streaming`: the producer-thread SSE helper; `inference`: `run_locked`, `cap`, `check_text_batch`, `offload_call`, `result_images`, `produce_each`; `fields`: shared Form-field parsing; `video_codec`: mp4/webm/frames encoding; `gguf`: llama.cpp loading; `client`: server URL and upload coercion; `probe_audio`: the probe WAV; see docs/MODALITIES.md). Image inputs go through `_common/images` (typed 413/400 errors, decode off-loop); image routes run backend call plus encoding through `_common/inference.offload_call`. Routes look backends up with `registry.get_or_404(modality, model)` and read `registry.capabilities(modality, model)` (always a dict). Positive size caps read `config.get_positive(key)`.
- `text_classification/` is muse's first modality whose internal MIME tag (`text/classification`) is broader than its primary URL route (`/v1/moderations`). The wire path is OpenAI-specific; v0.35.0 added the second route `/v1/text/classifications` on the same modality tag, sharing dataclasses + protocols. Two runtimes coexist: `HFTextClassifier` (existing, for fine-tuned classifier heads) and `HFZeroShotPipeline` (v0.35.0, wraps `transformers.pipeline("zero-shot-classification")`). Capability flags `supports_classification` and `supports_zero_shot` on the manifest gate which routes / runtime calls each model accepts: a request with `candidate_labels` against a model with `supports_zero_shot=False` returns 400 `zero_shot_not_supported`; the inverse mismatch returns 400 `candidate_labels_required`, and `/v1/moderations` against a zero-shot-only model returns 400 `classification_not_supported`. The HF plugin's `_resolve` dispatches to either runtime based on tag (`zero-shot-classification`) or repo-name fallback (`zero-shot`, or an `mnli`/`xnli`/`nli` name token). See docs/MODALITIES.md for the shared text caps.

Four distinct concepts worth keeping straight:

| Surface | Who writes it | Purpose |
|---|---|---|
| `muse/models/*.py` | bundled muse + users | public model scripts, one per model, discoverable |
| `modalities/*/runtimes/*.py` | muse internal | generic runtimes, one class serves many models (GGUF, ST) |
| `modalities/*/backends/*.py` | muse internal | private helpers shared inside a modality |
| `muse.core.runtime_helpers` | muse internal | cross-modality utilities every runtime imports: `select_device` (cuda/mps/cpu auto-detect), `dtype_for_name` (string-to-torch.dtype map with `fp16`/`bf16`/`fp32` aliases), `set_inference_mode` (no-grad switch; literal token kept out of caller bodies via shared helper), `LoadTimer` (opt-in load-time logging context), `GGUF_DEFAULT_CONTEXT_LENGTH` (the 8192 default both the GGUF runtime and the chat resolver use). Added v0.31.0; consolidated ~30 per-runtime copies. The meta-test `tests/core/test_runtime_helpers_meta.py` AST-walks every runtime and bundled script to flag re-implementations |

Each model script (`src/muse/models/<id>.py`) contains:
- Top-level `MANIFEST: dict` with required keys `model_id`, `modality`, `hf_repo` and optional `description`, `license`, `pip_extras`, `system_packages`, `capabilities`. Anything else passes through.
- A class named exactly `Model` (tests alias it: `from muse.models.kokoro_82m import Model as KokoroModel`).
  When a generic runtime already serves the model, the script just aliases it
  (`from muse.modalities.<m>.runtimes.<x> import <Runtime> as Model`) and puts
  per-model defaults in `MANIFEST["capabilities"]`, which `load_backend`
  splats into the constructor (plus `revision`); most bundled scripts do.

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

### Device placement precedence (v0.48.0+)

Which device a model loads on is resolved in `load_backend`, most
authoritative first:

1. **catalog `device_override`** -- an operator pin set via
   `muse models set-device <id> <auto|cpu|cuda|mps>` (clear with
   `--clear`). Stored as a top-level field on the model's catalog entry
   (NOT in the manifest), read live from `catalog.json` at load time.
   Beats everything below, so an operator can force a cpu-pinned model
   onto cuda (or pin a model to cpu to save VRAM) per deployment without
   editing the bundled script. `set-device <id> auto` un-pins to
   auto-detect.
2. **manifest `capabilities.device` pin** (model-author affinity) -- a
   value other than `"auto"` overrides the `--device` flag. Heavy
   GPU-only models declare `"cuda"`; the CPU-only models
   (`supertonic-3`, ONNX, and the weightless `pixel-tools`) declare `"cpu"`.
3. **`--device` server flag** (`muse serve --device ...`, default
   `auto`), folded into the runtime kwargs.
4. **`"auto"`** -- the runtime's `select_device` picks cuda if available,
   else mps, else cpu.

**Default is `auto`, not `cpu` (v0.48.0).** Every CUDA-safe bundled model
declares `device: "auto"` so it lands on the GPU automatically on a GPU
host and on CPU on a CPU-only host -- generally faster, and safe because
lazy-load + LRU/idle eviction size the live working set to fit VRAM. Only
`supertonic-3` (its ONNX runtime ignores a cuda device) and `pixel-tools`
(CPU-only Pillow/numpy) stay pinned `cpu`. `set-device` is catalog state: it takes effect on the model's
**next cold load**, so evict or restart an already-resident worker to
apply it.

**`auto` resolves to a concrete memory POOL consistently across the
control plane (v0.48.0).** An `auto` model that loads on the GPU must be
*sized, admitted, and evicted* against the VRAM pool, not host RAM -- else
the LoadDirector would think it "fits" against terabytes of host RAM while
the GPU OOMs. Three sites share one resolution: `LoadDirector._resolve_pool_device` (admission /
LRU eviction), `supervisor._servability_reason` (boot + request-path
capacity check), and `memory._resolve_auto_side` (the `/v1/admin/memory`
per-model breakdown, which also honors the supervisor `--device` flag
first). The sizing ladder reads an `auto` model's (and an `auto` LoRA's
base's) `measurements.<pool>` the same way (`supervisor._measurement_key`),
falling back to another device's measurement only when that pool has none,
so a catalog copied from a GPU box does not size a CPU-only host's load from
a VRAM peak. All four (director, servability, the ladder and the admin
breakdown) call one predicate, `memory_probe.auto_uses_cuda` (live NVML
reading, else the torch CUDA/ROCm detector). A probe whose inference fails
keeps an earlier, larger successful inference peak rather than replacing it
with its load-only peak. `mps` is its own case: it pools against host RAM for sizing
(pynvml is CUDA-only, and MPS is unified memory) but keeps its own `mps`
measurements bucket in `_record_observed_peak`, matching what
`muse models probe` persists. This bug never surfaced in CI because a
CPU-only host degrades `auto`->cpu, which happens to match the buggy
CPU-pool accounting; it only bites on a GPU host.

GGUF extras (details in `docs/INTERNALS.md`): `muse models set-gpu-layers <id> <N|--all|--clear>`
pins llama.cpp `n_gpu_layers` (pin > `capabilities.n_gpu_layers` > -1; refused for non-GGUF);
`install.llama_cpp_wheel: cu130` installs CUDA llama-cpp-python wheels binary-only, and
`llama_cuda.prepare_llama_cpp()` must run before `import llama_cpp`; the chat resolver caps
`context_length` at 8192. Storage: `muse.core.venv_dedupe` hardlinks identical venv files
(`muse storage dedupe`), `muse.core.hf_seed` hardlinks shared-HF-cache blobs into
`<catalog_dir>/weights` before downloads (see `docs/STORAGE.md`). A split GGUF is accounted
against one pool only (known limitation).

### No shared supertype across modalities

`AudioResult` and `ImageResult` do NOT share a common base. Streaming semantics
differ (audio chunks are time-ordered and playable immediately; diffusion steps
are progressive refinement of one frame). A `GenerationModel` abstract base
would be a leaky abstraction. Instead, `ModalityRegistry` treats models as
`Any`, and each modality's router + codec knows its own types.

## Process model

`muse serve` is a **supervisor**, not a single process. As of v0.40.0 it
is **lazy by default**: the gateway is healthy as soon as the supervisor
boots, and per-model workers spawn on the first request that names them
(see "Lazy load + LRU eviction" below for the full state machine):

```
User request
    |
    v
muse serve (supervisor, port 8000)              [healthy instantly; zero workers]
  ├── gateway FastAPI app (in-process)
  │    routes by request body `model` field
  │    on cold model: director.acquire() spawns the worker
  │
  └── subprocess per loaded model (spawned lazily):
       ├── worker (port 9001, venv-kokoro)   spawned on first /v1/audio/speech
       ├── worker (port 9002, venv-sd-turbo) spawned on first /v1/images/generations
       └── (other catalog-enabled models stay unloaded until requested)
```

Each pulled model gets its own venv at `~/.muse/venvs/<model-id>/`
with exactly the pip_extras it declares. Workers run the existing
`muse.cli_impl.worker.run_worker` logic via `muse _worker`
(hidden subcommand). The supervisor brings the gateway up immediately,
runs catalog boot validation (stamps an advisory `unservable_reason` on
models that are not sizable at all or whose estimate exceeds device
capacity; the stamp is re-checked live on first request, see "Lazy load
+ LRU eviction" below), then waits. Workers spawn on demand through
`LoadDirector.acquire`; the historical
"poll FIRST worker then promote others" boot dance is gone. Operators
who want eager loading run `muse models warmup <id>` for each model in
their startup script (see "Lazy load + LRU eviction" below).

The gateway extracts `model` from the request body (POST) or query
(GET), calls `director.acquire(model_id)` to ensure the worker is
loaded (cold loads pay one-shot latency; hot loads return in
sub-millisecond), forwards the request, and calls
`director.release(model_id)` on stream-close. `/v1/models` and
`/health` are aggregated across the live (loaded) workers via parallel
httpx calls; entries for catalog-enabled-but-unloaded models still
appear in `/v1/models` with `loaded: false` and the SDK can decide
whether to warm them.

This gives you dep isolation (transformers 4.46 for parler-tts
coexists with transformers 5.x for newer models), crash isolation
(a segfault in one worker does not kill the rest), and a uniform
HTTP surface (clients hit one port, do not care about internal venvs).

The supervisor also runs an auto-restart monitor thread. Every 5
seconds it polls each currently loaded worker's /health and checks for
process death through the worker's owned-process observation
(`muse.core.owned_process`, under the worker's `process_lock`, never
reaping outside the owner). After 3 consecutive failures (or
immediate process exit), the monitor terminates the existing process
and respawns it with exponential backoff (1s, 2s, 4s, ..., capped at
30s). After 10 unsuccessful restart attempts the worker is marked
dead; /health reports "degraded" and /v1/models skips its entries.
The monitor reads the dynamic loaded set from `state.director.loaded`
on each tick, so admin-driven loads and evictions show up immediately.

Use `muse models disable <id>` to mark a pulled model as inactive in
the catalog (the supervisor refuses to lazy-load it on request).
`muse models enable <id>` re-enables it. These flip the catalog
`enabled` bit; whether a model is *actually* in memory is a separate
runtime question handled by the LoadDirector (see "Lazy load + LRU
eviction" below). When the supervisor is running and `MUSE_ADMIN_TOKEN`
is set, the same CLI commands route through the admin API; otherwise
they edit the catalog only and take effect on the next request.

## Lazy load + LRU eviction (v0.40.0+)

Full design, sizing ladder and lock discipline: `docs/INTERNALS.md`. The invariants:

- `enabled` is catalog state; `loaded` is runtime state. Workers spawn on first request through
  `LoadDirector.acquire` (`muse.cli_impl.load_director`) and are evicted by LRU under memory
  pressure or by the idle sweeper (`server.idle_timeout_seconds`, default 600; per-model
  `capabilities.idle_timeout_seconds` wins; `<= 0` disables). Lazy-load/eviction use
  `load_model_into_worker` / `unload_model_from_worker`, which never flip `enabled`.
- The gateway runs `acquire` off the event loop (`asyncio.to_thread` under `shield`), coalesces
  same-model cold loads on one per-model future (`_acquire_coalesced`), and reserves in-flight
  memory per pool so concurrent cold loads of different models cannot over-commit. A cancelled
  request still releases its refcount (done-callback); `_forward` cleans up on `BaseException`.
- Memory is live (pynvml / psutil minus `server.*_headroom_gb`, capped by `server.*_budget_gb`).
  A model's size comes from the four-step ladder in `supervisor._has_memory_data`:
  (1) `capabilities.memory_gb` or a probe measurement for the resolved pool, (2) for a LoRA, its
  base's estimate (never the base's weights on disk), (3) weights on disk (one GGUF file; one
  variant per diffusers component), (4) unservable. Boot validation, the request-path re-check
  (`_servability_reason`) and `backfill_manifest_memory` share it; `auto` resolves to a pool
  through `memory_probe.auto_uses_cuda` everywhere. Observed peaks self-heal upward after solo
  loads only.
- The probe (`muse models probe`, also run by `muse pull`) is a gate on the pull: a failed load
  or inference (including degenerate image output) exits nonzero and disables the model; a
  failed probe never replaces a larger earlier inference peak.
- Request queueing (v0.55.0): per-model `capabilities.max_concurrency` (else
  `server.default_max_concurrency`, 0 = unlimited) parks requests FIFO on the loop; transient
  capacity 503s (`retryable=True`) wait on `CapacityNotifier` under `server.queue_timeout_seconds`
  (default 300); `server.max_queue_depth` fast-fails `queue_full`. A modality's literal
  `CONCURRENCY_CEILING` clamps every model of that modality (decision/evaluation: 1 / 32).

## Configuration

All server settings live in ONE declarative registry,
`muse.core.config.SETTINGS` (v0.52.0+). Each setting is one `Setting` row
(`key`, `env`, `type`, `default`, `group`, `help`); env reads, file reads,
`muse config`, and the generated template all derive from that single
list. There is no parallel lookup table. See docs/CONFIG.md for the full
settings inventory.

Resolution precedence (first wins): **CLI-override arg > `MUSE_*` env var >
`~/.muse/config.yaml` > built-in default.** The config file lives at
`<MUSE_CATALOG_DIR or ~/.muse>/config.yaml` (override the whole path with
`MUSE_CONFIG`). The two bootstrap keys (`paths.catalog_dir`,
`paths.config_file`) resolve from env+default ONLY -- the file cannot
redirect the path that locates the file. `config.get(key)` reads env LIVE
on every call and parses the file once (cached); it is LENIENT (a bad
value logs a warning and falls back to the default, never raises), so a
typo in one setting cannot 500 the request path. `muse config set` is the
STRICT path (validates and refuses a bad value).

`muse config` CLI verbs:
- `muse config generate [--force]` -- write a fully-commented `config.yaml`
  from the registry (every setting, its default, its env name; every leaf
  commented out, so it overrides nothing until edited).
- `muse config show [--json]` -- every setting's effective value AND source
  (default / file / env). `admin.token` is redacted to `set` / `unset`.
- `muse config path` -- print the resolved config-file path.
- `muse config get <key>` -- print one effective value.
- `muse config set <key> <value>` -- validate and write one value into the
  file (atomic, preserves other keys).
- `muse config unset <key>` -- remove one setting from the file so it falls
  back to env/default (the counterpart to `set`; no override value means
  "use the default", so reverting a key requires removing it).

Scope boundary: `config.yaml` is for SERVER / global settings only.
Per-model state (enable/disable, `device_override` from
`muse models set-device`, probe `measurements`) lives in
`~/.muse/catalog.json` and is edited via `muse models ...`. Shipped
recommendations live in `src/muse/curated.yaml` (package data, not
per-deployment editable). A setting can have both a global default and a
per-model override (idle-timeout is the canonical example: global
`server.idle_timeout_seconds` vs per-model
`capabilities.idle_timeout_seconds`, per-model wins).

## Admin REST API

`muse.admin/` provides twelve endpoints under `/v1/admin/*` for runtime
model control. Admin is closed-by-default: every request returns 503
`admin_disabled` until `MUSE_ADMIN_TOKEN` is set in the supervisor's
environment, after which `Authorization: Bearer <token>` is required (scheme
case-insensitive). `muse.admin.auth.check_bearer` is the one fastapi-free
bearer check; the dashboard uses it too. Bad bodies or path params are 400
`invalid_parameter` envelopes, never a 202 job that fails later (pull with
`{"identifier": ""}` is now a 400; omit it to use the path id).

Endpoints:

| Endpoint | Sync/async | Effect |
|---|---|---|
| `POST /v1/admin/models/{id}/enable` | async (202+job_id) | spawn or restart-in-place |
| `POST /v1/admin/models/{id}/disable` | sync | unload + catalog flip |
| `POST /v1/admin/models/{id}/warmup` | sync (blocks on the load) | load via the director without a refcount |
| `POST /v1/admin/models/{id}/probe` | async | run `muse models probe` in venv |
| `POST /v1/admin/models/_/pull` | async | run `muse pull <body.identifier>` |
| `DELETE /v1/admin/models/{id}?purge=bool` | sync | refuses 409 if loaded |
| `GET /v1/admin/models/{id}/status` | sync | merged catalog + worker view |
| `GET /v1/admin/workers` | sync | list workers + pid/uptime/restarts |
| `POST /v1/admin/workers/{port}/restart` | sync | SIGTERM; monitor handles bringup; revives a dead worker (status unhealthy, restart budget reset) only if its models are still in `director.loaded`, else 409 `worker_not_revivable` |
| `GET /v1/admin/memory` | sync | psutil + pynvml + per-model breakdown (incl. live `refcount` when a director is bound) |
| `GET /v1/admin/jobs/{job_id}` | sync | one job; 404 once reaped |
| `GET /v1/admin/jobs` | sync | recent jobs newest-first |

State management:
- `SupervisorState` (in `muse.cli_impl.supervisor`) is a module-level
  singleton holding the live worker list, device flag, and an RLock.
  `run_supervisor` registers it on boot; admin endpoints reach it via
  `get_supervisor_state()`. The auto-restart monitor reads
  `state.workers` directly so admin mutations show up on the next tick.
- `JobStore` (in `muse.admin.jobs`) is an in-memory map of async jobs
  with 10-minute retention (lazy reap on every list call). Job children
  never inherit `MUSE_ADMIN_TOKEN`; subprocess output is kept once, in
  `log_lines` (`result` is `{op, returncode}`). Each
  enable/pull/probe spawns a daemon thread tracked by the JobStore;
  `get_default_store().shutdown()` joins them on gateway exit.
- One global RLock guards SupervisorState mutations; per-model locks
  are deferred until contention becomes measurable.

Token leakage rules: the configured token is never echoed in any
error message, log line, or job record. `tests/admin/test_e2e_admin_router.py
::TestAuthEnvelope::test_token_never_appears_in_error_body` is a
regression watchdog.

Admin error envelope: auth failures use the bare OpenAI envelope
`{"error": {code, message, type}}`, matching the route-level admin errors
(which return it directly via `JSONResponse`). The auth dependency
(`verify_admin_token`) raises `HTTPException(detail={"error": ...})`,
which FastAPI's default handler would double-wrap as `{"detail":
{"error": ...}}`; `muse.admin.errors.install_admin_error_handler`
(installed by `build_gateway` right after mounting the admin router)
unwraps any HTTPException whose `detail` is already OpenAI-shaped, renders
`OperationError` from any admin route as the same envelope, and delegates to
FastAPI's default for everything else (so plain-string
details elsewhere keep their `{"detail": ...}` shape). Safe-degrading: an
includer that forgets the handler still gets a valid (if double-wrapped)
error, never a 500. `AdminClient` parses both shapes (v0.47.4).

`muse.admin.client.AdminClient` wraps every endpoint. `wait(job_id)`
polls `/jobs/{id}` until done/failed for "fire and block" usage. The
CLI's `muse models enable/disable` falls back to AdminClient when
`MUSE_ADMIN_TOKEN` is set and the supervisor is reachable, otherwise
to the legacy catalog-only mutation with a warning.

## Observability

`muse.observability` (details in `docs/INTERNALS.md`): a sparse single-table SQLite event store,
a fire-and-forget `record(...)` that never blocks or raises (bounded queue; drops are counted),
per-worker log ring buffers, a sampler, and `GET /dashboard` plus gated `/v1/telemetry/*`
endpoints. Its `__init__` imports the dashboard (fastapi) lazily: hot-path modules import
`muse.observability.recorder`, and `tests/observability/test_public_api.py` guards that
fastapi stays out of `muse --help`. Gated endpoints use the admin token, header-only, compared
with `secrets.compare_digest`; the SSE log tail authenticates with a short-lived ticket from
`POST /v1/telemetry/logs-ticket` instead (EventSource cannot set headers). `model_evict` is
emitted only after a successful teardown. Settings: `telemetry.*` (docs/CONFIG.md).

## Federation

`muse federate --port 8100 --node http://host:8000 ...` (or a `federation.yaml`) runs a coordinator
in front of unmodified `muse serve` nodes (`muse.federation` plus `muse.cli_impl.federation`).
It polls each node's `/v1/models` (and, with a per-node token, `/v1/telemetry/summary`) on a
background interval and routes each request by model locality: nodes listing the model, loaded
first, then lowest `in_flight` (in-flight requests; a node without telemetry is "unknown" and ties
with the lowest), else round-robin; one retry on connect failure, never mid-stream. Default node
names are `host:port` and must be unique. No candidate is 404, or a retryable 503 before the first
poll, with no reachable node, or when a now-unreachable node listed the model. It refuses to start
with zero nodes. The registry starts in the ASGI lifespan, so
plain `TestClient(app).get` tests never start it. Details: `docs/INTERNALS.md`; design:
`docs/superpowers/specs/2026-07-04-federation-design.md`.

## Agent discovery

A muse server describes itself from only its URL (full guide: `docs/AGENT_DISCOVERY.md`,
design: `docs/superpowers/specs/2026-10-21-agent-discovery-design.md`). Public,
read-only, never triggers a model load, computed off the event loop:

- `GET /` and `/.well-known/muse.json`: pointer document (the gateway no
  longer proxies `GET /`).
- `GET /v1/capabilities[?modality=|?model=]`: server info, modalities (summary,
  use_for/not_for, routes, examples, MCP tool names), enabled models (revision,
  components, notes: summary/strengths/limits/quality_tier, capability flags,
  live state). Built by `muse.core.capabilities` from per-modality `DISCOVERY`
  declarations (AST-read) and model notes.
- `GET /openapi.json` is the complete OpenAPI for every inference route;
  `GET /llms.txt` is rendered from the capabilities document.
- Every inference response carries `X-Muse-Model`, `X-Muse-Model-Revision`
  (40-hex HF commit, omitted if unknown) and `X-Muse-Model-Artifacts`
  (multi-component digest); chat sets `system_fingerprint` to
  `<id>@<revision[:12]>`. `/v1/models` gains `revision`. See
  `muse.core.provenance`.
- Federation: the coordinator serves the merged `/v1/capabilities`, `/`,
  `/llms.txt` and the first reachable node's `/openapi.json`; its `/docs` and
  `/redoc` are disabled.
- `modality_discovery()` adds `model_optional` to every route (true when the
  package lists the path in `MODEL_OPTIONAL_PATHS`), returning copies so the
  cached declarations are never mutated. The coordinator's merged document
  sets every `model_optional` to false (its proxy has no default-model seam).
- `/llms.txt` follows llmstxt.org (H1, blockquote, workflow paragraph,
  relative links, MCP section, only modalities with an enabled model, at most
  8 models each, an `Optional` section for the rest); the exact shape is in
  docs/AGENT_DISCOVERY.md. `server.mcp.install` pins
  `museq[mcp]>=<major>.<minor>` for a release version and leaves dev or local
  builds unpinned, so uvx does not reuse an older cached client.

## MCP server (Using muse from Claude Desktop / Claude Code)

`muse mcp` is a thin local HTTP client of a muse server. Install only the
light extra (`mcp = httpx, mcp>=1.19,<2, pillow`; the `server` extra pins the
same `mcp`):

    claude mcp add muse -e MUSE_SERVER=http://host:8000 -- uvx --from "museq[mcp]>=0.70" muse mcp

It lists `muse_describe` (the server's capabilities document), the inference
tools generated from the server's `/openapi.json` and `/v1/capabilities`
(`muse.mcp.generated`; cached 30 s), and the 11 unchanged admin tools
(`muse.mcp.tools.admin`, wrapping `/v1/admin/*`, gated by `MUSE_ADMIN_TOKEN`).
If the server is unreachable, only `muse_describe` and the admin tools are
listed, never stale generated tools. Tool counts are deliberately not
documented; they follow the server. The server advertises `tools.listChanged`
and sends `notifications/tools/list_changed` once per change (a model or
route added or removed, or the server coming up) to a session that has listed
tools: checked after each tool call / resource read, and every 30 s by a
stdio background watcher. HTTP mode is stateless, so it gets no push.
`model` is a plain string whose description lists the serving models, never
an `enum` (clients validate against their cached schema, so an enum would
reject a model enabled mid-session).

```
src/muse/mcp/
  server.py     MCPServer wrapping mcp.server.lowlevel.Server
  client.py     MuseClient (httpx + AdminClient delegation)
  generated.py  tools generated from the discovery document; request building and result rendering
  files.py      local file inputs (allowed prefixes) and hardened content-named outputs
  tools/admin.py   the 11 admin tools
src/muse/cli_impl/mcp_server.py  CLI entry (stdio or Streamable HTTP transport)
```

Transports: stdio (default) and `muse mcp --http --port 8088` (Starlette,
`/mcp` mount via `StreamableHTTPSessionManager`). `--filter all|admin|inference`
pins the surface. `--admin-token` / `$MUSE_ADMIN_TOKEN` is the admin bearer.
Long-running ops (pull, probe, enable) return a `job_id`; poll `muse_get_jobs`.

The server sends `instructions` at initialize (`muse.mcp.server.INSTRUCTIONS`):
the describe, pick a model, call, read files workflow; the path-input rules
and the `_paths` / `_urls` list forms; that admin tools need the token except
`muse_list_models` and `muse_search_models`, and that long jobs return a
job_id to poll. An admin-only or inference-only filter sends just its half.
Every tool has a `title` and `ToolAnnotations`. Generated inference tools are
never `readOnlyHint` (any can write a local file) nor `destructiveHint` (a
cold load evicting idle models is load management), are `openWorldHint` only
with `*_url` / `*_urls` inputs, and are idempotent on GET routes.
`muse_describe` and the admin read tools are read-only; `muse_remove_model`,
`muse_disable_model` and `muse_pull_model` are destructive (a re-pull
replaces weights; a failed probe disables the model); `muse_search_models`
and `muse_pull_model` are open-world. The generated `model` argument has
`minLength: 1` and is not required on a `model_optional` route.

I/O conventions:
- Media outputs are written to files and returned as path, sha256, model and
  revision. `out_path` names the file, else a content-named file lands in
  `mcp.output_dir` (env `MUSE_MCP_OUTPUT_DIR`, default `./muse-output`).
  `preview: true` on image tools adds one inline 512 px PNG. `X-Alignment` is
  surfaced as `response.alignment`.
- A DISCOVERY route may declare `mcp_defaults` (`{argument: value}`), default arguments for its generated MCP tool (`muse.core.discovery`, `muse.mcp.generated`); an explicit user argument always wins. `muse_autotile` uses `{"response_format": "zip"}` because MCP saves outputs under content-hash names, which would break the `.tres` / `.tsx` -> `<name>_blob47.png` reference; the zip keeps the five names intact.
- A DISCOVERY route may declare `text_media_fields` (`{json path: mime}`)
  for plain-text media; vectorize's `svg` is written as a UTF-8 `.svg` file.
- Path inputs (`<name>_path`) and `out_path` are allowed under the MCP
  process cwd and `mcp.output_dir` when `fetch.mcp_allowed_path_prefixes` is
  empty. A set value REPLACES the cwd (colon-separated); `mcp.output_dir` is
  always allowed, for reads and writes, so a generated file can be passed to
  another tool. A cwd of `/` (Claude Desktop) is never an implicit prefix,
  and relative entries (a relative `mcp.output_dir`, or `.` in the setting)
  then resolve under the home directory (`~/muse-output`), never `/`. (An empty setting
  used to disable path inputs; it now means cwd plus output dir.)
- `*_url` inputs on multipart routes are fetched by the MCP process (through
  `muse.core.net_fetch`'s SSRF guard) and uploaded; `*_url` inputs on JSON
  routes pass through to the server's SSRF-guarded fetch.
- An unfiltered `muse_describe` past 20,000 chars writes the full document
  to a file and returns an inline index (per modality: summary, tools,
  model ids; per model: id, modality, license, quality_tier, summary,
  loaded). With `modality` or `model` it returns the document inline.

Behaviour changes from the old hand-written tools: `muse_classify` maps to
`/v1/text/classifications` and `/v1/moderations` is now `muse_moderate`;
`muse_decide` is validated server-side (strict parser); JSON-route `*_url`
inputs are fetched by the server, multipart ones by the MCP process (above);
base64 inputs are gone (use a path or URL).

Claude Desktop config:

```json
{
  "mcpServers": {
    "muse": {
      "command": "uvx",
      "args": ["--from", "museq[mcp]>=0.70", "muse", "mcp"],
      "env": {
        "MUSE_SERVER": "http://localhost:8000",
        "MUSE_MCP_OUTPUT_DIR": "/Users/you/muse-output",
        "MUSE_ADMIN_TOKEN": "your-admin-token-here"
      }
    }
  }
}
```

Claude Desktop starts stdio servers with cwd `/`, so set
`MUSE_MCP_OUTPUT_DIR` to an absolute directory (without it, outputs go to
`~/muse-output`). With cwd `/` no path input is readable except files under
the output dir; to pass local files, set `MUSE_MCP_ALLOWED_PATH_PREFIXES` to
a colon-separated list of directories (it replaces the cwd default; the
output dir is always readable and writable, so outputs can be fed back in).

## Development commands

```bash
# Install (dev)
pip install -e ".[dev,server,audio,images]"

# Full dev stack (CPU torch) - what CI installs and what preflight expects
pip install -e ".[dev,server,audio,images,embeddings]" \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Gate the fast lane on a non-drifted venv (used in the release ritual)
python scripts/preflight.py

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

# CLI: admin surface (no per-modality verbs; generation is HTTP).
# Built on typer + rich since v0.39.0; `muse --help` and the per-
# subcommand help are auto-generated, color-coded, and respect
# NO_COLOR env. The list / search / refresh outputs auto-detect TTY:
# pretty rich.Table when interactive; plain text (no ANSI, no
# truncation) when piped or redirected, so `muse models list | grep`
# always sees full content. `--json` flag on list / probe / refresh
# emits machine-readable output.
muse --help                                    # top-level help (typer-styled)
muse models list                               # bundled + curated + pulled
muse models list --available --modality chat/completion
muse models list --json                        # deterministic JSON output
muse search qwen3 --modality chat/completion --max-size-gb 10
muse pull qwen3.5-4b-q4                        # curated alias
muse pull hf://unsloth/Qwen3.5-9B-GGUF@q4_k_m  # resolver URI
muse pull kokoro-82m                           # bundled bare id
muse models info <id>
muse models enable <id> / disable <id>
muse models set-device <id> cuda               # operator device pin (overrides manifest device)
muse models set-device <id> --clear            # remove the pin
muse models remove <id>
muse models refresh <id>                       # re-install muse[server,extras] into one venv
muse models refresh --all                      # all pulled venvs, alphabetical
muse models refresh --enabled                  # only enabled venvs
muse models refresh <id> --no-extras           # skip the model's pip_extras (only refresh muse[server])
muse models refresh --all --json               # machine-readable output
muse serve --device cuda
muse mcp                                       # MCP server for LLM clients (stdio mode)
muse mcp --http --port 8088                    # MCP server in Streamable HTTP mode
muse mcp --filter admin                        # only the admin tools
muse mcp --filter inference                    # only muse_describe + generated tools

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

Working in a git worktree: the editable install points at the main
checkout, so run tests with `PYTHONPATH=$PWD/src pytest ...` from the
worktree or you will test main's code.

Releasing (ask the owner before pushing, tagging or uploading):
1. Bump `version` in `pyproject.toml`; commit `chore(release): X.Y.Z` with a
   body summarising the release; annotated tag `vX.Y.Z` (message `vX.Y.Z`).
2. Build from a clean export, never the working tree (a stray `build/`
   dir leaks stale files into wheels): `git archive vX.Y.Z | tar -x -C
   <tmp>`, `python -m build --sdist --wheel`, `twine check`, rebuild the
   wheel from the sdist, and run `scripts/validate_installed_distribution.py`
   in a clean venv for both wheels.
3. Push `main` and the tag, wait for CI, then `twine upload` the two files
   (credentials in `~/.pypirc`). PyPI `museq` was 0.60.2 until 0.70.1;
   `uvx --from "museq[mcp]"` users get whatever PyPI has.

Commits follow Conventional Commits with a scope (`feat(pixelart): ...`,
`fix(owned-process): ...`, `docs(claude): ...`); see AGENTS.md.

## Project-specific conventions

- **Deferred imports:** `src/muse/__init__.py` and `src/muse/cli.py` MUST NOT
  import heavy libs (torch, diffusers, transformers). Each backend imports
  its heavy deps at module top-level inside a try/except so import of the
  backend module succeeds even without the deps. Tests mock at the module
  path where the library is imported. `muse --help` and `muse pull` work
  without any ML deps installed; pulling a model installs them on demand.
- **FakeModel-pattern tests:** Server and router tests use plain classes that
  satisfy the modality protocol, no real weights. Backend tests also mock
  heavy libs (see `tests/modalities/image_generation/runtimes/test_diffusers.py`).
- **No registry singleton:** each worker builds its own `ModalityRegistry()`,
  and tests create their own instances.
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
- **Env vars are registry settings (v0.52.0+).** Every `MUSE_*` knob is a
  row in the `muse.core.config` registry, so it can equivalently be set as
  an env var, in `~/.muse/config.yaml`, or via `muse config set`. See the
  "Configuration" section and docs/CONFIG.md for the full list. Common ones:
  `MUSE_SERVER` / `client.server_url` (client base URL), `MUSE_CATALOG_DIR`
  / `paths.catalog_dir` (catalog + config.yaml location, defaults
  `~/.muse/`), `MUSE_HOME` / `paths.home` (voices dir base),
  `MUSE_ALLOW_PRIVATE_FETCH` / `fetch.allow_private` (opt-in escape hatch
  for the SSRF guard in `muse.core.net_fetch`, which also requires http(s)
  on every redirect hop and bounds the whole fetch by a 120 s deadline;
  default off; needed only when operators on a trusted network want
  `image` URLs to reach internal services), `MUSE_IMAGE_INPUT_MAX_BYTES` / `limits.image_input_max_bytes`
  (per-request size cap for image uploads and data URLs; default 10485760 /
  10MB; read per-call so changes take effect without a server restart).
- **Auto-restart is always on.** No --no-autorestart flag in this iteration. Workers that can't stay up through 10 restart attempts are marked dead; `POST /v1/admin/workers/{port}/restart` revives one (or restart `muse serve`).
- **Enable/disable is catalog state; loaded/unloaded is runtime state (v0.40.0+).** The two are decoupled. `enabled: true` means "in service; allowed to lazy-load." `loaded` (visible via `/v1/models` and `muse models list`'s `enabled_loaded` glyph) means "currently resident on a worker." LRU eviction unloads without disabling; `muse models warmup` loads without bumping refcount. Operator-driven `muse models enable/disable` flips the catalog bit and (when the supervisor is running with `MUSE_ADMIN_TOKEN`) syncs runtime state via `enable_model` / `disable_model`; lazy-load and eviction paths use `load_model_into_worker` / `unload_model_from_worker` which skip the catalog flip.
- **Tool-use asymmetry (known landmine).** llama-cpp-python's `chatml-function-calling` handler parses tool calls *out* of a model's response into structured `tool_calls`, but does NOT format tool *result* messages (role=`tool`) back to the model in a way Qwen's chat template always recognizes. The muse-side contract is correct (verified by `tests/modalities/chat_completion/test_routes_messages_passthrough.py`); the asymmetry is upstream. Larger models (Qwen3.5-9B+) tolerate it in context; smaller models (Qwen3.5-4B) often ignore the tool result and give a generic "I don't have access to tools" reply. Tracked by `tests/integration/test_remote_tools.py::test_observe_tool_result_content_influences_next_response` (xfail-style watchdog). Upstream: [abetlen/llama-cpp-python#2063](https://github.com/abetlen/llama-cpp-python/issues/2063).
- **The `model` field in chat responses is the catalog id**, not the GGUF filesystem path. `LlamaCppModel._dict_to_chat_result` and `_dict_to_chat_chunk` override `response["model"]` with the muse catalog id (not the `resp.get("model") or fallback` pattern that lets llama-cpp's internal `model_path` win). Applies to both non-streaming responses and every streaming chunk.
- **CLI is typer + rich (v0.39.0+).** Per-subcommand parameter binding lives in `src/muse/cli.py` as typer command functions; the heavy logic lives in `src/muse/cli_impl/<command>.py`. New subcommands add a `@app.command(...)` plus a sibling `cli_impl/*.py` module; do not put logic in `cli.py`. Long-form output that goes to a TTY is rendered via `rich.Table` from `cli_impl/console.py`'s shared `Console`; non-TTY output (subprocess, pipe, redirect) is plain text with no ANSI and no truncation, so `muse models list | grep` always sees full content. Status encoding for `models list` and `refresh --all` uses single colored glyphs (`STATUS_STYLE` in `cli_impl/console.py`): `●` enabled, `○` disabled, `★` recommended, `·` available; `✓` ok, `✗` failed for refresh outcomes. `--json` is the canonical machine-readable output across `list` / `probe` / `refresh`.
- **CLI exit codes propagate through `main` (v0.48.0).** Commands signal errors with `raise typer.Exit(<code>)`. `muse.cli:main` runs the app with `standalone_mode=False`, where click handles `typer.Exit` itself and *returns* the code as `app()`'s value rather than raising; `main` now returns that int (None -> 0) so the shipped `muse` binary actually exits nonzero on error. Before v0.48.0 `main` discarded the return and always exited 0; the `python -m muse.cli` subprocess tests hit `app()` in standalone mode (correct exit), so they never caught it. New `main()`-level exit-code regressions belong in `tests/cli_impl/test_set_device_cli.py`.
- **CLI runtime state reads the PUBLIC `/v1/models`, not admin endpoints (v0.47.4).** `muse models list`'s `enabled_loaded` glyph and `muse models info`'s header + worker-status block derive "is this model loaded right now?" from the public `GET /v1/models` `loaded` flag (no `MUSE_ADMIN_TOKEN` required). The shared `muse.cli_impl.runtime_state` module (`fetch_public_models` + `loaded_ids`) centralizes the fetch, failure policy (None = unreachable; both `list` and the `info` "loaded?" predicate treat it identically via `loaded_ids`), and httpx-log quieting. `muse models info` still prefers the admin API first for rich worker pid/uptime/restart detail and falls back to the public loaded state, rendering `loaded (set MUSE_ADMIN_TOKEN for worker pid / uptime / restarts)` instead of faking detail or falsely claiming the supervisor is unreachable. Before v0.47.4 these read admin-only endpoints, so anyone without a token saw every model as `enabled_unloaded` and "supervisor unreachable" even against a live server.
- **Shared image preprocessing** (`muse.core.image_preprocessing`, v0.42.1+): four-tier dispatch ladder for image-side preprocessor selection. Public API: `read_encoder_hints(src)` (read encoder hyperparams from config.json), `DerivedImageProcessor` (synthesize a minimal preprocessor from explicit num_channels / image_size / image_mean / image_std), `build_image_processor(src, *, overrides, model_id)` (the orchestrator: override-first, then AutoImageProcessor, then encoder-hints-derived, then fail loud), and `ImageProcessorError` (structured exception pointing at the override hatch). Manifest hatch: `capabilities.image_processor_overrides: {num_channels, image_size, image_mean?, image_std?}` lets curated entries declare ground-truth preprocessing when AutoImageProcessor's sniff would be wrong (TexTeller is the canonical example: 1-channel grayscale at 448x448). Tier 3 (ViT defaults) was dropped in v0.42.1 because it produced silently wrong output for the very class of repos this fallback targets; failures now raise `ImageProcessorError` pointing at the override hatch. Currently consumed by `image_ocr`; other modalities can adopt as needed.

## Continuous integration

Two GitHub Actions workflows run on every push to `main` and every PR:

- `.github/workflows/tests.yml` runs the fast lane (`pytest -m "not slow"`)
  on Ubuntu with Python 3.10-3.14 and on macos-14 with Python 3.12 (no
  `os.waitid`, so the psutil leader-pinning path) and 3.13, with the full
  CPU stack (`pip install -e ".[dev,server,audio,images,embeddings]"`
  against the `download.pytorch.org/whl/cpu` index). A separate job is the
  wheel and sdist release gate: build both, `twine check`, rebuild the
  wheel from the sdist, and run `scripts/validate_installed_distribution.py`
  in a clean venv for each wheel. Dependencies float on latest (no
  lockfile), so an upstream breaking release surfaces here on the next push
  rather than shipping silently. Model tests are mocked, so no weights are
  downloaded.
- `.github/workflows/fresh-venv-smoke.yml` (below) validates the per-model
  fresh-venv install/load path.

Locally, run the fast lane through the preflight guard so a drifted venv
fails loudly instead of running a partial suite:

    python scripts/preflight.py            # check deps, then run the lane
    python scripts/preflight.py --check-only

## Fresh-venv smoke test (CI)

Bundled scripts under `src/muse/models/` declare `pip_extras` covering the
deps the runtime source-imports. The dev environment installs broad
extras (`muse[dev,server,audio,images,embeddings]`), so transitive deps
that `from_pretrained` (or sentence-transformers, or diffusers) ALSO
imports at load time happen to be present. Per-model venvs created via
`muse pull <id>` install only `muse[server]` plus the model's declared
`pip_extras`; transitive holes show up as ImportError when the worker
tries to load the model.

The v0.30.0 audit (#110, `tests/models/test_pip_extras_audit.py`)
catches direct-import gaps via AST scan but cannot see transitive
imports that `from_pretrained` triggers. v0.32.0 closes that gap with
a CI workflow:

- `.github/workflows/fresh-venv-smoke.yml` runs on every push to main
  and every PR. Matrix-tests the lightweight bundled models listed in
  the workflow's `model:` matrix (16 as of v0.70: kokoro-82m, dinov2-small,
  bart-large-cnn, bge-reranker-v2-m3, mert-v1-95m, supertonic-3, the two
  text classifiers, trocr-base-printed, the three image/cv models,
  ast-audioset, pixel-tools, utmos, audiobox-aesthetics).
- Each job creates a fresh venv, installs only what `muse pull` would
  install, then runs the in-venv probe worker (`muse _probe_worker
  --no-inference`) to verify load. Failure surfaces a structured label
  like `kokoro-82m: FAIL (missing dep: librosa)`.
- Heavy / GPU-only models (sd-turbo, animatediff, stable-audio, wan,
  large LLMs) are deferred until paid runner budget allows.

Local repro: `python scripts/smoke_fresh_venv.py --model_id <id>`.
Use `--json` for machine-readable output.

When you add a new bundled script and the smoke matrix should cover it,
add the id to the matrix in `.github/workflows/fresh-venv-smoke.yml`.
Lightweight models (under ~1 GB on disk, CPU-friendly) are good
candidates; heavier models need GPU runners and are out of scope for
the free-tier CI matrix.

## Benchmarks

`scripts/bench/` holds the reusable performance harness (v0.57.0):
`bench_llm.py` (muse-vs-ollama head-to-head: generation tok/s, streaming
TTFT/inter-token, long-prompt eval, multi-turn prefix-reuse detection,
worker-direct vs gateway split) and `bench_modalities.py` (per-modality
hot latencies against a live server). Reports land in `docs/benchmarks/`.
Protocol lesson from the 2026-07-08 baseline: on thermally-scaled boxes,
only alternating same-window comparisons are trustworthy; cross-window
medians can differ 3x. LlamaCppModel accepts optional perf kwargs
(n_threads, n_threads_batch, n_batch, flash_attn, use_mlock, type_k,
type_v) via manifest capabilities; defaults match llama-cpp-python (and
ollama's thread choice) and shipped untuned because the baseline showed
muse already competitive (non-streaming faster than ollama; streaming
comparison inconclusive on the noisy laptop test box -- muse 5.8-14.3 vs
ollama 7.6-10.9 tok/s in the committed alternating run, with muse's
within-run variance exceeding the delta).

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
`__init__.py`. The probe subprocess gets 600 s unless the modality exports a
literal `PROBE_TIMEOUT_SECONDS` (AST-read by `discovery.probe_timeouts`; the
larger wins): video/generation declares 2700, because Wan's 2-second probe clip
under sequential CPU offload takes ~16 min including load on a 12 GB card. The probe worker imports the modality at run time and
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
manifest). A resolver pull whose id names a model script is refused
outright (`ModelIdConflictError`), since the script would otherwise load
the new weights. A resolver pull whose synthesized id (the repo basename)
is already installed from a *different* repo is also refused before any
download, so `hf://alice/Foo-GGUF` and `hf://bob/Foo-GGUF` can't replace
each other; install the second with `muse pull <uri> --id <new-id>`.
Same-repo re-pulls (any case, any variant) and curated pulls, which own
their curated id, proceed as before. The check runs in
`catalog_pull._check_resolver_target`, once before and once under the model's
resource lease.

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
6. Inside the new runtime / bundled script, import device + dtype
   utilities from `muse.core.runtime_helpers` rather than rolling
   your own. The four utilities (`select_device`, `dtype_for_name`,
   `set_inference_mode`, `LoadTimer`) cover the common needs and the
   meta-test (`tests/core/test_runtime_helpers_meta.py`) AST-walks
   every runtime to flag re-implementations.
7. Add tests under `tests/modalities/<mime_name>/` (route + plugin)
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

Fast lane is `-m "not slow"`. `@pytest.mark.slow` marks tests that spawn
real subprocesses or supervisors (33 files, e.g.
`cli_impl/test_e2e_supervisor.py`); run the full lane before changing the
supervisor, gateway, owned-process or install paths. The real-process
suites are parametrized `waitid` / `no-waitid` (and `+darwin`, which
emulates macOS process-group rules via `tests/darwin_process_semantics.py`). The `tests/integration/` suite is separately
opt-in via `MUSE_REMOTE_SERVER` env var; fixtures auto-skip when the
server isn't reachable or the required model isn't loaded. The
`MUSE_CHAT_MODEL_ID` env var (default `qwen3.5-4b-q4`) lets the same
integration suite run against any chat model on the target server.

Test naming for integration tests:
- `test_protocol_*`: hard claims muse should always satisfy. Failure is a regression.
- `test_observe_*`: records what a particular model actually did. Usually xfail-style; useful as a watchdog (a passing xfail shows up as XPASS, signaling "upstream fixed something; promote to a hard assertion").
