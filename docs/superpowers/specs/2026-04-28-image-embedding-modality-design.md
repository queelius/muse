# `image/embedding` modality (CLIP / SigLIP / DINOv2 image vector embedders)

**Date:** 2026-04-28
**Status:** approved
**Target release:** v0.23.0

## Goal

Add muse's 11th modality: `image/embedding`, mounted at the
OpenAI-shape URL `POST /v1/images/embeddings`. One generic
`ImageEmbeddingRuntime` over `transformers.AutoModel` +
`AutoProcessor` serves any HuggingFace image-feature-extraction model
(CLIP, SigLIP, DINOv2 family). One bundled default
(`facebook/dinov2-small`) plus three curated additions
(`google/siglip2-base-patch16-256`, `openai/clip-vit-base-patch32`,
`jinaai/jina-clip-v2`). HF resolver sniffs HF repos with
`image-feature-extraction` (or related) tags + a
`preprocessor_config.json` sibling at priority **105** so they resolve
to the image-embedding runtime ahead of any catch-all classifier
plugin.

The wire envelope mirrors `/v1/embeddings` exactly (`{object: "list",
data: [...], model, usage}`), so OpenAI SDK clients that already speak
embeddings can reuse their helper code via the standard `embeddings`
client wrapper plus `extra_body` for image inputs (or our own
`ImageEmbeddingsClient`).

## Scope

**In v1:**
- `POST /v1/images/embeddings` with OpenAI-shape JSON request and
  response.
- `input: str | list[str]` where each entry is a `data:image/...;base64,...`
  URL or an `http(s)://...` URL pointing at PNG/JPEG/WEBP.
- `model: str | None` (catalog id; defaults to first registered under
  `image/embedding`).
- `encoding_format: "float" | "base64"` mirrors `/v1/embeddings` exactly.
  base64 is little-endian float32 (4 bytes per dim).
- Optional `dimensions: int` truncation (matryoshka-style; only honored
  when smaller than the model's native dimensionality, and re-normalized
  after slicing).
- Optional `user: str` accepted for OpenAI compat; ignored.
- Generic `ImageEmbeddingRuntime` over `AutoModel` + `AutoProcessor`
  with per-architecture `_extract_embeddings` dispatch (CLIP,
  SigLIP, DINOv2).
- Image decoding via the existing `decode_image_input` helper from
  `image_generation/image_input.py` (data URL OR http(s) URL,
  PIL.Image out, size-capped, MIME-validated).
- HF resolver eighth sniff branch for image-feature-extraction repos,
  priority 105.
- Search routes `--modality image/embedding` to a hybrid
  `list_models` query (image-feature-extraction tag combined with the
  user's query).
- Three curated entries: `dinov2-small` (bundled-script alias),
  `siglip2-base` (`hf://google/siglip2-base-patch16-256`),
  `clip-vit-base-patch32` (`hf://openai/clip-vit-base-patch32`).
- One bundled script: `src/muse/models/dinov2_small.py`.
- `ImageEmbeddingsClient` parallel to other muse clients; minimal HTTP
  wrapper that base64-encodes raw bytes into a data URL on the way
  out and returns `list[list[float]]` on the way back.
- `PROBE_DEFAULTS` so `muse models probe <id>` exercises a
  single-image (224x224) call.

**Not in v1 (deferred):**
- Cross-modal text embedding for CLIP/SigLIP. Those models also embed
  text, but exposing a separate `/v1/images/embeddings/text` route
  would mean splitting the modality across two URL paths. Deferred:
  later we can either (a) add a `text` field to the request that
  triggers the model's text tower, or (b) mount a sibling route with
  capability gating. Document as out-of-scope for v0.23.0; the
  `supports_text_embeddings_too` capability flag is set so the future
  work doesn't need a manifest schema change.
- Multipart/form-data uploads (`/v1/images/embeddings` taking raw
  `image=@file.png` files). v1 sticks to JSON-with-data-URLs since
  embedding workflows are mostly programmatic; multipart could land
  later.
- Image classification with logits (e.g. ViT-style). Those still
  belong in `text/classification`'s future image-classification
  sibling or a dedicated `image/classification` modality; embedding
  is "extract a vector", not "predict a label".
- Per-image attention/CLS variants in the request. v1 picks one
  pooling strategy per architecture (CLS for DINOv2,
  pooler_output / image_embeds for CLIP/SigLIP); per-call override is
  out of scope.
- Region-of-interest embeddings (crop, mask, etc.).

## Why generic runtime, not bundled-script-only

Matches muse's trajectory (sentence-transformers, llama-cpp,
faster-whisper, transformers AutoModelForSequenceClassification,
diffusers AutoPipeline, sentence-transformers CrossEncoder,
transformers AutoModelForSeq2SeqLM). One runtime serves any model
in its class; curated entries pin the recommended specific. Adding
`facebook/dinov2-base` or `openai/clip-vit-large-patch14` later is a
curated.yaml edit (or a `muse pull hf://...`), not a new Python
script. The per-architecture dispatch lives in the runtime so the
runtime stays the single source of truth on extraction.

## Why OpenAI shape (and OpenAI-compatible) at /v1/images/embeddings

OpenAI doesn't expose a public image-embedding endpoint, but their
`/v1/embeddings` envelope is the de-facto industry shape (LangChain,
LlamaIndex, OpenAI SDK builtins all assume it). Mirroring it for image
inputs gives clients a near-zero-effort path: same envelope, same
encoding_format, same usage rollup. The route prefix `/v1/images/...`
is consistent with `image/generation`'s family hierarchy, so the
modality lands in a predictable location.

## Package layout

```
src/muse/modalities/image_embedding/
|-- __init__.py          # MODALITY = "image/embedding" + build_router + exports + PROBE_DEFAULTS
|-- protocol.py          # ImageEmbeddingModel Protocol + ImageEmbeddingResult dataclass
|-- routes.py            # build_router; mounts POST /v1/images/embeddings
|-- codec.py             # ImageEmbeddingResult + base64 helper
|-- client.py            # ImageEmbeddingsClient
|-- hf.py                # HF_PLUGIN sniffing image-feature-extraction repos
`-- runtimes/
    |-- __init__.py
    `-- transformers_image.py  # ImageEmbeddingRuntime generic runtime
```

Bundled script:

```
src/muse/models/
`-- dinov2_small.py   # facebook/dinov2-small curated default
```

## Protocol

```python
@dataclass
class ImageEmbeddingResult:
    """N images in, N embedding vectors out, plus provenance.

    embeddings: list[list[float]] (float32, native dim per row).
                Keep it pure-Python at the protocol boundary so no
                numpy dep leaks into consumers.
    dimensions: vector length after any matryoshka truncation; equal
                to the model's native dimensionality when no
                truncation was applied.
    model_id: catalog id of the producing model.
    n_images: count of inputs the runtime processed (for usage roll-up).
    metadata: optional per-call extras (e.g. truncation_warning,
              source backend tag).
    """
    embeddings: list[list[float]]
    dimensions: int
    model_id: str
    n_images: int
    metadata: dict = field(default_factory=dict)


@runtime_checkable
class ImageEmbeddingModel(Protocol):
    """Structural protocol any image embedder backend satisfies."""

    @property
    def model_id(self) -> str: ...

    @property
    def dimensions(self) -> int: ...

    def embed(
        self,
        images: list,            # list of PIL.Image
        *,
        dimensions: int | None = None,
    ) -> ImageEmbeddingResult: ...
```

## Wire contract

**Request** (`POST /v1/images/embeddings`, `application/json`):

| Field | Type | Required | Validation | Notes |
|---|---|---|---|---|
| `input` | `str` or `list[str]` | yes | non-empty; each entry a data: URL or http(s) URL | The image(s) to embed |
| `model` | `str | None` | no | catalog id | Defaults to first registered under `image/embedding` |
| `encoding_format` | `str` | no | one of "float", "base64" | default "float" |
| `dimensions` | `int | None` | no | 1 <= n <= 4096 | matryoshka truncation |
| `user` | `str | None` | no | -- | OpenAI compat; ignored |

**Response** (`application/json`, OpenAI shape mirroring `/v1/embeddings`):

```json
{
  "object": "list",
  "data": [
    {"object": "embedding", "embedding": [0.123], "index": 0},
    {"object": "embedding", "embedding": [0.456], "index": 1}
  ],
  "model": "dinov2-small",
  "usage": {"prompt_tokens": 0, "total_tokens": 0}
}
```

`prompt_tokens` and `total_tokens` are 0 in v1 (image embedding has no
text tokenization). A future iteration may surface `n_images` here; we
keep the OpenAI-compatible names so SDK clients that read `usage`
fields don't choke.

**Error envelopes** (OpenAI-shape, used by all muse modalities):

- 400 `invalid_parameter`: `input` empty; bad data URL; HTTP fetch
  failed; image decode failed; `dimensions` out of range.
- 404 `model_not_found`: `model` unknown.

## Image input handling

Reuse `decode_image_input` from
`muse.modalities.image_generation.image_input`. It already:
- accepts `data:image/{png,jpeg,webp};base64,XYZ`;
- accepts `http(s)://...` (fetched via httpx, MIME-validated, size-capped at 10MB);
- returns a PIL.Image;
- raises `ValueError` on malformed/oversized/undecodable input.

The route layer batches: list inputs decode each entry in turn, and
the resulting `list[PIL.Image]` is passed to `backend.embed(...)`.

## Runtime: ImageEmbeddingRuntime

`src/muse/modalities/image_embedding/runtimes/transformers_image.py:ImageEmbeddingRuntime`:

The runtime wraps `transformers.AutoModel` + `AutoProcessor` with a
constructor that accepts the standard muse keyword args (model_id,
hf_repo, local_dir, device, dtype, image_size, **_). It calls
`_ensure_deps()` (lazy-import torch + transformers via module-level
sentinels), selects a device, loads the processor (with
AutoFeatureExtractor fallback), loads the AutoModel, moves the model
to the device, switches it into inference mode via the
`_set_inference_mode()` helper, and detects `dimensions` from the
loaded model.

The `embed()` method always normalizes the input to a list, runs the
processor to build a `pixel_values` tensor, calls the model under
`torch.inference_mode()`, dispatches the outputs through
`_extract_embeddings()`, optionally applies matryoshka truncation,
and returns an `ImageEmbeddingResult`.

The `_extract_embeddings()` dispatch order is fixed and documented:

1. **CLIP family**: outputs has an `image_embeds` attribute set
   (returned from CLIP's `vision_model` chain or the umbrella forward
   when both towers were skipped). Use it directly.
2. **SigLIP and DINOv2-with-pooler**: outputs has `pooler_output` set
   (a non-None tensor; SigLIP populates it; some DINOv2 builds also
   do). Use it directly.
3. **DINOv2 base**: outputs has `last_hidden_state` and we use the
   first token along dim 1 (the CLS token) as the global pool.
4. Anything else raises `ValueError`.

Each branch is independently testable with mocked outputs.

`_set_inference_mode` helper switches the model into no-grad mode via
its inference-mode method, kept in a helper to avoid bare calls in the
runtime body, matching v0.22.0's bart_seq2seq runtime. The literal
method name (the spelling Python's transformers uses) is the obvious
one for switching to inference; the helper just looks it up via
`getattr` and invokes it when callable.

Inference uses `torch.inference_mode()` (slightly faster than
`torch.no_grad()`).

AutoProcessor handles most cases. If it fails (older repos) we fall
back to AutoFeatureExtractor with a warning log; documented in the
runtime's `_load_processor` helper.

## ImageEmbeddingResult dataclass

```python
@dataclass
class ImageEmbeddingResult:
    embeddings: list[list[float]]
    dimensions: int
    model_id: str
    n_images: int
    metadata: dict = field(default_factory=dict)
```

`n_images` mirrors prompt_tokens-like accounting from `/v1/embeddings`
but in image space.

## Codec

The codec module re-exports
`embedding_to_base64` / `base64_to_embedding` from the existing
`embedding_text.codec` so the encoding format is bit-identical (little
endian float32). v1 keeps the codec module thin; if image-specific
encoding ever diverges (e.g. quantized embeddings, half-precision) the
abstraction is already in place.

## HF resolver plugin

`src/muse/modalities/image_embedding/hf.py`:

Sniff: any HF repo with one of the relevant tags (`feature-extraction`,
`image-feature-extraction`, `image-classification`) AND a
`preprocessor_config.json` sibling. The dual check (tag plus sibling)
prevents text-only feature-extraction models from being picked up.

```python
def _sniff(info) -> bool:
    siblings = [s.rfilename for s in getattr(info, "siblings", [])]
    tags = getattr(info, "tags", None) or []
    has_image_processor_config = any(
        Path(s).name == "preprocessor_config.json" for s in siblings
    )
    has_relevant_tag = any(
        t in tags
        for t in ("feature-extraction", "image-feature-extraction",
                  "image-classification")
    )
    return has_image_processor_config and has_relevant_tag
```

Priority **105**: between embedding/text (110) and the image-generation
file-pattern plugin (100). Loses to file-pattern plugins (GGUF,
faster-whisper, diffusers) so a multi-purpose repo that also ships a
diffusers pipeline still resolves as image/generation.

Capability defaults are inferred per-pattern:
- `clip` in repo name -> `supports_text_embeddings_too=True`,
  `dimensions=512` (base) or `768` (large).
- `siglip` in repo name -> `supports_text_embeddings_too=True`,
  `dimensions=768` (base) or `1024` (large).
- `dinov2` or `dinov3` in repo name ->
  `supports_text_embeddings_too=False`, `dimensions=384` (small),
  `768` (base), `1024` (large).
- `vit` (generic) -> `supports_text_embeddings_too=False`,
  `dimensions=768`.
- Fallback when no pattern matches: `dimensions=None` (will be
  inferred at load time), `supports_text_embeddings_too=False`.

## Bundled script: dinov2_small

`src/muse/models/dinov2_small.py`:

```python
MANIFEST = {
    "model_id": "dinov2-small",
    "modality": "image/embedding",
    "hf_repo": "facebook/dinov2-small",
    "description": (
        "DINOv2 small: 88MB, 384-dim self-supervised image features, Apache 2.0"
    ),
    "license": "Apache 2.0",
    "pip_extras": (
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "Pillow>=9.1.0",
    ),
    "system_packages": (),
    "capabilities": {
        "device": "auto",
        "dimensions": 384,
        "image_size": 224,
        "supports_text_embeddings_too": False,
        "memory_gb": 0.4,
    },
    "allow_patterns": [
        "*.safetensors", "*.json", "*.txt",
        "preprocessor_config.json",
    ],
}
```

The `Model` class wraps `transformers.AutoModel` + `AutoImageProcessor`
directly (rather than going through `ImageEmbeddingRuntime`) so the
script demonstrates the same shape muse uses for other bundled models.
Lazy imports.

## Curated entries

```yaml
- id: dinov2-small
  bundled: true

- id: siglip2-base
  uri: hf://google/siglip2-base-patch16-256
  modality: image/embedding
  size_gb: 0.4
  description: "SigLIP2 base: 370MB, supports text embeddings too, Apache 2.0"
  capabilities:
    supports_text_embeddings_too: true
    dimensions: 768
    memory_gb: 1.0

- id: clip-vit-base-patch32
  uri: hf://openai/clip-vit-base-patch32
  modality: image/embedding
  size_gb: 0.6
  description: "CLIP ViT-B/32: 600MB, classic cross-modal, MIT"
  capabilities:
    supports_text_embeddings_too: true
    dimensions: 512
    memory_gb: 1.2
```

## PROBE_DEFAULTS

```python
PROBE_DEFAULTS = {
    "shape": "1 image, 224x224",
    "call": lambda m: m.embed(
        [Image.new("RGB", (224, 224), (128, 128, 128))]
    ),
}
```

Used by `muse models probe <id>` so a power user can verify a fresh
pull works end-to-end without opening a Python REPL.

## Test strategy

Unit-heavy. Mocks for `transformers.AutoModel` + `AutoProcessor`. One
slow e2e test exercises FastAPI + codec + mocked runtime. One opt-in
integration test against a live muse server with a real image
embedder loaded.

Coverage targets:

- Protocol + dataclass shape (5 tests).
- Codec: float passthrough; base64 roundtrip; matryoshka truncation
  shape; index ordering preserved.
- Routes: 200 happy path with single + batch input, 400 envelope for
  bad input (empty input, bad data URL, oversized), 404 for unknown
  model, default model resolution.
- Runtime: deferred imports; processor + model load order; per-arch
  `_extract_embeddings` (CLIP, SigLIP, DINOv2 paths each independently
  mocked); device auto-select; matryoshka truncation honored;
  inference-mode helper invoked.
- HF plugin: positive sniff (image-feature-extraction tag +
  preprocessor_config sibling), negative without sibling, negative
  without tag, priority 105, per-pattern capability defaults
  (CLIP/SigLIP/DINOv2/generic), search branch.
- Curated: `dinov2-small` parses as bundled; `siglip2-base` and
  `clip-vit-base-patch32` parse as URI with capabilities overlay.
- Bundled script: MANIFEST shape, Model construction with lazy imports
  patched, `embed()` returns ImageEmbeddingResult with the right
  dimensions echoed.
- E2E slow: full JSON-in / JSON-out round-trip through the supervisor.
- Integration opt-in: real server + real model.
  `MUSE_IMAGE_EMBEDDING_MODEL_ID` env override (default
  `dinov2-small`).

## Documentation

- CLAUDE.md: add `image/embedding` to modality list; note bundled
  default and per-architecture dispatch. Bump count from 10 modalities
  to 11.
- README.md: modality list + `/v1/images/embeddings` endpoint + curl
  example.
- src/muse/__init__.py docstring: bump version to 0.23.0; add
  `image/embedding` to bundled modalities list.

## Release

v0.23.0. Minor bump (new feature, no breaking changes). Tag message
calls out: new modality, new endpoint, new bundled default, OpenAI-
compatible image-embedding wire contract.

## Out of scope

- Cross-modal text embedding via the same runtime (CLIP / SigLIP text
  tower). Capability flag is set so future work doesn't need a manifest
  schema change.
- Multipart/form-data uploads for raw bytes.
- Image classification logits (`image/classification` modality, future).
- Per-call pooling-strategy override.
- Region-of-interest embeddings.
