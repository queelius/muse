# `audio/embedding` modality (CLAP / MERT audio vector embedders)

**Date:** 2026-04-28
**Status:** approved
**Target release:** v0.24.0

## Goal

Add muse's 12th modality: `audio/embedding`, mounted at the
multipart-upload URL `POST /v1/audio/embeddings`. One generic
`AudioEmbeddingRuntime` over `transformers.AutoModel` +
`AutoFeatureExtractor` (with `AutoProcessor` fallback) plus
`librosa`-based audio decoding serves any HuggingFace audio
feature-extraction model (CLAP, MERT, wav2vec, audio-encoder family).
One bundled default (`m-a-p/MERT-v1-95M`) plus one curated addition
(`laion/clap-htsat-fused`). HF resolver sniffs HF repos with
`feature-extraction` tag and a name pattern matching CLAP / MERT /
audio-encoder / wav2vec at priority **105** so they resolve to the
audio-embedding runtime ahead of any catch-all classifier plugin.

Wire shape mirrors `/v1/audio/transcriptions`'s multipart-upload
contract for the request side and `/v1/embeddings` for the response
envelope (`{object: "list", data: [...], model, usage}`). OpenAI SDK
clients already speaking `/v1/embeddings` can reuse helpers via the
audio-specific `AudioEmbeddingsClient`. Audio inputs are decoded with
`librosa` (already in muse[audio] for Whisper); the runtime resamples
to each model's preferred rate (CLAP 48kHz, MERT 24kHz).

## Scope

**In v1:**
- `POST /v1/audio/embeddings` with multipart/form-data upload (one or
  more `file` parts plus `model`, `encoding_format`, `user`).
- Multiple audio inputs supported via repeated `file` field
  (multipart-style batching). Single-file uploads are the common path.
- `model: str | None` (catalog id; defaults to first registered under
  `audio/embedding`).
- `encoding_format: "float" | "base64"` mirrors `/v1/embeddings`.
  base64 is little-endian float32 (4 bytes per dim).
- Optional `user: str` accepted for OpenAI compat; ignored.
- Generic `AudioEmbeddingRuntime` over `AutoModel` +
  `AutoFeatureExtractor`/`AutoProcessor` with per-architecture
  `_extract_embeddings` dispatch (CLAP, MERT, generic).
- Audio decoding via `librosa.load(io.BytesIO(b), sr=<model_sr>,
  mono=True)`. The runtime resamples on the way in so callers can
  upload any sample rate.
- Size cap: env-tunable `MUSE_AUDIO_EMBEDDINGS_MAX_BYTES` (default
  50MB) on incoming file.
- Duration cap: env-tunable `MUSE_AUDIO_EMBEDDINGS_MAX_SECONDS`
  (default 60s); enforced post-decode by truncating samples.
- HF resolver ninth sniff branch for audio feature-extraction repos,
  priority 105.
- Search routes `--modality audio/embedding` to a hybrid HF
  `list_models` query (feature-extraction filter combined with the
  user query plus a name-pattern match).
- Two curated entries: `mert-v1-95m` (bundled-script alias) and
  `clap-htsat-fused` (`hf://laion/clap-htsat-fused`).
- One bundled script: `src/muse/models/mert_v1_95m.py`.
- `AudioEmbeddingsClient` parallel to other muse clients; minimal HTTP
  wrapper that posts multipart/form-data with the audio bytes attached
  as a `file` part and returns `list[list[float]]`.
- `PROBE_DEFAULTS` so `muse models probe <id>` exercises a 1-second
  24kHz sine-wave WAV.

**Not in v1 (deferred):**
- Cross-modal text embedding for CLAP. CLAP also embeds text, but
  exposing a separate `/v1/audio/embeddings/text` route would split
  the modality across two URL paths. Deferred: later we can either
  (a) add a `text` form field that routes to the model's text tower,
  or (b) mount a sibling route with capability gating. Document as
  out-of-scope for v0.24.0; the `supports_text_embeddings_too`
  capability flag is set so the future work doesn't need a manifest
  schema change.
- JSON-mode fallback at `POST /v1/audio/embeddings` taking
  `{input: <data URL>, ...}`. v1 sticks to multipart-only because
  audio payloads are typically larger than image payloads (1MB+
  common) and base64 inflation is wasteful. The image-embedding
  shape exists because images are smaller and JSON is convenient for
  inline encoding.
- Audio classification with logits (e.g. AST/wav2vec downstream
  heads). Those still belong in a future `audio/classification`
  modality; embedding is "extract a vector", not "predict a label".
- Per-call attention/CLS variants. v1 picks one pooling strategy per
  architecture (CLAP `audio_embeds`, MERT mean-pool over time,
  generic `pooler_output` then `last_hidden_state` mean).
- Region-of-interest embeddings (windowed/segmented audio).
- Streaming chunked embedding (one vector per N seconds).

## Why generic runtime, not bundled-script-only

Matches muse's trajectory (sentence-transformers, llama-cpp,
faster-whisper, transformers AutoModelForSequenceClassification,
diffusers AutoPipeline, sentence-transformers CrossEncoder,
transformers AutoModelForSeq2SeqLM, transformers AutoModel image).
One runtime serves any model in its class; curated entries pin the
recommended specific. Adding `m-a-p/MERT-v1-330M` later is a
curated.yaml edit (or `muse pull hf://...`), not a new Python script.
The per-architecture dispatch lives in the runtime so the runtime
stays the single source of truth on extraction.

## Why multipart upload (and OpenAI-style envelope) at /v1/audio/embeddings

OpenAI doesn't expose a public audio-embedding endpoint, but their
`/v1/audio/transcriptions` route is the de-facto industry shape for
audio uploads. Mirroring it (multipart/form-data with a `file` part)
gives clients a near-zero-effort path: the same wire shape they use
for speech-to-text now produces audio embeddings. The response
envelope mirrors `/v1/embeddings` exactly so SDK helpers that read
`{data: [{embedding: ...}], usage, model}` work unchanged. The route
prefix `/v1/audio/...` is consistent with `audio/transcription`'s
family hierarchy.

## Package layout

```
src/muse/modalities/audio_embedding/
|-- __init__.py          # MODALITY = "audio/embedding" + build_router + exports + PROBE_DEFAULTS
|-- protocol.py          # AudioEmbeddingModel Protocol + AudioEmbeddingResult dataclass
|-- routes.py            # build_router; mounts POST /v1/audio/embeddings
|-- codec.py             # re-exports embedding_to_base64 / base64_to_embedding
|-- client.py            # AudioEmbeddingsClient (multipart upload)
|-- hf.py                # HF_PLUGIN sniffing audio feature-extraction repos
`-- runtimes/
    |-- __init__.py
    `-- transformers_audio.py  # AudioEmbeddingRuntime generic runtime
```

Bundled script:

```
src/muse/models/
`-- mert_v1_95m.py   # m-a-p/MERT-v1-95M curated default
```

## Protocol

```python
@dataclass
class AudioEmbeddingResult:
    """N audio clips in, N embedding vectors out, plus provenance.

    embeddings: list[list[float]] (float32, native dim per row).
                Pure-Python at the protocol boundary; backends may use
                numpy internally and convert via `.tolist()` before
                returning.
    dimensions: vector length (model's native dim).
    model_id: catalog id of the producing model.
    n_audio_clips: count of inputs the runtime processed (for usage
                   roll-up).
    metadata: optional per-call extras (sample_rate_used,
              source backend tag, etc.).
    """
    embeddings: list[list[float]]
    dimensions: int
    model_id: str
    n_audio_clips: int
    metadata: dict = field(default_factory=dict)


@runtime_checkable
class AudioEmbeddingModel(Protocol):
    """Structural protocol any audio embedder backend satisfies."""

    @property
    def model_id(self) -> str: ...

    @property
    def dimensions(self) -> int: ...

    def embed(self, audio_bytes_list: list[bytes]) -> AudioEmbeddingResult: ...
```

## Wire contract

**Request** (`POST /v1/audio/embeddings`, `multipart/form-data`):

| Field | Type | Required | Validation | Notes |
|---|---|---|---|---|
| `file` | file (bytes) | yes | non-empty; size <= cap | Audio in any librosa-decodable format (wav/mp3/flac/ogg/...). Repeat for batched embedding. |
| `model` | str | no | catalog id | Defaults to first registered under `audio/embedding` |
| `encoding_format` | str | no | one of "float", "base64" | default "float" |
| `user` | str | no | -- | OpenAI compat; ignored |

**Response** (`application/json`, OpenAI shape mirroring `/v1/embeddings`):

```json
{
  "object": "list",
  "data": [
    {"object": "embedding", "embedding": [0.123], "index": 0}
  ],
  "model": "mert-v1-95m",
  "usage": {"prompt_tokens": 0, "total_tokens": 0}
}
```

`prompt_tokens` and `total_tokens` are 0 in v1 (audio embedding has no
text tokenization). A future iteration may surface `n_audio_clips`
here; we keep the OpenAI-compatible names so SDK clients reading
`usage` fields don't choke.

**Error envelopes** (OpenAI-shape, used by all muse modalities):

- 400 `invalid_parameter`: empty file; missing `file` field.
- 404 `model_not_found`: `model` unknown.
- 413 `payload_too_large`: file exceeds `MUSE_AUDIO_EMBEDDINGS_MAX_BYTES`.
- 415 `unsupported_media_type`: librosa decode failure.

## Audio input handling

`librosa.load` is the workhorse. It already lives in muse[audio]
(Whisper transcription pulls it transitively via faster-whisper or
direct install). The route layer:

1. Reads each `file` upload into bytes.
2. Enforces the byte cap before decode.
3. Hands raw bytes to `backend.embed([bytes_a, bytes_b, ...])`.
4. The runtime decodes each via `librosa.load(io.BytesIO(b),
   sr=self._sample_rate, mono=True)` and truncates to
   `self._max_duration_seconds`.
5. Decoded numpy arrays go to the feature extractor as a list, then
   one batched forward pass.
6. Outputs route through `_extract_embeddings(outputs)`.

`librosa` returns `(np.ndarray, sample_rate)`. We discard the returned
sample rate (always equals the requested `sr` after resample) and
keep only the array.

Decode failures (truncated WAV, unknown codec, non-audio bytes) raise
`ValueError`/`RuntimeError` from librosa; routes catch and return
415 envelopes.

## Runtime: AudioEmbeddingRuntime

`src/muse/modalities/audio_embedding/runtimes/transformers_audio.py:AudioEmbeddingRuntime`:

The runtime wraps `transformers.AutoModel` plus
`AutoFeatureExtractor`/`AutoProcessor` with a constructor that
accepts the standard muse keyword args (model_id, hf_repo, local_dir,
device, dtype, dimensions, sample_rate, max_duration_seconds,
trust_remote_code, **_). It calls `_ensure_deps()` (lazy-import
torch + transformers + librosa via module-level sentinels), selects
a device, loads the processor (preferring `AutoProcessor` then
`AutoFeatureExtractor` on AttributeError), loads the AutoModel,
moves the model to the device, switches it into inference mode via
the `_set_inference_mode()` helper (model's no-grad-switch method),
and detects `dimensions` from the loaded model's config.

The `embed()` method always normalizes the input to a list of bytes,
decodes each entry via librosa (resampling to the model's preferred
rate), truncates to `max_duration_seconds`, runs the feature
extractor to build the input tensors, calls the model under
`torch.inference_mode()`, dispatches the outputs through
`_extract_embeddings()`, and returns an `AudioEmbeddingResult`.

The `_extract_embeddings()` dispatch order is fixed and documented:

1. **CLAP family**: outputs has an `audio_embeds` attribute set
   (returned from CLAP's `audio_model` chain or the umbrella forward
   when both towers were skipped). Use it directly.
2. **Pooler-bearing models**: outputs has `pooler_output` set (a
   non-None tensor; many BERT-shaped audio models populate it).
   Use it directly.
3. **MERT and wav2vec base**: outputs has `last_hidden_state` shaped
   `[B, T, H]`; mean-pool over the time dimension (dim 1) to get
   `[B, H]`.
4. Anything else raises `ValueError`.

Each branch is independently testable with mocked outputs.

`_set_inference_mode` helper switches the model into no-grad mode via
its inference-mode method, kept in a helper to avoid bare calls in
the runtime body, matching the v0.22.0 / v0.23.0 pattern. The literal
method name is the obvious one; the helper just looks it up via
`getattr` and invokes it when callable.

Inference uses `torch.inference_mode()`.

`AutoProcessor` handles most cases. If it fails (typically because
the repo doesn't ship `processor_config.json`) we fall back to
`AutoFeatureExtractor` with a warning log; documented in the
runtime's `_load_processor` helper. Audio repos almost universally
ship a `preprocessor_config.json` so the fallback is the common path.

`trust_remote_code` is honored from the manifest. MERT in particular
ships custom feature-extractor code; the bundled MANIFEST sets
`trust_remote_code: true` and the runtime forwards it to both
`from_pretrained` calls.

## AudioEmbeddingResult dataclass

```python
@dataclass
class AudioEmbeddingResult:
    embeddings: list[list[float]]
    dimensions: int
    model_id: str
    n_audio_clips: int
    metadata: dict = field(default_factory=dict)
```

`n_audio_clips` mirrors `n_images` from image/embedding but in audio
space.

## Codec

The codec module re-exports `embedding_to_base64` /
`base64_to_embedding` from the existing `embedding_text.codec` so the
encoding format is bit-identical (little endian float32). v1 keeps
the codec module thin; if audio-specific encoding ever diverges (e.g.
quantized embeddings, half-precision) the abstraction is already in
place.

## HF resolver plugin

`src/muse/modalities/audio_embedding/hf.py`:

Sniff: any HF repo with `feature-extraction` tag AND a repo-name
pattern matching `clap`, `mert`, `audio-encoder`, `wav2vec`, or
`audio-embedding`. The dual check (tag plus name) prevents
text-only feature-extraction models from being picked up.

```python
def _sniff(info) -> bool:
    tags = getattr(info, "tags", None) or []
    if "feature-extraction" not in tags:
        return False
    repo_id = (getattr(info, "id", "") or "").lower()
    return any(s in repo_id for s in
               ("clap", "mert", "audio-encoder", "wav2vec",
                "audio-embedding"))
```

Priority **105**: between embedding/text (110) and the
image-generation file-pattern plugin (100). Loses to file-pattern
plugins (GGUF, faster-whisper, diffusers) so a multi-purpose repo
that also ships, say, a CT2 ASR model still resolves as
`audio/transcription`.

Capability defaults are inferred per-pattern:
- `clap` in repo name -> `dimensions=512`, `sample_rate=48000`,
  `supports_text_embeddings_too=True`.
- `mert` in repo name -> `dimensions=768`, `sample_rate=24000`,
  `supports_text_embeddings_too=False`.
- `wav2vec` in repo name -> `dimensions=768`, `sample_rate=16000`,
  `supports_text_embeddings_too=False`.
- Fallback when no pattern matches: `dimensions=512`,
  `sample_rate=16000`, `supports_text_embeddings_too=False`.

## Bundled script: mert_v1_95m

`src/muse/models/mert_v1_95m.py`:

```python
MANIFEST = {
    "model_id": "mert-v1-95m",
    "modality": "audio/embedding",
    "hf_repo": "m-a-p/MERT-v1-95M",
    "description": (
        "MERT v1 95M: music understanding, 768-dim audio embeddings, MIT"
    ),
    "license": "MIT",
    "pip_extras": (
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "librosa>=0.10.0",
    ),
    "system_packages": (),
    "capabilities": {
        "device": "auto",
        "dimensions": 768,
        "sample_rate": 24000,  # MERT was trained at 24kHz
        "max_duration_seconds": 60.0,
        "supports_text_embeddings_too": False,
        "trust_remote_code": True,  # MERT ships custom feature_extractor code
        "memory_gb": 0.5,
    },
    "allow_patterns": [
        "*.safetensors", "*.json", "*.txt", "*.md",
        "*.py",  # required by trust_remote_code path
        "preprocessor_config.json",
    ],
}
```

The `Model` class wraps `transformers.AutoModel` +
`AutoFeatureExtractor` directly (rather than going through
`AudioEmbeddingRuntime`) so the script demonstrates the same shape
muse uses for other bundled models. Lazy imports.

`trust_remote_code: True` is required because MERT ships a custom
feature extractor in the repo. Mirror Qwen3-Embedding's pattern (it
sets `trust_remote_code: true` in curated.yaml and the
SentenceTransformer runtime honors it).

## Curated entries

```yaml
- id: mert-v1-95m
  bundled: true

- id: clap-htsat-fused
  uri: hf://laion/clap-htsat-fused
  modality: audio/embedding
  size_gb: 0.8
  description: "CLAP HTSAT fused: 800MB, 512-dim audio embeddings, supports text too, BSD-3"
  capabilities:
    supports_text_embeddings_too: true
    dimensions: 512
    sample_rate: 48000
    memory_gb: 1.5
```

## PROBE_DEFAULTS

```python
def _make_probe_audio() -> bytes:
    """Generate a 1-second 24kHz mono sine wave WAV.

    Deferred so the probe doesn't import numpy at module import time.
    """
    import io, numpy as np, wave
    sr = 24000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = (0.3 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(audio.tobytes())
    return buf.getvalue()


PROBE_DEFAULTS = {
    "shape": "1s 24kHz sine wave",
    "call": lambda m: m.embed([_make_probe_audio()]),
}
```

Used by `muse models probe <id>` so a power user can verify a fresh
pull works end-to-end without opening a Python REPL.

## Test strategy

Unit-heavy. Mocks for `transformers.AutoModel` +
`AutoFeatureExtractor` plus `librosa.load`. One slow e2e test
exercises FastAPI + codec + mocked runtime. One opt-in integration
test against a live muse server with a real audio embedder loaded.

Coverage targets:

- Protocol + dataclass shape (5 tests).
- Codec: float passthrough; base64 roundtrip; matryoshka not
  required (skipped); index ordering preserved.
- Routes: 200 happy path with single + batch input, 400 envelope for
  empty file, 404 for unknown model, 413 for oversized, 415 for
  decoder failure, default model resolution.
- Runtime: deferred imports; processor + model load order; per-arch
  `_extract_embeddings` (CLAP, MERT, generic paths each
  independently mocked); device auto-select; inference-mode helper
  invoked; trust_remote_code threaded through.
- HF plugin: positive sniff (feature-extraction tag + clap/mert/
  wav2vec name), negative without tag, negative without name match,
  priority 105, per-pattern capability defaults, search branch.
- Curated: `mert-v1-95m` parses as bundled; `clap-htsat-fused` parses
  as URI with capabilities overlay.
- Bundled script: MANIFEST shape, Model construction with lazy
  imports patched, `embed()` returns AudioEmbeddingResult with the
  right dimensions echoed.
- E2E slow: full multipart-in / JSON-out round-trip through the
  supervisor.
- Integration opt-in: real server + real model.
  `MUSE_AUDIO_EMBEDDING_MODEL_ID` env override (default
  `mert-v1-95m`).

## Documentation

- CLAUDE.md: add `audio/embedding` to modality list; note bundled
  default and per-architecture dispatch. Bump count from 11
  modalities to 12.
- README.md: modality list + `/v1/audio/embeddings` endpoint + curl
  example.
- src/muse/__init__.py docstring: bump version to 0.24.0; add
  `audio/embedding` to bundled modalities list.

## Release

v0.24.0. Minor bump (new feature, no breaking changes). Tag message
calls out: new modality, new endpoint, new bundled default,
multipart-upload audio-embedding wire contract.

## Out of scope

- Cross-modal text embedding via the same runtime (CLAP text tower).
  Capability flag is set so future work doesn't need a manifest
  schema change.
- JSON-mode fallback at the same path with data-URL inputs.
- Audio classification logits (`audio/classification` modality,
  future).
- Per-call pooling-strategy override.
- Region-of-interest / windowed embeddings.
- Streaming chunked embedding (one vector per N seconds of audio).
