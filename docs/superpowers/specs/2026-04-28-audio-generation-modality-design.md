# `audio/generation` modality (music + SFX text-to-audio)

**Date:** 2026-04-28
**Status:** approved
**Target release:** v0.20.0

## Goal

Add muse's 9th modality: `audio/generation`, mounted at two routes that
share a single registry surface and a single codec:

- `POST /v1/audio/music` for musical content
- `POST /v1/audio/sfx` for sound effects / non-musical audio

Both routes accept the same body shape (mirroring `/v1/audio/speech`,
with `prompt` instead of `input`) and return the same audio bytes
encoding (`wav` default; `mp3`, `opus`, `flac` opt-in).

The first bundled model is **Stable Audio Open 1.0** (Apache 2.0,
~3.4GB at fp16, up to 47s, 44.1kHz stereo). It advertises both
`supports_music: True` and `supports_sfx: True`. Future models can pin
either flag (a MusicGen-only or AudioGen-only model would surface 400
on the unsupported route).

A generic `StableAudioRuntime` wraps `diffusers.StableAudioPipeline`
so any Stable Audio Open-shaped repo on HuggingFace works via the
resolver. The HF plugin sniffs at priority 105 (more specific than
embedding/text's 110, less specific than file-pattern plugins at 100)
to avoid disturbing other plugins.

## Why two routes, one modality

The user-facing intent for music vs. SFX differs sharply. A user
posting "footsteps on gravel" to `/v1/audio/music` for 30s gets a
30-second loop of footsteps that the model treated as music; the
output is wrong but the request never errored. Posting to
`/v1/audio/sfx` makes that intent legible to the model (and to muse
operators reading logs).

The two-route shape also lets future models stay isolatable:

- A MusicGen-only model declares `supports_music: True,
  supports_sfx: False`. The route handler returns 400 on
  `/v1/audio/sfx` with `"model X does not support sfx generation"`.
- An AudioGen-only model declares the inverse and 400s on
  `/v1/audio/music`.

Same codec, same client, same registry, same runtime path: only the
URL hint and the capability gate differ. That keeps the cost of two
routes low while preserving the legibility win.

## Why mirror `/v1/audio/speech` (and not OpenAI's nonexistent T2A)

OpenAI does not currently expose a text-to-audio API beyond TTS
(`/v1/audio/speech`) and ASR (`/v1/audio/transcriptions`). There is
no industry-standard wire shape for music/SFX generation: ElevenLabs,
Stable Audio's own demo, MusicGen, Bark, and Suno all ship different
JSON shapes. We pick the muse-native shape that matches our existing
TTS route most closely:

- `prompt` (text input) replaces `input`
- `model` (catalog id) is identical
- `response_format` (wav default) is identical
- `seed`, `duration`, `steps`, `guidance`, `negative_prompt` are
  generation-specific muse extensions

This keeps the wire surface uniform across modalities (request shape +
response shape are well-known to anyone who's used `/v1/audio/speech`)
and avoids picking a vendor lock-in pattern.

## Scope

**In v1:**

- `POST /v1/audio/music` and `POST /v1/audio/sfx`, identical request
  shape, share the codec and registry plumbing.
- Per-route capability gating via `capabilities.supports_music` and
  `capabilities.supports_sfx`. 400 returned when a route does not
  match the model's capabilities.
- Generic `StableAudioRuntime` wrapping `diffusers.StableAudioPipeline`.
  Any Stable Audio Open-shaped repo works.
- HF resolver fifth sniff branch for stable-audio repos (priority 105).
  Narrow sniff: text-to-audio tag AND repo name contains `stable-audio`
  AND `model_index.json` sibling.
- One curated entry: `stable-audio-open-1.0` (bundled-script alias to
  `stabilityai/stable-audio-open-1.0`).
- One bundled script: `src/muse/models/stable_audio_open_1_0.py`.
- `MusicClient` and `SFXClient` parallel to other muse clients;
  minimal HTTP wrappers.
- Codec layer: `wav` always; `flac` via `soundfile` (already in
  muse[audio]); `mp3` and `opus` via `pydub` + ffmpeg.
- `UnsupportedFormatError` mirrors the v0.18.0 image_animation pattern;
  routes catch and 400.
- `PROBE_DEFAULTS` exercises a 5-second music probe so `muse models
  probe stable-audio-open-1.0` works.

**Not in v1 (deferred):**

- MusicGen / AudioGen / AudioLDM2 runtimes. The architectures differ
  enough from Stable Audio that each needs its own runtime; they can
  ship as bundled scripts (or future per-runtime plugins) when needed.
- Audio-to-audio conditioning (init_audio + strength). Stable Audio
  Open 1.0 supports it; out of scope for v1 wire surface. Future task.
- Streaming. Diffusion is per-step refinement of the whole clip, not
  time-ordered chunks; not naturally streamable. Bundled future TTS-
  style audio gen models could stream via SSE if needed.
- Long-form generation (>47s native limit). Stable Audio Open caps at
  ~47s; future models or chained inference could lift this.
- Batched generation (`n` parameter). Stable Audio is heavy; one clip
  per request keeps memory predictable. Future task if a small model
  ships that supports it cleanly.

## Why generic runtime, not bundled-script-only

Matches muse's trajectory (sentence-transformers, llama-cpp,
faster-whisper, transformers AutoModelForSequenceClassification,
diffusers AutoPipelineForText2Image, sentence_transformers
CrossEncoder, AnimateDiff). One runtime serves any Stable Audio
Open-shaped repo. The bundled script `stable_audio_open_1_0.py`
exists for first-found-wins curated alias and discovery, but the
runtime under `runtimes/stable_audio.py` is what powers
resolver-pulled repos.

Adding `stable-audio-open-1.5` (when it lands) is a curated.yaml edit
or a `muse pull hf://...`, not a new Python file. Adding MusicGen
takes a new bundled script (different pipeline class) but reuses the
modality, codec, routes, client, capability flags.

## Why audio/generation (broad MIME) at /v1/audio/music + /v1/audio/sfx

Same precedent as `text/classification` at `/v1/moderations` (broad
MIME tag, narrow OpenAI-compat URL): the modality tag describes the
backing model class, the URL describes the wire surface. This time
the broad tag hosts two URL routes simultaneously (not a single primary
+ deferred secondaries). Both routes share:

- One `audio_generation/` package
- One `AudioGenerationModel` Protocol
- One `AudioGenerationResult` dataclass
- One codec
- One registry surface
- Two client classes (`MusicClient`, `SFXClient`) for ergonomic Python
  clients

Future audio generation routes (e.g., `/v1/audio/ambient`,
`/v1/audio/foley`) can mount on the same modality without a second
package.

## Package layout

```
src/muse/modalities/audio_generation/
|-- __init__.py          # MODALITY = "audio/generation" + build_router + exports + PROBE_DEFAULTS
|-- protocol.py          # AudioGenerationModel Protocol + AudioGenerationResult dataclass
|-- routes.py            # build_router; mounts POST /v1/audio/music and /v1/audio/sfx
|-- codec.py             # encode_wav/mp3/opus/flac + UnsupportedFormatError
|-- client.py            # MusicClient + SFXClient
|-- hf.py                # HF_PLUGIN sniffing stable-audio repos
`-- runtimes/
    |-- __init__.py
    `-- stable_audio.py  # StableAudioRuntime generic runtime
```

Bundled script:

```
src/muse/models/
`-- stable_audio_open_1_0.py   # Stable Audio Open 1.0 curated default
```

## Protocol

```python
@dataclass
class AudioGenerationResult:
    """One generated audio clip plus provenance metadata.

    audio: numpy float32 array, shape (samples,) for mono or
           (samples, channels) for stereo+. Codec converts to
           int16 PCM at output.
    sample_rate: per-model output sample rate (Hz). Stable Audio
           Open is 44100; MusicGen typically 32000; AudioGen 16000.
    channels: 1 for mono, 2 for stereo. Codec respects this when
           writing the WAV/FLAC header.
    duration_seconds: real duration of the synthesized clip,
           which may differ from the requested duration when the
           model rounds to its internal grid.
    metadata: dict of model-specific provenance (prompt, steps,
           guidance, seed, model_id).
    """
    audio: np.ndarray
    sample_rate: int
    channels: int
    duration_seconds: float
    metadata: dict


@runtime_checkable
class AudioGenerationModel(Protocol):
    """Structural protocol any audio-generation backend satisfies."""

    @property
    def model_id(self) -> str: ...

    def generate(
        self,
        prompt: str,
        *,
        duration: float | None = None,
        seed: int | None = None,
        steps: int | None = None,
        guidance: float | None = None,
        negative_prompt: str | None = None,
        **kwargs,
    ) -> AudioGenerationResult: ...
```

## Wire contract

**Request** (`POST /v1/audio/music` and `POST /v1/audio/sfx`,
`application/json`):

| Field | Type | Required | Validation | Notes |
|---|---|---|---|---|
| `prompt` | `str` | yes | `1 <= len <= 4000` | Generation prompt |
| `model` | `str | None` | no | catalog id | Defaults to first registered under `audio/generation` |
| `duration` | `float | None` | no | `0.5 <= duration <= 120.0` when set | Defaults to model's `default_duration` capability |
| `seed` | `int | None` | no | non-negative when set | Reproducibility seed |
| `response_format` | `str` | no | `^(wav|mp3|opus|flac)$` | Default `wav` |
| `steps` | `int | None` | no | `1 <= steps <= 200` when set | Defaults to model's `default_steps` |
| `guidance` | `float | None` | no | `0.0 <= guidance <= 20.0` when set | Defaults to model's `default_guidance` |
| `negative_prompt` | `str | None` | no | None | Negative prompt (Stable Audio supports it) |

**Response:** raw audio bytes, `Content-Type` set per `response_format`:

- `wav` -> `audio/wav`
- `mp3` -> `audio/mpeg`
- `opus` -> `audio/ogg`
- `flac` -> `audio/flac`

Mirror `/v1/audio/speech`'s response shape: bytes only, no JSON
envelope. Set `Content-Length` and content type. No streaming in v1.

**Error envelopes** (OpenAI-shape):

- 400 `invalid_parameter`:
  - `prompt` empty (handled by Pydantic min_length).
  - `duration`, `steps`, `guidance`, `seed` out of range
    (handled by Pydantic Field).
  - `response_format` not in supported set (Pydantic).
  - `response_format` requires deps that aren't present (codec).
  - Model does not support the requested generation kind
    (`supports_music: False` for /v1/audio/music, etc.).
- 404 `model_not_found`: `model` unknown.

## Capability flags

Every audio/generation manifest declares:

| Key | Type | Default | Notes |
|---|---|---|---|
| `supports_music` | `bool` | `True` | Gate for `/v1/audio/music` |
| `supports_sfx` | `bool` | `True` | Gate for `/v1/audio/sfx` |
| `default_duration` | `float` | model-specific | Honored when request omits `duration` |
| `min_duration` | `float` | 0.5 | Lower bound advertised to clients |
| `max_duration` | `float` | model-specific | Upper bound (ignored on request validation; route enforces) |
| `default_sample_rate` | `int` | 44100 | Output sample rate (Hz) |
| `default_steps` | `int` | 50 | Honored when request omits `steps` |
| `default_guidance` | `float` | 7.0 | Honored when request omits `guidance` |
| `memory_gb` | `float` | model-specific | Annotation for `muse models list` |
| `device` | `str` | "auto" | "auto" / "cpu" / "cuda" |

## Routes

The handler logic is shared between the two routes; the only
difference is the capability key checked.

```python
def build_router(registry):
    router = APIRouter(prefix="/v1/audio", tags=["audio/generation"])

    @router.post("/music")
    async def music(req: AudioGenerationRequest):
        return await _handle(registry, req, kind="music")

    @router.post("/sfx")
    async def sfx(req: AudioGenerationRequest):
        return await _handle(registry, req, kind="sfx")

    return router


async def _handle(registry, req, *, kind):
    try:
        model = registry.get(MODALITY, req.model)
    except KeyError:
        raise ModelNotFoundError(model_id=req.model or "<default>", modality=MODALITY)
    manifest = registry.manifest(MODALITY, model.model_id) or {}
    capabilities = manifest.get("capabilities") or {}
    cap_key = "supports_music" if kind == "music" else "supports_sfx"
    if not capabilities.get(cap_key, True):
        return error_response(
            400, "invalid_parameter",
            f"model {model.model_id!r} does not support {kind} generation",
        )
    # offload sync inference to a thread, encode per response_format,
    # return Response(content=bytes, media_type=...)
```

## Runtime: StableAudioRuntime

`src/muse/modalities/audio_generation/runtimes/stable_audio.py:StableAudioRuntime`:

```python
class StableAudioRuntime:
    """Generic runtime over diffusers.StableAudioPipeline.

    Construction kwargs (set by catalog at load_backend, sourced from
    manifest fields and capabilities):
      - hf_repo, local_dir, device, dtype: standard
      - model_id: catalog id (response metadata echoes this)
      - default_duration, default_steps, default_guidance,
        default_sample_rate, max_duration: defaults from manifest
    """

    def __init__(
        self,
        *,
        model_id: str,
        hf_repo: str,
        local_dir: str | None = None,
        device: str = "auto",
        dtype: str = "float16",
        default_duration: float = 10.0,
        default_steps: int = 50,
        default_guidance: float = 7.0,
        default_sample_rate: int = 44100,
        max_duration: float = 47.0,
        **_: Any,
    ) -> None: ...

    def generate(
        self,
        prompt: str,
        *,
        duration: float | None = None,
        seed: int | None = None,
        steps: int | None = None,
        guidance: float | None = None,
        negative_prompt: str | None = None,
        **_: Any,
    ) -> AudioGenerationResult: ...
```

Key points:

- Lazy-import torch + diffusers (sentinel pattern).
- `_select_device` mirrors siblings (auto/cuda/mps/cpu).
- Honors max_duration as a hard cap; clamps requested duration.
- StableAudioPipeline returns `out.audios[0]`: numpy array, shape
  `(channels, samples)` or `(samples,)`. Runtime normalizes to
  `(samples, channels)` for soundfile compatibility (this is the
  shape soundfile expects).
- `audio_end_in_s=duration` is the StableAudioPipeline knob for clip
  length.

## Codec

```python
class UnsupportedFormatError(Exception):
    """Raised when a response_format requires deps that aren't installed."""


def encode_wav(audio: np.ndarray, sample_rate: int, channels: int) -> bytes:
    """Stdlib wave; 16-bit PCM. Always available."""

def encode_flac(audio: np.ndarray, sample_rate: int, channels: int) -> bytes:
    """soundfile-based FLAC encoder. Raises UnsupportedFormatError if
    soundfile is missing (it ships with muse[audio])."""

def encode_mp3(audio: np.ndarray, sample_rate: int, channels: int) -> bytes:
    """pydub+ffmpeg-based MP3. Raises UnsupportedFormatError when either
    dep is missing."""

def encode_opus(audio: np.ndarray, sample_rate: int, channels: int) -> bytes:
    """pydub+ffmpeg-based Opus. Raises UnsupportedFormatError when either
    dep is missing."""
```

The audio array shape contract is `(samples,)` for mono or
`(samples, channels)` for multi-channel. Codec handles either.

`encode_wav` uses stdlib `wave` and is always available. `encode_flac`
uses `soundfile` (already in muse[audio] for Kokoro). `encode_mp3` and
`encode_opus` use `pydub` + ffmpeg via shutil.which. If any dep is
missing, raises `UnsupportedFormatError`; the route catches and 400s
with `"response_format X requires Y; install or use wav/flac"`.

For mp3/opus, the runtime first writes WAV to an in-memory buffer,
then transcodes via pydub. This keeps the pipeline simple and avoids
multiple format-specific encoders.

## HF resolver plugin

`src/muse/modalities/audio_generation/hf.py`:

Sniff: `text-to-audio` tag in tags AND `model_index.json` in siblings
AND `stable-audio` substring in repo id (case-insensitive).

```python
def _sniff(info) -> bool:
    tags = getattr(info, "tags", None) or []
    if "text-to-audio" not in tags:
        return False
    siblings = [s.rfilename for s in getattr(info, "siblings", [])]
    if "model_index.json" not in siblings:
        return False
    repo_id = (getattr(info, "id", "") or "").lower()
    return "stable-audio" in repo_id
```

Priority **105**: more specific than embedding/text (110) since the
sniff requires both file shape AND tag AND name pattern; less
specific than the other priority-100 plugins which match very
specific shapes (.gguf, model_index.json + text-to-image, CT2
faster-whisper). The narrowness here is by name match (only Stable
Audio Open-shaped repos), which constrains false positives more
than a tag match alone would.

Capability defaults (hardcoded): Stable Audio Open lineage:

- `default_steps=50`
- `default_guidance=7.0`
- `default_sample_rate=44100`
- `default_duration=10.0`
- `min_duration=1.0`
- `max_duration=47.0`
- `supports_music=True`
- `supports_sfx=True`

`HFResolver.search` adds a branch for `modality == "audio/generation"`:
search HF for the query string, filter by `text-to-audio` tag (with
the same `stable-audio` substring guard).

## Bundled script: stable_audio_open_1_0

`src/muse/models/stable_audio_open_1_0.py`:

```python
MANIFEST = {
    "model_id": "stable-audio-open-1.0",
    "modality": "audio/generation",
    "hf_repo": "stabilityai/stable-audio-open-1.0",
    "description": "Stable Audio Open 1.0: 47s music + SFX, 44.1kHz stereo",
    "license": "Apache 2.0",
    "pip_extras": (
        "torch>=2.1.0",
        "diffusers>=0.27.0",
        "transformers>=4.36.0",
        "accelerate",
        "soundfile",
    ),
    "system_packages": ("ffmpeg",),
    "capabilities": {
        "device": "cuda",
        "supports_music": True,
        "supports_sfx": True,
        "default_duration": 10.0,
        "min_duration": 1.0,
        "max_duration": 47.0,
        "default_sample_rate": 44100,
        "default_steps": 50,
        "default_guidance": 7.0,
        "memory_gb": 6.0,
    },
    "allow_patterns": [
        "*.safetensors", "*.json", "*.txt", "*.md",
        "scheduler/*.json",
        "transformer/*.fp16.safetensors", "transformer/*.json",
        "vae/*.fp16.safetensors", "vae/*.json",
        "text_encoder/*.fp16.safetensors", "text_encoder/*.json",
        "tokenizer/*",
    ],
}
```

The `Model` class wraps `diffusers.StableAudioPipeline` directly
(rather than going through `StableAudioRuntime`) so the script
demonstrates the same shape muse uses for other bundled models. Lazy
imports for torch + diffusers.

`device: cuda` is the default in capabilities because Stable Audio
inference is impractical on CPU (10-second clips take 5+ minutes on
CPU vs. 20-30s on a 12GB GPU). Users without a GPU can override via
`muse pull` capability overlay.

## Curated entry

```yaml
# ---------- audio/generation (music + sfx) ----------

- id: stable-audio-open-1.0
  bundled: true
```

For now, no second entry. AudioLDM2 and MusicGen could be added later
when their runtimes ship.

## PROBE_DEFAULTS

```python
PROBE_DEFAULTS = {
    "shape": "5s music",
    "call": lambda m: m.generate("ambient piano", duration=5.0),
}
```

Used by `muse models probe stable-audio-open-1.0` to verify a fresh
pull works end-to-end without opening a Python REPL. 5s keeps probe
time low (under 30s on a 12GB GPU).

## Test strategy

Unit-heavy. Mocks for `diffusers.StableAudioPipeline`. One slow e2e
test exercises FastAPI + codec + mocked runtime. One opt-in
integration test against a live muse server with a real Stable Audio
Open loaded.

Coverage targets:

- Protocol + dataclass shape.
- Codec: wav (always), flac (with soundfile), mp3 (UnsupportedFormatError
  when pydub missing), opus (likewise). Round-trip tests where deps
  available; assertion-only when not.
- Routes: 200 happy path on both /v1/audio/music and /v1/audio/sfx,
  400 envelope for capability mismatch, 400 for unsupported format,
  404 for unknown model, content-type header per format.
- Runtime: deferred imports, generate() returns AudioGenerationResult,
  default capabilities applied when request omits fields, max_duration
  clamping.
- HF plugin: positive sniff (stable-audio tag + name + model_index),
  negatives (text-to-audio without stable-audio in name; stable-audio
  in name without model_index.json; only one of three conditions),
  priority correctness, search routing.
- Curated: stable-audio-open-1.0 entry parses as bundled.
- Bundled script: MANIFEST shape, Model construction with lazy imports
  patched.
- E2E slow: full bytes-out round-trip through the supervisor.
- Integration opt-in: real server + real stable-audio-open-1.0.

## Documentation

- CLAUDE.md: add `audio/generation` to modality list; note that this
  modality is muse's first to mount two routes on one modality;
  document the supports_music/supports_sfx capability gates.
- README.md: modality list + `/v1/audio/music` + `/v1/audio/sfx`
  endpoints + curl examples.

## Release

v0.20.0. Minor bump (new feature, no breaking changes). Tag message
calls out: 9th modality, two new endpoints sharing one modality + one
codec, capability-gated routing (precedent for future single-modality
multi-route designs).

## Out of scope

- MusicGen / AudioGen / AudioLDM2 runtimes (different architectures).
- Audio-to-audio conditioning (init_audio + strength).
- Streaming.
- Long-form generation beyond 47s.
- Batched generation (`n` parameter).
