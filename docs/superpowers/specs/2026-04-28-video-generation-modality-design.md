# `video/generation` modality design

**Date:** 2026-04-28
**Driver:** ship muse's 15th modality (`video/generation`), exposing
`POST /v1/video/generations` for text-to-video synthesis. Wan-AI/Wan2.1-T2V-1.3B
is the v0.27.0 default bundle (Apache 2.0, ~3GB at fp16, fits 8GB GPUs).
THUDM/CogVideoX-2b ships as a second curated entry. LTX-Video and
larger-VRAM options (Mochi-1, HunyuanVideo) are curated only.

This is the largest modality muse has shipped: heaviest models, most
complex codec (mp4 + webm + frames_b64), tightest VRAM budgets, and
the first modality to ship with two distinct runtimes side-by-side
(WanRuntime + CogVideoXRuntime) under one MIME tag.

## Goal

1. Mount `POST /v1/video/generations` (JSON body) on a new
   `video/generation` modality. Inputs: prompt + duration + fps + size
   + steps + guidance + negative_prompt + seed + n + response_format.
   Behavior: an mp4 / webm / list-of-frames asset bytes-as-base64.
2. Bundle `Wan-AI/Wan2.1-T2V-1.3B` (~3GB at fp16, 5-second clips at
   832x480, fits on 8GB cards, Apache 2.0) as the default low-VRAM
   model, alias-mapped via `curated.yaml`.
3. New `WanRuntime` generic runtime in `video_generation/runtimes/wan_runtime.py`,
   wrapping `diffusers.WanPipeline` (or `DiffusionPipeline.from_pretrained`
   when `WanPipeline` isn't yet a top-level export).
4. New `CogVideoXRuntime` generic runtime in
   `video_generation/runtimes/cogvideox_runtime.py`, wrapping
   `diffusers.CogVideoXPipeline`.
5. New HF plugin sniffing `text-to-video`-tagged repos whose name
   matches one of the supported architecture patterns (`wan`, `cogvideox`,
   `ltx`, `mochi`, `hunyuan`). Priority 105.
6. New `VideoGenerationClient` HTTP client (JSON body, returns mp4/webm
   bytes by default; list-of-PNG-bytes for `frames_b64`).
7. New codec module: `encode_mp4` (h264 via imageio), `encode_webm`
   (vp9 via imageio), `encode_frames_b64` (one base64 PNG per frame).
   The mp4 path is reused logic from image_animation; webm is new.
8. Three curated entries: bundled `wan2-1-t2v-1-3b`, plus
   `cogvideox-2b` (~6GB, 12GB-GPU friendly), plus `ltx-video` (~13GB,
   16GB+ GPU). Mochi-1 and HunyuanVideo are documented as curated
   add-ons but not enabled in v0.27.0 (they need 24GB+ VRAM and the
   architecture-specific runtimes are deferred).

## Non-goals

- **Image-to-video conditioning.** Wan supports it via `WanImageToVideoPipeline`;
  CogVideoX-1.5 has an I2V variant. Text-only is the v0.27.0 surface.
  `supports_image_to_video` is reserved at the manifest level for
  future runtimes; the wire layer doesn't yet expose an `image` field.
  Filed for v1.next.
- **Audio sync.** No audio in v1. Generated videos are silent.
- **Streaming frame-by-frame.** Like image_animation: route waits for
  all frames before encoding. Diffusion is progressive refinement of
  the entire clip at once, not time-sequential.
- **Mochi-1 + HunyuanVideo runtimes.** The HF plugin sniffs them, but
  the v0.27.0 fallback routes them to the closest existing runtime
  (Wan) which won't load them correctly. Manifest is synthesized but
  the user sees a clear runtime error; full support filed for v1.next.
- **Composite URIs** (`hf://base+vae`). Wan and CogVideoX are
  single-component repos; no composite resolution needed.
- **`/v1/images/animations` overlap.** image_animation and
  video_generation are deliberately separate modalities. Animation is
  short coherent loops (16 frames @ 8fps = 2s, default loop=true,
  WebP); video is narrative clips (5s+, single play, mp4). The wire
  defaults reflect the use case difference, even though the codec
  patterns rhyme.

## Architecture

```
client.video.generate(prompt="...", model="wan2-1-t2v-1-3b")
   |
   v  HTTP POST /v1/video/generations, application/json
   |  body: {prompt, model, duration_seconds?, fps?, size?, steps?,
   |         guidance?, negative_prompt?, seed?, response_format?, n?}
   v
routes.video_generations handler
   - validate request via VideoGenerationRequest pydantic
   - lookup model via registry; unknown -> 404 with OpenAI envelope
   - parse size as WxH; clamp to manifest min/max; honor capabilities
   - for i in range(n):
       result = backend.generate(prompt, **kwargs)
   - encode each result via codec (mp4 / webm / frames_b64)
   - assemble OpenAI-shape envelope
   v
WanRuntime.generate(prompt, ...)  /  CogVideoXRuntime.generate(prompt, ...)
   - lazy-load pipeline once, cache as self._pipe
   - compute num_frames from duration_seconds * fps, align to model
     native frame count if needed
   - call self._pipe(prompt=..., num_frames=..., width=..., height=...,
                      num_inference_steps=..., guidance_scale=..., generator=...)
   - extract frames via out.frames[0] (list of PIL.Image)
   - return VideoResult(frames, fps, width, height, duration_seconds, seed, metadata)
   v
codec.encode_mp4 / encode_webm / encode_frames_b64
   v
JSON envelope: {data: [{b64_json: ...}], model, metadata: {frames, fps, ...}}
```

The MIME tag is `video/generation`; the URL is `/v1/video/generations`
(plural to mirror OpenAI's `/v1/images/generations` even though
n<=2 in practice). The directory is `src/muse/modalities/video_generation/`
(underscore in dir name; slash in MIME tag).

## Wire contract

`POST /v1/video/generations` (application/json):

| Field | Type | Required | Default | Validation |
|---|---|---|---|---|
| `prompt` | str | yes | - | `1 <= len <= 4000` |
| `model` | str | no | None | bundled or pulled model id |
| `duration_seconds` | float | no | from manifest | `0.5 <= x <= 30.0` |
| `fps` | int | no | from manifest | `1 <= x <= 60` |
| `size` | str | no | from manifest | regex `^\d+x\d+$`, e.g. `"832x480"` |
| `seed` | int | no | None | for reproducibility |
| `negative_prompt` | str | no | None | up to 4000 chars |
| `steps` | int | no | from manifest | `1 <= x <= 200` |
| `guidance` | float | no | from manifest | `0.0 <= x <= 20.0` |
| `response_format` | str | no | `"mp4"` | one of `mp4`, `webm`, `frames_b64` |
| `n` | int | no | 1 | `1 <= x <= 2` (videos are heavy) |

Response (mirrors `/v1/images/animations` envelope):

```json
{
  "data": [
    {"b64_json": "<base64 of mp4 bytes>"}
  ],
  "model": "wan2-1-t2v-1-3b",
  "metadata": {
    "frames": 25,
    "fps": 5,
    "duration_seconds": 5.0,
    "format": "mp4",
    "size": [832, 480]
  }
}
```

For `response_format: "frames_b64"`, `data` is a list of per-frame
base64 PNGs (mirror image_animation behavior):

```json
{
  "data": [
    {"b64_json": "<png frame 1>"},
    {"b64_json": "<png frame 2>"},
    ...
  ],
  "model": "wan2-1-t2v-1-3b",
  "metadata": {"frames": 25, "fps": 5, "duration_seconds": 5.0, ...}
}
```

For `n>1` with non-frames format, `data` contains `n` entries (one
encoded video each). For `n>1` with `frames_b64`, frames from each
result are appended in order; this matches image_animation's behavior
but is admittedly ambiguous. v1 sticks with this shape; a future
revision may add explicit per-result grouping.

## Capability flags

Per-model capabilities declared in MANIFEST or curated overlay:

| Capability | Type | Purpose |
|---|---|---|
| `default_duration_seconds` | float | manifest-default clip length |
| `default_fps` | int | manifest-default frame rate |
| `min_duration_seconds` | float | per-model minimum (clip protocol-side) |
| `max_duration_seconds` | float | per-model maximum (often hard model limit) |
| `default_size` | tuple[int, int] | resolution default |
| `default_steps` | int | denoise steps default |
| `default_guidance` | float | guidance scale default |
| `supports_image_to_video` | bool | future: I2V capability gate |
| `memory_gb` | float | peak inference VRAM (with activations) |
| `device` | str | "cuda" for v0.27.0 (video models are GPU-only) |

For Wan 2.1 T2V 1.3B defaults: `duration=5.0, fps=5, size=(832, 480),
steps=30, guidance=5.0, supports_image_to_video=False, memory_gb=6.0,
device="cuda"`.

For CogVideoX-2b defaults: `duration=6.0, fps=8, size=(720, 480),
steps=50, guidance=6.0, memory_gb=9.0, device="cuda"`.

## Modality structure

```
src/muse/modalities/video_generation/
|-- __init__.py            # MODALITY="video/generation", build_router, exports, PROBE_DEFAULTS
|-- protocol.py            # VideoResult dataclass, VideoGenerationModel Protocol
|-- codec.py               # encode_mp4 / encode_webm / encode_frames_b64
|-- routes.py              # POST /v1/video/generations
|-- client.py              # VideoGenerationClient
|-- hf.py                  # HF_PLUGIN sniffing text-to-video repos
`-- runtimes/
    |-- __init__.py
    |-- wan_runtime.py     # WanRuntime (diffusers WanPipeline / DiffusionPipeline)
    `-- cogvideox_runtime.py  # CogVideoXRuntime (diffusers CogVideoXPipeline)
```

Bundled script: `src/muse/models/wan2_1_t2v_1_3b.py`.

## `VideoResult` dataclass

```python
@dataclass
class VideoResult:
    """One generated video clip plus timing + provenance.

    `frames` is loosely typed (Any) at the protocol boundary so backends
    can return PIL.Image, numpy arrays, or torch tensors. The codec
    layer normalizes to PIL.Image before encoding.

    `duration_seconds` is the actual clip duration (frames / fps), not
    the requested duration; runtimes that align to model-native frame
    counts may return slightly different values than requested.
    """
    frames: list[Any]
    fps: int
    width: int
    height: int
    duration_seconds: float
    seed: int
    metadata: dict = field(default_factory=dict)
```

## `VideoGenerationModel` Protocol

```python
@runtime_checkable
class VideoGenerationModel(Protocol):
    @property
    def model_id(self) -> str: ...

    def generate(
        self,
        prompt: str,
        *,
        negative_prompt: str | None = None,
        duration_seconds: float | None = None,
        fps: int | None = None,
        width: int | None = None,
        height: int | None = None,
        steps: int | None = None,
        guidance: float | None = None,
        seed: int | None = None,
        **kwargs: Any,
    ) -> VideoResult: ...
```

Duck-typed; backends only need to satisfy the structural shape.

## Codec

`muse.modalities.video_generation.codec`:

- `encode_mp4(frames, fps) -> bytes`: h264 via `imageio.mimwrite(...,
  codec="h264", format="mp4", quality=8)`. Same approach as
  image_animation. Lazy-imports imageio; raises `UnsupportedFormatError`
  when imageio[ffmpeg] isn't installed.
- `encode_webm(frames, fps) -> bytes`: vp9 via `imageio.mimwrite(...,
  codec="vp9", format="webm")`. New for video_generation. Same lazy
  import. If the bundled ffmpeg lacks vp9, falls back to `vp8` and
  logs a warning. If vp8 also fails, raises UnsupportedFormatError.
- `encode_frames_b64(frames) -> list[str]`: per-frame base64 PNG.
  Same as image_animation.

The codec is designed so image_animation and video_generation share
the encode_mp4 implementation conceptually, but each modality keeps
its own copy for clarity. Duplication is a few dozen lines; coupling
the two modalities through a shared module would tangle their evolution.

```python
class UnsupportedFormatError(Exception):
    """Raised when a response_format requires deps that aren't installed."""


def encode_mp4(frames: list[Any], fps: int) -> bytes:
    imageio = _try_import_imageio()
    if imageio is None:
        raise UnsupportedFormatError(
            "mp4 response_format requires imageio[ffmpeg]; "
            "install via `pip install imageio[ffmpeg]` or use frames_b64"
        )
    if not frames:
        raise ValueError("encode_mp4: frames list is empty")
    import numpy as np
    arrays = [np.array(_to_pil(f).convert("RGB")) for f in frames]
    buf = io.BytesIO()
    imageio.mimwrite(buf, arrays, fps=fps, codec="h264", format="mp4", quality=8)
    return buf.getvalue()


def encode_webm(frames: list[Any], fps: int) -> bytes:
    imageio = _try_import_imageio()
    if imageio is None:
        raise UnsupportedFormatError(
            "webm response_format requires imageio[ffmpeg]; "
            "install via `pip install imageio[ffmpeg]` or use frames_b64"
        )
    if not frames:
        raise ValueError("encode_webm: frames list is empty")
    import numpy as np
    arrays = [np.array(_to_pil(f).convert("RGB")) for f in frames]
    buf = io.BytesIO()
    try:
        imageio.mimwrite(buf, arrays, fps=fps, codec="vp9", format="webm")
    except Exception as e:
        logger.warning("vp9 encode failed (%s); falling back to vp8", e)
        buf = io.BytesIO()
        try:
            imageio.mimwrite(buf, arrays, fps=fps, codec="vp8", format="webm")
        except Exception as e2:
            raise UnsupportedFormatError(
                f"webm encode failed (vp9 and vp8): {e2}"
            ) from e2
    return buf.getvalue()


def encode_frames_b64(frames: list[Any]) -> list[str]:
    out: list[str] = []
    for f in frames:
        buf = io.BytesIO()
        _to_pil(f).save(buf, format="PNG")
        out.append(base64.b64encode(buf.getvalue()).decode("ascii"))
    return out
```

`_to_pil(frame)` is a small normalization helper that accepts
PIL.Image, numpy ndarray (HWC uint8), or torch tensor (HWC, normalized
to [0, 255] uint8).

## Runtime: `WanRuntime`

```python
class WanRuntime:
    def __init__(
        self,
        *,
        hf_repo: str,
        local_dir: str | None = None,
        device: str = "auto",
        dtype: str = "float16",
        model_id: str,
        default_duration_seconds: float = 5.0,
        default_fps: int = 5,
        default_size: tuple[int, int] = (832, 480),
        default_steps: int = 30,
        default_guidance: float = 5.0,
        **_: Any,
    ) -> None:
        # lazy-import torch + diffusers
        # Try WanPipeline.from_pretrained(...) if available; else
        # fall back to DiffusionPipeline.from_pretrained(...) which
        # autodetects from model_index.json. The fallback path is the
        # safer default since WanPipeline is recent (diffusers >=0.32).
        # honor device + dtype

    def generate(self, prompt, *, negative_prompt=None,
                 duration_seconds=None, fps=None, width=None, height=None,
                 steps=None, guidance=None, seed=None, **_) -> VideoResult:
        # compute target_frames = round(duration_seconds * fps)
        # align to model-native frame count where required (Wan: 81 frames
        # native; the runtime computes from request and the pipeline
        # silently aligns); record actual_duration = actual_frames / fps
        # call pipeline; extract via out.frames[0]; return VideoResult
```

Mirrors `AnimateDiffRuntime`'s shape: lazy-import sentinels at module
scope, `_ensure_deps()` with short-circuit, `_select_device()` helper,
the rest of the class. The runtime doesn't subclass anything; it
satisfies the Protocol structurally.

## Runtime: `CogVideoXRuntime`

```python
class CogVideoXRuntime:
    def __init__(
        self,
        *,
        hf_repo: str,
        local_dir: str | None = None,
        device: str = "auto",
        dtype: str = "float16",
        model_id: str,
        default_duration_seconds: float = 6.0,
        default_fps: int = 8,
        default_size: tuple[int, int] = (720, 480),
        default_steps: int = 50,
        default_guidance: float = 6.0,
        **_: Any,
    ) -> None:
        # lazy-import torch + diffusers.CogVideoXPipeline
        # CogVideoXPipeline.from_pretrained(...)

    def generate(self, prompt, *, ...) -> VideoResult:
        # CogVideoX-2b expects 49 frames @ 8fps for ~6s clips at 720x480
        # Pipeline call: self._pipe(prompt, num_videos_per_prompt=1,
        #   num_inference_steps=..., num_frames=..., guidance_scale=...,
        #   generator=...).frames[0]
```

Same structure as `WanRuntime` but with CogVideoX-specific defaults
and pipeline class. The two runtimes share enough that a base class
could merge them, but for v0.27.0 they live separately for clarity.
A second pass (consolidation #117) can merge after both are stable.

## Bundled `wan2_1_t2v_1_3b.py`

```python
MANIFEST = {
    "model_id": "wan2-1-t2v-1-3b",
    "modality": "video/generation",
    "hf_repo": "Wan-AI/Wan2.1-T2V-1.3B",
    "description": (
        "Wan 2.1 T2V 1.3B: ~3GB, 5s videos at 832x480, "
        "fits 8GB GPUs, Apache 2.0"
    ),
    "license": "Apache 2.0",
    "pip_extras": (
        "torch>=2.1.0",
        "diffusers>=0.32.0",
        "transformers>=4.36.0",
        "accelerate",
        "Pillow>=9.1.0",
        "imageio[ffmpeg]>=2.31.0",
    ),
    "system_packages": (),
    "capabilities": {
        "device": "cuda",
        "default_duration_seconds": 5.0,
        "default_fps": 5,
        "default_size": (832, 480),
        "min_duration_seconds": 1.0,
        "max_duration_seconds": 10.0,
        "default_steps": 30,
        "default_guidance": 5.0,
        "supports_image_to_video": False,
        "memory_gb": 6.0,
    },
    "allow_patterns": [
        "*.safetensors", "*.json", "*.txt", "*.md",
        "scheduler/*.json",
        "transformer/*.safetensors", "transformer/*.json",
        "vae/*.safetensors", "vae/*.json",
        "text_encoder/*.safetensors", "text_encoder/*.json",
        "tokenizer/*",
    ],
}
```

The `Model` class constructor robustly handles two cases:
1. `WanPipeline` is a top-level export of diffusers (newer versions):
   load via `WanPipeline.from_pretrained(...)` directly.
2. `WanPipeline` is missing: fall back to
   `DiffusionPipeline.from_pretrained(...)` which auto-detects from
   `model_index.json`. This is the safer default and works across
   diffusers versions.

If neither path resolves a pipeline class, the constructor raises a
clear RuntimeError pointing the user at `muse pull wan2-1-t2v-1-3b`
to refresh the per-model venv.

## HF plugin

Sniff: tag `text-to-video` AND repo name contains one of `wan`,
`cogvideox`, `ltx`, `mochi`, `hunyuan` (case-insensitive). Priority
105 (more specific than `image/animation`'s 110 since the architecture
name match is required, and earlier in the discovery sort than the
catch-all priority-200 plugins).

```python
def _sniff(info) -> bool:
    tags = getattr(info, "tags", None) or []
    if "text-to-video" not in tags:
        return False
    repo_id = (getattr(info, "id", "") or "").lower()
    patterns = ("wan", "cogvideox", "ltx-video", "mochi", "hunyuan")
    return any(p in repo_id for p in patterns)
```

Resolve infers per-architecture defaults and dispatches to the
correct runtime path:

| Pattern | Runtime path | Defaults |
|---|---|---|
| `*wan2*` / `*wan-*` | `WanRuntime` | duration=5, fps=5, size=(832,480), steps=30, guidance=5.0, memory_gb=6 |
| `*cogvideox*` | `CogVideoXRuntime` | duration=6, fps=8, size=(720,480), steps=50, guidance=6.0, memory_gb=9 |
| `*ltx-video*` | `WanRuntime` (fallback) | duration=5, fps=30, size=(1216,704), steps=20, guidance=3.0, memory_gb=16 |
| `*mochi*` | `WanRuntime` (fallback) | duration=5, fps=30, size=(848,480), steps=64, guidance=4.5, memory_gb=24 |
| `*hunyuan*` | `WanRuntime` (fallback) | duration=5, fps=24, size=(1280,720), steps=50, guidance=6.0, memory_gb=60 |
| fallback | `WanRuntime` | conservative Wan defaults |

For LTX-Video, Mochi, and Hunyuan, the fallback `WanRuntime` will
fail at load time (the pipeline class won't match the repo's
model_index.json structure) but the manifest is synthesized so the
catalog entry exists. Documented in spec; the user sees a clear
"pipeline class mismatch" RuntimeError rather than a silent failure.
LTXVideoRuntime, MochiRuntime, and HunyuanRuntime are filed for
v1.next.

`_search` filters by `text-to-video` tag.

## Curated entries

```yaml
# ---------- video/generation ----------

- id: wan2-1-t2v-1-3b
  bundled: true

- id: cogvideox-2b
  uri: hf://THUDM/CogVideoX-2b
  modality: video/generation
  size_gb: 6.0
  description: "CogVideoX 2B: ~6GB at fp16, 6s videos at 720x480, fits 12GB, Apache 2.0"
  capabilities:
    memory_gb: 9.0
    default_size: [720, 480]
    default_duration_seconds: 6.0
    default_fps: 8
    default_steps: 50
    default_guidance: 6.0
    device: cuda

- id: ltx-video
  uri: hf://Lightricks/LTX-Video
  modality: video/generation
  size_gb: 13.0
  description: "LTX-Video: ~13GB, fast generation at 30fps, requires 16GB+ GPU"
  capabilities:
    memory_gb: 16.0
    default_size: [1216, 704]
    default_fps: 30
    default_duration_seconds: 5.0
    default_steps: 20
    default_guidance: 3.0
    device: cuda
```

Mochi-1 and HunyuanVideo are intentionally not in `curated.yaml`:
they need 24GB+ VRAM (Mochi) or 60GB+ (Hunyuan), and their runtimes
aren't shipped in v0.27.0. Users can still pull them via
`muse pull hf://genmo/mochi-1-preview` if they accept the trade-offs;
the HF plugin synthesizes a manifest with conservative defaults but
the runtime fails to load (documented).

## PROBE_DEFAULTS

```python
PROBE_DEFAULTS = {
    "shape": "2-second clip at default size, steps=10",
    "call": lambda m: m.generate(
        "a flag waving in the wind",
        duration_seconds=2.0,
        steps=10,
    ),
}
```

(Use a very short duration plus few steps for the probe so the
measurement is bounded; real inference reveals true peak VRAM. On a
12GB card the probe should complete in roughly 30-60 seconds for Wan
1.3B.)

## Behavioral resilience

1. **`WanPipeline` availability across diffusers versions.** Wan
   support landed in diffusers 0.32.0. If the per-model venv has an
   older diffusers, the bundled script's `Model.__init__` falls back
   to `DiffusionPipeline.from_pretrained(...)` which auto-detects from
   `model_index.json`. The pip_extras pin diffusers>=0.32.0 so the
   fallback should rarely trigger; it exists because pip resolution
   sometimes downgrades to satisfy other constraints.

2. **Frame extraction.** Different diffusers pipelines return frames
   in different shapes:
   - Wan: `pipe(...).frames[0]` (list of PIL.Image, similar to AnimateDiff).
   - CogVideoX: `pipe(...).frames[0]` (same shape).
   - The runtime extracts via `out.frames[0]` and converts tensor to
     PIL if needed (helper shared with image_animation).

3. **VRAM accounting** (peak with activations):
   - Wan 2.1 1.3B: ~3GB weights, ~6GB peak.
   - CogVideoX-2b: ~6GB weights, ~9GB peak.
   - LTX-Video: ~13GB weights, ~16GB peak.
   - Mochi-1: ~10GB weights, ~24GB peak.
   - HunyuanVideo: ~13B params at fp16, ~60GB peak.
   The probe will measure real peaks at runtime.

4. **fps and duration interplay.** `num_frames = round(duration_seconds * fps)`,
   then aligned to the model's native frame count if needed. Wan's
   native is 81 frames at 16fps (~5s); the runtime computes target
   from request, then the pipeline silently aligns. Actual returned
   `duration_seconds` reflects `actual_frames / fps`, which may differ
   from the requested duration by up to one frame's worth.

5. **Codec h264 and vp9 support via imageio.** imageio[ffmpeg] bundles
   ffmpeg with both h264 and vp9 codecs. The mp4 path is the same
   well-trodden code from image_animation. The webm path tries vp9
   first, falls back to vp8 with a warning, and raises only if both
   fail. Documented in the codec module.

6. **Probe runtime cost.** The probe will actually run inference if
   the user has Wan pulled. Default probe is 2s @ 10 steps; on a 12GB
   GPU this completes in ~30-60s. The measurement is bounded to that
   range, not the full inference cost. Documented in PROBE_DEFAULTS.

7. **GPU-only.** v0.27.0 video models all declare `device: "cuda"`.
   CPU inference would take 10-30 minutes per clip; not a useful
   default. The runtime's `_select_device("auto")` resolves to "cuda"
   when available, "cpu" when not, but capabilities mark the model
   as cuda-required so the supervisor logs a clear warning when the
   user has CUDA disabled.

8. **LTX/Mochi/Hunyuan runtime fallback.** The HF plugin synthesizes
   a manifest pointing at `WanRuntime` for these architectures. The
   pipeline classes don't match, so the runtime constructor raises a
   RuntimeError at load time. Documented; v1.next adds dedicated
   runtimes per architecture.

9. **`n>1` for video.** Each generated video is heavy (multi-second
   diffusion). The wire contract caps `n <= 2` to avoid pathological
   request shapes. For `n>1` with `frames_b64`, frames from each
   result are appended in order without per-result grouping; this
   matches image_animation's behavior. Documented as "ambiguous;
   future revision may add per-result grouping."

## Modality `__init__.py` exports

```python
from muse.modalities.video_generation.client import VideoGenerationClient
from muse.modalities.video_generation.protocol import (
    VideoGenerationModel, VideoResult,
)
from muse.modalities.video_generation.routes import build_router

MODALITY = "video/generation"

PROBE_DEFAULTS = {
    "shape": "2-second clip at default size, steps=10",
    "call": lambda m: m.generate(
        "a flag waving in the wind",
        duration_seconds=2.0,
        steps=10,
    ),
}

__all__ = [
    "MODALITY",
    "PROBE_DEFAULTS",
    "build_router",
    "VideoGenerationClient",
    "VideoGenerationModel",
    "VideoResult",
]
```

## Distinction from image/animation

| Dimension | image/animation | video/generation |
|---|---|---|
| Modality MIME | `image/animation` | `video/generation` |
| URL | `/v1/images/animations` | `/v1/video/generations` |
| Default duration | 16 frames at 8 fps = 2s | 5 seconds |
| Default loop | true (infinite) | false (single play, no loop field) |
| Default format | `webp` (animated WebP) | `mp4` (h264) |
| Architectures | AnimateDiff (UNet + motion adapter) | Wan / CogVideoX (transformer) |
| VRAM target | 8-12 GB | 8 GB (Wan 1.3B) up to 60+ GB (Hunyuan) |
| Use case | Looping shorts, GIF replacements | Narrative clips, demos |

The two modalities deliberately don't overlap: a request for a 2s
looping animation goes to `image/animation`; a request for a 6s
narrative clip goes to `video/generation`. Users pick the modality
that matches their use case; muse doesn't try to auto-route between
them.

## Migration / risk

- **New modality directory; no impact on existing routes.** Pure addition.
- **HF plugin priority 105.** Sits below image/animation (110) but
  with a more specific filter (architecture name match in addition
  to the text-to-video tag). Both can coexist; muse routes via the
  first plugin whose `_sniff` returns True at the expected priority,
  and architecture name match disambiguates.
- **New per-model venv when users `muse pull wan2-1-t2v-1-3b`.** Pip
  extras include diffusers>=0.32.0, transformers, torch, imageio[ffmpeg].
- **Release-note caveat: video models are GPU-required.** Even Wan
  1.3B at fp16 is tight on 8GB cards; users should plan for 12GB+ for
  comfortable headroom. CogVideoX-2b realistically wants 16GB.
  LTX-Video 24GB+. Documented prominently.
- **Codec sets a precedent.** webm support arrives via imageio[ffmpeg]'s
  vp9 codec; if vp9 isn't bundled with the user's ffmpeg, vp8 fallback
  fires, and if that also fails, the user gets a clear UnsupportedFormatError.

## Test coverage

```
tests/modalities/video_generation/
|-- test_protocol.py      # VideoResult shape, VideoGenerationModel duck-typing
|-- test_codec.py         # encode_mp4 / encode_webm / encode_frames_b64 round-trip
|-- test_routes.py        # POST /v1/video/generations: happy paths + error envelopes
|-- test_client.py        # VideoGenerationClient HTTP shape
|-- test_hf_plugin.py     # HF_PLUGIN keys, sniff (Wan/Cog/LTX/etc), resolve, search
`-- runtimes/
    |-- __init__.py
    |-- test_wan_runtime.py      # patched WanPipeline / DiffusionPipeline
    `-- test_cogvideox_runtime.py # patched CogVideoXPipeline

tests/models/test_wan2_1_t2v_1_3b.py             # bundled-script tests

tests/cli_impl/test_e2e_video_generation.py      # @pytest.mark.slow
tests/integration/test_remote_video_generation.py # opt-in MUSE_REMOTE_SERVER
```

Specifically:

- `test_protocol.py`: VideoResult fields, VideoGenerationModel
  duck-typing.
- `test_codec.py`:
  - `encode_mp4` returns mp4 bytes (matches imageio call signature
    fps=..., codec="h264", format="mp4").
  - `encode_mp4` raises `UnsupportedFormatError` when imageio absent.
  - `encode_webm` calls imageio with codec="vp9", format="webm".
  - `encode_webm` falls back to vp8 on vp9 failure (warns).
  - `encode_webm` raises UnsupportedFormatError when both vp9 and
    vp8 fail.
  - `encode_frames_b64` returns one base64 PNG per frame.
- `test_hf_plugin.py`:
  - `_sniff(Wan-AI/Wan2.1-T2V-1.3B)` returns True (text-to-video + wan).
  - `_sniff(THUDM/CogVideoX-2b)` returns True.
  - `_sniff(Lightricks/LTX-Video)` returns True.
  - `_sniff(genmo/mochi-1-preview)` returns True.
  - `_sniff(tencent/HunyuanVideo)` returns True.
  - `_sniff(stable-diffusion-2-1)` returns False.
  - `_sniff(some-text-to-video-without-known-arch)` returns False.
  - `_resolve(Wan-AI/Wan2.1-T2V-1.3B, ...)` produces WanRuntime path.
  - `_resolve(THUDM/CogVideoX-2b, ...)` produces CogVideoXRuntime path.
  - `_resolve(Lightricks/LTX-Video, ...)` produces WanRuntime fallback path.
  - `_search` filters by text-to-video.
- `test_routes.py`:
  - POST happy path (mp4 by default) returns 200 + envelope.
  - response_format="webm" returns 200 + webm bytes.
  - response_format="frames_b64" returns 200 + list of PNG entries.
  - n=2 returns 2 entries (or n*frames for frames_b64).
  - unknown model returns 404.
  - prompt missing returns 422.
  - duration out-of-range returns 422.
  - response_format="avi" returns 400 or 422.
  - size mis-formatted returns 422.
- `test_client.py`:
  - default base url + MUSE_SERVER override.
  - body shape (prompt + model + format).
  - decodes mp4/webm to bytes; frames_b64 to list[bytes].
  - non-200 raises RuntimeError.
- `runtimes/test_wan_runtime.py`:
  - lazy-import sentinels.
  - constructor with `WanPipeline` patched: calls `from_pretrained`.
  - constructor falls back to `DiffusionPipeline` when WanPipeline
    is None.
  - constructor honors device + dtype.
  - generate returns VideoResult.
  - generate forwards num_frames, num_inference_steps, guidance_scale,
    width, height to pipeline.
  - generate forwards negative_prompt when set.
  - generate omits negative_prompt when None.
  - construction absorbs unknown kwargs.
- `runtimes/test_cogvideox_runtime.py`:
  - mirror of test_wan_runtime.py for CogVideoXPipeline.
- `models/test_wan2_1_t2v_1_3b.py`:
  - manifest required fields.
  - manifest pip_extras has torch + diffusers + imageio.
  - manifest capabilities advertise expected defaults.
  - Model loads via patched WanPipeline.
  - Model falls back to DiffusionPipeline when WanPipeline absent.
  - Model.generate returns VideoResult.
- `cli_impl/test_e2e_video_generation.py`: full TestClient flow with
  a fake backend that satisfies the protocol structurally.
  `@pytest.mark.slow`.
- `integration/test_remote_video_generation.py`: protocol assertion
  on a real server, optional via MUSE_REMOTE_SERVER. Skips if no
  video model loaded.

## Open questions

None blocking. Three follow-ups deferred:
- Whether `n=2` should be allowed at all given the VRAM cost. v0.27.0
  keeps it but caps lower than image/animation.
- Whether per-result grouping is needed for `frames_b64` with `n>1`.
  Future revision may add `result_index` to per-frame entries.
- Whether `guidance_scale` should be split into negative + positive
  for finer control. Wan and CogVideoX accept a scalar; left as-is.

## Out of scope (filed for later)

- LTXVideoRuntime, MochiRuntime, HunyuanRuntime: per-architecture
  runtimes. Track in v1.next.
- Image-to-video (I2V) support: WanImageToVideoPipeline / CogVideoX-1.5
  I2V. Track in v1.next.
- Audio sync (talking-head video, sound effects matched to motion).
  Track in v1.next.
- Frame-by-frame streaming. Track in v1.next.
- ControlNet conditioning for video (depth, pose, optical flow).
  Track in v1.next.
- Video upscale / interpolation (RIFE, FILM). Separate modality.

## Estimated build size

Largest modality muse has shipped. 8 tasks (protocol+codec, two
runtimes, routes+init, client, bundled script, plugin+curated, docs+release).
Mirrors the image_animation + image_segmentation cadence but with two
runtimes. Target: v0.27.0.
