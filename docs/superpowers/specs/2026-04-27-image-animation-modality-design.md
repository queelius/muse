# `image/animation` modality (AnimateDiff and friends)

**Date:** 2026-04-27
**Driver:** ship a modality dedicated to coherent frame sequences. Today's options on muse are text-to-image (single frame) and chained img2img (which the v0.17.x experiments confirmed cannot do real temporal evolution: low strength preserves composition but ignores prompt; high strength evolves prompt but breaks composition; AnimateDiff's temporal layers learn time-as-a-dimension which is the missing capability).

## Goal

1. New modality MIME tag `image/animation`.
2. New route `POST /v1/images/animations`.
3. New protocol (`AnimationModel`, `AnimationResult`).
4. New runtime + bundled script for AnimateDiff (SD 1.5 + motion adapter v1-5-3, the well-trodden first config).
5. New HF resolver plugin sniffing fused-checkpoint AnimateDiff repos (AnimateLCM, etc.).
6. Default response is animated WebP. Alternates: GIF, MP4 (requires extra dep), frames_b64.

## Non-goals

- **Stable Video Diffusion (SVD).** img2vid only, single-component. Different shape, future modality work; could land as a second bundled script when the contract is stable.
- **Composite URIs** like `hf://base+adapter`. AnimateDiff configurations require base + motion-adapter pairs. v1 handles only fused checkpoints (single repo); arbitrary pairs come later.
- **Streaming frame-by-frame.** Confirmed: route waits for all frames.
- **Audio sync.** No audio in v1. Animations are silent.
- **`/v1/video/generations`.** Real video generation (CogVideoX, LTX-Video, Mochi: longer clips with motion priors, narrative continuity, audio) is task #107. Animation is the shorter, looping sibling.
- **Modifying the existing `/v1/images/generations` route.** Animation is a distinct modality; routes don't overlap.

## Constraints

1. **MIME-tag discipline.** `image/animation` is the modality tag. Singular file output (one WebP/GIF/MP4 per request, or one list of frames if requested) keeps it close to `image/generation`'s one-image-per-request contract.
2. **OpenAI compat: not directly applicable.** OpenAI has no animation API. We get to define the wire shape but should mirror the `/v1/images/generations` envelope shape (`{"data": [{"b64_json": "..."}]}`) so OpenAI SDK clients can use it via custom HTTP wrappers.
3. **Plugin discipline.** HF plugin must follow the established pattern: stdlib + huggingface_hub + muse.core only, no relative imports, no heavy deps at module top.
4. **Capability flags per model**, mirroring the v0.17 img2img pattern.

## Wire contract

Route: `POST /v1/images/animations`

Request body (`AnimationsRequest`, pydantic):

| Field | Type | Default | Validation |
|---|---|---|---|
| `prompt` | `str` | required | `1 <= len <= 4000` |
| `model` | `str | None` | `None` | falls back to modality default |
| `n` | `int` | `1` | `1 <= n <= 4` (lower than image_gen because each is heavier) |
| `frames` | `int | None` | from manifest | `4 <= frames <= 64` when set |
| `fps` | `int | None` | from manifest | `1 <= fps <= 30` when set |
| `loop` | `bool` | `True` | infinite loop in webp/gif; ignored by mp4 |
| `negative_prompt` | `str | None` | `None` | optional |
| `steps` | `int | None` | from manifest | `1 <= steps <= 100` when set |
| `guidance` | `float | None` | from manifest | `0 <= guidance <= 20` when set |
| `seed` | `int | None` | `None` | for reproducibility |
| `image` | `str | None` | `None` | data URL or `https?://...`, used by img2vid-capable models |
| `strength` | `float | None` | `None` | `0 <= strength <= 1`, img2vid only |
| `response_format` | `str` | `"webp"` | one of `webp`, `gif`, `mp4`, `frames_b64` |
| `size` | `str` | from manifest | `^\d+x\d+$`, common values `512x512`, `768x768`, `1024x1024` |

Response shape (mirrors image_generation envelope):

```json
{
  "data": [
    {"b64_json": "..."}
  ],
  "model": "animatediff-motion-v3",
  "metadata": {
    "frames": 16,
    "fps": 8,
    "duration_seconds": 2.0,
    "format": "webp",
    "size": [512, 512]
  }
}
```

For `response_format: "frames_b64"`:

```json
{
  "data": [
    {"b64_json": "<png frame 1>"},
    {"b64_json": "<png frame 2>"},
    ...
  ],
  "model": "...",
  "metadata": {"frames": 16, "fps": 8, ...}
}
```

For other formats, `data` is a single-element list with the binary asset base64-encoded.

## Capability flags

Per-model capabilities declared in MANIFEST or curated overlay:

| Capability | Type | Purpose |
|---|---|---|
| `supports_text_to_animation` | `bool` | model can take text only (txt2vid) |
| `supports_image_to_animation` | `bool` | model can take an image input (img2vid) |
| `default_frames` | `int` | manifest-default frame count |
| `default_fps` | `int` | manifest-default frame rate |
| `min_frames` | `int` | minimum for this model (clip protocol-side) |
| `max_frames` | `int` | maximum for this model (often hard model limit) |
| `default_size` | `[int, int]` | resolution default |
| `default_steps` | `int` | denoise steps default |
| `default_guidance` | `float` | guidance scale default |

For AnimateDiff (SD 1.5 motion-v3) defaults: `frames=16, fps=8, size=[512,512], steps=25, guidance=7.5, supports_text_to_animation=True, supports_image_to_animation=False, min_frames=8, max_frames=24`.

Capability gate: a request with `image` set against a model whose `supports_image_to_animation: False` returns 400 with `code=invalid_parameter`, `message="model X does not support image-to-animation; use a model with supports_image_to_animation=True"`.

## Modality structure

```
src/muse/modalities/image_animation/
├── __init__.py             # MODALITY="image/animation", build_router, exports
├── protocol.py             # AnimationModel Protocol, AnimationResult dataclass
├── routes.py               # POST /v1/images/animations handler
├── codec.py                # encode_animation: frames -> webp/gif/mp4/list bytes
├── client.py               # AnimationsClient (HTTP)
├── hf.py                   # HF_PLUGIN for fused AnimateDiff checkpoints
└── runtimes/
    ├── __init__.py
    └── animatediff.py      # AnimateDiffRuntime (generic; reads manifest caps)
```

Bundled scripts:

```
src/muse/models/
└── animatediff_motion_v3.py   # SD 1.5 base + motion-adapter-v1-5-3 (curated config)
```

`AnimationResult` dataclass:

```python
@dataclass
class AnimationResult:
    frames: list[Any]      # list of PIL.Image (one per frame)
    fps: int
    width: int
    height: int
    seed: int
    metadata: dict
```

`AnimationModel` Protocol:

```python
@runtime_checkable
class AnimationModel(Protocol):
    @property
    def model_id(self) -> str: ...

    def generate(
        self,
        prompt: str,
        *,
        negative_prompt: str | None = None,
        frames: int | None = None,
        fps: int | None = None,
        width: int | None = None,
        height: int | None = None,
        steps: int | None = None,
        guidance: float | None = None,
        seed: int | None = None,
        init_image: Any = None,        # img2vid path
        strength: float | None = None,
        **kwargs,
    ) -> AnimationResult: ...
```

## Codec

`muse.modalities.image_animation.codec`:

- `encode_webp(frames, fps, *, loop=True, lossless=False) -> bytes`: uses `Pillow.save(format='WEBP', save_all=True, append_images=..., duration=..., loop=...)`. Pillow 9.1+ has solid animated WebP support.
- `encode_gif(frames, fps, *, loop=True) -> bytes`: Pillow GIF, much larger files but universally playable.
- `encode_mp4(frames, fps) -> bytes`: uses `imageio[ffmpeg]` (lazy import; new optional dep declared in `muse[images]`). Returns h264-encoded MP4. If `imageio` unavailable, `mp4` response_format returns 400 with a clear message.
- `encode_frames_b64(frames) -> list[str]`: base64-encoded PNG per frame.

Duration math: `duration_ms_per_frame = 1000 / fps`. Animated WebP/GIF use this; MP4 carries fps directly.

## Bundled script: animatediff_motion_v3

`src/muse/models/animatediff_motion_v3.py`:

```python
MANIFEST = {
    "model_id": "animatediff-motion-v3",
    "modality": "image/animation",
    "hf_repo": "guoyww/animatediff-motion-adapter-v1-5-3",  # the motion adapter
    "description": "AnimateDiff motion v3 + SD 1.5 base, 16 frames @ 8fps, 512x512",
    "license": "Apache 2.0",
    "pip_extras": (
        "torch>=2.1.0",
        "diffusers>=0.27.0",
        "transformers>=4.36.0",
        "accelerate",
        "Pillow>=9.1.0",         # animated WebP support
        "safetensors",
    ),
    "capabilities": {
        "supports_text_to_animation": True,
        "supports_image_to_animation": False,  # base AnimateDiff is txt2vid
        "default_frames": 16,
        "default_fps": 8,
        "min_frames": 8,
        "max_frames": 24,
        "default_size": [512, 512],
        "default_steps": 25,
        "default_guidance": 7.5,
        "device": "cuda",  # AnimateDiff is heavy; prefer GPU. Errors clearly on CPU-only host.
    },
}
```

The `Model` class in this script wraps `AnimateDiffPipeline` from diffusers, attaching `MotionAdapter.from_pretrained(...)` to a base SD 1.5 checkpoint (which it pulls separately via `snapshot_download` at first construction). This is the only bundled script that has a 2-component download.

## HF plugin: fused-checkpoint repos

`src/muse/modalities/image_animation/hf.py`:

Sniffs HF repos that are *single-component* AnimateDiff variants. The known patterns:
- `wangfuyun/AnimateLCM`: AnimateDiff distilled to fewer steps, fused checkpoint.
- `wangfuyun/AnimateLCM-SVD-xt`: SVD-based, different shape.
- Some community-fused checkpoints with motion adapter pre-merged into the UNet.

Sniff heuristic: `model_index.json` sibling AND tag includes `text-to-video` AND filename contains "animate" or "motion" (case-insensitive). Priority 110 (more specific than text/classification's tag-only 200, less specific than file-pattern-only 100 of GGUF/CT2/diffusers-text2image).

Defaults inferred per-pattern:
- `*animatelcm*`: 4 steps (LCM-distilled), guidance=1.0
- `*animatediff*`: 25 steps, guidance=7.5
- fallback: 25 steps, guidance=7.5

Resolver-pulled models advertise `supports_text_to_animation: True`. `supports_image_to_animation: False` unless the repo name suggests otherwise (e.g., `*svd*` or `*img2vid*`).

This plugin handles the simple case well enough for v1. Composite URIs (base + adapter pairs) are out of scope.

## Curated entries

```yaml
- id: animatediff-motion-v3
  bundled: true   # the bundled script handles the 2-component download

- id: animatelcm
  uri: hf://wangfuyun/AnimateLCM
  modality: image/animation
  size_gb: 5.0
  description: "AnimateLCM: distilled AnimateDiff, 4 steps, ~3-5x faster than v3"
```

## Tests

Mirror the established modality test structure:

```
tests/modalities/image_animation/
├── test_protocol.py        # AnimationResult shape; AnimationModel structural compliance
├── test_codec.py           # encode_webp / gif / mp4 / frames_b64 round-trip
├── test_routes.py          # /v1/images/animations: happy paths, error envelopes
├── test_client.py          # AnimationsClient HTTP shape
├── test_hf_plugin.py       # HF_PLUGIN keys, sniff, resolve, search
└── runtimes/
    └── test_animatediff.py # generic runtime, mocked diffusers
tests/models/
└── test_animatediff_motion_v3.py  # bundled script (2-component construction mocked)
```

## Migration / risk

- **Pure addition.** No existing modality changes. No existing route changes.
- **AnimateDiff is heavy.** SD 1.5 base + motion adapter at fp16 is ~5-7GB VRAM during inference. 12GB cards: comfortable. 8GB cards: tight, may need offload.
- **First request cold-start is long** (~30s) because diffusers needs to load both base and adapter, fuse them, move to GPU. Subsequent requests are warm.
- **Animated WebP requires Pillow >= 9.1.** Most modern installs have this; pip_extras pins it explicitly.
- **MP4 requires imageio[ffmpeg].** Optional. If missing, `response_format: mp4` returns a clean 400.

## Out of scope (filed for later)

- **SVD (img2vid only).** Could land as `models/svd_xt.py` after this modality stabilizes. Same modality tag, different capability set.
- **CogVideoX, LTX-Video, Mochi.** Real video generation with longer clips. Task #107 (`video/generation` modality).
- **AnimateDiff with arbitrary base + adapter pairs.** Requires composite URI resolution; future plugin work.
- **ControlNet conditioning** (e.g., depth-guided or pose-guided animation).
- **Audio synthesis sync** for animations with talking characters etc.
- **Frame-by-frame streaming** (server pushes frames as generated).

## Estimated build size

Comparable to audio/transcription (#95) which took ~10 commits. Single bundled script + plugin + 4 modality files + tests + 1 release. Target: v0.18.0.
