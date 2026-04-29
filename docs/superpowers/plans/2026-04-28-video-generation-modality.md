# Implementation plan: `video/generation` modality (v0.27.0)

**Date:** 2026-04-28
**Spec:** `docs/superpowers/specs/2026-04-28-video-generation-modality-design.md`
**Closes:** task #107

## Tasks (one commit each)

- **A** Protocol + codec (mp4 + webm + frames_b64)
- **B** WanRuntime (diffusers WanPipeline / DiffusionPipeline fallback)
- **C** CogVideoXRuntime (diffusers CogVideoXPipeline)
- **D** Routes + modality `__init__.py` (capability gating + n-loop)
- **E** VideoGenerationClient
- **F** Bundled `wan2_1_t2v_1_3b.py`
- **G** HF plugin + curated entries (3)
- **H** Documentation + slow e2e + integration + v0.27.0 release

Each task ends with `pytest tests/ -q -m "not slow"`. Push at H only.

## Task A: protocol + codec

Files:
- `src/muse/modalities/video_generation/__init__.py` (skeleton; finalized in D)
- `src/muse/modalities/video_generation/protocol.py`
- `src/muse/modalities/video_generation/codec.py`
- `tests/modalities/video_generation/__init__.py`
- `tests/modalities/video_generation/test_protocol.py`
- `tests/modalities/video_generation/test_codec.py`

`protocol.py`:

```python
"""Modality protocol for video/generation.

VideoResult holds a list of PIL.Image frames + timing metadata. The
codec layer transforms the list into mp4/webm/frames_b64 bytes for
the HTTP response.

video/generation is muse's narrative-clip sibling to image/animation:
longer durations, single play, mp4 default, transformer-based
backbones (Wan, CogVideoX) instead of UNet+motion-adapter pairs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class VideoResult:
    frames: list[Any]
    fps: int
    width: int
    height: int
    duration_seconds: float
    seed: int
    metadata: dict = field(default_factory=dict)


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

`codec.py`:

```python
"""Encoding helpers for video/generation responses.

Pure functions: list[PIL.Image] + timing -> bytes (or list[base64 PNG]).

mp4 uses h264 via imageio[ffmpeg]; webm uses vp9 via imageio (with vp8
fallback if vp9 isn't bundled in the user's ffmpeg). frames_b64 is
per-frame base64 PNG.

The mp4 path is conceptually shared with image_animation but kept
independent here for clarity. Each modality owns its codec; coupling
through a shared module would tangle their evolution.
"""
from __future__ import annotations

import base64
import io
import logging
from typing import Any


logger = logging.getLogger(__name__)


class UnsupportedFormatError(Exception):
    """Raised when a response_format requires deps that aren't installed."""


def _try_import_imageio():
    try:
        import imageio
        return imageio
    except ImportError:
        return None


def _to_pil(frame: Any):
    """Normalize frame to PIL.Image. Accepts PIL, numpy ndarray, torch tensor."""
    from PIL import Image
    if isinstance(frame, Image.Image):
        return frame
    import numpy as np
    arr = np.asarray(frame)
    if arr.ndim == 2:
        return Image.fromarray(arr.astype(np.uint8), mode="L")
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def encode_mp4(frames: list[Any], fps: int) -> bytes:
    """Encode frames as h264 MP4 via imageio. Raises UnsupportedFormatError
    when imageio[ffmpeg] is not installed."""
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
    imageio.mimwrite(
        buf, arrays, fps=fps, codec="h264", format="mp4", quality=8,
    )
    return buf.getvalue()


def encode_webm(frames: list[Any], fps: int) -> bytes:
    """Encode frames as vp9/vp8 WebM via imageio.

    Tries vp9 first (better quality at the same bitrate). If vp9 isn't
    bundled in the user's ffmpeg build, falls back to vp8 with a
    warning. Raises UnsupportedFormatError if both fail.
    """
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
        return buf.getvalue()
    except Exception as e:
        logger.warning("vp9 encode failed (%s); falling back to vp8", e)
    buf = io.BytesIO()
    try:
        imageio.mimwrite(buf, arrays, fps=fps, codec="vp8", format="webm")
        return buf.getvalue()
    except Exception as e2:
        raise UnsupportedFormatError(
            f"webm encode failed (vp9 and vp8 both errored): {e2}"
        ) from e2


def encode_frames_b64(frames: list[Any]) -> list[str]:
    """Each frame as a standalone base64-encoded PNG."""
    out: list[str] = []
    for f in frames:
        buf = io.BytesIO()
        _to_pil(f).save(buf, format="PNG")
        out.append(base64.b64encode(buf.getvalue()).decode("ascii"))
    return out
```

Tests:

- `test_protocol.py`: VideoResult shape (default metadata), Protocol
  duck-typing (a fake backend with `model_id` + `generate` satisfies
  it). No imports of heavy deps.
- `test_codec.py`:
  - `encode_mp4` returns bytes via fake imageio (capture mimwrite kwargs:
    fps, codec="h264", format="mp4").
  - `encode_mp4` raises UnsupportedFormatError when imageio absent
    (patch `_try_import_imageio` to return None).
  - `encode_webm` calls fake imageio with codec="vp9", format="webm".
  - `encode_webm` falls back to vp8 when vp9 raises (patch fake
    mimwrite to raise on first call, succeed on second).
  - `encode_webm` raises UnsupportedFormatError when both calls fail.
  - `encode_frames_b64` returns one base64 PNG per frame; PNG header
    check.

Skeleton `__init__.py`:

```python
"""video/generation modality (placeholder).

Final exports added in Task D once routes + client land.
"""
from muse.modalities.video_generation.protocol import (
    VideoGenerationModel, VideoResult,
)

MODALITY = "video/generation"

__all__ = [
    "MODALITY",
    "VideoGenerationModel",
    "VideoResult",
]
```

Commit: `feat(video-gen): protocol + codec (mp4/webm/frames_b64)`.

## Task B: WanRuntime

Files:
- `src/muse/modalities/video_generation/runtimes/__init__.py`
- `src/muse/modalities/video_generation/runtimes/wan_runtime.py`
- `tests/modalities/video_generation/runtimes/__init__.py`
- `tests/modalities/video_generation/runtimes/test_wan_runtime.py`

The runtime mirrors `AnimateDiffRuntime`: lazy-import sentinels at
module scope, `_ensure_deps()` with short-circuit, `_select_device()`
helper, the class with structural protocol satisfaction.

```python
"""Generic Wan runtime via diffusers.WanPipeline (or DiffusionPipeline
fallback).

Wan was added to diffusers in 0.32.0 as a top-level export. Older
diffusers versions can still load the model via DiffusionPipeline
since model_index.json points at the right pipeline class. The
runtime tries WanPipeline first, falls back to DiffusionPipeline when
WanPipeline is None.

Lazy-import sentinel pattern matches sd_turbo, animatediff, and the
other generic runtimes.
"""
from __future__ import annotations

import logging
from typing import Any

from muse.modalities.video_generation.protocol import VideoResult


logger = logging.getLogger(__name__)


# Sentinels patched by tests; populated by _ensure_deps at runtime.
torch: Any = None
WanPipeline: Any = None
DiffusionPipeline: Any = None


def _ensure_deps() -> None:
    global torch, WanPipeline, DiffusionPipeline
    if torch is None:
        try:
            import torch as _t
            torch = _t
        except Exception as e:  # noqa: BLE001
            logger.debug("wan: torch unavailable: %s", e)
    if WanPipeline is None:
        try:
            from diffusers import WanPipeline as _p
            WanPipeline = _p
        except Exception as e:  # noqa: BLE001
            logger.debug("wan: WanPipeline not in diffusers (%s); "
                         "will fall back to DiffusionPipeline", e)
    if DiffusionPipeline is None:
        try:
            from diffusers import DiffusionPipeline as _d
            DiffusionPipeline = _d
        except Exception as e:  # noqa: BLE001
            logger.debug("wan: DiffusionPipeline unavailable: %s", e)


def _select_device(device: str) -> str:
    if device != "auto":
        return device
    if torch is None:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class WanRuntime:
    """Wan runtime over diffusers.WanPipeline / DiffusionPipeline.

    Construction kwargs:
      - hf_repo: model repo id (manifest's hf_repo)
      - local_dir: local cache for weights (from `muse pull`)
      - device, dtype, model_id: standard
      - default_duration_seconds, default_fps, default_size, default_steps,
        default_guidance: manifest-driven defaults injected via capabilities splat
      - **kwargs: absorbed (future capability flags)
    """

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
        _ensure_deps()
        pipeline_cls = WanPipeline or DiffusionPipeline
        if pipeline_cls is None:
            raise RuntimeError(
                "diffusers is not installed; ensure muse[images] extras are "
                "installed in the per-model venv via `muse pull <model-id>`"
            )
        self.model_id = model_id
        self._default_duration = default_duration_seconds
        self._default_fps = default_fps
        self._default_size = tuple(default_size)
        self._default_steps = default_steps
        self._default_guidance = default_guidance
        self._device = _select_device(device)

        import muse.modalities.video_generation.runtimes.wan_runtime as _mod
        _torch = _mod.torch
        torch_dtype = None
        if _torch is not None:
            torch_dtype = {
                "float16": _torch.float16,
                "float32": _torch.float32,
                "bfloat16": _torch.bfloat16,
            }[dtype]

        src = local_dir or hf_repo
        cls_name = getattr(pipeline_cls, "__name__", "Pipeline")
        logger.info(
            "loading %s from %s (model_id=%s, device=%s, dtype=%s)",
            cls_name, src, model_id, self._device, dtype,
        )
        self._pipe = pipeline_cls.from_pretrained(src, torch_dtype=torch_dtype)
        if self._device != "cpu":
            self._pipe = self._pipe.to(self._device)

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
        **_: Any,
    ) -> VideoResult:
        dur = duration_seconds if duration_seconds is not None else self._default_duration
        out_fps = fps if fps is not None else self._default_fps
        w = width or self._default_size[0]
        h = height or self._default_size[1]
        n_steps = steps if steps is not None else self._default_steps
        cfg = guidance if guidance is not None else self._default_guidance
        n_frames = max(1, round(dur * out_fps))

        gen = None
        if seed is not None:
            import muse.modalities.video_generation.runtimes.wan_runtime as _mod
            _torch = _mod.torch
            if _torch is not None:
                gen = _torch.Generator(device=self._device).manual_seed(seed)

        call_kwargs: dict = {
            "prompt": prompt,
            "num_frames": n_frames,
            "num_inference_steps": n_steps,
            "guidance_scale": cfg,
            "width": w,
            "height": h,
        }
        if negative_prompt is not None:
            call_kwargs["negative_prompt"] = negative_prompt
        if gen is not None:
            call_kwargs["generator"] = gen

        out = self._pipe(**call_kwargs)
        # Wan's pipeline returns out.frames as list-of-lists (one video,
        # multiple frames). Take the first video.
        frames_list = out.frames[0]
        first = frames_list[0]
        actual_frames = len(frames_list)
        actual_duration = round(actual_frames / max(out_fps, 1), 3)
        first_w = getattr(first, "size", (w, h))[0]
        first_h = getattr(first, "size", (w, h))[1]
        return VideoResult(
            frames=list(frames_list),
            fps=out_fps,
            width=int(first_w),
            height=int(first_h),
            duration_seconds=actual_duration,
            seed=seed if seed is not None else -1,
            metadata={
                "prompt": prompt,
                "frames": actual_frames,
                "fps": out_fps,
                "steps": n_steps,
                "guidance": cfg,
                "model": self.model_id,
            },
        )
```

Tests mock `WanPipeline` (or `DiffusionPipeline`) and assert:

- constructor calls `from_pretrained` on WanPipeline when present.
- constructor falls back to DiffusionPipeline when WanPipeline is None.
- constructor honors device + dtype.
- `generate` returns VideoResult with frames + fps + width + height +
  duration_seconds.
- `generate` forwards num_frames, num_inference_steps, guidance_scale,
  width, height to pipeline.
- `generate` forwards negative_prompt when set; omits when None.
- duration_seconds * fps -> num_frames math works.
- construction absorbs unknown kwargs.

Commit: `feat(video-gen): WanRuntime (WanPipeline/DiffusionPipeline fallback)`.

## Task C: CogVideoXRuntime

Files:
- `src/muse/modalities/video_generation/runtimes/cogvideox_runtime.py`
- `tests/modalities/video_generation/runtimes/test_cogvideox_runtime.py`

Mirror of `WanRuntime` with CogVideoX-specific defaults and pipeline
class:

```python
"""Generic CogVideoX runtime via diffusers.CogVideoXPipeline.

CogVideoX-2b / 5b are THUDM's transformer-based T2V models. Pipeline
defaults: 49 frames at 8fps for ~6s clips at 720x480.

Lazy-import sentinel pattern matches WanRuntime.
"""
from __future__ import annotations

import logging
from typing import Any

from muse.modalities.video_generation.protocol import VideoResult


logger = logging.getLogger(__name__)


torch: Any = None
CogVideoXPipeline: Any = None


def _ensure_deps() -> None:
    global torch, CogVideoXPipeline
    if torch is None:
        try:
            import torch as _t
            torch = _t
        except Exception as e:  # noqa: BLE001
            logger.debug("cogvideox: torch unavailable: %s", e)
    if CogVideoXPipeline is None:
        try:
            from diffusers import CogVideoXPipeline as _p
            CogVideoXPipeline = _p
        except Exception as e:  # noqa: BLE001
            logger.debug("cogvideox: CogVideoXPipeline unavailable: %s", e)


def _select_device(device: str) -> str:
    if device != "auto":
        return device
    if torch is None:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class CogVideoXRuntime:
    """CogVideoX runtime over diffusers.CogVideoXPipeline."""

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
        _ensure_deps()
        if CogVideoXPipeline is None:
            raise RuntimeError(
                "diffusers CogVideoXPipeline is not installed; ensure "
                "muse[images] extras are installed in the per-model venv"
            )
        self.model_id = model_id
        self._default_duration = default_duration_seconds
        self._default_fps = default_fps
        self._default_size = tuple(default_size)
        self._default_steps = default_steps
        self._default_guidance = default_guidance
        self._device = _select_device(device)

        import muse.modalities.video_generation.runtimes.cogvideox_runtime as _mod
        _torch = _mod.torch
        torch_dtype = None
        if _torch is not None:
            torch_dtype = {
                "float16": _torch.float16,
                "float32": _torch.float32,
                "bfloat16": _torch.bfloat16,
            }[dtype]

        src = local_dir or hf_repo
        logger.info(
            "loading CogVideoXPipeline from %s (model_id=%s, device=%s, dtype=%s)",
            src, model_id, self._device, dtype,
        )
        self._pipe = CogVideoXPipeline.from_pretrained(src, torch_dtype=torch_dtype)
        if self._device != "cpu":
            self._pipe = self._pipe.to(self._device)

    def generate(self, prompt, *, negative_prompt=None, duration_seconds=None,
                 fps=None, width=None, height=None, steps=None, guidance=None,
                 seed=None, **_) -> VideoResult:
        # Body identical to WanRuntime.generate but with CogVideoX defaults.
        # The two runtimes can be merged into a base class in v1.next.
        dur = duration_seconds if duration_seconds is not None else self._default_duration
        out_fps = fps if fps is not None else self._default_fps
        w = width or self._default_size[0]
        h = height or self._default_size[1]
        n_steps = steps if steps is not None else self._default_steps
        cfg = guidance if guidance is not None else self._default_guidance
        n_frames = max(1, round(dur * out_fps))

        gen = None
        if seed is not None:
            import muse.modalities.video_generation.runtimes.cogvideox_runtime as _mod
            _torch = _mod.torch
            if _torch is not None:
                gen = _torch.Generator(device=self._device).manual_seed(seed)

        call_kwargs: dict = {
            "prompt": prompt,
            "num_frames": n_frames,
            "num_inference_steps": n_steps,
            "guidance_scale": cfg,
            "width": w,
            "height": h,
        }
        if negative_prompt is not None:
            call_kwargs["negative_prompt"] = negative_prompt
        if gen is not None:
            call_kwargs["generator"] = gen

        out = self._pipe(**call_kwargs)
        frames_list = out.frames[0]
        first = frames_list[0]
        actual_frames = len(frames_list)
        actual_duration = round(actual_frames / max(out_fps, 1), 3)
        first_w = getattr(first, "size", (w, h))[0]
        first_h = getattr(first, "size", (w, h))[1]
        return VideoResult(
            frames=list(frames_list),
            fps=out_fps,
            width=int(first_w),
            height=int(first_h),
            duration_seconds=actual_duration,
            seed=seed if seed is not None else -1,
            metadata={
                "prompt": prompt,
                "frames": actual_frames,
                "fps": out_fps,
                "steps": n_steps,
                "guidance": cfg,
                "model": self.model_id,
            },
        )
```

Tests mirror WanRuntime tests (mocked CogVideoXPipeline, structural
assertions on call kwargs).

Commit: `feat(video-gen): CogVideoXRuntime (CogVideoXPipeline)`.

## Task D: routes + modality `__init__.py`

Files:
- `src/muse/modalities/video_generation/routes.py`
- `src/muse/modalities/video_generation/__init__.py` (update)
- `tests/modalities/video_generation/test_routes.py`

`routes.py`:

```python
"""POST /v1/video/generations.

Wire contract documented in
docs/superpowers/specs/2026-04-28-video-generation-modality-design.md.
"""
from __future__ import annotations

import asyncio
import base64
import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from muse.core.errors import ModelNotFoundError, error_response
from muse.core.registry import ModalityRegistry
from muse.modalities.video_generation.codec import (
    encode_mp4, encode_webm, encode_frames_b64, UnsupportedFormatError,
)


MODALITY = "video/generation"

logger = logging.getLogger(__name__)


class VideoGenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    model: str | None = None
    duration_seconds: float | None = Field(default=None, ge=0.5, le=30.0)
    fps: int | None = Field(default=None, ge=1, le=60)
    size: str | None = Field(default=None, pattern=r"^\d+x\d+$")
    seed: int | None = None
    negative_prompt: str | None = None
    steps: int | None = Field(default=None, ge=1, le=200)
    guidance: float | None = Field(default=None, ge=0.0, le=20.0)
    response_format: str = Field(
        default="mp4", pattern="^(mp4|webm|frames_b64)$",
    )
    n: int = Field(default=1, ge=1, le=2)


def build_router(registry: ModalityRegistry) -> APIRouter:
    router = APIRouter(prefix="/v1/video", tags=["video/generation"])

    @router.post("/generations")
    async def generations(req: VideoGenerationRequest):
        try:
            model = registry.get(MODALITY, req.model)
        except KeyError:
            raise ModelNotFoundError(
                model_id=req.model or "<default>", modality=MODALITY,
            )

        width = height = None
        if req.size is not None:
            width, height = map(int, req.size.split("x"))

        def _call_one(seed_offset: int):
            kwargs = {
                "negative_prompt": req.negative_prompt,
                "duration_seconds": req.duration_seconds,
                "fps": req.fps,
                "width": width,
                "height": height,
                "steps": req.steps,
                "guidance": req.guidance,
            }
            if req.seed is not None:
                kwargs["seed"] = req.seed + seed_offset
            return model.generate(req.prompt, **kwargs)

        results = []
        for i in range(req.n):
            r = await asyncio.to_thread(_call_one, i)
            results.append(r)

        data = []
        for r in results:
            try:
                encoded = _encode(req.response_format, r)
            except UnsupportedFormatError as e:
                return error_response(400, "invalid_parameter", str(e))
            if req.response_format == "frames_b64":
                for s in encoded:
                    data.append({"b64_json": s})
            else:
                data.append({
                    "b64_json": base64.b64encode(encoded).decode("ascii"),
                })

        head = results[0]
        body = {
            "data": data,
            "model": model.model_id,
            "metadata": {
                "frames": len(head.frames),
                "fps": head.fps,
                "duration_seconds": head.duration_seconds,
                "format": req.response_format,
                "size": [head.width, head.height],
            },
        }
        return JSONResponse(content=body)

    return router


def _encode(fmt, result):
    if fmt == "mp4":
        return encode_mp4(result.frames, result.fps)
    if fmt == "webm":
        return encode_webm(result.frames, result.fps)
    if fmt == "frames_b64":
        return encode_frames_b64(result.frames)
    raise ValueError(f"unknown format: {fmt}")
```

Update `__init__.py`:

```python
"""video/generation modality.

Public surface:
  - MODALITY: str (MIME-shaped tag; used by discover_modalities)
  - build_router(registry) -> APIRouter (mounted by the worker)
  - VideoGenerationModel Protocol, VideoResult dataclass
  - VideoGenerationClient (HTTP)

Wire contract: POST /v1/video/generations
"""
from muse.modalities.video_generation.client import VideoGenerationClient
from muse.modalities.video_generation.protocol import (
    VideoGenerationModel,
    VideoResult,
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

Tests for routes: happy path mp4, webm, frames_b64, n=2, unknown
model 404, missing prompt 422, duration out-of-range, malformed size.

Commit: `feat(video-gen): POST /v1/video/generations route + modality wiring`.

## Task E: VideoGenerationClient

Files:
- `src/muse/modalities/video_generation/client.py`
- `tests/modalities/video_generation/test_client.py`

```python
"""HTTP client for /v1/video/generations.

By default returns the encoded video bytes (mp4/webm) for the
configured response_format. For response_format='frames_b64', returns
list[bytes] (one PNG per frame).
"""
from __future__ import annotations

import base64
import os
from typing import Any

import requests


class VideoGenerationClient:
    def __init__(self, server_url: str | None = None, timeout: float = 600.0) -> None:
        server_url = server_url or os.environ.get(
            "MUSE_SERVER", "http://localhost:8000",
        )
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        duration_seconds: float | None = None,
        fps: int | None = None,
        size: str | None = None,
        seed: int | None = None,
        negative_prompt: str | None = None,
        steps: int | None = None,
        guidance: float | None = None,
        response_format: str = "mp4",
        n: int = 1,
    ) -> Any:
        body: dict = {
            "prompt": prompt, "n": n, "response_format": response_format,
        }
        for k, v in [
            ("model", model), ("duration_seconds", duration_seconds),
            ("fps", fps), ("size", size), ("seed", seed),
            ("negative_prompt", negative_prompt), ("steps", steps),
            ("guidance", guidance),
        ]:
            if v is not None:
                body[k] = v

        r = requests.post(
            f"{self.server_url}/v1/video/generations",
            json=body, timeout=self.timeout,
        )
        if r.status_code != 200:
            raise RuntimeError(f"server returned {r.status_code}: {r.text[:500]}")
        payload = r.json()
        data = payload["data"]
        if response_format == "frames_b64":
            return [base64.b64decode(e["b64_json"]) for e in data]
        if n == 1:
            return base64.b64decode(data[0]["b64_json"])
        return [base64.b64decode(e["b64_json"]) for e in data]
```

Tests: env var honored, default URL, body shape, decodes mp4/webm to
bytes when n=1, list when n>1, frames_b64 always returns list,
non-200 raises RuntimeError.

Commit: `feat(video-gen): VideoGenerationClient HTTP client`.

## Task F: bundled wan2_1_t2v_1_3b script

Files:
- `src/muse/models/wan2_1_t2v_1_3b.py`
- `tests/models/test_wan2_1_t2v_1_3b.py`

Mirrors `animatediff_motion_v3.py`'s shape: lazy imports, MANIFEST,
`_select_device`, Model class with `_ensure_deps`. Loads the pipeline
class robustly: tries `WanPipeline` first, falls back to
`DiffusionPipeline.from_pretrained` when WanPipeline isn't present.

```python
"""Wan 2.1 T2V 1.3B: Apache 2.0, ~3GB at fp16, 5s clips at 832x480.

Default low-VRAM video generation bundle. Fits comfortably on 8GB
cards but tight; 12GB+ recommended for headroom.

Constructs robustly across diffusers versions:
  - WanPipeline.from_pretrained when WanPipeline is exported
  - DiffusionPipeline.from_pretrained as a fallback (auto-detects from
    model_index.json)
"""
from __future__ import annotations

import logging
from typing import Any

from muse.modalities.video_generation.protocol import VideoResult


logger = logging.getLogger(__name__)


torch: Any = None
WanPipeline: Any = None
DiffusionPipeline: Any = None


def _ensure_deps() -> None:
    global torch, WanPipeline, DiffusionPipeline
    if torch is None:
        try:
            import torch as _t
            torch = _t
        except Exception as e:  # noqa: BLE001
            logger.debug("wan2_1_t2v_1_3b: torch unavailable: %s", e)
    if WanPipeline is None:
        try:
            from diffusers import WanPipeline as _p
            WanPipeline = _p
        except Exception as e:  # noqa: BLE001
            logger.debug("wan2_1_t2v_1_3b: WanPipeline unavailable: %s", e)
    if DiffusionPipeline is None:
        try:
            from diffusers import DiffusionPipeline as _d
            DiffusionPipeline = _d
        except Exception as e:  # noqa: BLE001
            logger.debug("wan2_1_t2v_1_3b: DiffusionPipeline unavailable: %s", e)


MANIFEST = {
    "model_id": "wan2-1-t2v-1-3b",
    "modality": "video/generation",
    "hf_repo": "Wan-AI/Wan2.1-T2V-1.3B",
    "description": (
        "Wan 2.1 T2V 1.3B: ~3GB, 5s videos at 832x480, fits 8GB GPUs, Apache 2.0"
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


def _select_device(device: str) -> str:
    if device != "auto":
        return device
    if torch is None:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Model:
    """Wan 2.1 T2V 1.3B backend."""

    model_id = MANIFEST["model_id"]

    def __init__(
        self,
        *,
        hf_repo: str,
        local_dir: str | None = None,
        device: str = "auto",
        dtype: str = "float16",
        **_: Any,
    ) -> None:
        _ensure_deps()
        pipeline_cls = WanPipeline or DiffusionPipeline
        if pipeline_cls is None:
            raise RuntimeError(
                "diffusers is not installed; run "
                "`muse pull wan2-1-t2v-1-3b` to set up the per-model venv"
            )
        caps = MANIFEST["capabilities"]
        self._default_duration = caps["default_duration_seconds"]
        self._default_fps = caps["default_fps"]
        self._default_size = tuple(caps["default_size"])
        self._default_steps = caps["default_steps"]
        self._default_guidance = caps["default_guidance"]
        self._device = _select_device(device)

        import muse.models.wan2_1_t2v_1_3b as _mod
        _torch = _mod.torch
        torch_dtype = None
        if _torch is not None:
            torch_dtype = {
                "float16": _torch.float16,
                "float32": _torch.float32,
                "bfloat16": _torch.bfloat16,
            }[dtype]

        src = local_dir or hf_repo
        cls_name = getattr(pipeline_cls, "__name__", "Pipeline")
        logger.info(
            "loading %s from %s (device=%s, dtype=%s)",
            cls_name, src, self._device, dtype,
        )
        self._pipe = pipeline_cls.from_pretrained(src, torch_dtype=torch_dtype)
        if self._device != "cpu":
            self._pipe = self._pipe.to(self._device)

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
        **_: Any,
    ) -> VideoResult:
        dur = duration_seconds if duration_seconds is not None else self._default_duration
        out_fps = fps if fps is not None else self._default_fps
        w = width or self._default_size[0]
        h = height or self._default_size[1]
        n_steps = steps if steps is not None else self._default_steps
        cfg = guidance if guidance is not None else self._default_guidance
        n_frames = max(1, round(dur * out_fps))

        gen = None
        if seed is not None:
            import muse.models.wan2_1_t2v_1_3b as _mod
            _torch = _mod.torch
            if _torch is not None:
                gen = _torch.Generator(device=self._device).manual_seed(seed)

        call_kwargs: dict = {
            "prompt": prompt,
            "num_frames": n_frames,
            "num_inference_steps": n_steps,
            "guidance_scale": cfg,
            "width": w,
            "height": h,
        }
        if negative_prompt is not None:
            call_kwargs["negative_prompt"] = negative_prompt
        if gen is not None:
            call_kwargs["generator"] = gen

        out = self._pipe(**call_kwargs)
        frames_list = out.frames[0]
        first = frames_list[0]
        actual_frames = len(frames_list)
        actual_duration = round(actual_frames / max(out_fps, 1), 3)
        first_w = getattr(first, "size", (w, h))[0]
        first_h = getattr(first, "size", (w, h))[1]
        return VideoResult(
            frames=list(frames_list),
            fps=out_fps,
            width=int(first_w),
            height=int(first_h),
            duration_seconds=actual_duration,
            seed=seed if seed is not None else -1,
            metadata={
                "prompt": prompt,
                "frames": actual_frames,
                "fps": out_fps,
                "steps": n_steps,
                "guidance": cfg,
                "model": self.model_id,
            },
        )
```

Tests use `importlib.import_module` per call, patch
`muse.models.wan2_1_t2v_1_3b.WanPipeline` and `DiffusionPipeline`,
and exercise:
- manifest required fields.
- pip_extras has torch + diffusers >= 0.32 + imageio.
- capabilities advertise expected defaults.
- Model loads via patched WanPipeline.
- Model falls back to DiffusionPipeline when WanPipeline=None.
- Model.generate returns VideoResult.

Commit: `feat(video-gen): bundled wan2-1-t2v-1-3b model script`.

## Task G: HF plugin + curated entries

Files:
- `src/muse/modalities/video_generation/hf.py`
- `tests/modalities/video_generation/test_hf_plugin.py`
- `src/muse/curated.yaml` (append three entries)

`hf.py`:

```python
"""HF resolver plugin for text-to-video repos.

Sniffs HF repos with the `text-to-video` tag and a repo name matching
one of the supported architecture patterns: wan, cogvideox, ltx,
mochi, hunyuan. Priority 105.

Per-architecture dispatch:
  - *wan2*, *wan-* -> WanRuntime (production-ready)
  - *cogvideox*    -> CogVideoXRuntime (production-ready)
  - *ltx*, *mochi*, *hunyuan* -> WanRuntime fallback (manifest synthesized
    but pipeline class won't match; v1.next adds dedicated runtimes)

Loaded via single-file import; no relative imports.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from huggingface_hub import HfApi, snapshot_download

from muse.core.resolvers import ResolvedModel, SearchResult


_WAN_RUNTIME_PATH = (
    "muse.modalities.video_generation.runtimes.wan_runtime:WanRuntime"
)
_COGVIDEOX_RUNTIME_PATH = (
    "muse.modalities.video_generation.runtimes.cogvideox_runtime:CogVideoXRuntime"
)
_PIP_EXTRAS = (
    "torch>=2.1.0",
    "diffusers>=0.32.0",
    "transformers>=4.36.0",
    "accelerate",
    "Pillow>=9.1.0",
    "imageio[ffmpeg]>=2.31.0",
)


def _model_id(repo_id: str) -> str:
    return repo_id.split("/", 1)[-1].lower()


def _repo_license(info) -> str | None:
    card = getattr(info, "card_data", None)
    if card is None:
        return None
    return getattr(card, "license", None)


def _infer_defaults(repo_id: str) -> tuple[str, dict[str, Any]]:
    """Returns (runtime_path, capabilities) per architecture pattern."""
    rid = repo_id.lower()
    if "cogvideox" in rid:
        caps = {
            "device": "cuda",
            "default_duration_seconds": 6.0,
            "default_fps": 8,
            "default_size": [720, 480],
            "min_duration_seconds": 1.0,
            "max_duration_seconds": 10.0,
            "default_steps": 50,
            "default_guidance": 6.0,
            "supports_image_to_video": False,
            "memory_gb": 9.0,
        }
        return _COGVIDEOX_RUNTIME_PATH, caps
    if "wan" in rid:
        caps = {
            "device": "cuda",
            "default_duration_seconds": 5.0,
            "default_fps": 5,
            "default_size": [832, 480],
            "min_duration_seconds": 1.0,
            "max_duration_seconds": 10.0,
            "default_steps": 30,
            "default_guidance": 5.0,
            "supports_image_to_video": False,
            "memory_gb": 6.0,
        }
        return _WAN_RUNTIME_PATH, caps
    if "ltx-video" in rid or "ltx_video" in rid or "ltxvideo" in rid:
        caps = {
            "device": "cuda",
            "default_duration_seconds": 5.0,
            "default_fps": 30,
            "default_size": [1216, 704],
            "default_steps": 20,
            "default_guidance": 3.0,
            "supports_image_to_video": False,
            "memory_gb": 16.0,
        }
        return _WAN_RUNTIME_PATH, caps  # fallback path; v1.next: LTXVideoRuntime
    if "mochi" in rid:
        caps = {
            "device": "cuda",
            "default_duration_seconds": 5.0,
            "default_fps": 30,
            "default_size": [848, 480],
            "default_steps": 64,
            "default_guidance": 4.5,
            "supports_image_to_video": False,
            "memory_gb": 24.0,
        }
        return _WAN_RUNTIME_PATH, caps  # fallback path; v1.next: MochiRuntime
    if "hunyuan" in rid:
        caps = {
            "device": "cuda",
            "default_duration_seconds": 5.0,
            "default_fps": 24,
            "default_size": [1280, 720],
            "default_steps": 50,
            "default_guidance": 6.0,
            "supports_image_to_video": False,
            "memory_gb": 60.0,
        }
        return _WAN_RUNTIME_PATH, caps  # fallback path; v1.next: HunyuanRuntime
    # Generic fallback
    caps = {
        "device": "cuda",
        "default_duration_seconds": 5.0,
        "default_fps": 8,
        "default_size": [768, 432],
        "default_steps": 30,
        "default_guidance": 5.0,
        "supports_image_to_video": False,
        "memory_gb": 8.0,
    }
    return _WAN_RUNTIME_PATH, caps


_KNOWN_PATTERNS = ("wan", "cogvideox", "ltx-video", "ltx_video", "ltxvideo",
                   "mochi", "hunyuan")


def _sniff(info) -> bool:
    tags = getattr(info, "tags", None) or []
    if "text-to-video" not in tags:
        return False
    repo_id = (getattr(info, "id", "") or "").lower()
    return any(p in repo_id for p in _KNOWN_PATTERNS)


def _resolve(repo_id: str, variant: str | None, info) -> ResolvedModel:
    runtime_path, capabilities = _infer_defaults(repo_id)
    manifest = {
        "model_id": _model_id(repo_id),
        "modality": "video/generation",
        "hf_repo": repo_id,
        "description": f"text-to-video model: {repo_id}",
        "license": _repo_license(info),
        "pip_extras": list(_PIP_EXTRAS),
        "system_packages": [],
        "capabilities": capabilities,
    }

    def _download(cache_root: Path) -> Path:
        return Path(snapshot_download(
            repo_id=repo_id,
            cache_dir=str(cache_root) if cache_root else None,
        ))

    return ResolvedModel(
        manifest=manifest,
        backend_path=runtime_path,
        download=_download,
    )


def _search(api: HfApi, query: str, *, sort: str, limit: int) -> Iterable[SearchResult]:
    repos = api.list_models(
        search=query, filter="text-to-video",
        sort=sort, limit=limit,
    )
    for repo in repos:
        yield SearchResult(
            uri=f"hf://{repo.id}",
            model_id=_model_id(repo.id),
            modality="video/generation",
            size_gb=None,
            downloads=getattr(repo, "downloads", None),
            license=None,
            description=repo.id,
        )


HF_PLUGIN = {
    "modality": "video/generation",
    "runtime_path": _WAN_RUNTIME_PATH,  # default; resolve dispatches per-arch
    "pip_extras": _PIP_EXTRAS,
    "system_packages": (),
    "priority": 105,
    "sniff": _sniff,
    "resolve": _resolve,
    "search": _search,
}
```

Tests: positive sniff for Wan / CogVideoX / LTX-Video / Mochi /
Hunyuan; negative sniff for SD / non-text-to-video / text-to-video
without architecture match; resolve dispatches correct runtime path
per arch; search filters tag.

Append curated entries to `src/muse/curated.yaml`:

```yaml
# ---------- video/generation (Wan, CogVideoX, LTX) ----------

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

Commit: `feat(video-gen): HF plugin + curated entries (wan/cogvideox/ltx)`.

## Task H: documentation + slow e2e + integration + v0.27.0 release

Files:
- `tests/cli_impl/test_e2e_video_generation.py` (slow)
- `tests/integration/test_remote_video_generation.py` (opt-in)
- `tests/integration/conftest.py` (add `video_model` fixture if needed)
- `pyproject.toml` (version bump to 0.27.0)
- `src/muse/__init__.py` (docstring update; add video/generation; bump v reference)
- `CLAUDE.md` (modality list + new section on video/generation)
- `README.md` (modality list + endpoint section + curl/Python examples)
- `git tag v0.27.0 && git push origin main v0.27.0`
- GitHub release via `gh release create`

Slow e2e: full TestClient flow with FakeVideoBackend that satisfies
the protocol structurally; covers each format (mp4, webm, frames_b64),
n=1 and n=2, the response envelope. Decorated with
`@pytest.mark.slow`.

Integration test: opt-in via `MUSE_REMOTE_SERVER`; `video_model`
fixture defaults to `wan2-1-t2v-1-3b`. Tests are protocol-style
assertions that must always hold (test_protocol_*). Skips when
the model isn't loaded on the remote server (typical home setup).

CLAUDE.md update: bump count to 15 modalities, add video/generation
to the modality list, add a new section explaining the
distinction from image/animation, document the VRAM caveats.

README.md update: bump count to 15 modalities, add video/generation
row to the modality list, add a "Video Generation" subsection with
curl + Python examples and explicit VRAM warnings.

Commit: `chore(release): v0.27.0`.

Tag, push, release.

## Acceptance

- 15 modalities discovered.
- `pytest tests/ -m "not slow"` green.
- `pytest tests/ -q` (slow lane) green.
- `muse models list` shows wan2-1-t2v-1-3b / cogvideox-2b / ltx-video.
- `muse search wan --modality video/generation` returns Wan family.
- `muse pull wan2-1-t2v-1-3b` would create a venv with diffusers
  >=0.32 + torch + transformers + imageio[ffmpeg].
- POST /v1/video/generations with valid mp4/webm/frames_b64 payloads
  succeeds end-to-end (FakeVideoBackend in tests; real Wan in
  integration when the user has the model pulled).
- v0.27.0 tagged, pushed, GitHub release published.
