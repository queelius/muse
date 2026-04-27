# `image/animation` Modality Implementation Plan (#144)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `image/animation` modality with `POST /v1/images/animations`. Bundled `animatediff-motion-v3` script (SD 1.5 + motion adapter v1-5-3) and HF plugin for fused-checkpoint AnimateDiff variants (AnimateLCM). Default response: animated WebP. Alternates: GIF, MP4 (optional dep), frames_b64.

**Architecture:** Mirror existing modalities. New `image_animation/` package with protocol, codec, routes, client, and `runtimes/animatediff.py` generic runtime. New bundled `animatediff_motion_v3.py`. New `image_animation/hf.py` plugin (sniffs `text-to-video` tag + animated/motion in repo name).

**Spec:** `docs/superpowers/specs/2026-04-27-image-animation-modality-design.md`

**Target version:** v0.18.0

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `src/muse/modalities/image_animation/__init__.py` | create | exports `MODALITY`, `build_router`, Protocol, Result, Client |
| `src/muse/modalities/image_animation/protocol.py` | create | `AnimationResult` dataclass, `AnimationModel` Protocol |
| `src/muse/modalities/image_animation/codec.py` | create | `encode_webp/gif/mp4/frames_b64` |
| `src/muse/modalities/image_animation/routes.py` | create | `POST /v1/images/animations`, capability gates |
| `src/muse/modalities/image_animation/client.py` | create | `AnimationsClient` (HTTP) |
| `src/muse/modalities/image_animation/runtimes/__init__.py` | create | empty marker |
| `src/muse/modalities/image_animation/runtimes/animatediff.py` | create | `AnimateDiffRuntime` generic runtime |
| `src/muse/modalities/image_animation/hf.py` | create | HF plugin for fused-checkpoint AnimateDiff |
| `src/muse/models/animatediff_motion_v3.py` | create | bundled (SD 1.5 base + motion adapter) |
| `src/muse/curated.yaml` | modify | +2 entries: `animatediff-motion-v3` (bundled), `animatelcm` (uri) |
| `pyproject.toml` | modify | bump 0.17.2 to 0.18.0; add `imageio[ffmpeg]` to images extras (optional via mp4 path) |
| `src/muse/__init__.py` | modify | docstring v0.18.0; add `image/animation` to bundled-modalities list |
| `CLAUDE.md` | modify | document the new modality |
| `README.md` | modify | add image/animation to the route list |
| `tests/modalities/image_animation/` (full tree) | create | protocol, codec, routes, client, hf_plugin, runtimes |
| `tests/models/test_animatediff_motion_v3.py` | create | bundled-script tests |
| `tests/cli_impl/test_e2e_supervisor.py` | modify | extend slow e2e to include the new modality |
| `tests/integration/test_remote_animations.py` | create | opt-in integration tests |

---

## Task A: Protocol + Codec

Smallest, most isolated. No callers. Foundation for everything else.

**Files:**
- Create: `src/muse/modalities/image_animation/__init__.py` (initially empty re-exports)
- Create: `src/muse/modalities/image_animation/protocol.py`
- Create: `src/muse/modalities/image_animation/codec.py`
- Create: `tests/modalities/image_animation/__init__.py` (empty)
- Create: `tests/modalities/image_animation/test_protocol.py`
- Create: `tests/modalities/image_animation/test_codec.py`

- [ ] **Step 1: Write the protocol test**

Create `tests/modalities/image_animation/test_protocol.py`:

```python
"""Tests for the image_animation protocol surface."""
from unittest.mock import MagicMock

from muse.modalities.image_animation.protocol import (
    AnimationModel,
    AnimationResult,
)


def test_animation_result_dataclass_shape():
    fake_frame = MagicMock()
    r = AnimationResult(
        frames=[fake_frame, fake_frame],
        fps=8,
        width=512, height=512,
        seed=42,
        metadata={"prompt": "x"},
    )
    assert len(r.frames) == 2
    assert r.fps == 8
    assert r.width == 512
    assert r.metadata["prompt"] == "x"


def test_animation_model_protocol_is_runtime_checkable():
    """Plain duck-typed class satisfies the Protocol structurally."""
    class FakeModel:
        model_id = "fake"
        def generate(self, prompt, **kwargs):
            return AnimationResult(frames=[], fps=8, width=0, height=0, seed=-1, metadata={})

    assert isinstance(FakeModel(), AnimationModel)
```

- [ ] **Step 2: Run, expect ImportError**

```bash
pytest tests/modalities/image_animation/test_protocol.py -v
```

- [ ] **Step 3: Implement protocol**

Create `src/muse/modalities/image_animation/protocol.py`:

```python
"""Modality protocol for image/animation.

AnimationResult holds a list of PIL.Image frames + timing metadata. The
codec layer transforms the list into webp/gif/mp4/frames_b64 bytes for
the HTTP response.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class AnimationResult:
    """One generated animation: ordered frames + timing + provenance.

    frames: list[Any] (typed loosely to avoid forcing PIL on the protocol
    boundary; codec normalizes to PIL before encoding).
    """
    frames: list[Any]
    fps: int
    width: int
    height: int
    seed: int
    metadata: dict = field(default_factory=dict)


@runtime_checkable
class AnimationModel(Protocol):
    """Protocol for animation backends."""

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
        init_image: Any = None,
        strength: float | None = None,
        **kwargs,
    ) -> AnimationResult: ...
```

- [ ] **Step 4: Write the codec test**

Create `tests/modalities/image_animation/test_codec.py`:

```python
"""Tests for image_animation codec.

WebP/GIF use Pillow (always available). MP4 uses imageio (optional;
test verifies the lazy import + clean error message when unavailable).
"""
from unittest.mock import patch

import pytest
from PIL import Image

from muse.modalities.image_animation.codec import (
    encode_webp,
    encode_gif,
    encode_mp4,
    encode_frames_b64,
    UnsupportedFormatError,
)


def _frames(n=3, w=64, h=64):
    return [Image.new("RGB", (w, h), color=(i*40, 100, 200)) for i in range(n)]


def test_encode_webp_returns_bytes_with_riff_header():
    out = encode_webp(_frames(), fps=8, loop=True)
    assert isinstance(out, bytes)
    # WebP files start with "RIFF" (4 bytes) ... "WEBP" (at offset 8).
    assert out[:4] == b"RIFF"
    assert out[8:12] == b"WEBP"


def test_encode_webp_loop_count_is_honored():
    """loop=False (single play) and loop=True (infinite) produce different output."""
    looped = encode_webp(_frames(), fps=8, loop=True)
    once = encode_webp(_frames(), fps=8, loop=False)
    assert looped != once  # different loop counts encode to different bytes


def test_encode_gif_returns_bytes_with_gif_header():
    out = encode_gif(_frames(), fps=8, loop=True)
    assert out[:6] in (b"GIF87a", b"GIF89a")


def test_encode_frames_b64_returns_per_frame_base64():
    out = encode_frames_b64(_frames(n=3))
    assert isinstance(out, list)
    assert len(out) == 3
    import base64
    # Each entry must be base64-decodable into PNG bytes.
    for entry in out:
        png = base64.b64decode(entry)
        assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_encode_mp4_raises_when_imageio_unavailable():
    """mp4 requires imageio[ffmpeg]; if absent, raise a clean error."""
    with patch(
        "muse.modalities.image_animation.codec._try_import_imageio",
        return_value=None,
    ):
        with pytest.raises(UnsupportedFormatError, match="imageio"):
            encode_mp4(_frames(), fps=8)


def test_encode_mp4_calls_imageio_when_available():
    """mp4 uses imageio.mimwrite to produce h264 bytes."""
    fake_imageio = type("ii", (), {})()
    captured = {}
    def fake_mimwrite(buf, frames, *, fps, codec, **_):
        captured["fps"] = fps
        captured["codec"] = codec
        captured["n_frames"] = len(frames)
        buf.write(b"fakeMP4DATA")
    fake_imageio.mimwrite = fake_mimwrite

    with patch(
        "muse.modalities.image_animation.codec._try_import_imageio",
        return_value=fake_imageio,
    ):
        out = encode_mp4(_frames(n=3), fps=12)

    assert out == b"fakeMP4DATA"
    assert captured["fps"] == 12
    assert captured["codec"] == "h264"
    assert captured["n_frames"] == 3
```

- [ ] **Step 5: Run, expect ImportError**

- [ ] **Step 6: Implement codec**

Create `src/muse/modalities/image_animation/codec.py`:

```python
"""Encoding helpers for image/animation responses.

Pure functions: list[PIL.Image] + timing -> bytes (or list[base64-str]).

WebP and GIF use Pillow (always installed via the modality's pip_extras).
MP4 uses imageio[ffmpeg] (optional dep; raises UnsupportedFormatError
when missing so the route can return 400 with a clear message).
"""
from __future__ import annotations

import base64
import io
from typing import Any


class UnsupportedFormatError(Exception):
    """Raised when a response_format requires deps that aren't installed."""


def encode_webp(
    frames: list[Any], fps: int, *, loop: bool = True, lossless: bool = False,
) -> bytes:
    """Encode frames as animated WebP.

    duration_ms_per_frame = round(1000 / fps).
    loop=True -> 0 (infinite). loop=False -> 1 (single play).
    """
    if not frames:
        raise ValueError("encode_webp: frames list is empty")
    duration = max(1, round(1000 / max(fps, 1)))
    loop_count = 0 if loop else 1
    buf = io.BytesIO()
    head = frames[0]
    head.save(
        buf, format="WEBP",
        save_all=True,
        append_images=list(frames[1:]),
        duration=duration,
        loop=loop_count,
        lossless=lossless,
        quality=85,
    )
    return buf.getvalue()


def encode_gif(frames: list[Any], fps: int, *, loop: bool = True) -> bytes:
    if not frames:
        raise ValueError("encode_gif: frames list is empty")
    duration = max(1, round(1000 / max(fps, 1)))
    loop_count = 0 if loop else 1
    buf = io.BytesIO()
    head = frames[0]
    head.save(
        buf, format="GIF",
        save_all=True,
        append_images=list(frames[1:]),
        duration=duration,
        loop=loop_count,
        disposal=2,
    )
    return buf.getvalue()


def encode_mp4(frames: list[Any], fps: int) -> bytes:
    """Encode frames as h264 MP4 via imageio. Raises UnsupportedFormatError
    when imageio[ffmpeg] is not installed."""
    imageio = _try_import_imageio()
    if imageio is None:
        raise UnsupportedFormatError(
            "mp4 response_format requires imageio[ffmpeg]; "
            "install via `pip install imageio[ffmpeg]` or use webp/gif"
        )
    if not frames:
        raise ValueError("encode_mp4: frames list is empty")
    import numpy as np
    arrays = [np.array(f.convert("RGB")) for f in frames]
    buf = io.BytesIO()
    imageio.mimwrite(buf, arrays, fps=fps, codec="h264", format="mp4", quality=8)
    return buf.getvalue()


def encode_frames_b64(frames: list[Any]) -> list[str]:
    """Each frame as a standalone base64-encoded PNG."""
    out: list[str] = []
    for f in frames:
        buf = io.BytesIO()
        f.save(buf, format="PNG")
        out.append(base64.b64encode(buf.getvalue()).decode("ascii"))
    return out


def _try_import_imageio():
    """Lazy import isolated for test patching."""
    try:
        import imageio
        return imageio
    except ImportError:
        return None
```

- [ ] **Step 7: Run all tests, expect pass**

```bash
pytest tests/modalities/image_animation/ -v
```

Expected: ~10 tests pass.

- [ ] **Step 8: Run full fast lane**

```bash
pytest tests/ -q -m "not slow"
```

- [ ] **Step 9: Commit**

```bash
git add src/muse/modalities/image_animation/__init__.py \
        src/muse/modalities/image_animation/protocol.py \
        src/muse/modalities/image_animation/codec.py \
        tests/modalities/image_animation/__init__.py \
        tests/modalities/image_animation/test_protocol.py \
        tests/modalities/image_animation/test_codec.py
git commit -m "feat(image-animation): protocol + codec (#144)

AnimationResult dataclass and AnimationModel Protocol. Codec helpers
encode_webp/gif/mp4/frames_b64. WebP/GIF use Pillow (always available).
MP4 uses imageio[ffmpeg], lazy-imported with clean error when missing.

No callers yet; route + client + runtime + bundled script wire it in
across the next tasks."
```

For now `__init__.py` re-exports just the protocol + codec. MODALITY constant and build_router come in Task C.

---

## Task B: AnimateDiffRuntime generic runtime

The meatiest single task. Wraps `diffusers.AnimateDiffPipeline` + `MotionAdapter`. Lazy imports. Parameterized by manifest capabilities.

**Files:**
- Create: `src/muse/modalities/image_animation/runtimes/__init__.py` (empty)
- Create: `src/muse/modalities/image_animation/runtimes/animatediff.py`
- Create: `tests/modalities/image_animation/runtimes/__init__.py` (empty)
- Create: `tests/modalities/image_animation/runtimes/test_animatediff.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/modalities/image_animation/runtimes/test_animatediff.py`:

```python
"""Tests for AnimateDiffRuntime: generic runtime over diffusers AnimateDiff.

Mirrors the lazy-import sentinel pattern from sd_turbo and runtimes/diffusers.
Stubs AnimateDiffPipeline + MotionAdapter so no real diffusion runs.
"""
from unittest.mock import MagicMock, patch

import pytest

from muse.modalities.image_animation.protocol import AnimationResult
from muse.modalities.image_animation.runtimes.animatediff import (
    AnimateDiffRuntime,
)


def _patched_pipe():
    """Fake pipeline whose .from_pretrained yields a callable returning
    a fake output object with .frames[0] = list[PIL-shaped images]."""
    fake_pipe = MagicMock()
    fake_frame = MagicMock()
    fake_frame.size = (512, 512)
    # AnimateDiffPipeline output: out.frames is list of lists (per video, per frame)
    fake_pipe.return_value.frames = [[fake_frame, fake_frame, fake_frame]]
    return fake_pipe


def test_construction_loads_pipeline_and_adapter():
    fake_pipe_class = MagicMock()
    fake_pipe_class.from_pretrained.return_value = _patched_pipe()
    fake_adapter_class = MagicMock()
    fake_adapter_class.from_pretrained.return_value = MagicMock()

    with patch(
        "muse.modalities.image_animation.runtimes.animatediff.AnimateDiffPipeline",
        fake_pipe_class,
    ), patch(
        "muse.modalities.image_animation.runtimes.animatediff.MotionAdapter",
        fake_adapter_class,
    ), patch(
        "muse.modalities.image_animation.runtimes.animatediff.torch",
        MagicMock(),
    ):
        m = AnimateDiffRuntime(
            hf_repo="guoyww/animatediff-motion-adapter-v1-5-3",
            local_dir="/fake/adapter",
            device="cpu",
            model_id="adv3",
            base_model="emilianJR/epiCRealism",
        )
    fake_adapter_class.from_pretrained.assert_called_once()
    fake_pipe_class.from_pretrained.assert_called_once()
    assert m.model_id == "adv3"


def test_generate_returns_animation_result():
    fake_pipe = _patched_pipe()
    fake_pipe_class = MagicMock()
    fake_pipe_class.from_pretrained.return_value = fake_pipe
    fake_adapter_class = MagicMock()
    fake_adapter_class.from_pretrained.return_value = MagicMock()

    with patch(
        "muse.modalities.image_animation.runtimes.animatediff.AnimateDiffPipeline",
        fake_pipe_class,
    ), patch(
        "muse.modalities.image_animation.runtimes.animatediff.MotionAdapter",
        fake_adapter_class,
    ), patch(
        "muse.modalities.image_animation.runtimes.animatediff.torch",
        MagicMock(),
    ):
        m = AnimateDiffRuntime(
            hf_repo="guoyww/animatediff-motion-adapter-v1-5-3",
            local_dir="/fake/adapter",
            device="cpu",
            model_id="adv3",
            base_model="emilianJR/epiCRealism",
            default_frames=16, default_fps=8,
            default_size=(512, 512), default_steps=25, default_guidance=7.5,
        )
        r = m.generate("a cat")
    assert isinstance(r, AnimationResult)
    assert len(r.frames) == 3  # the fake pipe returned 3 frames
    assert r.fps == 8
    assert r.width == 512


def test_generate_request_overrides_defaults():
    fake_pipe = _patched_pipe()
    fake_pipe_class = MagicMock()
    fake_pipe_class.from_pretrained.return_value = fake_pipe
    fake_adapter_class = MagicMock()
    fake_adapter_class.from_pretrained.return_value = MagicMock()

    with patch(
        "muse.modalities.image_animation.runtimes.animatediff.AnimateDiffPipeline",
        fake_pipe_class,
    ), patch(
        "muse.modalities.image_animation.runtimes.animatediff.MotionAdapter",
        fake_adapter_class,
    ), patch(
        "muse.modalities.image_animation.runtimes.animatediff.torch",
        MagicMock(),
    ):
        m = AnimateDiffRuntime(
            hf_repo="x", local_dir="/fake", device="cpu",
            model_id="m", base_model="b",
            default_frames=16, default_fps=8, default_steps=25, default_guidance=7.5,
        )
        m.generate("a fox", frames=8, fps=12, steps=50, guidance=9.0)
    kwargs = fake_pipe.call_args.kwargs
    assert kwargs["num_frames"] == 8
    assert kwargs["num_inference_steps"] == 50
    assert kwargs["guidance_scale"] == 9.0


def test_generate_passes_negative_prompt_when_set():
    fake_pipe = _patched_pipe()
    fake_pipe_class = MagicMock()
    fake_pipe_class.from_pretrained.return_value = fake_pipe
    fake_adapter_class = MagicMock()
    fake_adapter_class.from_pretrained.return_value = MagicMock()

    with patch(
        "muse.modalities.image_animation.runtimes.animatediff.AnimateDiffPipeline",
        fake_pipe_class,
    ), patch(
        "muse.modalities.image_animation.runtimes.animatediff.MotionAdapter",
        fake_adapter_class,
    ), patch(
        "muse.modalities.image_animation.runtimes.animatediff.torch",
        MagicMock(),
    ):
        m = AnimateDiffRuntime(
            hf_repo="x", local_dir="/fake", device="cpu",
            model_id="m", base_model="b",
        )
        m.generate("a fox", negative_prompt="blurry, ugly")
    assert fake_pipe.call_args.kwargs.get("negative_prompt") == "blurry, ugly"


def test_generate_omits_negative_prompt_when_none():
    fake_pipe = _patched_pipe()
    fake_pipe_class = MagicMock()
    fake_pipe_class.from_pretrained.return_value = fake_pipe
    fake_adapter_class = MagicMock()
    fake_adapter_class.from_pretrained.return_value = MagicMock()

    with patch(
        "muse.modalities.image_animation.runtimes.animatediff.AnimateDiffPipeline",
        fake_pipe_class,
    ), patch(
        "muse.modalities.image_animation.runtimes.animatediff.MotionAdapter",
        fake_adapter_class,
    ), patch(
        "muse.modalities.image_animation.runtimes.animatediff.torch",
        MagicMock(),
    ):
        m = AnimateDiffRuntime(
            hf_repo="x", local_dir="/fake", device="cpu",
            model_id="m", base_model="b",
        )
        m.generate("a fox")
    assert "negative_prompt" not in fake_pipe.call_args.kwargs


def test_construction_absorbs_unknown_kwargs():
    """Future capability flags must not crash the constructor."""
    fake_pipe_class = MagicMock()
    fake_pipe_class.from_pretrained.return_value = _patched_pipe()
    fake_adapter_class = MagicMock()
    fake_adapter_class.from_pretrained.return_value = MagicMock()

    with patch(
        "muse.modalities.image_animation.runtimes.animatediff.AnimateDiffPipeline",
        fake_pipe_class,
    ), patch(
        "muse.modalities.image_animation.runtimes.animatediff.MotionAdapter",
        fake_adapter_class,
    ), patch(
        "muse.modalities.image_animation.runtimes.animatediff.torch",
        MagicMock(),
    ):
        AnimateDiffRuntime(
            hf_repo="x", local_dir="/fake", device="cpu", model_id="m",
            base_model="b",
            future_unrecognized_flag="whatever",
            supports_text_to_animation=True,
        )
```

- [ ] **Step 2: Run, expect ImportError**

- [ ] **Step 3: Implement the runtime**

Create `src/muse/modalities/image_animation/runtimes/__init__.py` (empty).

Create `src/muse/modalities/image_animation/runtimes/animatediff.py`:

```python
"""Generic AnimateDiff runtime via diffusers AnimateDiffPipeline.

Two-component model: a base SD 1.5 (or compatible) checkpoint provides
text encoder + UNet + VAE; a MotionAdapter provides the temporal layers.
The base is referenced by manifest field `base_model` (a Python string
identifying an HF repo); the motion adapter is what `local_dir`/`hf_repo`
points to.

Lazy-import sentinel pattern matches sd_turbo and runtimes/diffusers.
"""
from __future__ import annotations

import logging
from typing import Any

from muse.modalities.image_animation.protocol import AnimationResult


logger = logging.getLogger(__name__)


# Sentinels patched by tests; populated by _ensure_deps at runtime.
torch: Any = None
AnimateDiffPipeline: Any = None
MotionAdapter: Any = None


def _ensure_deps() -> None:
    global torch, AnimateDiffPipeline, MotionAdapter
    if torch is None:
        try:
            import torch as _t
            torch = _t
        except Exception as e:  # noqa: BLE001
            logger.debug("animatediff: torch unavailable: %s", e)
    if AnimateDiffPipeline is None:
        try:
            from diffusers import AnimateDiffPipeline as _p
            AnimateDiffPipeline = _p
        except Exception as e:  # noqa: BLE001
            logger.debug("animatediff: AnimateDiffPipeline unavailable: %s", e)
    if MotionAdapter is None:
        try:
            from diffusers import MotionAdapter as _m
            MotionAdapter = _m
        except Exception as e:  # noqa: BLE001
            logger.debug("animatediff: MotionAdapter unavailable: %s", e)


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


class AnimateDiffRuntime:
    """AnimateDiff runtime backed by AnimateDiffPipeline + MotionAdapter.

    Construction kwargs:
      - hf_repo: motion adapter repo id (the manifest's hf_repo)
      - local_dir: local cache for the adapter (from `muse pull`)
      - device, dtype, model_id: standard
      - base_model: HF repo id of the base SD 1.5 (or compatible) checkpoint
      - default_frames, default_fps, default_size, default_steps, default_guidance:
        manifest-driven defaults injected via capabilities splat
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
        base_model: str,
        default_frames: int = 16,
        default_fps: int = 8,
        default_size: tuple[int, int] = (512, 512),
        default_steps: int = 25,
        default_guidance: float = 7.5,
        **_: Any,
    ) -> None:
        _ensure_deps()
        if AnimateDiffPipeline is None or MotionAdapter is None:
            raise RuntimeError(
                "diffusers AnimateDiff is not installed; ensure muse[images] "
                "extras are installed in the per-model venv"
            )
        self.model_id = model_id
        self.default_size = tuple(default_size)
        self._default_frames = default_frames
        self._default_fps = default_fps
        self._default_steps = default_steps
        self._default_guidance = default_guidance
        self._device = _select_device(device)

        import muse.modalities.image_animation.runtimes.animatediff as _mod
        _torch = _mod.torch
        torch_dtype = None
        if _torch is not None:
            torch_dtype = {
                "float16": _torch.float16,
                "float32": _torch.float32,
                "bfloat16": _torch.bfloat16,
            }[dtype]

        adapter_src = local_dir or hf_repo
        logger.info(
            "loading MotionAdapter from %s (model_id=%s, dtype=%s)",
            adapter_src, model_id, dtype,
        )
        adapter = MotionAdapter.from_pretrained(adapter_src, torch_dtype=torch_dtype)

        logger.info(
            "loading AnimateDiffPipeline base=%s + adapter (device=%s, dtype=%s)",
            base_model, self._device, dtype,
        )
        self._pipe = AnimateDiffPipeline.from_pretrained(
            base_model,
            motion_adapter=adapter,
            torch_dtype=torch_dtype,
        )
        if self._device != "cpu":
            self._pipe = self._pipe.to(self._device)

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
        init_image: Any = None,
        strength: float | None = None,
        **_: Any,
    ) -> AnimationResult:
        # AnimateDiff base does not support img2vid in v1; the route layer
        # gates this via supports_image_to_animation. If init_image lands
        # here on this runtime, it's a programming error or a model with
        # mis-set capability; surface clearly.
        if init_image is not None:
            raise NotImplementedError(
                "AnimateDiffRuntime base path does not support init_image; "
                "use a model whose capability supports_image_to_animation=True"
            )

        n_frames = frames if frames is not None else self._default_frames
        out_fps = fps if fps is not None else self._default_fps
        w = width or self.default_size[0]
        h = height or self.default_size[1]
        n_steps = steps if steps is not None else self._default_steps
        cfg = guidance if guidance is not None else self._default_guidance

        gen = None
        if seed is not None:
            import muse.modalities.image_animation.runtimes.animatediff as _mod
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
        # AnimateDiffPipeline returns out.frames as list-of-lists (one
        # video, multiple frames). Take the first video.
        frames_list = out.frames[0]
        first = frames_list[0]
        return AnimationResult(
            frames=list(frames_list),
            fps=out_fps,
            width=first.size[0],
            height=first.size[1],
            seed=seed if seed is not None else -1,
            metadata={
                "prompt": prompt,
                "frames": n_frames,
                "fps": out_fps,
                "steps": n_steps,
                "guidance": cfg,
                "model": self.model_id,
            },
        )
```

- [ ] **Step 4: Run, expect pass**

```bash
pytest tests/modalities/image_animation/runtimes/test_animatediff.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Run full fast lane**

- [ ] **Step 6: Commit**

```bash
git add src/muse/modalities/image_animation/runtimes/ \
        tests/modalities/image_animation/runtimes/
git commit -m "feat(image-animation): AnimateDiffRuntime generic runtime (#144)

Wraps diffusers AnimateDiffPipeline + MotionAdapter. Two-component
construction: motion adapter from local_dir/hf_repo, base SD 1.5 from
manifest's base_model field. Capability defaults (frames, fps, size,
steps, guidance) injected via constructor kwargs. Lazy-import sentinel
pattern matches sd_turbo and runtimes/diffusers.

No callers yet; routes + bundled script wire it in next."
```

---

## Task C: Routes + modality `__init__.py`

**Files:**
- Modify: `src/muse/modalities/image_animation/__init__.py` (add MODALITY + build_router exports)
- Create: `src/muse/modalities/image_animation/routes.py`
- Create: `tests/modalities/image_animation/test_routes.py`

- [ ] **Step 1: Write the failing route tests**

Create `tests/modalities/image_animation/test_routes.py`:

```python
"""Tests for POST /v1/images/animations."""
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from muse.core.registry import ModalityRegistry
from muse.core.server import create_app
from muse.modalities.image_animation import (
    MODALITY, AnimationResult, build_router,
)


class RecordingModel:
    """Captures the kwargs each generate() received."""
    model_id = "fake-anim"
    def __init__(self, capabilities=None):
        self._caps = capabilities or {}
        self.last_kwargs = None
    def generate(self, prompt, **kwargs):
        self.last_kwargs = kwargs
        # Return a 4-frame "animation"
        frames = [Image.new("RGB", (64, 64), (i*50, 100, 150)) for i in range(4)]
        return AnimationResult(
            frames=frames, fps=8, width=64, height=64, seed=-1,
            metadata={"prompt": prompt},
        )


@pytest.fixture
def client_text_only():
    backend = RecordingModel()
    reg = ModalityRegistry()
    reg.register(MODALITY, backend, manifest={
        "model_id": "fake-anim",
        "capabilities": {
            "supports_text_to_animation": True,
            "supports_image_to_animation": False,
        },
    })
    app = create_app(registry=reg, routers={MODALITY: build_router(reg)})
    return TestClient(app), backend


@pytest.fixture
def client_img2vid():
    backend = RecordingModel()
    reg = ModalityRegistry()
    reg.register(MODALITY, backend, manifest={
        "model_id": "fake-anim",
        "capabilities": {
            "supports_text_to_animation": True,
            "supports_image_to_animation": True,
        },
    })
    app = create_app(registry=reg, routers={MODALITY: build_router(reg)})
    return TestClient(app), backend


def test_post_returns_webp_by_default(client_text_only):
    client, _backend = client_text_only
    r = client.post("/v1/images/animations", json={
        "prompt": "a cat playing", "model": "fake-anim",
    })
    assert r.status_code == 200
    body = r.json()
    assert "data" in body
    assert len(body["data"]) == 1
    import base64
    asset = base64.b64decode(body["data"][0]["b64_json"])
    assert asset[:4] == b"RIFF"
    assert asset[8:12] == b"WEBP"
    assert body["metadata"]["format"] == "webp"


def test_post_response_format_gif(client_text_only):
    client, _ = client_text_only
    r = client.post("/v1/images/animations", json={
        "prompt": "x", "model": "fake-anim", "response_format": "gif",
    })
    assert r.status_code == 200
    import base64
    asset = base64.b64decode(r.json()["data"][0]["b64_json"])
    assert asset[:6] in (b"GIF87a", b"GIF89a")


def test_post_response_format_frames_b64(client_text_only):
    client, _ = client_text_only
    r = client.post("/v1/images/animations", json={
        "prompt": "x", "model": "fake-anim", "response_format": "frames_b64",
    })
    assert r.status_code == 200
    body = r.json()
    assert len(body["data"]) == 4  # 4 frames
    for entry in body["data"]:
        import base64
        png = base64.b64decode(entry["b64_json"])
        assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_post_image_with_text_only_model_returns_400(client_text_only):
    """Model with supports_image_to_animation=False rejects image input."""
    client, _ = client_text_only
    r = client.post("/v1/images/animations", json={
        "prompt": "x", "model": "fake-anim",
        "image": "data:image/png;base64,iVBORw0KGgo=",
    })
    assert r.status_code == 400
    assert "image-to-animation" in r.json()["error"]["message"].lower()


def test_post_image_with_img2vid_model_passes_through(client_img2vid):
    """Model with supports_image_to_animation=True accepts image input."""
    import base64, io
    img = Image.new("RGB", (64, 64), (255, 0, 0))
    buf = io.BytesIO(); img.save(buf, format="PNG")
    data_url = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

    client, backend = client_img2vid
    r = client.post("/v1/images/animations", json={
        "prompt": "x", "model": "fake-anim",
        "image": data_url, "strength": 0.6,
    })
    assert r.status_code == 200
    # Backend received a PIL image as init_image
    assert backend.last_kwargs.get("init_image") is not None
    assert backend.last_kwargs.get("strength") == 0.6


def test_post_unknown_model_returns_404(client_text_only):
    client, _ = client_text_only
    r = client.post("/v1/images/animations", json={
        "prompt": "x", "model": "nonexistent",
    })
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "model_not_found"


def test_post_loop_default_true(client_text_only):
    """Default loop=true encodes infinite-loop WebP."""
    client, _ = client_text_only
    r1 = client.post("/v1/images/animations", json={
        "prompt": "x", "model": "fake-anim",
    })
    r2 = client.post("/v1/images/animations", json={
        "prompt": "x", "model": "fake-anim", "loop": False,
    })
    assert r1.json()["data"][0]["b64_json"] != r2.json()["data"][0]["b64_json"]


def test_post_invalid_response_format_returns_400(client_text_only):
    client, _ = client_text_only
    r = client.post("/v1/images/animations", json={
        "prompt": "x", "model": "fake-anim", "response_format": "avi",
    })
    assert r.status_code in (400, 422)


def test_post_frames_out_of_range_rejected(client_text_only):
    client, _ = client_text_only
    r = client.post("/v1/images/animations", json={
        "prompt": "x", "model": "fake-anim", "frames": 999,
    })
    assert r.status_code in (400, 422)
```

- [ ] **Step 2: Run, expect ImportError**

- [ ] **Step 3: Implement routes + __init__**

Create `src/muse/modalities/image_animation/routes.py`:

```python
"""POST /v1/images/animations.

Wire contract documented in docs/superpowers/specs/2026-04-27-image-animation-modality-design.md.
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
from muse.modalities.image_animation.codec import (
    encode_webp, encode_gif, encode_mp4, encode_frames_b64,
    UnsupportedFormatError,
)
from muse.modalities.image_generation.image_input import decode_image_input


MODALITY = "image/animation"

logger = logging.getLogger(__name__)


class AnimationsRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    model: str | None = None
    n: int = Field(default=1, ge=1, le=4)
    frames: int | None = Field(default=None, ge=4, le=64)
    fps: int | None = Field(default=None, ge=1, le=30)
    loop: bool = True
    negative_prompt: str | None = None
    steps: int | None = Field(default=None, ge=1, le=100)
    guidance: float | None = Field(default=None, ge=0.0, le=20.0)
    seed: int | None = None
    image: str | None = None
    strength: float | None = Field(default=None, ge=0.0, le=1.0)
    response_format: str = Field(default="webp", pattern="^(webp|gif|mp4|frames_b64)$")
    size: str | None = Field(default=None, pattern=r"^\d+x\d+$")


def build_router(registry: ModalityRegistry) -> APIRouter:
    router = APIRouter(prefix="/v1/images", tags=["image/animation"])

    @router.post("/animations")
    async def animations(req: AnimationsRequest):
        try:
            model = registry.get(MODALITY, req.model)
        except KeyError:
            raise ModelNotFoundError(
                model_id=req.model or "<default>", modality=MODALITY,
            )
        manifest = registry.manifest(MODALITY, model.model_id) or {}
        capabilities = manifest.get("capabilities") or {}

        # Image-to-animation gate
        init_image = None
        if req.image is not None:
            if not capabilities.get("supports_image_to_animation"):
                return error_response(
                    400, "invalid_parameter",
                    f"model {model.model_id!r} does not support image-to-animation; "
                    f"use a model with supports_image_to_animation=True",
                )
            try:
                init_image = decode_image_input(req.image)
            except ValueError as e:
                return error_response(
                    400, "invalid_parameter", f"image decode failed: {e}",
                )

        width = height = None
        if req.size is not None:
            width, height = map(int, req.size.split("x"))

        def _call_one(seed_offset: int):
            kwargs = {
                "negative_prompt": req.negative_prompt,
                "frames": req.frames,
                "fps": req.fps,
                "width": width, "height": height,
                "steps": req.steps, "guidance": req.guidance,
                "init_image": init_image,
                "strength": req.strength,
            }
            if req.seed is not None:
                kwargs["seed"] = req.seed + seed_offset
            return model.generate(req.prompt, **kwargs)

        results = []
        for i in range(req.n):
            r = await asyncio.to_thread(_call_one, i)
            results.append(r)

        # Encode each result according to response_format
        data = []
        for r in results:
            try:
                encoded = _encode(req.response_format, r, loop=req.loop)
            except UnsupportedFormatError as e:
                return error_response(400, "invalid_parameter", str(e))
            if req.response_format == "frames_b64":
                # encoded is list[str]; expand into per-frame data entries
                for s in encoded:
                    data.append({"b64_json": s})
            else:
                data.append({
                    "b64_json": base64.b64encode(encoded).decode("ascii"),
                })

        # Use the first result for top-level metadata (n=1 is common)
        head = results[0]
        body = {
            "data": data,
            "model": model.model_id,
            "metadata": {
                "frames": len(head.frames),
                "fps": head.fps,
                "duration_seconds": round(len(head.frames) / max(head.fps, 1), 3),
                "format": req.response_format,
                "size": [head.width, head.height],
            },
        }
        return JSONResponse(content=body)

    return router


def _encode(fmt, result, *, loop):
    if fmt == "webp":
        return encode_webp(result.frames, result.fps, loop=loop)
    if fmt == "gif":
        return encode_gif(result.frames, result.fps, loop=loop)
    if fmt == "mp4":
        return encode_mp4(result.frames, result.fps)
    if fmt == "frames_b64":
        return encode_frames_b64(result.frames)
    raise ValueError(f"unknown format: {fmt}")
```

Update `src/muse/modalities/image_animation/__init__.py`:

```python
"""image/animation modality.

Public surface:
  - MODALITY: str (MIME-shaped tag; used by discover_modalities)
  - build_router(registry) -> APIRouter (mounted by the worker)
  - AnimationModel Protocol, AnimationResult dataclass
  - AnimationsClient (HTTP)

Wire contract: POST /v1/images/animations
"""
from muse.modalities.image_animation.protocol import (
    AnimationModel,
    AnimationResult,
)
from muse.modalities.image_animation.routes import build_router

MODALITY = "image/animation"

__all__ = [
    "MODALITY",
    "build_router",
    "AnimationModel",
    "AnimationResult",
]
```

(Client gets added in Task D.)

- [ ] **Step 4: Run, expect pass**

```bash
pytest tests/modalities/image_animation/test_routes.py -v
```

Expected: 9 passed.

- [ ] **Step 5: Run full fast lane** (also verifies MODALITY is discovered)

- [ ] **Step 6: Commit**

```bash
git add src/muse/modalities/image_animation/__init__.py \
        src/muse/modalities/image_animation/routes.py \
        tests/modalities/image_animation/test_routes.py
git commit -m "feat(image-animation): POST /v1/images/animations route (#144)

Capability gates: supports_image_to_animation gates image input
(returns 400 if model is text-only and image is sent). Encodes
per response_format: webp (default), gif, mp4 (lazy imageio), or
frames_b64 (per-frame PNG list). Loop default True for webp/gif.

Modality __init__.py exports MODALITY + build_router so
discover_modalities picks it up."
```

---

## Task D: AnimationsClient

Mirror `SpeechClient` / `GenerationsClient` shape.

**Files:**
- Create: `src/muse/modalities/image_animation/client.py`
- Create: `tests/modalities/image_animation/test_client.py`
- Modify: `src/muse/modalities/image_animation/__init__.py` (export AnimationsClient)

- [ ] **Step 1: Write the failing client test**

Create `tests/modalities/image_animation/test_client.py`:

```python
"""Tests for AnimationsClient HTTP client."""
import base64
from unittest.mock import patch, MagicMock

from muse.modalities.image_animation.client import AnimationsClient


def test_client_returns_webp_bytes_by_default():
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {
        "data": [{"b64_json": base64.b64encode(b"RIFFfakeWEBP").decode()}],
        "metadata": {"format": "webp"},
    }
    with patch(
        "muse.modalities.image_animation.client.requests.post",
        return_value=fake_resp,
    ) as mock_post:
        c = AnimationsClient(server_url="http://x")
        out = c.animate("a cat", model="anim")
    assert out == b"RIFFfakeWEBP"
    payload = mock_post.call_args.kwargs["json"]
    assert payload["prompt"] == "a cat"
    assert payload["model"] == "anim"


def test_client_response_format_frames_returns_list_of_pngs():
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {
        "data": [
            {"b64_json": base64.b64encode(b"png1").decode()},
            {"b64_json": base64.b64encode(b"png2").decode()},
        ],
        "metadata": {"format": "frames_b64"},
    }
    with patch(
        "muse.modalities.image_animation.client.requests.post",
        return_value=fake_resp,
    ):
        c = AnimationsClient(server_url="http://x")
        out = c.animate("x", response_format="frames_b64")
    assert out == [b"png1", b"png2"]


def test_client_passes_optional_fields():
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {
        "data": [{"b64_json": base64.b64encode(b"x").decode()}],
        "metadata": {"format": "webp"},
    }
    with patch(
        "muse.modalities.image_animation.client.requests.post",
        return_value=fake_resp,
    ) as mock_post:
        c = AnimationsClient(server_url="http://x")
        c.animate(
            "x", model="m", frames=24, fps=12, loop=False,
            negative_prompt="bad", steps=30, guidance=8.0, seed=7,
        )
    payload = mock_post.call_args.kwargs["json"]
    assert payload["frames"] == 24
    assert payload["fps"] == 12
    assert payload["loop"] is False
    assert payload["negative_prompt"] == "bad"
    assert payload["seed"] == 7


def test_client_raises_on_non_200():
    fake_resp = MagicMock()
    fake_resp.status_code = 400
    fake_resp.text = '{"error": {"message": "bad"}}'
    with patch(
        "muse.modalities.image_animation.client.requests.post",
        return_value=fake_resp,
    ):
        c = AnimationsClient(server_url="http://x")
        try:
            c.animate("x")
        except RuntimeError as e:
            assert "400" in str(e)
        else:
            assert False, "expected RuntimeError"
```

- [ ] **Step 2: Run, expect ImportError**

- [ ] **Step 3: Implement client**

Create `src/muse/modalities/image_animation/client.py`:

```python
"""HTTP client for /v1/images/animations.

By default returns the encoded animation bytes (webp/gif/mp4) for the
configured response_format. For response_format='frames_b64', returns
list[bytes] (one PNG per frame).
"""
from __future__ import annotations

import base64
import os
from typing import Any

import requests


class AnimationsClient:
    def __init__(self, server_url: str | None = None, timeout: float = 300.0) -> None:
        server_url = server_url or os.environ.get(
            "MUSE_SERVER", "http://localhost:8000",
        )
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout

    def animate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        n: int = 1,
        frames: int | None = None,
        fps: int | None = None,
        loop: bool | None = None,
        negative_prompt: str | None = None,
        steps: int | None = None,
        guidance: float | None = None,
        seed: int | None = None,
        image: str | None = None,
        strength: float | None = None,
        response_format: str = "webp",
        size: str | None = None,
    ) -> Any:
        body: dict = {"prompt": prompt, "n": n, "response_format": response_format}
        for k, v in [
            ("model", model), ("frames", frames), ("fps", fps), ("loop", loop),
            ("negative_prompt", negative_prompt), ("steps", steps),
            ("guidance", guidance), ("seed", seed), ("image", image),
            ("strength", strength), ("size", size),
        ]:
            if v is not None:
                body[k] = v

        r = requests.post(
            f"{self.server_url}/v1/images/animations",
            json=body, timeout=self.timeout,
        )
        if r.status_code != 200:
            raise RuntimeError(f"server returned {r.status_code}: {r.text[:500]}")
        payload = r.json()
        data = payload["data"]
        if response_format == "frames_b64":
            return [base64.b64decode(e["b64_json"]) for e in data]
        return base64.b64decode(data[0]["b64_json"])
```

- [ ] **Step 4: Run client tests, expect pass**

- [ ] **Step 5: Update __init__**

Add `AnimationsClient` to the imports and `__all__` in `src/muse/modalities/image_animation/__init__.py`.

- [ ] **Step 6: Run full fast lane**

- [ ] **Step 7: Commit**

```bash
git add src/muse/modalities/image_animation/client.py \
        src/muse/modalities/image_animation/__init__.py \
        tests/modalities/image_animation/test_client.py
git commit -m "feat(image-animation): AnimationsClient (#144)"
```

---

## Task E: Bundled script `animatediff_motion_v3`

The trickiest file because of 2-component handling. The `hf_repo` field points at the motion adapter (the small ~50MB component); the base SD 1.5 is referenced by a `base_model` capability and fetched at construction time if not in cache.

**Files:**
- Create: `src/muse/models/animatediff_motion_v3.py`
- Create: `tests/models/test_animatediff_motion_v3.py`

- [ ] **Step 1: Write the failing test**

Create `tests/models/test_animatediff_motion_v3.py`:

```python
"""Tests for the bundled animatediff_motion_v3 script."""
from unittest.mock import MagicMock, patch

import pytest

from muse.models.animatediff_motion_v3 import MANIFEST, Model
from muse.modalities.image_animation.protocol import AnimationResult


def test_manifest_shape():
    assert MANIFEST["model_id"] == "animatediff-motion-v3"
    assert MANIFEST["modality"] == "image/animation"
    assert MANIFEST["hf_repo"] == "guoyww/animatediff-motion-adapter-v1-5-3"
    caps = MANIFEST["capabilities"]
    assert caps["supports_text_to_animation"] is True
    assert caps["supports_image_to_animation"] is False
    assert caps["default_frames"] == 16
    assert caps["default_fps"] == 8
    assert caps["device"] == "cuda"
    # Base model is referenced
    assert "base_model" in caps


def _patched_pipe():
    fake_pipe = MagicMock()
    fake_frame = MagicMock()
    fake_frame.size = (512, 512)
    fake_pipe.return_value.frames = [[fake_frame] * 16]
    return fake_pipe


def test_construction_loads_adapter_and_base():
    fake_pipe_class = MagicMock()
    fake_pipe_class.from_pretrained.return_value = _patched_pipe()
    fake_adapter_class = MagicMock()
    fake_adapter_class.from_pretrained.return_value = MagicMock()

    with patch(
        "muse.models.animatediff_motion_v3.AnimateDiffPipeline",
        fake_pipe_class,
    ), patch(
        "muse.models.animatediff_motion_v3.MotionAdapter",
        fake_adapter_class,
    ), patch(
        "muse.models.animatediff_motion_v3.torch",
        MagicMock(),
    ):
        m = Model(
            hf_repo="guoyww/animatediff-motion-adapter-v1-5-3",
            local_dir="/fake/adapter",
            device="cpu",
        )
    fake_adapter_class.from_pretrained.assert_called_once()
    fake_pipe_class.from_pretrained.assert_called_once()
    # The pipe is loaded with base_model from MANIFEST capabilities
    pipe_call = fake_pipe_class.from_pretrained.call_args
    assert pipe_call.args[0] == MANIFEST["capabilities"]["base_model"]
    assert m.model_id == MANIFEST["model_id"]


def test_generate_returns_animation_result():
    fake_pipe = _patched_pipe()
    fake_pipe_class = MagicMock()
    fake_pipe_class.from_pretrained.return_value = fake_pipe
    fake_adapter_class = MagicMock()
    fake_adapter_class.from_pretrained.return_value = MagicMock()

    with patch(
        "muse.models.animatediff_motion_v3.AnimateDiffPipeline",
        fake_pipe_class,
    ), patch(
        "muse.models.animatediff_motion_v3.MotionAdapter",
        fake_adapter_class,
    ), patch(
        "muse.models.animatediff_motion_v3.torch",
        MagicMock(),
    ):
        m = Model(hf_repo="x", local_dir="/fake", device="cpu")
        r = m.generate("a cat")

    assert isinstance(r, AnimationResult)
    assert len(r.frames) == 16
    assert r.fps == MANIFEST["capabilities"]["default_fps"]
```

- [ ] **Step 2: Run, expect ImportError**

- [ ] **Step 3: Implement bundled script**

Create `src/muse/models/animatediff_motion_v3.py`:

```python
"""AnimateDiff motion v3 + SD 1.5 base.

Two-component model. `muse pull animatediff-motion-v3` fetches the
motion adapter (~50MB). On first construction, the SD 1.5 base
(emilianJR/epiCRealism, ~3GB) is fetched if not already in the
HuggingFace cache. Subsequent constructions are warm.

Trade-off: muse pull is fast and small, but first-request cold start
may take 30-60s on a fresh machine while the base downloads. After
that, both components are cached locally.
"""
from __future__ import annotations

import logging
from typing import Any

from muse.modalities.image_animation.protocol import AnimationResult


logger = logging.getLogger(__name__)


# Sentinels (lazy-import pattern matches sd_turbo).
torch: Any = None
AnimateDiffPipeline: Any = None
MotionAdapter: Any = None


def _ensure_deps() -> None:
    global torch, AnimateDiffPipeline, MotionAdapter
    if torch is None:
        try:
            import torch as _t
            torch = _t
        except Exception as e:  # noqa: BLE001
            logger.debug("animatediff_motion_v3: torch unavailable: %s", e)
    if AnimateDiffPipeline is None:
        try:
            from diffusers import AnimateDiffPipeline as _p
            AnimateDiffPipeline = _p
        except Exception as e:  # noqa: BLE001
            logger.debug("animatediff_motion_v3: AnimateDiffPipeline unavailable: %s", e)
    if MotionAdapter is None:
        try:
            from diffusers import MotionAdapter as _m
            MotionAdapter = _m
        except Exception as e:  # noqa: BLE001
            logger.debug("animatediff_motion_v3: MotionAdapter unavailable: %s", e)


MANIFEST = {
    "model_id": "animatediff-motion-v3",
    "modality": "image/animation",
    "hf_repo": "guoyww/animatediff-motion-adapter-v1-5-3",
    "description": "AnimateDiff motion v3 + SD 1.5 base, 16 frames @ 8fps, 512x512",
    "license": "Apache 2.0",
    "pip_extras": (
        "torch>=2.1.0",
        "diffusers>=0.27.0",
        "transformers>=4.36.0",
        "accelerate",
        "Pillow>=9.1.0",
        "safetensors",
    ),
    "system_packages": (),
    "capabilities": {
        "supports_text_to_animation": True,
        "supports_image_to_animation": False,
        "default_frames": 16,
        "default_fps": 8,
        "min_frames": 8,
        "max_frames": 24,
        "default_size": (512, 512),
        "default_steps": 25,
        "default_guidance": 7.5,
        "device": "cuda",
        "base_model": "emilianJR/epiCRealism",
    },
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
    """AnimateDiff motion v3 backend.

    The catalog passes hf_repo (motion adapter) + local_dir (cached
    adapter weights) + device (resolved per capability). The base SD 1.5
    is read from MANIFEST capabilities[base_model]; diffusers fetches
    it on first construction if not in cache.
    """

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
        if AnimateDiffPipeline is None or MotionAdapter is None:
            raise RuntimeError(
                "diffusers AnimateDiff is not installed; run "
                "`muse pull animatediff-motion-v3`"
            )
        caps = MANIFEST["capabilities"]
        self._default_frames = caps["default_frames"]
        self._default_fps = caps["default_fps"]
        self._default_size = tuple(caps["default_size"])
        self._default_steps = caps["default_steps"]
        self._default_guidance = caps["default_guidance"]
        self._device = _select_device(device)

        import muse.models.animatediff_motion_v3 as _mod
        _torch = _mod.torch
        torch_dtype = None
        if _torch is not None:
            torch_dtype = {
                "float16": _torch.float16,
                "float32": _torch.float32,
                "bfloat16": _torch.bfloat16,
            }[dtype]

        adapter_src = local_dir or hf_repo
        logger.info("loading MotionAdapter from %s", adapter_src)
        adapter = MotionAdapter.from_pretrained(adapter_src, torch_dtype=torch_dtype)

        base = caps["base_model"]
        logger.info(
            "loading AnimateDiffPipeline base=%s + adapter (device=%s, dtype=%s) "
            "(first run downloads base if not cached)",
            base, self._device, dtype,
        )
        self._pipe = AnimateDiffPipeline.from_pretrained(
            base,
            motion_adapter=adapter,
            torch_dtype=torch_dtype,
        )
        if self._device != "cpu":
            self._pipe = self._pipe.to(self._device)

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
        init_image: Any = None,
        strength: float | None = None,
        **_: Any,
    ) -> AnimationResult:
        if init_image is not None:
            raise NotImplementedError(
                "animatediff-motion-v3 does not support init_image; route layer "
                "should have gated this via supports_image_to_animation"
            )
        n_frames = frames if frames is not None else self._default_frames
        out_fps = fps if fps is not None else self._default_fps
        w = width or self._default_size[0]
        h = height or self._default_size[1]
        n_steps = steps if steps is not None else self._default_steps
        cfg = guidance if guidance is not None else self._default_guidance

        gen = None
        if seed is not None:
            import muse.models.animatediff_motion_v3 as _mod
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
        return AnimationResult(
            frames=list(frames_list),
            fps=out_fps,
            width=first.size[0],
            height=first.size[1],
            seed=seed if seed is not None else -1,
            metadata={
                "prompt": prompt,
                "frames": n_frames,
                "fps": out_fps,
                "steps": n_steps,
                "guidance": cfg,
                "model": self.model_id,
                "base_model": MANIFEST["capabilities"]["base_model"],
            },
        )
```

- [ ] **Step 4: Run, expect pass**

- [ ] **Step 5: Run full fast lane**

- [ ] **Step 6: Commit**

```bash
git add src/muse/models/animatediff_motion_v3.py \
        tests/models/test_animatediff_motion_v3.py
git commit -m "feat(animation): bundled animatediff_motion_v3 script (#144)

Two-component model: hf_repo points at the motion adapter (~50MB);
base SD 1.5 is referenced via capabilities[base_model] and fetched
on first construction if not cached. Trade-off documented: muse pull
is small/fast, first construction may take 30-60s for base download.

Capability device=cuda (AnimateDiff is too slow on CPU to be useful).
"
```

---

## Task F: HF plugin (fused-checkpoint AnimateDiff)

**Files:**
- Create: `src/muse/modalities/image_animation/hf.py`
- Create: `tests/modalities/image_animation/test_hf_plugin.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/modalities/image_animation/test_hf_plugin.py` with 8 tests covering: required keys, metadata, sniff (true on text-to-video tag + animatediff-shape; false on plain text-to-image), resolve (animatelcm pattern), search.

(Reference shape: see `tests/modalities/image_generation/test_hf_plugin.py`. Same structure.)

- [ ] **Step 2: Run, expect ImportError**

- [ ] **Step 3: Implement plugin**

Create `src/muse/modalities/image_animation/hf.py` following the established pattern. Key bits:

- `_RUNTIME_PATH = "muse.modalities.image_animation.runtimes.animatediff:AnimateDiffRuntime"`
- `_PIP_EXTRAS = ("torch>=2.1.0", "diffusers>=0.27.0", "transformers>=4.36.0", "accelerate", "Pillow>=9.1.0", "safetensors")`
- `priority = 110` (medium specific: tag + repo-name pattern)
- `_sniff(info)`: True iff `model_index.json` sibling AND `text-to-video` in tags AND repo name contains `animate` or `motion` (case-insensitive)
- `_infer_defaults(repo_id)`: animatelcm gets steps=4 guidance=1.0; default animatediff gets steps=25 guidance=7.5
- `_resolve` synthesizes manifest with `base_model` set if pattern matches a known fused checkpoint; otherwise omitted (the runtime would need an explicit base from elsewhere; for v1 we only support patterns we recognize)

Reasonable v1 sniff: only AnimateLCM-class fused checkpoints are claimed. Other AnimateDiff repos that need an explicit base+adapter pairing don't match (and would return "no plugin matched" with a descriptive message).

- [ ] **Step 4: Run, expect pass**

- [ ] **Step 5: Verify plugin discovery sees 6 plugins now**

```bash
python -c "from muse.core.resolvers_hf import HFResolver; print([(p['priority'], p['modality']) for p in HFResolver()._plugins])"
```

Expected: 6 entries; `image/animation` at priority 110.

- [ ] **Step 6: Run full fast lane, commit**

---

## Task G: Curated entries

- [ ] Add `animatediff-motion-v3` (bundled) and `animatelcm` (uri) to `src/muse/curated.yaml`.
- [ ] Add a test in `tests/core/test_curated.py` asserting both are present with correct fields.
- [ ] Run + commit.

---

## Task H: Slow e2e + integration tests

- [ ] **Slow e2e**: extend `tests/cli_impl/test_e2e_supervisor.py` (the `@pytest.mark.slow` test) to include `image/animation` in the modalities discovered. Mock the runtime so no real diffusion runs but verify the route mounts and a request returns the right envelope shape.

- [ ] **Integration**: create `tests/integration/test_remote_animations.py` mirroring `test_remote_moderations.py`. Tests are opt-in via `MUSE_REMOTE_SERVER`. Cover: webp default, gif format, frames_b64 format, seed reproducibility, capability gate (image input on text-only model returns 400).

- [ ] Run, commit.

---

## Task I: Docs + v0.18.0 release

- [ ] **Update `CLAUDE.md`**: add `image/animation` to the modality list at the top. Note v0.18.0 in version reference. Document the bundled `animatediff-motion-v3` (with the cold-start caveat) and the `animatelcm` curated id.
- [ ] **Update `README.md`**: add the `/v1/images/animations` route to the bullet list.
- [ ] **Update `src/muse/__init__.py` docstring**: "As of v0.18.0" + add `image/animation: /v1/images/animations`.
- [ ] **Bump `pyproject.toml` to 0.18.0**. Add `imageio[ffmpeg]` to the `images` extras as optional (so `mp4` response_format works for users who install with `[images]`).
- [ ] Run full test suite (slow lane included).
- [ ] Em-dash check on all changed files.
- [ ] Commit:
  ```
  chore(release): v0.18.0
  
  image/animation modality (#144). POST /v1/images/animations.
  Bundled animatediff-motion-v3 (SD 1.5 + motion adapter v3,
  16 frames @ 8fps @ 512x512). HF plugin sniffs fused-checkpoint
  AnimateDiff variants (AnimateLCM curated). Default response
  format: animated WebP. Alternates: GIF, MP4 (requires
  imageio[ffmpeg]), frames_b64.
  
  Closes #144.
  ```
- [ ] Tag `v0.18.0`, push commits + tag, create GitHub release with notes.

---

## Self-review checklist

1. **Spec coverage:** every spec section has a corresponding task. Wire contract -> Task C. Capability flags -> Task C+E. Codec -> Task A. Runtime -> Task B+E. Plugin -> Task F. Curated -> Task G. Tests -> all tasks. Docs + release -> Task I.
2. **Placeholder scan:** zero TBD/TODO/XXX/FIXME outside the self-review meta-text.
3. **Type consistency:** `frames`, `fps`, `seed`, `init_image`, `strength` flow through pydantic -> runtime kwargs -> diffusers pipe call with same names. `base_model` is consistent (capability key) between bundled script and runtime.
4. **Migration safety:** purely additive. No existing modality changes. No existing route changes.
5. **Behavior preservation:** existing 754 fast-lane tests must keep passing through every task.
6. **Two-component download caveat:** documented in Task E (animatediff_motion_v3 bundled script). First-cold-start may take 30-60s for the base model download. Acceptable for v1.
