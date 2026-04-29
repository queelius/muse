"""Generic CogVideoX runtime via diffusers.CogVideoXPipeline.

CogVideoX-2b / 5b are THUDM's transformer-based T2V models. Pipeline
defaults: 49 frames at 8fps for ~6s clips at 720x480.

Lazy-import sentinel pattern matches WanRuntime; the two runtimes
share enough that a base class could merge them in v1.next, but for
v0.27.0 they live separately for clarity.
"""
from __future__ import annotations

import logging
from typing import Any

from muse.modalities.video_generation.protocol import VideoResult


logger = logging.getLogger(__name__)


# Sentinels patched by tests; populated by _ensure_deps at runtime.
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
    """CogVideoX runtime over diffusers.CogVideoXPipeline.

    Construction kwargs:
      - hf_repo: model repo id (manifest's hf_repo)
      - local_dir: local cache for weights (from `muse pull`)
      - device, dtype, model_id: standard
      - default_duration_seconds, default_fps, default_size, default_steps,
        default_guidance: manifest-driven defaults injected via the
        capabilities splat
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
            "loading CogVideoXPipeline from %s "
            "(model_id=%s, device=%s, dtype=%s)",
            src, model_id, self._device, dtype,
        )
        self._pipe = CogVideoXPipeline.from_pretrained(
            src, torch_dtype=torch_dtype,
        )
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
        dur = (
            duration_seconds
            if duration_seconds is not None
            else self._default_duration
        )
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
        # CogVideoX's pipeline returns out.frames as list-of-lists (one
        # video, multiple frames). Take the first video.
        frames_list = out.frames[0]
        first = frames_list[0]
        actual_frames = len(frames_list)
        actual_duration = round(actual_frames / max(out_fps, 1), 3)
        first_size = getattr(first, "size", (w, h))
        first_w, first_h = int(first_size[0]), int(first_size[1])
        return VideoResult(
            frames=list(frames_list),
            fps=out_fps,
            width=first_w,
            height=first_h,
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
