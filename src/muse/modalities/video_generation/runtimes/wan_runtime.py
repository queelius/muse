"""Generic Wan runtime via diffusers.WanPipeline (or DiffusionPipeline fallback).

Wan was added to diffusers in 0.32.0 as a top-level export. Older
diffusers versions can still load the model via DiffusionPipeline
since the repo's `model_index.json` points at the right pipeline
class. The runtime tries WanPipeline first and falls back to
DiffusionPipeline when WanPipeline is None.

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
            logger.debug(
                "wan: WanPipeline not in diffusers (%s); "
                "will fall back to DiffusionPipeline",
                e,
            )
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
                "diffusers is not installed; ensure muse[images] extras "
                "are installed in the per-model venv via "
                "`muse pull <model-id>`"
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
