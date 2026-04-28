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
