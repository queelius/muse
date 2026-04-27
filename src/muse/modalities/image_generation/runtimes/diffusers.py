"""Generic text-to-image runtime via diffusers.AutoPipelineForText2Image.

One runtime serves many models. The model_id, default_size, default_steps,
and default_guidance are injected at construction time from manifest
capabilities. Heavy imports (torch, diffusers) are lazy: discovery must
work without them on the host python.

Mirrors the lazy-import pattern from src/muse/models/sd_turbo.py: tests
patch the module-level `torch` and `AutoPipelineForText2Image` sentinels
directly; `_ensure_deps()` short-circuits when sentinels are non-None
(mocked) so the patches survive.
"""
from __future__ import annotations

import logging
import math
from typing import Any

from muse.modalities.image_generation.protocol import ImageResult


logger = logging.getLogger(__name__)


# Sentinels patched by tests; populated by _ensure_deps at runtime.
torch: Any = None
AutoPipelineForText2Image: Any = None
AutoPipelineForImage2Image: Any = None


def _ensure_deps() -> None:
    global torch, AutoPipelineForText2Image, AutoPipelineForImage2Image
    if torch is None:
        try:
            import torch as _t
            torch = _t
        except Exception as e:  # noqa: BLE001
            logger.debug("diffusers runtime: torch unavailable: %s", e)
    if AutoPipelineForText2Image is None:
        try:
            from diffusers import AutoPipelineForText2Image as _p
            AutoPipelineForText2Image = _p
        except Exception as e:  # noqa: BLE001
            logger.debug("diffusers runtime: AutoPipelineForText2Image unavailable: %s", e)
    if AutoPipelineForImage2Image is None:
        try:
            from diffusers import AutoPipelineForImage2Image as _p
            AutoPipelineForImage2Image = _p
        except Exception as e:  # noqa: BLE001
            logger.debug("diffusers runtime: AutoPipelineForImage2Image unavailable: %s", e)


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


class DiffusersText2ImageModel:
    """Text-to-image runtime backed by diffusers.AutoPipelineForText2Image.

    Construction kwargs (set by catalog at load_backend time, sourced from
    manifest fields and capabilities):
      - hf_repo, local_dir, device, dtype: standard
      - model_id: catalog id (response envelope echoes this)
      - default_size: (width, height) when request omits size
      - default_steps: num_inference_steps default
      - default_guidance: guidance_scale default
    """

    def __init__(
        self,
        *,
        hf_repo: str,
        local_dir: str | None = None,
        device: str = "auto",
        dtype: str = "float16",
        model_id: str,
        default_size: tuple[int, int] = (512, 512),
        default_steps: int = 1,
        default_guidance: float = 0.0,
        **_: Any,
    ) -> None:
        _ensure_deps()
        if AutoPipelineForText2Image is None:
            raise RuntimeError(
                "diffusers is not installed; ensure muse[images] extras are "
                "installed in the per-model venv"
            )
        self.model_id = model_id
        # JSON round-trips turn (512, 512) into [512, 512]; coerce back to
        # the documented tuple shape so consumers can rely on it.
        self.default_size = tuple(default_size)
        self._default_steps = default_steps
        self._default_guidance = default_guidance
        self._device = _select_device(device)
        # Stash for lazy img2img pipeline load (same checkpoint + dtype).
        self._src = local_dir or hf_repo
        self._dtype = dtype
        self._i2i_pipe = None

        # Access torch through this module so tests' patches survive.
        import muse.modalities.image_generation.runtimes.diffusers as _mod
        _torch = _mod.torch
        torch_dtype = None
        if _torch is not None:
            torch_dtype = {
                "float16": _torch.float16,
                "float32": _torch.float32,
                "bfloat16": _torch.bfloat16,
            }[dtype]
        self._torch_dtype = torch_dtype

        logger.info(
            "loading diffusers pipeline from %s (model_id=%s, device=%s, dtype=%s)",
            self._src, model_id, self._device, dtype,
        )
        self._pipe = AutoPipelineForText2Image.from_pretrained(
            self._src,
            torch_dtype=torch_dtype,
            variant="fp16" if dtype == "float16" else None,
        )
        if self._device != "cpu":
            self._pipe = self._pipe.to(self._device)

    def generate(
        self,
        prompt: str,
        *,
        negative_prompt: str | None = None,
        width: int | None = None,
        height: int | None = None,
        steps: int | None = None,
        guidance: float | None = None,
        seed: int | None = None,
        init_image: Any = None,
        strength: float | None = None,
        **_: Any,
    ) -> ImageResult:
        if init_image is not None:
            return self._generate_img2img(
                prompt,
                init_image=init_image,
                strength=strength,
                negative_prompt=negative_prompt,
                steps=steps,
                guidance=guidance,
                seed=seed,
            )
        w = width or self.default_size[0]
        h = height or self.default_size[1]
        n_steps = steps if steps is not None else self._default_steps
        cfg = guidance if guidance is not None else self._default_guidance

        gen = None
        if seed is not None:
            import muse.modalities.image_generation.runtimes.diffusers as _mod
            _torch = _mod.torch
            if _torch is not None:
                gen = _torch.Generator(device=self._device).manual_seed(seed)

        call_kwargs: dict = {
            "prompt": prompt,
            "width": w,
            "height": h,
            "num_inference_steps": n_steps,
            "guidance_scale": cfg,
        }
        if negative_prompt is not None:
            call_kwargs["negative_prompt"] = negative_prompt
        if gen is not None:
            call_kwargs["generator"] = gen

        out = self._pipe(**call_kwargs)
        img = out.images[0]
        return ImageResult(
            image=img,
            width=img.size[0],
            height=img.size[1],
            seed=seed if seed is not None else -1,
            metadata={
                "prompt": prompt,
                "steps": n_steps,
                "guidance": cfg,
                "model": self.model_id,
            },
        )

    def _generate_img2img(
        self,
        prompt: str,
        *,
        init_image: Any,
        strength: float | None,
        negative_prompt: str | None,
        steps: int | None,
        guidance: float | None,
        seed: int | None,
    ) -> ImageResult:
        # Lazy-load the img2img pipeline, cached on the instance. Diffusers
        # shares the underlying model objects between pipelines loaded from
        # the same checkpoint, so the marginal cost is small.
        if self._i2i_pipe is None:
            import muse.modalities.image_generation.runtimes.diffusers as _mod
            _i2i = _mod.AutoPipelineForImage2Image
            if _i2i is None:
                raise RuntimeError(
                    "diffusers AutoPipelineForImage2Image is not available; "
                    "ensure muse[images] extras are installed in the per-model venv"
                )
            logger.info(
                "loading img2img pipeline from %s (model_id=%s, device=%s, dtype=%s)",
                self._src, self.model_id, self._device, self._dtype,
            )
            pipe = _i2i.from_pretrained(
                self._src,
                torch_dtype=self._torch_dtype,
                variant="fp16" if self._dtype == "float16" else None,
            )
            if self._device != "cpu":
                pipe = pipe.to(self._device)
            self._i2i_pipe = pipe

        n_steps = steps if steps is not None else self._default_steps
        cfg = guidance if guidance is not None else self._default_guidance
        s = strength if strength is not None else 0.5

        # img2img diffusers contract: num_inference_steps * strength must be >= 1
        # (lower values round to 0 effective denoise steps and crash the VAE).
        # Bump steps to satisfy the contract while preserving the user's
        # requested strength.
        min_steps_for_strength = max(1, math.ceil(1.0 / max(s, 0.01)))
        if n_steps < min_steps_for_strength:
            logger.info(
                "img2img bumping num_inference_steps from %d to %d to satisfy "
                "strength=%.2f * steps >= 1 contract",
                n_steps, min_steps_for_strength, s,
            )
            n_steps = min_steps_for_strength

        gen = None
        if seed is not None:
            import muse.modalities.image_generation.runtimes.diffusers as _mod
            _torch = _mod.torch
            if _torch is not None:
                gen = _torch.Generator(device=self._device).manual_seed(seed)

        call_kwargs: dict = {
            "prompt": prompt,
            "image": init_image,
            "strength": s,
            "num_inference_steps": n_steps,
            "guidance_scale": cfg,
        }
        if negative_prompt is not None:
            call_kwargs["negative_prompt"] = negative_prompt
        if gen is not None:
            call_kwargs["generator"] = gen

        out = self._i2i_pipe(**call_kwargs)
        img = out.images[0]
        return ImageResult(
            image=img,
            width=img.size[0],
            height=img.size[1],
            seed=seed if seed is not None else -1,
            metadata={
                "prompt": prompt,
                "steps": n_steps,
                "guidance": cfg,
                "strength": s,
                "model": self.model_id,
                "mode": "img2img",
            },
        )
