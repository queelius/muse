"""Generic image upscale runtime via diffusers.StableDiffusionUpscalePipeline.

One runtime serves many models. The model_id, default_scale,
supported_scales, default_steps, and default_guidance are injected at
construction time from manifest capabilities.

Heavy imports (torch, diffusers) are lazy: discovery must work without
them on the host python. Tests patch the module-level `torch` and
`StableDiffusionUpscalePipeline` sentinels directly; `_ensure_deps()`
short-circuits when sentinels are non-None (mocked) so the patches
survive.
"""
from __future__ import annotations

import logging
from typing import Any

from muse.core.runtime_helpers import dtype_for_name, select_device
from muse.modalities.image_upscale.protocol import UpscaleResult


logger = logging.getLogger(__name__)


# Sentinels patched by tests; populated by _ensure_deps at runtime.
torch: Any = None
StableDiffusionUpscalePipeline: Any = None


def _ensure_deps() -> None:
    global torch, StableDiffusionUpscalePipeline
    if torch is None:
        try:
            import torch as _t
            torch = _t
        except Exception as e:  # noqa: BLE001
            logger.debug("upscale runtime: torch unavailable: %s", e)
    if StableDiffusionUpscalePipeline is None:
        try:
            from diffusers import StableDiffusionUpscalePipeline as _p
            StableDiffusionUpscalePipeline = _p
        except Exception as e:  # noqa: BLE001
            logger.debug(
                "upscale runtime: StableDiffusionUpscalePipeline unavailable: %s",
                e,
            )


def _select_device(device: str) -> str:
    """Thin delegator preserved for test imports. Real logic in runtime_helpers."""
    return select_device(device, torch_module=torch)


class DiffusersUpscaleRuntime:
    """Image super-resolution runtime backed by diffusers' upscaler pipeline.

    Construction kwargs (set by catalog at load_backend time, sourced
    from manifest fields and capabilities):
      - hf_repo, local_dir, device, dtype: standard
      - model_id: catalog id (response envelope echoes this)
      - default_scale: scale factor used when caller omits it (also the
        single supported scale for SD x4)
      - supported_scales: list[int] of permitted scale values
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
        default_scale: int = 4,
        supported_scales: list[int] | None = None,
        default_steps: int = 20,
        default_guidance: float = 9.0,
        **_: Any,
    ) -> None:
        _ensure_deps()
        if StableDiffusionUpscalePipeline is None:
            raise RuntimeError(
                "diffusers is not installed; ensure muse[images] extras are "
                "installed in the per-model venv"
            )
        self.model_id = model_id
        self._default_scale = int(default_scale)
        self._supported_scales = (
            [int(s) for s in supported_scales]
            if supported_scales
            else [self._default_scale]
        )
        self._default_steps = default_steps
        self._default_guidance = default_guidance
        self._device = _select_device(device)
        self._src = local_dir or hf_repo
        self._dtype = dtype

        # Access torch through this module so tests' patches survive.
        import muse.modalities.image_upscale.runtimes.diffusers_upscaler as _mod
        _torch = _mod.torch
        torch_dtype = dtype_for_name(dtype, _torch)
        self._torch_dtype = torch_dtype

        logger.info(
            "loading diffusers upscale pipeline from %s "
            "(model_id=%s, device=%s, dtype=%s)",
            self._src, model_id, self._device, dtype,
        )
        self._pipe = StableDiffusionUpscalePipeline.from_pretrained(
            self._src,
            torch_dtype=torch_dtype,
        )
        if self._device != "cpu":
            self._pipe = self._pipe.to(self._device)

    @property
    def supported_scales(self) -> list[int]:
        return list(self._supported_scales)

    def upscale(
        self,
        image: Any,
        *,
        scale: int | None = None,
        prompt: str | None = None,
        negative_prompt: str | None = None,
        steps: int | None = None,
        guidance: float | None = None,
        seed: int | None = None,
        **_: Any,
    ) -> UpscaleResult:
        """Upscale an image. The route layer pre-validates `scale`."""
        # Original size taken from the caller's PIL.Image. Tests pass real
        # Image.new(...) instances, so .size is reliable.
        ow, oh = image.size

        actual_scale = int(scale) if scale is not None else self._default_scale
        n_steps = steps if steps is not None else self._default_steps
        cfg = guidance if guidance is not None else self._default_guidance
        prompt_str = prompt if prompt is not None else ""

        gen = None
        if seed is not None:
            import muse.modalities.image_upscale.runtimes.diffusers_upscaler as _mod
            _torch = _mod.torch
            if _torch is not None:
                gen = _torch.Generator(device=self._device).manual_seed(seed)

        call_kwargs: dict = {
            "prompt": prompt_str,
            "image": image,
            "num_inference_steps": n_steps,
            "guidance_scale": cfg,
        }
        if negative_prompt is not None:
            call_kwargs["negative_prompt"] = negative_prompt
        if gen is not None:
            call_kwargs["generator"] = gen

        out = self._pipe(**call_kwargs)
        upscaled = out.images[0]
        uw, uh = upscaled.size
        return UpscaleResult(
            image=upscaled,
            original_width=ow,
            original_height=oh,
            upscaled_width=uw,
            upscaled_height=uh,
            scale=actual_scale,
            seed=seed if seed is not None else -1,
            metadata={
                "prompt": prompt_str,
                "steps": n_steps,
                "guidance": cfg,
                "model": self.model_id,
            },
        )
