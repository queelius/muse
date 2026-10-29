"""Reference-image diffusers runtime (FLUX.2 klein and similar).

One runtime for pipelines that take a list of reference images next to the
prompt (`Flux2KleinPipeline`, diffusers>=0.37). The pipeline class is named
by the manifest (`pipeline_class`) and looked up on the diffusers module, so
a sibling pipeline needs no new code. Text-to-image and multi-reference
edits share one pipeline object; there is no img2img/inpaint path.

LoRA adapters (``lora_adapter``) follow ``runtimes/diffusers.py``: the
pipeline loads from the resolved ``base_model`` and the adapter is layered
on unfused, before offload placement, so a per-request ``lora_scale`` can be
applied with ``set_adapters``.

Heavy imports (torch, diffusers) are lazy sentinels populated by
`_ensure_deps()`; tests patch `torch` and `diffusers` on this module.
"""
from __future__ import annotations

import inspect
import logging
from typing import Any

from muse.core.runtime_helpers import (
    LoadTimer,
    dtype_for_name,
    resolve_model_source,
    select_device,
)
from muse.modalities._common.offload import place_pipeline
from muse.modalities.image_generation.protocol import ImageResult
from muse.modalities.image_generation.runtimes._lora import resolve_lora_base


logger = logging.getLogger(__name__)

# Sentinels patched by tests; populated by _ensure_deps at runtime.
torch: Any = None
diffusers: Any = None


def _ensure_deps() -> None:
    global torch, diffusers
    if torch is None:
        try:
            import torch as _t
            torch = _t
        except Exception as e:  # noqa: BLE001
            logger.debug("reference runtime: torch unavailable: %s", e)
    if diffusers is None:
        try:
            import diffusers as _d
            diffusers = _d
        except Exception as e:  # noqa: BLE001
            logger.debug("reference runtime: diffusers unavailable: %s", e)


class DiffusersReferenceModel:
    """Text-to-image plus multi-reference editing over one diffusers pipeline.

    Construction kwargs come from the manifest and its capabilities:
      - pipeline_class: attribute of the diffusers module to load
      - default_size / default_steps / default_guidance: request defaults
      - cpu_offload ("model" | "sequential" | None) and vae_tiling:
        placement, via the shared helper; the video-only global override
        does not apply here
      - max_reference_images: advertised cap enforced by the route
      - lora_adapter: True when hf_repo/local_dir hold a LoRA adapter;
        the pipeline then loads from base_model (a muse catalog id or an
        HF repo id), with the same errors as the SDXL diffusers runtime
      - lora_scale: default adapter weight, overridable per request on
        both generate() and edit()
    """

    _ADAPTER = "lora"

    supports_reference_images = True

    def __init__(
        self,
        *,
        hf_repo: str,
        local_dir: str | None = None,
        device: str = "auto",
        dtype: str = "bfloat16",
        model_id: str,
        pipeline_class: str = "Flux2KleinPipeline",
        default_size: tuple[int, int] = (1024, 1024),
        default_steps: int = 4,
        default_guidance: float = 1.0,
        cpu_offload: Any = None,
        vae_tiling: bool = False,
        max_reference_images: int = 4,
        lora_adapter: bool = False,
        base_model: str | None = None,
        lora_scale: float = 1.0,
        **_: Any,
    ) -> None:
        _ensure_deps()
        cls = getattr(diffusers, pipeline_class, None) if diffusers is not None else None
        if cls is None:
            raise RuntimeError(
                f"diffusers.{pipeline_class} is not available; install a "
                "diffusers release that ships it (>=0.37 for Flux2KleinPipeline)"
            )
        self.model_id = model_id
        self.default_size = tuple(default_size)
        self.max_reference_images = max_reference_images
        self._default_steps = default_steps
        self._default_guidance = default_guidance
        self._device = select_device(device, torch_module=torch)
        self._lora_adapter = bool(lora_adapter)
        self._default_lora_scale = float(lora_scale)
        # The adapter weight last handed to set_adapters; None forces the
        # first LoRA request to set it explicitly.
        self._applied_lora_scale: float | None = None

        self._src = local_dir or hf_repo
        if self._lora_adapter:
            # Module-level name so tests' patches of resolve_model_source apply.
            load_src = resolve_lora_base(
                model_id, base_model, resolve=resolve_model_source,
            )
        else:
            load_src = self._src
        logger.info(
            "loading %s from %s (model_id=%s, device=%s, dtype=%s)",
            pipeline_class, load_src, model_id, self._device, dtype,
        )
        with LoadTimer(model_id, logger):
            pipe = cls.from_pretrained(
                load_src, torch_dtype=dtype_for_name(dtype, torch),
            )
            if self._lora_adapter:
                # Unfused, and before placement: offload hooks then wrap
                # the LoRA-injected modules, and set_adapters can still
                # change the weight per request.
                logger.info(
                    "loading LoRA adapter %s onto base %s", self._src, base_model,
                )
                pipe.load_lora_weights(self._src, adapter_name=self._ADAPTER)
            self._pipe = place_pipeline(
                pipe, self._device,
                cpu_offload=cpu_offload, vae_tiling=vae_tiling,
                override_key=None,
            )
        try:
            params = inspect.signature(type(self._pipe).__call__).parameters
        except (TypeError, ValueError):
            params = {}
        self._accepts_negative = "negative_prompt" in params

    def _run(
        self,
        prompt: str,
        *,
        images: list | None,
        negative_prompt: str | None,
        width: int | None,
        height: int | None,
        steps: int | None,
        guidance: float | None,
        seed: int | None,
        lora_scale: float | None,
    ) -> ImageResult:
        lora_weight = self._apply_lora_scale(lora_scale)
        w = width or self.default_size[0]
        h = height or self.default_size[1]
        n_steps = steps if steps is not None else self._default_steps
        cfg = guidance if guidance is not None else self._default_guidance
        kwargs: dict = {
            "prompt": prompt,
            "width": w,
            "height": h,
            "num_inference_steps": n_steps,
            "guidance_scale": cfg,
        }
        if images:
            kwargs["image"] = images
        if negative_prompt is not None and self._accepts_negative:
            kwargs["negative_prompt"] = negative_prompt
        if seed is not None:
            # CPU generator: reproducible regardless of offload placement.
            kwargs["generator"] = torch.Generator(device="cpu").manual_seed(seed)
        img = self._pipe(**kwargs).images[0]
        return ImageResult(
            image=img,
            width=img.size[0],
            height=img.size[1],
            seed=seed if seed is not None else -1,
            metadata={
                "prompt": prompt,
                "steps": n_steps,
                "guidance": cfg,
                "reference_images": len(images or []),
                **({"lora_scale": lora_weight} if lora_weight is not None else {}),
            },
        )

    def _apply_lora_scale(self, lora_scale: float | None) -> float | None:
        """Set the adapter weight for this call; returns it (None without LoRA).

        Callers hold the route's inference lock, so the weight cannot change
        under a running call. set_adapters only rewrites the PEFT layers'
        scaling factors, so it is skipped when the weight is unchanged.
        """
        if not self._lora_adapter:
            return None
        scale = float(lora_scale if lora_scale is not None else self._default_lora_scale)
        if scale != self._applied_lora_scale:
            self._pipe.set_adapters([self._ADAPTER], adapter_weights=[scale])
            self._applied_lora_scale = scale
        return scale

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
        lora_scale: float | None = None,
        **_: Any,
    ) -> ImageResult:
        if init_image is not None:
            raise ValueError(
                f"{self.model_id} does not support img2img; "
                "use /v1/images/edits with reference images"
            )
        return self._run(
            prompt, images=None, negative_prompt=negative_prompt,
            width=width, height=height, steps=steps, guidance=guidance,
            seed=seed, lora_scale=lora_scale,
        )

    def edit(
        self,
        prompt: str,
        *,
        images: Any,
        width: int | None = None,
        height: int | None = None,
        steps: int | None = None,
        guidance: float | None = None,
        seed: int | None = None,
        negative_prompt: str | None = None,
        lora_scale: float | None = None,
        **_: Any,
    ) -> ImageResult:
        return self._run(
            prompt, images=list(images), negative_prompt=negative_prompt,
            width=width, height=height, steps=steps, guidance=guidance,
            seed=seed, lora_scale=lora_scale,
        )
