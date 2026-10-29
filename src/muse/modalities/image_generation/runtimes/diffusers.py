"""Generic text-to-image runtime via diffusers.AutoPipelineForText2Image.

One runtime serves many models. The model_id, default_size, default_steps,
and default_guidance are injected at construction time from manifest
capabilities. Heavy imports (torch, diffusers) are lazy: discovery must
work without them on the host python.

The bundled sd-turbo script aliases this class as its Model. Tests
patch the module-level `torch` and `AutoPipelineForText2Image` sentinels
directly; `_ensure_deps()` short-circuits when sentinels are non-None
(mocked) so the patches survive.
"""
from __future__ import annotations

import logging
import math
import threading
from pathlib import Path
from typing import Any

from muse.core.runtime_helpers import (
    dtype_for_name,
    resolve_model_source,
    select_device,
)
from muse.modalities.image_generation.protocol import ImageResult
from muse.modalities.image_generation.runtimes._lora import (
    positive_factor,
    resolve_lora_base,
    resolve_lora_scale_factor,
)


logger = logging.getLogger(__name__)


def _local_has_fp16_variant(src: str) -> bool:
    """Whether the local snapshot at ``src`` holds ``.fp16.`` variant weights.

    The image_generation HF downloader fetches ONLY the ``.fp16.`` files when
    a repo ships them (sd-turbo/sdxl-turbo) and ONLY the bare weights
    otherwise (flux-schnell and other bf16-only repos). The runtime must
    request ``variant="fp16"`` in the first case (so from_pretrained finds the
    fp16 files) and ``variant=None`` in the second (else from_pretrained
    raises because no fp16 files exist). Mirroring the downloader, detect the
    variant from the files actually on disk. Returns False when ``src`` is a
    bare repo id (not a local dir), which keeps the safe no-variant path.
    """
    p = Path(src)
    if not p.is_dir():
        return False
    return any(".fp16." in f.name for f in p.rglob("*.safetensors")) or any(
        ".fp16." in f.name for f in p.rglob("*.bin")
    )


# Sentinels patched by tests; populated by _ensure_deps at runtime.
torch: Any = None
AutoPipelineForText2Image: Any = None
AutoPipelineForImage2Image: Any = None
AutoPipelineForInpainting: Any = None


def _ensure_deps() -> None:
    global torch, AutoPipelineForText2Image, AutoPipelineForImage2Image
    global AutoPipelineForInpainting
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
    if AutoPipelineForInpainting is None:
        try:
            from diffusers import AutoPipelineForInpainting as _p
            AutoPipelineForInpainting = _p
        except Exception as e:  # noqa: BLE001
            logger.debug("diffusers runtime: AutoPipelineForInpainting unavailable: %s", e)


def _select_device(device: str) -> str:
    """Thin delegator preserved for test imports. Real logic in runtime_helpers."""
    return select_device(device, torch_module=torch)


class DiffusersText2ImageModel:
    """Text-to-image runtime backed by diffusers.AutoPipelineForText2Image.

    Construction kwargs (set by catalog at load_backend time, sourced from
    manifest fields and capabilities):
      - hf_repo, local_dir, device, dtype: standard
      - model_id: catalog id (response envelope echoes this)
      - default_size: (width, height) when request omits size
      - default_steps: num_inference_steps default
      - default_guidance: guidance_scale default
      - lora_adapter: True when hf_repo/local_dir hold a LoRA adapter,
        not a full pipeline; the base pipeline loads from base_model
      - base_model: muse catalog id or HF repo id of the base pipeline
        this adapter layers onto (required when lora_adapter is True)
      - lora_scale: default cross-attention scale for the adapter.
        Overridable per-request on /v1/images/generations, both the
        text-to-image and img2img paths (generate(..., lora_scale=...)
        with or without init_image); edits (inpaint) and variations
        (vary) have no per-request scale channel and always use this
        configured default.
      - lora_scale_factor: multiplier from lora_scale to the scale handed
        to diffusers. When not set it is derived from the adapter's
        safetensors header (runtimes/_lora.py): sqrt(r)/alpha for an
        rsLoRA adapter without .alpha tensors, else 1.0. An explicit
        value always wins
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
        lora_adapter: bool = False,
        base_model: str | None = None,
        lora_scale: float = 1.0,
        lora_scale_factor: float | None = None,
        **_: Any,
    ) -> None:
        # An explicit factor is validated before any weights load.
        if lora_scale_factor is not None:
            positive_factor(lora_scale_factor)
        _ensure_deps()
        if AutoPipelineForText2Image is None:
            raise RuntimeError(
                "diffusers is not installed; ensure museq[images] extras are "
                "installed in the per-model venv"
            )
        self.model_id = model_id
        # JSON round-trips turn (512, 512) into [512, 512]; coerce back to
        # the documented tuple shape so consumers can rely on it.
        self.default_size = tuple(default_size)
        self._default_steps = default_steps
        self._default_guidance = default_guidance
        self._device = _select_device(device)
        # Stash for lazy img2img / inpaint pipeline loads (same checkpoint + dtype).
        # For LoRA entries self._src is the ADAPTER source; the pipeline
        # itself loads from the resolved BASE below.
        self._src = local_dir or hf_repo
        self._dtype = dtype
        self._i2i_pipe = None
        self._inp_pipe = None
        # Guard the lazy img2img/inpaint builds against a double-build race
        # (two concurrent first requests both calling from_pipe, briefly
        # doubling VRAM and leaking one pipeline).
        self._i2i_lock = threading.Lock()
        self._inp_lock = threading.Lock()
        self._lora_adapter = bool(lora_adapter)
        self._default_lora_scale = float(lora_scale)
        self._lora_scale_factor = 1.0

        if self._lora_adapter:
            # Module-level name so tests' patches of resolve_model_source apply.
            load_src = resolve_lora_base(
                model_id, base_model, resolve=resolve_model_source,
            )
            self._lora_scale_factor = resolve_lora_scale_factor(
                model_id, self._src, explicit=lora_scale_factor,
            )
        else:
            load_src = self._src

        # Access torch through this module so tests' patches survive.
        import muse.modalities.image_generation.runtimes.diffusers as _mod
        _torch = _mod.torch
        torch_dtype = dtype_for_name(dtype, _torch)
        self._torch_dtype = torch_dtype

        logger.info(
            "loading diffusers pipeline from %s (model_id=%s, device=%s, dtype=%s)",
            load_src, model_id, self._device, dtype,
        )
        # Request the fp16 variant only when the local snapshot actually
        # holds .fp16. weights; otherwise from_pretrained raises for repos
        # that ship no fp16 files (e.g. flux-schnell). torch_dtype still
        # governs the compute dtype independently of variant (H6).
        use_fp16_variant = dtype == "float16" and _local_has_fp16_variant(load_src)
        self._pipe = AutoPipelineForText2Image.from_pretrained(
            load_src,
            torch_dtype=torch_dtype,
            variant="fp16" if use_fp16_variant else None,
        )
        if self._lora_adapter:
            # Unfused: fuse_lora would bake the adapter into the base
            # weights and make per-request lora_scale impossible.
            logger.info(
                "loading LoRA adapter %s onto base %s", self._src, base_model,
            )
            self._pipe.load_lora_weights(self._src)
        if self._device != "cpu":
            # Diffusers pipelines move in place. Keep the object created
            # above rather than trusting a fluent return value; wrappers and
            # test doubles need not return ``self`` from ``to()``.
            self._pipe.to(self._device)

    def _lora_weight(self, lora_scale: float | None) -> float:
        """The scale handed to diffusers: the request's lora_scale (else the
        configured default) times lora_scale_factor."""
        scale = lora_scale if lora_scale is not None else self._default_lora_scale
        return float(scale) * self._lora_scale_factor

    def _call_pipe(self, pipe: Any, call_kwargs: dict) -> Any:
        """Run one pipeline call, restoring the VAE dtype if it raises.

        SDXL-style pipelines upcast a float16 VAE to float32 for the
        decode and cast it back only after the decode returns. A failure
        mid-decode (typically CUDA OOM at a large size) leaves the shared
        VAE at float32, and every later call then fails on a Half/float
        mismatch because the pipeline no longer sees a float16 VAE to
        upcast. Restoring it keeps one failed request from breaking the
        worker for good.
        """
        try:
            return pipe(**call_kwargs)
        except BaseException:
            vae = getattr(pipe, "vae", None)
            if vae is not None and getattr(vae, "dtype", None) != self._torch_dtype:
                try:
                    vae.to(dtype=self._torch_dtype)
                except Exception:  # noqa: BLE001
                    logger.warning(
                        "%s: could not restore VAE dtype after a failed call",
                        self.model_id, exc_info=True,
                    )
            raise

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
            return self._generate_img2img(
                prompt,
                init_image=init_image,
                strength=strength,
                negative_prompt=negative_prompt,
                steps=steps,
                guidance=guidance,
                seed=seed,
                lora_scale=lora_scale,
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
        if self._lora_adapter:
            call_kwargs["cross_attention_kwargs"] = {
                "scale": self._lora_weight(lora_scale),
            }

        out = self._call_pipe(self._pipe, call_kwargs)
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
        lora_scale: float | None = None,
    ) -> ImageResult:
        # Lazy-load the img2img pipeline, cached on the instance.
        if self._i2i_pipe is None:
            with self._i2i_lock:
                if self._i2i_pipe is None:
                    import muse.modalities.image_generation.runtimes.diffusers as _mod
                    _i2i = _mod.AutoPipelineForImage2Image
                    if _i2i is None:
                        raise RuntimeError(
                            "diffusers AutoPipelineForImage2Image is not available; "
                            "ensure museq[images] extras are installed in the per-model venv"
                        )
                    logger.info(
                        "loading img2img pipeline from %s (model_id=%s, device=%s, dtype=%s)",
                        self._src, self.model_id, self._device, self._dtype,
                    )
                    # Share UNet/VAE/text-encoders with the already-loaded t2i pipeline so we
                    # don't double VRAM on small GPUs. AutoPipelineForImage2Image.from_pipe
                    # is the canonical diffusers idiom for this; from_pretrained allocates a
                    # fresh copy and OOMs.
                    # Reference: https://huggingface.co/docs/diffusers/api/pipelines/auto_pipeline
                    self._i2i_pipe = _i2i.from_pipe(self._pipe)

        n_steps = steps if steps is not None else self._default_steps
        cfg = guidance if guidance is not None else self._default_guidance
        s = strength if strength is not None else 0.5

        # img2img diffusers contract: num_inference_steps * strength must be >= 1
        # (lower values round to 0 effective denoise steps and crash the VAE).
        # Floor the strength itself, not just the step computation: the schema
        # allows strength=0.0 (ge=0.0), and with strength=0 the effective steps
        # int(num_inference_steps * 0) stay 0 no matter how high we bump steps,
        # so the timestep schedule empties and the VAE crashes. Clamping the
        # value we actually pass to the pipeline makes steps*strength >= 1
        # reachable.
        s = max(s, 0.01)
        min_steps_for_strength = max(1, math.ceil(1.0 / s))
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
        if self._lora_adapter:
            call_kwargs["cross_attention_kwargs"] = {
                "scale": self._lora_weight(lora_scale),
            }

        out = self._call_pipe(self._i2i_pipe, call_kwargs)
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

    def inpaint(
        self,
        prompt: str,
        *,
        init_image: Any,
        mask_image: Any,
        negative_prompt: str | None = None,
        width: int | None = None,
        height: int | None = None,
        steps: int | None = None,
        guidance: float | None = None,
        seed: int | None = None,
        strength: float | None = None,
        **_: Any,
    ) -> ImageResult:
        """Inpainting: regenerate the masked region per prompt.

        Mask convention follows diffusers: white pixels mark the region
        to regenerate, black pixels are preserved. The route layer hands
        us a PIL.Image; we normalize to single-channel grayscale (mode
        L) so RGB/RGBA masks (e.g. PNG with alpha) work without surprise.

        Lazy-loads AutoPipelineForInpainting via from_pipe(self._pipe)
        so the inpaint pipeline shares UNet/VAE/text-encoders with the
        already-loaded t2i pipeline. from_pretrained would allocate a
        fresh copy and OOM on small GPUs (regression seen in v0.17.2
        for img2img; same risk applies here).
        """
        if self._inp_pipe is None:
            with self._inp_lock:
                if self._inp_pipe is None:
                    import muse.modalities.image_generation.runtimes.diffusers as _mod
                    _inp = _mod.AutoPipelineForInpainting
                    if _inp is None:
                        raise RuntimeError(
                            "diffusers AutoPipelineForInpainting is not available; "
                            "ensure museq[images] extras are installed in the per-model venv"
                        )
                    logger.info(
                        "loading inpaint pipeline from %s (model_id=%s, device=%s, dtype=%s)",
                        self._src, self.model_id, self._device, self._dtype,
                    )
                    self._inp_pipe = _inp.from_pipe(self._pipe)

        # Normalize mask to grayscale (mode L). Diffusers accepts L or
        # RGB; mode L is the documented contract. RGBA masks (alpha as
        # the regenerate region) get flattened by .convert("L").
        try:
            mask_mode = getattr(mask_image, "mode", None)
        except Exception:  # noqa: BLE001
            mask_mode = None
        if mask_mode is not None and mask_mode != "L":
            mask_image = mask_image.convert("L")

        n_steps = steps if steps is not None else self._default_steps
        cfg = guidance if guidance is not None else self._default_guidance
        s = strength if strength is not None else 0.99

        # Same steps * strength >= 1 contract as img2img.
        min_steps_for_strength = max(1, math.ceil(1.0 / max(s, 0.01)))
        if n_steps < min_steps_for_strength:
            logger.info(
                "inpaint bumping num_inference_steps from %d to %d to satisfy "
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
            "mask_image": mask_image,
            "strength": s,
            "num_inference_steps": n_steps,
            "guidance_scale": cfg,
        }
        if negative_prompt is not None:
            call_kwargs["negative_prompt"] = negative_prompt
        if width is not None:
            call_kwargs["width"] = width
        if height is not None:
            call_kwargs["height"] = height
        if gen is not None:
            call_kwargs["generator"] = gen
        if self._lora_adapter:
            call_kwargs["cross_attention_kwargs"] = {
                "scale": self._lora_weight(None),
            }

        out = self._call_pipe(self._inp_pipe, call_kwargs)
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
                "mode": "inpaint",
            },
        )

    def vary(
        self,
        *,
        init_image: Any,
        width: int | None = None,
        height: int | None = None,
        steps: int | None = None,
        guidance: float | None = None,
        seed: int | None = None,
        strength: float | None = None,
        **_: Any,
    ) -> ImageResult:
        """Variations: img2img with empty prompt and high strength.

        Reuses the existing img2img path via _generate_img2img. The
        OpenAI variations route has no prompt, so we pass empty string.
        Default strength 0.85 lands in the territory where the output
        is recognizably similar but visibly different (lower values
        look almost identical to the input; higher values drift too
        far for a "variation").
        """
        result = self._generate_img2img(
            prompt="",
            init_image=init_image,
            strength=strength if strength is not None else 0.85,
            negative_prompt=None,
            steps=steps,
            guidance=guidance,
            seed=seed,
        )
        # Override the underlying img2img label with the variations
        # mode so callers (and metadata-driven tests) see the right tag.
        result.metadata["mode"] = "variations"
        return result
