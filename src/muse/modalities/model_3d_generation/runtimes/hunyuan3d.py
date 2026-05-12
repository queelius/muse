"""Hunyuan3DRuntime: dual-direction 3D generation via Tencent's Hunyuan3D-2 SDK.

First muse 3D runtime supporting BOTH image_to_3d and text_to_3d from
one model. Wraps the Tencent SDK (hy3dgen package, installed from
GitHub). API verified against the real SDK at implementation time;
mocks reflect the actual return shapes, not speculation.

API VERIFIED (2026-05-06):

  Shape pipeline: Hunyuan3DDiTFlowMatchingPipeline from hy3dgen.shapegen
  Text-to-image pipeline: HunyuanDiTPipeline from hy3dgen.text2image

  TEXT-TO-3D ARCHITECTURE (verified from gradio_app.py):
    Text-to-3D is a two-stage pipeline: text -> HunyuanDiTPipeline ->
    PIL image -> Hunyuan3DDiTFlowMatchingPipeline -> mesh.
    The shape pipeline is image-only; there is NO native text-conditioned
    3D pipeline. The T2I pipeline is loaded lazily (optional dep; only
    required when text_to_3d is actually called).

  Shape pipeline from_pretrained signature (from pipelines.py):
    Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        model_path,
        device='cuda',
        dtype=torch.float16,
        use_safetensors=True,
        variant='fp16',
        subfolder='hunyuan3d-dit-v2-0',
    )
    Device placement: .to(device, dtype) method exists and is used
    internally by from_pretrained; device/dtype are passed directly
    to from_pretrained, not chained afterward.

  Shape pipeline call signature:
    pipeline(
        image=<PIL.Image.Image or path>,
        num_inference_steps=50,
        guidance_scale=5.0,
        generator=None,
        octree_resolution=384,
        output_type='trimesh',  # default
        **kwargs,
    ) -> List[List[trimesh.Trimesh]]
    Result is result[0][0] for the first mesh (outer list = batch,
    inner list = samples per batch element).

  Text-to-image pipeline:
    HunyuanDiTPipeline(
        'Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled',
        device=device,
    )
    Call: t2i_pipeline(prompt) -> PIL.Image.Image

  Mesh attributes: standard trimesh .vertices and .faces (the SDK's
  internal export_to_trimesh already converts from .mesh_v/.mesh_f
  before returning the trimesh.Trimesh objects).

  Device placement: device and dtype kwargs to from_pretrained.
  The .to() method is also available for explicit chaining.

Verified against:
  - https://github.com/Tencent/Hunyuan3D-2/blob/main/hy3dgen/shapegen/pipelines.py
  - https://github.com/Tencent/Hunyuan3D-2/blob/main/gradio_app.py
  - https://github.com/Tencent/Hunyuan3D-2/blob/main/minimal_demo.py
  on 2026-05-06.

If the SDK changes upstream, update mocks in
tests/modalities/model_3d_generation/runtimes/test_hunyuan3d.py and the
_HUNYUAN3D_PIPELINE / _HUNYUAN3D_T2I_PIPELINE sentinel logic here.

Deferred-imports pattern: sentinels populated by _ensure_deps. Tests
patch sentinels directly; _ensure_deps short-circuits on non-None.
"""
from __future__ import annotations

import logging
from typing import Any

from muse.core.runtime_helpers import (
    LoadTimer, dtype_for_name, select_device, set_inference_mode,
)
from muse.modalities.model_3d_generation.protocol import Generation3DResult


logger = logging.getLogger(__name__)


# Deferred-import sentinels. Tests pre-populate these with mocks;
# _ensure_deps short-circuits when they are non-None.
torch: Any = None
_HUNYUAN3D_PIPELINE: Any = None      # Hunyuan3DDiTFlowMatchingPipeline
_HUNYUAN3D_T2I_PIPELINE: Any = None  # HunyuanDiTPipeline (optional, text-to-3D only)
trimesh: Any = None
_LAST_IMPORT_ERROR: Exception | None = None


def _ensure_deps() -> None:
    """Lazy-import torch + Hunyuan3D shape pipeline + trimesh.

    The T2I pipeline (_HUNYUAN3D_T2I_PIPELINE) is intentionally NOT
    loaded here; it is loaded on first call to text_to_3d via
    _ensure_t2i_dep(). This avoids loading the heavier text-to-image
    model for operators who only use image-to-3D.
    """
    global torch, _HUNYUAN3D_PIPELINE, trimesh, _LAST_IMPORT_ERROR
    if torch is None:
        try:
            import torch as _t
            torch = _t
        except Exception as e:  # noqa: BLE001
            logger.debug("Hunyuan3DRuntime torch unavailable: %s", e)
            _LAST_IMPORT_ERROR = e
    if _HUNYUAN3D_PIPELINE is None:
        try:
            from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline as _p
            _HUNYUAN3D_PIPELINE = _p
        except Exception as e:  # noqa: BLE001
            logger.debug("Hunyuan3DRuntime SDK unavailable: %s", e)
            _LAST_IMPORT_ERROR = e
    if trimesh is None:
        try:
            import trimesh as _tm
            trimesh = _tm
        except Exception as e:  # noqa: BLE001
            logger.debug("Hunyuan3DRuntime trimesh unavailable: %s", e)
            _LAST_IMPORT_ERROR = e


def _ensure_t2i_dep() -> None:
    """Lazy-import HunyuanDiTPipeline for text-to-3D.

    Called only on the first text_to_3d request. Populates the
    _HUNYUAN3D_T2I_PIPELINE sentinel. Uses the same short-circuit
    pattern: skips if already populated (tests pre-populate with mock).
    """
    global _HUNYUAN3D_T2I_PIPELINE, _LAST_IMPORT_ERROR
    if _HUNYUAN3D_T2I_PIPELINE is None:
        try:
            from hy3dgen.text2image import HunyuanDiTPipeline as _p
            _HUNYUAN3D_T2I_PIPELINE = _p
        except Exception as e:  # noqa: BLE001
            logger.debug("Hunyuan3DRuntime HunyuanDiTPipeline unavailable: %s", e)
            _LAST_IMPORT_ERROR = e


class Hunyuan3DRuntime:
    """Dual-direction 3D generation runtime over Hunyuan3D-2.

    Wraps Tencent's Hunyuan3D-2 SDK for both image-to-3D and text-to-3D.
    Text-to-3D is a two-stage pipeline: text -> HunyuanDiTPipeline ->
    PIL image -> Hunyuan3DDiTFlowMatchingPipeline -> GLB mesh.

    Constructor kwargs (sourced from manifest's capabilities merged in
    by the registry at load_backend time):

      - ``model_id`` (required): catalog id; echoed in result envelope.
      - ``hf_repo``, ``local_dir``: standard weight source.
      - ``device``, ``dtype``: standard device + dtype selection.
      - ``trust_remote_code``: accepted but not forwarded to from_pretrained
        (Hunyuan3D-2 uses a pip-installed SDK, not transformers
        trust_remote_code). The flag is in capabilities for documentation;
        it means "install the hy3dgen package from GitHub."
      - ``num_inference_steps`` (default 50): denoising steps for the
        shape pipeline. The SDK default is 50; lower values are faster.
      - ``guidance_scale`` (default 5.0): classifier-free guidance for
        shape generation. SDK default.
      - ``t2i_model_id`` (default 'Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled'):
        HF repo for the text-to-image model used in text_to_3d.
        Lazily loaded on first text_to_3d call.
    """

    model_id: str
    supports_image_to_3d: bool = True
    supports_text_to_3d: bool = True

    def __init__(
        self,
        *,
        model_id: str,
        hf_repo: str,
        local_dir: str | None = None,
        device: str = "auto",
        dtype: str = "fp16",
        trust_remote_code: bool = True,
        seed: int | None = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        t2i_model_id: str = "Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled",
        **_: Any,
    ) -> None:
        _ensure_deps()
        if torch is None:
            raise RuntimeError(
                "torch is not installed; run "
                f"`muse models refresh {model_id}` or install "
                "`torch>=2.1.0` into this venv"
            )
        if _HUNYUAN3D_PIPELINE is None:
            raise RuntimeError(
                "Hunyuan3D-2 SDK not loadable; run "
                f"`muse models refresh {model_id}`. The SDK is the "
                "`hy3dgen` package from Tencent's GitHub. If the git "
                "pip-install failed during pull, follow Tencent's "
                "setup.sh manually in the per-model venv at "
                f"~/.muse/venvs/{model_id}/. "
                "See: https://github.com/Tencent/Hunyuan3D-2"
            )
        if trimesh is None:
            raise RuntimeError(
                "trimesh is not installed; needed for GLB export. "
                f"Run `muse models refresh {model_id}`."
            )
        self.model_id = model_id
        self._device = select_device(device, torch_module=torch)
        self._dtype = dtype_for_name(dtype, torch_module=torch)
        self._default_seed = seed
        self._default_num_inference_steps = int(num_inference_steps)
        self._default_guidance_scale = float(guidance_scale)
        self._t2i_model_id = t2i_model_id
        # Lazily loaded on first text_to_3d call.
        self._t2i_pipeline: Any = None

        src = local_dir or hf_repo
        with LoadTimer(f"loading Hunyuan3D-2 shape pipeline from {src}", logger):
            # from_pretrained signature (verified 2026-05-06):
            #   from_pretrained(model_path, device, dtype, use_safetensors,
            #                   variant, subfolder)
            # device and dtype are forwarded directly; device placement is
            # handled inside from_pretrained (calls .to(device, dtype) internally).
            self._pipeline = _HUNYUAN3D_PIPELINE.from_pretrained(
                src,
                device=self._device,
                dtype=self._dtype,
            )
            # Belt-and-suspenders: chain .to() in case from_pretrained
            # did not apply device/dtype (SDK behavior may vary across versions).
            if hasattr(self._pipeline, "to"):
                self._pipeline = self._pipeline.to(self._device, self._dtype)
        set_inference_mode(self._pipeline)

    def _load_t2i_pipeline(self) -> Any:
        """Load the text-to-image pipeline on first text_to_3d call.

        Uses _HUNYUAN3D_T2I_PIPELINE sentinel so tests can inject a mock
        without triggering a real download.
        """
        _ensure_t2i_dep()
        if _HUNYUAN3D_T2I_PIPELINE is None:
            raise RuntimeError(
                "HunyuanDiTPipeline (text-to-image) is not loadable; the "
                "hy3dgen.text2image module is required for text_to_3d. "
                f"Run `muse models refresh {self.model_id}`. "
                "See: https://github.com/Tencent/Hunyuan3D-2"
            )
        with LoadTimer("loading Hunyuan3D-2 text-to-image pipeline", logger):
            return _HUNYUAN3D_T2I_PIPELINE(
                self._t2i_model_id,
                device=self._device,
            )

    def image_to_3d(
        self, image: Any, **kwargs: Any,
    ) -> list[Generation3DResult]:
        """Generate one or more 3D meshes from a single image.

        Parameters
        ----------
        image:
            A PIL.Image.Image (or path/URL string; the SDK accepts both).
            The route layer decodes the incoming multipart/form-data file
            into a PIL image before calling this method.
        kwargs:
            Forwarded to pipeline():
            - ``n`` (int, default 1): number of samples.
            - ``seed`` (int): random seed.
            - ``num_inference_steps`` (int): denoising steps.
            - ``guidance_scale`` (float): classifier-free guidance scale.

        Returns
        -------
        list[Generation3DResult]
            One item per sample (n items total). Each item carries a
            geometry-only GLB blob.
        """
        n = int(kwargs.get("n", 1))
        seed = kwargs.get("seed", self._default_seed)
        num_inference_steps = int(
            kwargs.get("num_inference_steps", self._default_num_inference_steps)
        )
        guidance_scale = float(
            kwargs.get("guidance_scale", self._default_guidance_scale)
        )

        generator = None
        if seed is not None and torch is not None:
            generator = torch.Generator().manual_seed(int(seed))

        results: list[Generation3DResult] = []
        for _ in range(max(1, n)):
            # Verified call signature (2026-05-06):
            #   pipeline(image=..., num_inference_steps=...,
            #            guidance_scale=..., generator=None, ...)
            # Returns List[List[trimesh.Trimesh]] when output_type='trimesh'
            # (the default). result[0][0] is the first mesh.
            output = self._pipeline(
                image=image,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )
            # output is List[List[trimesh.Trimesh]]: outer = batch, inner = samples.
            mesh = output[0][0]
            glb_bytes = mesh.export(file_type="glb")
            results.append(Generation3DResult(
                glb_bytes=bytes(glb_bytes),
                model_id=self.model_id,
                format="glb",
            ))
        return results

    def text_to_3d(
        self, prompt: str, **kwargs: Any,
    ) -> list[Generation3DResult]:
        """Generate one or more 3D meshes from a text prompt.

        Implements the two-stage Hunyuan3D-2 text-to-3D pipeline:
          1. HunyuanDiTPipeline converts the text prompt to a PIL image.
          2. Hunyuan3DDiTFlowMatchingPipeline converts the image to a mesh.

        The T2I pipeline is loaded lazily on the first call to this
        method; subsequent calls reuse the cached pipeline instance.

        Parameters
        ----------
        prompt:
            Text description of the 3D object to generate.
        kwargs:
            - ``n`` (int, default 1): number of samples.
            - ``seed`` (int): random seed.
            - ``num_inference_steps`` (int): denoising steps for shape gen.
            - ``guidance_scale`` (float): guidance scale for shape gen.

        Returns
        -------
        list[Generation3DResult]
            One item per sample (n items total).
        """
        n = int(kwargs.get("n", 1))
        seed = kwargs.get("seed", self._default_seed)
        num_inference_steps = int(
            kwargs.get("num_inference_steps", self._default_num_inference_steps)
        )
        guidance_scale = float(
            kwargs.get("guidance_scale", self._default_guidance_scale)
        )

        # Lazy-load the text-to-image pipeline on first use.
        if self._t2i_pipeline is None:
            self._t2i_pipeline = self._load_t2i_pipeline()

        generator = None
        if seed is not None and torch is not None:
            generator = torch.Generator().manual_seed(int(seed))

        results: list[Generation3DResult] = []
        for _ in range(max(1, n)):
            # Stage 1: text -> image via HunyuanDiTPipeline.
            # Verified call signature: t2i_pipeline(prompt) -> PIL.Image.Image
            image = self._t2i_pipeline(prompt)

            # Stage 2: image -> mesh via Hunyuan3DDiTFlowMatchingPipeline.
            output = self._pipeline(
                image=image,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )
            mesh = output[0][0]
            glb_bytes = mesh.export(file_type="glb")
            results.append(Generation3DResult(
                glb_bytes=bytes(glb_bytes),
                model_id=self.model_id,
                format="glb",
            ))
        return results
