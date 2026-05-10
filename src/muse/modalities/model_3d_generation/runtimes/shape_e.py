"""ShapERuntime: text-to-3D via diffusers.ShapEPipeline.

Wraps `diffusers.ShapEPipeline` for OpenAI Shap-E text-to-3D
generation. Returns GLB-encoded mesh output through the existing
model_3d_generation codec.

Shap-E base is text-only. The route layer's `supports_image_to_3d`
capability gate prevents the image-to-3D route from invoking this
runtime; the runtime additionally raises NotImplementedError on
`image_to_3d` as a defensive check.

Deferred-imports pattern: torch, ShapEPipeline, trimesh as module-top
sentinels populated by _ensure_deps(). Tests patch sentinels directly;
_ensure_deps short-circuits on non-None.
"""
from __future__ import annotations

import logging
from typing import Any

from muse.core.runtime_helpers import (
    LoadTimer, dtype_for_name, select_device, set_inference_mode,
)
from muse.modalities.model_3d_generation.protocol import Generation3DResult


logger = logging.getLogger(__name__)


# Deferred-import sentinels.
torch: Any = None
ShapEPipeline: Any = None
trimesh: Any = None
_LAST_IMPORT_ERROR: Exception | None = None


def _ensure_deps() -> None:
    """Lazy-import heavy deps into module-level sentinels.

    Each branch records the last failure so the constructor can report
    which dep is missing. Test fixtures pre-populate the sentinels;
    those branches short-circuit.
    """
    global torch, ShapEPipeline, trimesh, _LAST_IMPORT_ERROR
    if torch is None:
        try:
            import torch as _t
            torch = _t
        except Exception as e:  # noqa: BLE001
            logger.debug("ShapERuntime torch unavailable: %s", e)
            _LAST_IMPORT_ERROR = e
    if ShapEPipeline is None:
        try:
            from diffusers import ShapEPipeline as _p
            ShapEPipeline = _p
        except Exception as e:  # noqa: BLE001
            logger.debug("ShapERuntime ShapEPipeline unavailable: %s", e)
            _LAST_IMPORT_ERROR = e
    if trimesh is None:
        try:
            import trimesh as _tm
            trimesh = _tm
        except Exception as e:  # noqa: BLE001
            logger.debug("ShapERuntime trimesh unavailable: %s", e)
            _LAST_IMPORT_ERROR = e


class ShapERuntime:
    """Generic Shap-E text-to-3D runtime.

    Constructor kwargs (sourced from manifest's capabilities, merged
    in by the registry at load_backend time):

      - ``model_id`` (required): catalog id; echoed in result envelope.
      - ``hf_repo``, ``local_dir``: standard weight source.
      - ``device``, ``dtype``: standard device + dtype selection.
      - ``guidance_scale`` (default 15.0): classifier-free guidance
        strength; Shap-E's official default. Higher values push the
        output closer to the text prompt at the cost of diversity.
      - ``num_inference_steps`` (default 64): denoising steps.
      - ``frame_size`` (default 256): resolution of the ShapE rendering
        grid; higher values produce finer meshes.
    """

    model_id: str
    supports_text_to_3d: bool = True
    supports_image_to_3d: bool = False

    def __init__(
        self,
        *,
        model_id: str,
        hf_repo: str,
        local_dir: str | None = None,
        device: str = "auto",
        dtype: str = "fp16",
        guidance_scale: float = 15.0,
        num_inference_steps: int = 64,
        frame_size: int = 256,
        **_: Any,
    ) -> None:
        _ensure_deps()
        # Validate each dep with a specific error message so operators
        # can fix the right thing without trial-and-error.
        if torch is None:
            raise RuntimeError(
                "torch is not installed; run "
                f"`muse models refresh {model_id}` or install "
                "`torch>=2.1.0` into this venv"
            )
        if ShapEPipeline is None:
            raise RuntimeError(
                "diffusers is not installed or too old (needs >= 0.27.0 "
                "for ShapEPipeline). Run "
                f"`muse models refresh {model_id}`."
            )
        if trimesh is None:
            raise RuntimeError(
                "trimesh is not installed; needed for GLB export. "
                f"Run `muse models refresh {model_id}`."
            )
        self.model_id = model_id
        self._device = select_device(device, torch_module=torch)
        self._dtype = dtype_for_name(dtype, torch_module=torch)
        self._default_guidance_scale = float(guidance_scale)
        self._default_num_inference_steps = int(num_inference_steps)
        self._default_frame_size = int(frame_size)
        src = local_dir or hf_repo
        with LoadTimer(f"loading Shap-E from {src}", logger):
            self._pipeline = ShapEPipeline.from_pretrained(
                src, torch_dtype=self._dtype,
            )
            self._pipeline = self._pipeline.to(self._device)
        set_inference_mode(self._pipeline)

    def text_to_3d(
        self, prompt: str, **kwargs: Any,
    ) -> list[Generation3DResult]:
        """Generate one or more 3D assets from a text prompt.

        Returns a list of ``Generation3DResult``, each carrying the GLB
        bytes for one mesh. ``n`` is capped at 2 by the route layer.
        ShapE sampling is stochastic, so n>1 may yield different meshes.
        """
        n = int(kwargs.get("n", 1))
        guidance_scale = float(
            kwargs.get("guidance_scale", self._default_guidance_scale)
        )
        num_inference_steps = int(
            kwargs.get("num_inference_steps", self._default_num_inference_steps)
        )
        frame_size = int(
            kwargs.get("frame_size", self._default_frame_size)
        )

        results: list[Generation3DResult] = []
        for _ in range(max(1, n)):
            result = self._pipeline(
                prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                frame_size=frame_size,
            )

            # ShapEPipeline returns .meshes as a list of mesh-like objects
            # with .vertices and .faces tensors. Adapt to trimesh.Trimesh,
            # then export to GLB bytes through the existing codec contract.
            mesh_data = result.meshes[0]
            vertices = mesh_data.vertices.cpu().numpy()
            faces = mesh_data.faces.cpu().numpy()
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            glb_bytes = mesh.export(file_type="glb")
            results.append(Generation3DResult(
                glb_bytes=bytes(glb_bytes),
                model_id=self.model_id,
            ))
        return results

    def image_to_3d(
        self, image_path: str, **kwargs: Any,
    ) -> list[Generation3DResult]:
        """Not supported: Shap-E base is text-to-3D only.

        The capability flag ``supports_image_to_3d=False`` prevents the
        route layer from calling this method; this raise is a defensive
        check for callers that bypass the route layer.
        """
        raise NotImplementedError(
            "Shap-E base is text-to-3D only; image-to-3D would require "
            "the separate openai/shap-e-img2img variant which is not "
            "supported in v0.43.0."
        )
