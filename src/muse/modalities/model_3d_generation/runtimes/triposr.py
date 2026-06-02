"""TripoSRRuntime: image-to-3d via Stability AI's TripoSR.

Wraps `tsr.system.TSR` (TripoSR's own loader; not a transformers
AutoModel) plus `trimesh` to convert the model's mesh output to a
GLB blob. Image-to-3d only; the route layer's `supports_text_to_3d`
capability gate prevents the unrelated text route from invoking
this runtime.

Surface:
- ``TSR.from_pretrained(src, config_name="config.yaml",
  weight_name="model.ckpt")`` returns a ``TSR`` instance.
- ``model.renderer.set_chunk_size(n)`` configures surface-extraction
  chunking; smaller values reduce VRAM at the cost of speed.
- ``model([image], device=device)`` (TSR.forward) returns
  ``scene_codes``.
- ``model.extract_mesh(scene_codes, has_vertex_color, resolution=256)``
  returns ``list[trimesh.Trimesh]`` (one per input image).
- ``mesh.export(file_type="glb")`` returns the GLB bytes directly.

Foreground isolation: TripoSR's official ``run.py`` chooses between
``--no-remove-bg`` (RGB-with-gray-background, foreground-isolated by
the caller) and rembg + ``resize_foreground`` (which requires RGBA).
This runtime uses the no-remove-bg path: it opens the image as RGB
and forwards it. Operators who want background removal should preprocess
client-side or layer a ``rembg`` step before calling. Adding rembg as
a runtime dependency is rejected because rembg pulls in onnxruntime
and ~200MB of model weights for a step that is best done at the edge.

Deferred-imports pattern: ``torch``, ``TSR``, ``trimesh``, ``PIL_Image``
are module-top sentinels populated by ``_ensure_deps()``. Tests patch
the sentinels directly; ``_ensure_deps`` short-circuits on non-None.
"""
from __future__ import annotations

import logging
from typing import Any

from muse.core.runtime_helpers import (
    LoadTimer, dtype_for_name, select_device, set_inference_mode,
)
from muse.modalities.model_3d_generation.codec import mesh_to_glb_result
from muse.modalities.model_3d_generation.protocol import Generation3DResult


logger = logging.getLogger(__name__)


# Deferred-import sentinels.
torch: Any = None
TSR: Any = None
trimesh: Any = None
PIL_Image: Any = None
_LAST_IMPORT_ERROR: Exception | None = None


def _ensure_deps() -> None:
    """Lazy-import heavy deps into module-level sentinels.

    Each branch records the last failure so the constructor can report
    which dep is missing. Test fixtures pre-populate the sentinels;
    those branches short-circuit.
    """
    global torch, TSR, trimesh, PIL_Image, _LAST_IMPORT_ERROR
    if torch is None:
        try:
            import torch as _t
            torch = _t
        except Exception as e:  # noqa: BLE001
            logger.debug("TripoSRRuntime torch unavailable: %s", e)
            _LAST_IMPORT_ERROR = e
    if TSR is None:
        try:
            from tsr.system import TSR as _T
            TSR = _T
        except Exception as e:  # noqa: BLE001
            logger.debug("TripoSRRuntime tsr unavailable: %s", e)
            _LAST_IMPORT_ERROR = e
    if trimesh is None:
        try:
            import trimesh as _tm
            trimesh = _tm
        except Exception as e:  # noqa: BLE001
            logger.debug("TripoSRRuntime trimesh unavailable: %s", e)
            _LAST_IMPORT_ERROR = e
    if PIL_Image is None:
        try:
            from PIL import Image as _PI
            PIL_Image = _PI
        except Exception as e:  # noqa: BLE001
            logger.debug("TripoSRRuntime PIL unavailable: %s", e)
            _LAST_IMPORT_ERROR = e


class TripoSRRuntime:
    """Stability AI's TripoSR: single-image to 3D mesh.

    Constructor kwargs (sourced from manifest's capabilities, merged
    in by the registry at load_backend time):

      - ``model_id`` (required): catalog id; echoed in result envelope.
      - ``hf_repo``, ``local_dir``: standard weight source.
      - ``device``, ``dtype``: standard device + dtype selection.
      - ``chunk_size`` (default 8192): renderer chunk; smaller -> lower
        VRAM, higher -> faster.
      - ``mc_resolution`` (default 256): marching-cubes grid; higher ->
        finer mesh but more VRAM + time.
      - ``has_vertex_color`` (default False): when True, ``extract_mesh``
        also queries vertex colors. Default off because TripoSR's
        density-only meshes are noticeably lighter to ship and most
        downstream tooling will texture-bake separately.
      - ``foreground_ratio`` (kept for parity with the official run.py
        path; not used in the runtime today, since we follow the
        no-remove-bg flow). Reserved.

    """

    model_id: str

    def __init__(
        self,
        *,
        model_id: str,
        hf_repo: str,
        local_dir: str | None = None,
        device: str = "auto",
        dtype: str = "fp32",
        chunk_size: int = 8192,
        mc_resolution: int = 256,
        has_vertex_color: bool = False,
        config_name: str = "config.yaml",
        weight_name: str = "model.ckpt",
        **_: Any,
    ) -> None:
        _ensure_deps()
        # Validate each dep with a specific error message so operators
        # can fix the right thing without trial-and-error.
        if torch is None:
            raise RuntimeError(
                "torch is not installed; run `muse pull` or install "
                "`torch>=2.1.0` into this venv"
            )
        if TSR is None:
            raise RuntimeError(
                f"`tsr` package is not installed in this venv; "
                f"run `muse models refresh {model_id}`. "
                f"(Last import error: {_LAST_IMPORT_ERROR})"
            )
        if trimesh is None:
            raise RuntimeError(
                f"`trimesh` is not installed in this venv; "
                f"run `muse models refresh {model_id}`."
            )
        if PIL_Image is None:
            raise RuntimeError(
                f"`Pillow` is not installed in this venv; "
                f"run `muse models refresh {model_id}`."
            )

        self.model_id = model_id
        self._device = select_device(device, torch_module=torch)
        self._dtype = dtype_for_name(dtype, torch_module=torch)
        self._chunk_size = int(chunk_size)
        self._mc_resolution = int(mc_resolution)
        self._has_vertex_color = bool(has_vertex_color)

        src = local_dir or hf_repo
        with LoadTimer(f"loading TripoSR from {src}", logger):
            # TripoSR's loader is NOT transformers' from_pretrained;
            # it takes config_name + weight_name explicitly. Defaults
            # match the stabilityai/TripoSR repo layout (config.yaml +
            # model.ckpt). Curated TripoSR variants that ship
            # safetensors can override via the manifest capabilities.
            self._model = TSR.from_pretrained(
                src,
                config_name=config_name,
                weight_name=weight_name,
            )
            # Renderer chunking trades VRAM for compute. 8192 is the
            # upstream default and works on 8GB cards at the default
            # 256x256 input resolution.
            self._model.renderer.set_chunk_size(self._chunk_size)
            self._model.to(self._device)
        # Switch the model to no-grad mode. set_inference_mode() is a
        # no-op when the model lacks the standard switch method (e.g.,
        # a bare nn.Module subclass without transformers conventions),
        # so this is safe even though TSR is not a transformers model.
        set_inference_mode(self._model)

    def image_to_3d(
        self,
        image_path: str,
        *,
        n: int = 1,
        seed: int | None = None,
        **_: Any,
    ) -> list[Generation3DResult]:
        """Generate ``n`` 3D meshes from one input image.

        Returns a list of ``Generation3DResult``, each carrying the GLB
        bytes for one mesh. ``n`` is capped at 2 by the route layer.

        TripoSR is deterministic per-image: there is no stochastic
        sampling, so ``n>1`` returns ``n`` identical meshes. The
        ``seed`` kwarg is accepted for protocol uniformity (other
        backends use it) but ignored by TripoSR.
        """
        # Open as RGB. TripoSR's no-remove-bg path expects an RGB image
        # with the foreground already isolated against a near-uniform
        # background. The runtime intentionally does NOT call rembg or
        # resize_foreground (the latter requires RGBA input from rembg
        # anyway); operators wanting that pipeline preprocess upstream.
        image = PIL_Image.open(image_path).convert("RGB")

        results: list[Generation3DResult] = []
        # Single forward pass produces one set of scene_codes that all
        # n outputs share, since TripoSR is deterministic. We still run
        # the mesh extraction n times to keep the path uniform with
        # stochastic backends; the cost is small (extract_mesh is O(N)
        # in the marching-cubes grid, not in the model forward).
        with torch.no_grad():
            scene_codes = self._model([image], device=self._device)

        for _ in range(max(1, n)):
            with torch.no_grad():
                meshes = self._model.extract_mesh(
                    scene_codes,
                    self._has_vertex_color,
                    resolution=self._mc_resolution,
                )
            # Single-image input always yields one scene_code, hence one
            # mesh. Take the first; defensive against empty in case the
            # marching-cubes step fails to find any surface (which we
            # surface as a clear error rather than an empty GLB).
            if not meshes:
                raise RuntimeError(
                    f"TripoSR extract_mesh produced no meshes for "
                    f"{image_path!r}; the input may not have a clear "
                    f"foreground subject"
                )
            mesh = meshes[0]
            results.append(mesh_to_glb_result(mesh, self.model_id))
        return results

