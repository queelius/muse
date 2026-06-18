# TRELLIS.2 3D Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `microsoft/TRELLIS.2-4B` as a curated image-to-3D model in muse's `3d/generation` modality, served by a new `TRELLIS2Runtime`, shipped as v0.47.0.

**Architecture:** Mirror the v0.44 TRELLIS / v0.45 Hunyuan3D adds. TRELLIS.2 uses a distinct SDK (`trellis2` + `o_voxel` packages, not the original `trellis`), so it gets its own runtime file and a new `_Family` entry ordered *before* the colliding `trellis` name-hint. Wire is unchanged (`POST /v1/3d/from-image` already routes a PIL image to image-to-3d backends).

**Tech Stack:** Python, `trellis2.pipelines.Trellis2ImageTo3DPipeline`, `o_voxel.postprocess.to_glb`, the existing `mesh_to_glb_result` codec, pytest with mocked SDK.

## Global Constraints

- **ASCII only in committed files.** A pre-commit soul-voice hook rejects em-dashes ( - ), en-dashes (-), and other non-ASCII. Use `-`, `:`, `,`, `()`, `->`.
- **Deferred-imports sentinel pattern.** Heavy/SDK imports live in module-level sentinels populated by `_ensure_deps()`; tests pre-populate the sentinels with mocks. `muse --help` / `muse pull` must import the runtime module without the SDK installed.
- **Use `muse.core.runtime_helpers`** (`select_device`, `dtype_for_name`, `set_inference_mode`, `LoadTimer`) - do not re-implement; `tests/core/test_runtime_helpers_meta.py` AST-walks every runtime and fails on re-implementations.
- **GPU-only.** TRELLIS.2-4B requires an NVIDIA GPU with >=24GB VRAM, Linux, CUDA 12.4 toolchain. `device: cuda`. NOT added to the free-tier CI fresh-venv smoke matrix.
- **Image-to-3D only.** `supports_image_to_3d: True`, `supports_text_to_3d: False`.
- **Curated alongside, not replacing.** Add `trellis2-image`; keep `trellis-image`.
- **Real-API discipline (B1).** The runtime code below is derived from the TRELLIS.2-4B model-card README usage block. Task 1 confirms it against the real installed SDK before the runtime ships.

Verified API facts (TRELLIS.2-4B README, https://github.com/microsoft/TRELLIS.2):
```python
from trellis2.pipelines import Trellis2ImageTo3DPipeline
import o_voxel
pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
pipeline.cuda()                       # .cuda(), not .to(device)
mesh = pipeline.run(image)[0]         # image: PIL.Image; returns a list
mesh.simplify(16777216)               # nvdiffrast vertex limit
glb = o_voxel.postprocess.to_glb(
    vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs,
    coords=mesh.coords, attr_layout=mesh.layout, voxel_size=mesh.voxel_size,
    aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
    decimation_target=1000000, texture_size=4096,
    remesh=True, remesh_band=1, remesh_project=0, verbose=False,
)                                     # glb is a trimesh.Scene-like object
glb.export("out.glb", extension_webp=True)   # path export; file_type= returns bytes
```

---

### Task 1: B1 verification of the real trellis2 SDK (GPU host)

**Files:**
- Reference only (no code commit): record findings in the spec's "Open items" section and in the Task 3 runtime docstring.

This task runs on a Linux GPU host (>=24GB VRAM, CUDA 12.4) where the SDK can actually install and load. It is NOT a TDD code task; its deliverable is a set of confirmed facts that gate Task 3.

- [ ] **Step 1: Install the SDK in a scratch venv**

```bash
python -m venv /tmp/trellis2-b1 && . /tmp/trellis2-b1/bin/activate
pip install torch torchvision
# Confirm the exact install recipe from https://github.com/microsoft/TRELLIS.2
# (trellis2 + o_voxel are NOT on PyPI; expect a git install + setup script
#  with CUDA-compiled deps similar to the original TRELLIS).
```

- [ ] **Step 2: Confirm the API surface**

Verify and record each fact (expected value in parens; correct the plan/runtime if reality differs):
1. `from trellis2.pipelines import Trellis2ImageTo3DPipeline` importable. (yes)
2. `Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")` signature; no `trust_remote_code` kwarg. (positional repo/path)
3. `pipeline.cuda()` is the device call (not `.to()`). (yes)
4. `pipeline.run(image)` returns a list; `run(image)[0]` is one mesh. (yes)
5. The mesh object exposes `.vertices`, `.faces`, `.attrs`, `.coords`, `.layout`, `.voxel_size`, and `.simplify(int)`. (per README)
6. `o_voxel.postprocess.to_glb(...)` import path + the kwarg names above. (per README)
7. **GLB bytes extraction:** confirm `bytes(glb.export(file_type="glb"))` returns valid GLB bytes (i.e. the `to_glb` output is trimesh-compatible, so the existing `mesh_to_glb_result(glb, model_id)` codec helper works). If `glb` is NOT trimesh-compatible, record the bytes-extraction recipe (e.g. export to a temp `.glb` and read it back) so Task 3 uses that instead.
8. The **git install URL(s)** for `trellis2` and `o_voxel`.

- [ ] **Step 3: Record findings**

Append a dated "B1 findings" note to `docs/superpowers/specs/2026-06-18-trellis2-3d-runtime-design.md` capturing the 8 answers, especially (7) the GLB-bytes recipe and (8) the install URLs. These feed Task 3 (runtime docstring) and Task 4 (pip_extras). No commit of code in this task; commit only the spec note:

```bash
git add docs/superpowers/specs/2026-06-18-trellis2-3d-runtime-design.md
git commit -m "docs(spec): record TRELLIS.2 B1 SDK-verification findings"
```

---

### Task 2: Dispatch - new trellis2 _Family ordered before trellis

**Files:**
- Modify: `src/muse/modalities/model_3d_generation/hf.py` (add constants near the other `_*_RUNTIME_PATH` / `_*_PIP_EXTRAS`; insert a `_Family` into `_FAMILIES` before the `trellis` entry; add to `_NAME_HINTS`)
- Test: `tests/modalities/model_3d_generation/test_hf_per_family_dispatch.py`

**Interfaces:**
- Consumes: existing `_Family` dataclass, `_family_for(repo_id) -> _Family`, `_FAMILIES`, `_NAME_HINTS`.
- Produces: `_TRELLIS2_RUNTIME_PATH = "muse.modalities.model_3d_generation.runtimes.trellis2:TRELLIS2Runtime"`; a `_Family` whose `runtime_path` is that, matched by repo names containing `trellis2` / `trellis.2`.

- [ ] **Step 1: Write the failing test**

Add to `tests/modalities/model_3d_generation/test_hf_per_family_dispatch.py`:

```python
def test_trellis2_repo_routes_to_trellis2_runtime():
    """microsoft/TRELLIS.2-4B must route to TRELLIS2Runtime, NOT the original
    TRELLISRuntime (the generic 'trellis' hint also matches 'trellis.2-4b',
    so the trellis2 family must win by ordering)."""
    from muse.modalities.model_3d_generation.hf import _family_for
    fam = _family_for("microsoft/TRELLIS.2-4B")
    assert fam.runtime_path.endswith(":TRELLIS2Runtime")
    assert fam.capability_overrides.get("supports_image_to_3d") is True
    assert fam.capability_overrides.get("supports_text_to_3d") is False


def test_original_trellis_repo_still_routes_to_trellis_runtime():
    """A non-.2 TRELLIS repo must still pick the original TRELLISRuntime."""
    from muse.modalities.model_3d_generation.hf import _family_for
    fam = _family_for("JeffreyXiang/TRELLIS-image-large")
    assert fam.runtime_path.endswith(":TRELLISRuntime")
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/modalities/model_3d_generation/test_hf_per_family_dispatch.py::test_trellis2_repo_routes_to_trellis2_runtime -v`
Expected: FAIL (routes to `:TRELLISRuntime` because the trellis2 family does not exist yet).

- [ ] **Step 3: Add the runtime-path constant + pip_extras placeholder**

In `src/muse/modalities/model_3d_generation/hf.py`, after the existing `_HUNYUAN3D_PIP_EXTRAS` block, add:

```python
_TRELLIS2_RUNTIME_PATH = (
    "muse.modalities.model_3d_generation.runtimes.trellis2:TRELLIS2Runtime"
)
# TRELLIS.2 uses Microsoft's `trellis2` + `o_voxel` standalone packages (NOT
# transformers/diffusers, and a DIFFERENT SDK from the original `trellis`).
# Native-build CUDA deps; GPU-only (>=24GB VRAM), Linux, CUDA 12.4 toolchain.
# pip may fail without that toolchain; fallback is the upstream setup script
# inside the per-model venv at ~/.muse/venvs/trellis2-image/. URLs confirmed
# in B1 (Task 1). See https://github.com/microsoft/TRELLIS.2
_TRELLIS2_PIP_EXTRAS: tuple[str, ...] = (
    "torch>=2.1.0",
    "torchvision>=0.16.0",
    "trimesh",
    "Pillow",
    "numpy",
    "opencv-python",
    "trellis2 @ git+https://github.com/microsoft/TRELLIS.2.git",
)
```

- [ ] **Step 4: Insert the _Family entry BEFORE the trellis entry**

In `_FAMILIES`, immediately before the existing `_Family(  # TRELLIS (unchanged from v0.44.0)` entry, insert:

```python
    _Family(  # TRELLIS.2 (v0.47.0): distinct SDK (trellis2) from original TRELLIS
        name_hints=("trellis2", "trellis.2"),
        runtime_path=_TRELLIS2_RUNTIME_PATH,
        pip_extras=_TRELLIS2_PIP_EXTRAS,
        capability_overrides={
            "supports_image_to_3d": True,
            "supports_text_to_3d": False,
        },
        trust_remote_code=False,
    ),
```

Then add `"trellis2"` to the `_NAME_HINTS` tuple (after `"trellis"`).

- [ ] **Step 5: Run to verify both dispatch tests pass**

Run: `pytest tests/modalities/model_3d_generation/test_hf_per_family_dispatch.py -v`
Expected: PASS (both the new trellis2 routing test and the original-trellis test).

- [ ] **Step 6: Commit**

```bash
git add src/muse/modalities/model_3d_generation/hf.py tests/modalities/model_3d_generation/test_hf_per_family_dispatch.py
git commit -m "feat(3d): trellis2 _Family dispatch ordered before trellis"
```

---

### Task 3: TRELLIS2Runtime

**Files:**
- Create: `src/muse/modalities/model_3d_generation/runtimes/trellis2.py`
- Test: `tests/modalities/model_3d_generation/runtimes/test_trellis2.py`

**Interfaces:**
- Consumes: `mesh_to_glb_result(mesh, model_id) -> Generation3DResult` (codec; does `bytes(mesh.export(file_type="glb"))` with a zero-byte guard), `Generation3DResult`, `runtime_helpers`.
- Produces: `TRELLIS2Runtime(model_id=, hf_repo=, local_dir=, device=, **_)` with `image_to_3d(image, **kwargs) -> list[Generation3DResult]`, `text_to_3d(...)` raising `NotImplementedError`, class attrs `supports_image_to_3d=True`, `supports_text_to_3d=False`.

- [ ] **Step 1: Write the failing test**

Create `tests/modalities/model_3d_generation/runtimes/test_trellis2.py`:

```python
"""TRELLIS2Runtime: mocked-dep tests. Mocks reflect the REAL trellis2 SDK API
from the TRELLIS.2-4B README (Trellis2ImageTo3DPipeline + o_voxel.postprocess.to_glb)."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import muse.modalities.model_3d_generation.runtimes.trellis2 as mod


@pytest.fixture(autouse=True)
def _reset_module_sentinels():
    orig = (mod.torch, mod._TRELLIS2_PIPELINE, mod._o_voxel, mod._LAST_IMPORT_ERROR)
    yield
    (mod.torch, mod._TRELLIS2_PIPELINE, mod._o_voxel, mod._LAST_IMPORT_ERROR) = orig


@pytest.mark.parametrize(
    "sentinel_name,match_str",
    [
        ("torch", "torch is not installed"),
        ("_TRELLIS2_PIPELINE", "TRELLIS.2 SDK"),
        ("_o_voxel", "o_voxel"),
    ],
)
def test_raises_when_dep_not_installed(monkeypatch, sentinel_name, match_str):
    monkeypatch.setattr(mod, "_ensure_deps", lambda: None)
    mod.torch = MagicMock()
    mod.torch.cuda.is_available.return_value = False
    mod.torch.backends = MagicMock(mps=None)
    mod._TRELLIS2_PIPELINE = MagicMock()
    mod._o_voxel = MagicMock()
    setattr(mod, sentinel_name, None)
    with pytest.raises(RuntimeError, match=match_str):
        mod.TRELLIS2Runtime(model_id="m", hf_repo="x", device="cpu")


def _wire_runtime():
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False
    fake_torch.backends = MagicMock(mps=None)
    mod.torch = fake_torch

    pipeline = MagicMock()
    pipeline.cuda = MagicMock(return_value=pipeline)
    fake_mesh = MagicMock()  # one O-Voxel mesh; pipeline.run(image) -> [mesh]
    pipeline.run = MagicMock(return_value=[fake_mesh])

    pipe_factory = MagicMock()
    pipe_factory.from_pretrained = MagicMock(return_value=pipeline)
    mod._TRELLIS2_PIPELINE = pipe_factory

    # o_voxel.postprocess.to_glb(...) -> trimesh-compatible object whose
    # .export(file_type="glb") returns bytes (consumed by mesh_to_glb_result).
    fake_glb = MagicMock()
    fake_glb.export = MagicMock(return_value=b"GLB_BYTES")
    fake_o_voxel = MagicMock()
    fake_o_voxel.postprocess.to_glb = MagicMock(return_value=fake_glb)
    mod._o_voxel = fake_o_voxel
    return pipeline, fake_mesh, fake_o_voxel, fake_glb


def test_image_to_3d_returns_glb_in_result():
    pipeline, fake_mesh, fake_o_voxel, fake_glb = _wire_runtime()
    runtime = mod.TRELLIS2Runtime(
        model_id="trellis2-image", hf_repo="microsoft/TRELLIS.2-4B", device="cpu",
    )
    results = runtime.image_to_3d(MagicMock(name="pil_image"))
    assert len(results) == 1
    assert results[0].model_id == "trellis2-image"
    assert results[0].glb_bytes == b"GLB_BYTES"
    pipeline.run.assert_called_once()
    fake_mesh.simplify.assert_called_once()           # nvdiffrast limit applied
    fake_o_voxel.postprocess.to_glb.assert_called_once()
    fake_glb.export.assert_called_once_with(file_type="glb")


def test_image_to_3d_uses_run_index_zero():
    pipeline, _, _, _ = _wire_runtime()
    runtime = mod.TRELLIS2Runtime(model_id="m", hf_repo="x", device="cpu")
    runtime.image_to_3d(MagicMock())
    assert pipeline.run.call_count == 1
    assert pipeline.call_count == 0  # run(), not __call__


def test_image_to_3d_loops_over_n():
    pipeline, _, _, _ = _wire_runtime()
    runtime = mod.TRELLIS2Runtime(model_id="m", hf_repo="x", device="cpu")
    results = runtime.image_to_3d(MagicMock(), n=2)
    assert len(results) == 2
    assert pipeline.run.call_count == 2


def test_text_to_3d_raises_not_implemented():
    _wire_runtime()
    runtime = mod.TRELLIS2Runtime(model_id="m", hf_repo="x", device="cpu")
    with pytest.raises(NotImplementedError, match="image-only"):
        runtime.text_to_3d("a chair")


def test_supports_capability_attrs():
    assert mod.TRELLIS2Runtime.supports_image_to_3d is True
    assert mod.TRELLIS2Runtime.supports_text_to_3d is False


def test_local_dir_preferred_over_hf_repo():
    _wire_runtime()
    mod.TRELLIS2Runtime(
        model_id="m", hf_repo="microsoft/TRELLIS.2-4B",
        local_dir="/tmp/local-trellis2", device="cpu",
    )
    src = mod._TRELLIS2_PIPELINE.from_pretrained.call_args.args[0]
    assert src == "/tmp/local-trellis2"


def test_from_pretrained_gets_no_trust_remote_code():
    _wire_runtime()
    mod.TRELLIS2Runtime(
        model_id="m", hf_repo="microsoft/TRELLIS.2-4B", device="cpu",
        trust_remote_code=True,
    )
    assert "trust_remote_code" not in mod._TRELLIS2_PIPELINE.from_pretrained.call_args.kwargs
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/modalities/model_3d_generation/runtimes/test_trellis2.py -v`
Expected: FAIL with `ModuleNotFoundError: ...runtimes.trellis2` (module not created yet).

- [ ] **Step 3: Implement the runtime**

Create `src/muse/modalities/model_3d_generation/runtimes/trellis2.py`:

```python
"""TRELLIS2Runtime: image-to-3D via Microsoft's TRELLIS.2 SDK.

Wraps Trellis2ImageTo3DPipeline from the standalone `trellis2` package and
o_voxel.postprocess.to_glb for GLB export. This is a DIFFERENT SDK from the
original `trellis` package (see runtimes/trellis.py): trellis2 produces
O-Voxel meshes (not iso-surface meshes) and exports GLB via o_voxel, not by
constructing a trimesh.Trimesh from .vertices/.faces.

API (from the TRELLIS.2-4B model-card README; confirmed in B1):
  from trellis2.pipelines import Trellis2ImageTo3DPipeline
  pipeline = Trellis2ImageTo3DPipeline.from_pretrained(repo_or_path)
  pipeline.cuda()                      # .cuda(), not .to(device)
  mesh = pipeline.run(image)[0]        # image: PIL.Image; returns a list
  mesh.simplify(16777216)              # nvdiffrast vertex limit
  glb = o_voxel.postprocess.to_glb(vertices=, faces=, attr_volume=, coords=,
        attr_layout=, voxel_size=, aabb=, decimation_target=, texture_size=,
        remesh=, remesh_band=, remesh_project=, verbose=)
  # glb is trimesh-compatible: bytes(glb.export(file_type="glb")) -> GLB bytes

GPU-only (>=24GB VRAM), Linux, CUDA 12.4 toolchain.

Deferred-imports pattern: sentinels populated by _ensure_deps; tests patch
sentinels directly.
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

torch: Any = None
_TRELLIS2_PIPELINE: Any = None  # Trellis2ImageTo3DPipeline from trellis2.pipelines
_o_voxel: Any = None            # o_voxel package (postprocess.to_glb)
_LAST_IMPORT_ERROR: Exception | None = None

# o_voxel.postprocess.to_glb defaults (from the README usage block).
_AABB = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]
_DECIMATION_TARGET = 1_000_000
_TEXTURE_SIZE = 4096
_NVDIFFRAST_VERTEX_LIMIT = 16_777_216


def _ensure_deps() -> None:
    global torch, _TRELLIS2_PIPELINE, _o_voxel, _LAST_IMPORT_ERROR
    if torch is None:
        try:
            import torch as _t
            torch = _t
        except Exception as e:  # noqa: BLE001
            logger.debug("TRELLIS2Runtime torch unavailable: %s", e)
            _LAST_IMPORT_ERROR = e
    if _TRELLIS2_PIPELINE is None:
        try:
            from trellis2.pipelines import Trellis2ImageTo3DPipeline as _p
            _TRELLIS2_PIPELINE = _p
        except Exception as e:  # noqa: BLE001
            logger.debug("TRELLIS2Runtime pipeline unavailable: %s", e)
            _LAST_IMPORT_ERROR = e
    if _o_voxel is None:
        try:
            import o_voxel as _ov
            _o_voxel = _ov
        except Exception as e:  # noqa: BLE001
            logger.debug("TRELLIS2Runtime o_voxel unavailable: %s", e)
            _LAST_IMPORT_ERROR = e


class TRELLIS2Runtime:
    """Image-to-3D runtime over Microsoft's TRELLIS.2 SDK."""

    model_id: str
    supports_image_to_3d: bool = True
    supports_text_to_3d: bool = False
    supports_tools: bool = False

    def __init__(
        self,
        *,
        model_id: str,
        hf_repo: str,
        local_dir: str | None = None,
        device: str = "auto",
        dtype: str = "fp16",
        trust_remote_code: bool = False,
        seed: int | None = None,
        **_: Any,
    ) -> None:
        _ensure_deps()
        if torch is None:
            raise RuntimeError(
                "torch is not installed; run "
                f"`muse models refresh {model_id}` or install torch>=2.1.0"
            )
        if _TRELLIS2_PIPELINE is None:
            raise RuntimeError(
                "TRELLIS.2 SDK not available: `trellis2` package not installed. "
                f"Run `muse models refresh {model_id}`. trellis2 installs from "
                "git+https://github.com/microsoft/TRELLIS.2 and needs a CUDA "
                "toolchain; see its setup instructions if pip install fails."
            )
        if _o_voxel is None:
            raise RuntimeError(
                "o_voxel is not installed; needed for TRELLIS.2 GLB export. "
                f"Run `muse models refresh {model_id}`."
            )
        self.model_id = model_id
        self._device = select_device(device, torch_module=torch)
        self._dtype = dtype_for_name(dtype, torch_module=torch)
        self._default_seed = seed
        src = local_dir or hf_repo
        with LoadTimer(f"loading TRELLIS.2 from {src}", logger):
            self._pipeline = _TRELLIS2_PIPELINE.from_pretrained(src)
            if self._device == "cuda":
                self._pipeline.cuda()
            elif hasattr(self._pipeline, "to"):
                self._pipeline.to(self._device)
        set_inference_mode(self._pipeline)

    def image_to_3d(self, image: Any, **kwargs: Any) -> list[Generation3DResult]:
        """Generate one or more textured GLB meshes from a single image.

        The route layer passes a PIL.Image (v0.45.7 image-to-3d contract).
        kwargs: ``n`` (int, default 1) number of samples.
        """
        n = max(1, int(kwargs.get("n", 1)))
        results: list[Generation3DResult] = []
        for _ in range(n):
            mesh = self._pipeline.run(image)[0]
            mesh.simplify(_NVDIFFRAST_VERTEX_LIMIT)
            glb = _o_voxel.postprocess.to_glb(
                vertices=mesh.vertices,
                faces=mesh.faces,
                attr_volume=mesh.attrs,
                coords=mesh.coords,
                attr_layout=mesh.layout,
                voxel_size=mesh.voxel_size,
                aabb=_AABB,
                decimation_target=_DECIMATION_TARGET,
                texture_size=_TEXTURE_SIZE,
                remesh=True,
                remesh_band=1,
                remesh_project=0,
                verbose=False,
            )
            # to_glb returns a trimesh-compatible object; mesh_to_glb_result
            # does bytes(glb.export(file_type="glb")) with a zero-byte guard.
            results.append(mesh_to_glb_result(glb, self.model_id))
        return results

    def text_to_3d(self, prompt: str, **kwargs: Any) -> list[Generation3DResult]:
        raise NotImplementedError(
            "TRELLIS2Runtime is image-only; TRELLIS.2-4B does not support "
            "text-to-3D generation. Use Shap-E or Hunyuan3D for text-to-3D."
        )
```

- [ ] **Step 4: Run to verify all runtime tests pass**

Run: `pytest tests/modalities/model_3d_generation/runtimes/test_trellis2.py -v`
Expected: PASS (all tests).

- [ ] **Step 5: Run the runtime-helpers meta-test (no re-implementations)**

Run: `pytest tests/core/test_runtime_helpers_meta.py -v`
Expected: PASS (the new runtime uses the shared helpers).

- [ ] **Step 6: Commit**

```bash
git add src/muse/modalities/model_3d_generation/runtimes/trellis2.py tests/modalities/model_3d_generation/runtimes/test_trellis2.py
git commit -m "feat(3d): TRELLIS2Runtime (trellis2 + o_voxel image-to-3D)"
```

---

### Task 4: Curated entry + resolver routing test

**Files:**
- Modify: `src/muse/curated.yaml` (add `trellis2-image` in the 3d/generation section, after `hunyuan3d-2` / near `trellis-image`)
- Test: `tests/core/test_curated.py`

**Interfaces:**
- Consumes: `muse.core.curated.load_curated()`, the trellis2 `_Family` from Task 2.
- Produces: curated id `trellis2-image` -> `hf://microsoft/TRELLIS.2-4B`, modality `3d/generation`.

- [ ] **Step 1: Write the failing test**

Add to `tests/core/test_curated.py`:

```python
def test_load_curated_includes_trellis2_image():
    """v0.47.0: TRELLIS.2 curated entry, alongside the original trellis-image."""
    by_id = {e.id: e for e in load_curated()}
    assert "trellis-image" in by_id          # original kept (not replaced)
    assert "trellis2-image" in by_id
    e = by_id["trellis2-image"]
    assert e.modality == "3d/generation"
    assert e.uri == "hf://microsoft/TRELLIS.2-4B"
    assert (e.capabilities or {}).get("supports_image_to_3d") is True
    assert (e.capabilities or {}).get("supports_text_to_3d") is False
    assert (e.capabilities or {}).get("device") == "cuda"
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/core/test_curated.py::test_load_curated_includes_trellis2_image -v`
Expected: FAIL (`trellis2-image` not in curated).

- [ ] **Step 3: Add the curated entry**

In `src/muse/curated.yaml`, in the `3d/generation` section after the `hunyuan3d-2` entry, add:

```yaml
# TRELLIS.2 4B: newer TRELLIS, distinct SDK (trellis2 + o_voxel). Kept
# ALONGSIDE trellis-image (the original install is proven; trellis2's is
# newer). GPU-only: >=24GB VRAM, Linux, CUDA 12.4 toolchain, git-install SDK.
- id: trellis2-image
  uri: hf://microsoft/TRELLIS.2-4B
  modality: 3d/generation
  size_gb: 9.0
  description: "TRELLIS.2 4B: image-to-3D, textured PBR meshes, MIT (GPU-only, git-install SDK; >=24GB VRAM)"
  capabilities:
    device: cuda
    supports_image_to_3d: true
    supports_text_to_3d: false
    memory_gb: 24.0
```

- [ ] **Step 4: Run the curated tests**

Run: `pytest tests/core/test_curated.py -v`
Expected: PASS (new test plus the existing curated tests).

- [ ] **Step 5: Verify the curated id resolves to TRELLIS2Runtime (metadata only, network)**

Run:
```bash
python -c "
import muse.core.resolvers_hf as rhf
r = rhf.HFResolver()
m = r.resolve_via_modality('hf://microsoft/TRELLIS.2-4B', '3d/generation')
print(m.manifest['modality'], m.backend_path)
assert m.manifest['modality'] == '3d/generation'
assert m.backend_path.endswith(':TRELLIS2Runtime')
print('OK')
"
```
Expected: prints `3d/generation ...:TRELLIS2Runtime` then `OK`.

- [ ] **Step 6: Commit**

```bash
git add src/muse/curated.yaml tests/core/test_curated.py
git commit -m "feat(3d): curated trellis2-image entry (alongside trellis-image)"
```

---

### Task 5: Slow e2e supervisor test + opt-in GPU integration test

**Files:**
- Test: `tests/modalities/model_3d_generation/test_routes.py` (or the existing 3D slow-e2e test file) - add a slow e2e with a fake TRELLIS2Runtime-shaped backend
- Test: `tests/integration/test_remote_3d.py` (extend the opt-in suite)

**Interfaces:**
- Consumes: existing `/v1/3d/from-image` route + the test harness used by the current TRELLIS/Hunyuan3D e2e tests.

- [ ] **Step 1: Add the slow e2e test (mocked backend, no SDK, no GPU)**

Mirror the existing 3D route e2e: register a fake backend exposing
`supports_image_to_3d=True`, `supports_text_to_3d=False`, and
`image_to_3d(image, **kw) -> [Generation3DResult(glb_bytes=b"GLB", model_id="trellis2-image", format="glb")]`; POST a tiny PNG to `/v1/3d/from-image` with `model=trellis2-image`; assert 200 and a GLB data-url / b64 in the response. Mark `@pytest.mark.slow` if the sibling tests are.

```python
def test_from_image_trellis2_backend_returns_glb(<existing fixtures>):
    # register fake trellis2-image backend (image_to_3d -> [Generation3DResult])
    # POST multipart image -> assert 200 and GLB payload, model == "trellis2-image"
    ...
```

- [ ] **Step 2: Run it**

Run: `pytest tests/modalities/model_3d_generation/ -v -k trellis2`
Expected: PASS.

- [ ] **Step 3: Add the opt-in GPU integration test**

In `tests/integration/test_remote_3d.py`, add a `test_observe_*` test (xfail/skip-style, gated on `MUSE_REMOTE_SERVER` and the model being loaded) that POSTs an image to `/v1/3d/from-image` with `model=trellis2-image` and asserts a non-empty GLB. Auto-skips when the server/model is absent (mirror the existing 3D integration tests).

- [ ] **Step 4: Run the fast lane to confirm nothing regressed**

Run: `pytest tests/ -m "not slow" -q`
Expected: PASS (all green; integration tests auto-skip without `MUSE_REMOTE_SERVER`).

- [ ] **Step 5: Commit**

```bash
git add tests/modalities/model_3d_generation/ tests/integration/test_remote_3d.py
git commit -m "test(3d): trellis2 slow e2e + opt-in GPU integration"
```

---

### Task 6: Docs + v0.47.0 release

**Files:**
- Modify: `CLAUDE.md` (3d/generation paragraph: note TRELLIS.2 / TRELLIS2Runtime as the fourth 3D family)
- Modify: `pyproject.toml` (version bump)

- [ ] **Step 1: Update CLAUDE.md**

In the `3d/generation` section, add one sentence: TRELLIS.2-4B is served by `TRELLIS2Runtime` over the standalone `trellis2` + `o_voxel` SDK (distinct from the original `trellis`); image-to-3D only, GPU-only, curated as `trellis2-image` alongside `trellis-image`. Note the per-family dispatch now has five runtimes (TripoSR, Shap-E, TRELLIS, Hunyuan3D, TRELLIS.2). ASCII only.

- [ ] **Step 2: Bump version**

```bash
sed -i 's/^version = "0.46.1"/version = "0.47.0"/' pyproject.toml
grep '^version' pyproject.toml   # expect: version = "0.47.0"
```

- [ ] **Step 3: Full fast lane (pre-publish gate)**

Run: `pytest tests/ -m "not slow" -q`
Expected: PASS (all green).

- [ ] **Step 4: Build + twine check + WHEEL smoke-install**

```bash
rm -rf dist/ build/ src/museq.egg-info src/muse.egg-info
python -m build
twine check dist/*
python -m venv /tmp/museq-v0470 && /tmp/museq-v0470/bin/pip install -q dist/museq-0.47.0-py3-none-any.whl
/tmp/museq-v0470/bin/python -c "
from muse import __version__
from muse.core import curated
ids = [e.id for e in curated.load_curated()]
print('version:', __version__)
print('trellis2-image present:', 'trellis2-image' in ids)
print('curated count:', len(ids))
"
```
Expected: version `0.47.0`, `trellis2-image present: True`, curated count 78. (The wheel smoke-install is mandatory per the v0.46.0 packaging lesson.)

- [ ] **Step 5: Commit, tag, push, publish, GitHub release**

```bash
git add CLAUDE.md pyproject.toml
git commit -m "chore(release): v0.47.0 (TRELLIS.2 image-to-3D runtime)"
twine upload dist/*
git tag -a v0.47.0 -m "v0.47.0: TRELLIS.2 image-to-3D runtime"
git push origin main && git push origin v0.47.0
gh release create v0.47.0 --title "v0.47.0: TRELLIS.2 image-to-3D" --notes "<summary: new TRELLIS2Runtime over trellis2 + o_voxel; curated trellis2-image alongside trellis-image; GPU-only; first of four runtime upgrades>"
```

- [ ] **Step 6: Verify the release**

```bash
curl -s https://pypi.org/pypi/museq/0.47.0/json | python -c "import json,sys;print('PyPI:', json.load(sys.stdin)['info']['version'])"
gh release view v0.47.0 --json tagName,isDraft -q '"GH: \(.tagName) draft=\(.isDraft)"'
```
Expected: `PyPI: 0.47.0`, `GH: v0.47.0 draft=false`.

---

## Self-Review

**Spec coverage:**
- Why-new-runtime (different SDK) -> Tasks 2+3. Name-collision ordering -> Task 2 (+ regression test). Capabilities (image-only, GPU, textured) -> Task 3 + Task 4 curated caps. B1 gate -> Task 1. pip_extras/install -> Task 2 Step 3. Curated alongside -> Task 4. Wire unchanged -> no task needed (verified in Task 5 e2e). Testing (unit/dispatch/resolver/slow-e2e/opt-in GPU) -> Tasks 2,3,4,5. CI smoke exclusion -> Global Constraints (not added to matrix). Release + wheel smoke -> Task 6. All covered.

**Placeholder scan:** The runtime code, tests, _Family entry, and curated entry are concrete. The one genuine unknown (the `trellis2`/`o_voxel` git URL and the GLB-bytes recipe) is gated in Task 1 (B1) with the README-derived best-evidence values used in Tasks 2-4; Task 1 Step 2(7)/(8) explicitly resolves them and instructs correcting the pip_extras URL + the bytes path if reality differs. `size_gb`/`memory_gb` use conservative estimates (9GB/24GB) confirmable via `muse models probe`. No lazy TBDs.

**Type consistency:** `TRELLIS2Runtime`, `image_to_3d(image, **kwargs) -> list[Generation3DResult]`, sentinels `torch`/`_TRELLIS2_PIPELINE`/`_o_voxel`/`_LAST_IMPORT_ERROR`, `_TRELLIS2_RUNTIME_PATH` ending `:TRELLIS2Runtime`, curated id `trellis2-image` - all consistent across Tasks 2-6 and the tests.
