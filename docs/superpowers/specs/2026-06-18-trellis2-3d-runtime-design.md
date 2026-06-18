# TRELLIS.2 3D runtime - design (v0.47.0)

**Status:** approved (brainstorm), pending spec review -> writing-plans.

**Goal:** Add Microsoft's **TRELLIS.2-4B** as a curated image-to-3D model in
muse's `3d/generation` modality, served by a new `TRELLIS2Runtime`. First of
four runtime-upgrade sub-projects (TRELLIS.2 -> ACE-Step -> Supertonic -> Wan2.2).

## Why a new runtime (not a curated.yaml line)

`microsoft/TRELLIS.2-4B` is a **different SDK** from the TRELLIS muse already
serves:

- `library: trellis2` (the original is `trellis`); HF tag `trellis2`;
  ships `pipeline.json` + `texturing_pipeline.json`.
- muse's existing `TRELLISRuntime` imports `TrellisImageTo3DPipeline` from the
  original `trellis` package (`trellis.pipelines`). TRELLIS.2 uses the
  `trellis2` package with a different (to-be-verified) API.

So a curated entry alone would mis-resolve. We need a new family + runtime.

**Name-collision (must handle):** `_family_for()` matches by name-hint and the
existing `trellis` hint matches `microsoft/trellis.2-4b` (the `.` is a word
boundary). The new `trellis2` family MUST be ordered **before** the `trellis`
family in `_FAMILIES` (first-match-wins).

## Capabilities (from the model card)

- `pipeline_tag: image-to-3d`; tags include `image-to-3d` only.
 -> `supports_image_to_3d: True`, `supports_text_to_3d: False`.
- License: MIT. Not gated.
- 4B params -> GPU-only (`device: cuda`); curated entry, never a bundled default
  (TripoSR remains the small bundled default).
- Has a texturing pipeline -> emit **textured** GLB by default.

## Architecture

Mirrors the v0.44.0 TRELLIS and v0.45.0 Hunyuan3D adds exactly (the per-family
dispatch was designed for this: "append one `_Family` entry plus a runtime
file"). No protocol/route/codec changes.

### 1. Dispatch (`modalities/model_3d_generation/hf.py`)

- Add module constants:
  - `_TRELLIS2_RUNTIME_PATH = "muse.modalities.model_3d_generation.runtimes.trellis2:TRELLIS2Runtime"`
  - `_TRELLIS2_PIP_EXTRAS` (see Install).
- Insert a new `_Family` **before** the existing `trellis` entry in `_FAMILIES`:
  ```python
  _Family(  # TRELLIS.2 (v0.47.0): different SDK from original TRELLIS
      name_hints=("trellis2", "trellis.2"),
      runtime_path=_TRELLIS2_RUNTIME_PATH,
      pip_extras=_TRELLIS2_PIP_EXTRAS,
      capability_overrides={
          "supports_image_to_3d": True,
          "supports_text_to_3d": False,
      },
      trust_remote_code=<set per B1 finding>,
  )
  ```
- Add `"trellis2"` (and rely on `trellis.2` via the hint) to `_NAME_HINTS`.
- Verify `_matches_hint("microsoft/trellis.2-4b", "trellis2")` and `"trellis.2"`
  both match, and that the new family wins over `trellis` (ordering test).

### 2. Runtime (`modalities/model_3d_generation/runtimes/trellis2.py`)

- New `TRELLIS2Runtime`, deferred-imports sentinel pattern (module-level
  sentinels for torch / the trellis2 pipeline class / trimesh; `_ensure_deps()`
  lazy-populates them; tests pre-populate with mocks; `muse --help` / `muse pull`
  must import without the SDK present).
- Use `muse.core.runtime_helpers` (`select_device`, `set_inference_mode`,
  `LoadTimer`) rather than re-implementing - the meta-test enforces this.
- Constructor accepts `hf_repo=`, `local_dir=`, `device=`, `**_`; prefers
  `local_dir`.
- `image_to_3d(image, **kwargs) -> Generation3DResult`: the route passes a
  PIL.Image (the v0.45.7 H1 contract - image-to-3d backends receive a PIL
  image, not a path). Convert/normalize as the SDK needs, run the pipeline,
  extract a `trimesh.Trimesh` (with texture), and return via the existing
  `mesh_to_glb_result(mesh, model_id)` codec helper (which already guards
  zero-byte exports, v0.45.8 M7).
- Document the **B1-verified** real API in the module docstring, exactly like
  `runtimes/trellis.py` does for the original (pipeline class + import path,
  load signature, run signature, device placement, mesh/texture extraction).

### 3. B1 verification gate (the one real risk)

Before writing the runtime body or any mock, verify against the **real
downloaded `trellis2` SDK**:

1. Pipeline class name + import path (e.g. `from trellis2.pipelines import ...`).
2. Load API: `from_pretrained(repo_or_local)` vs other; whether
   `trust_remote_code` applies.
3. `run(...)` signature: image arg, seed, formats, texturing toggle.
4. Device placement: `.cuda()` vs `.to(device)` (original TRELLIS needs
   `.cuda()` because the pipeline isn't an `nn.Module`).
5. Mesh + texture extraction -> `trimesh.Trimesh` with materials.
6. **The git install URL** for the `trellis2` package (confirm from the
   TRELLIS.2-4B README; it is NOT on PyPI).

This is the discipline that caught the v0.43 Shap-E mock-vs-reality break.
Record findings in the runtime docstring.

### 4. Install (`_TRELLIS2_PIP_EXTRAS`)

Mirror the TRELLIS / Hunyuan3D pip_extras shape:
```
torch>=2.1.0, torchvision>=0.16.0, transformers>=4.46.0, diffusers>=0.27.0,
trimesh, accelerate, Pillow, numpy,
"trellis2 @ git+<url confirmed in B1>",
```
Plus the same comment those carry: standalone GitHub SDK with native CUDA
build deps (kaolin/xformers/flash-attn/nvdiffrast class) that pip may fail to
install on hosts without a CUDA toolchain; documented fallback is to follow the
upstream setup script inside the per-model venv at `~/.muse/venvs/trellis2-image/`.

### 5. Curated entry (`curated.yaml`) - alongside, not replacing

Add `trellis2-image` next to the existing `trellis-image` (keep both: the
original's install is already proven; trellis2's is newer/less-validated, so
the old entry stays as a fallback). Explicit `modality: 3d/generation`.
```yaml
- id: trellis2-image
  uri: hf://microsoft/TRELLIS.2-4B
  modality: 3d/generation
  size_gb: <confirm at probe>
  description: "TRELLIS.2 4B: image-to-3D, textured meshes, MIT (GPU-only, git-install SDK)"
  capabilities:
    device: cuda
    supports_image_to_3d: true
    supports_text_to_3d: false
    trust_remote_code: <per B1>
    memory_gb: <~12-16, confirm at probe>
```

### 6. Wire - unchanged

`POST /v1/3d/from-image` already exists and routes image-to-3d backends a PIL
image; GLB output flows through the existing codec. No protocol, route, client,
or codec changes.

## Testing

- **Unit** (`tests/modalities/model_3d_generation/runtimes/test_trellis2.py`):
  mock the `trellis2` SDK (sentinels), mirror `test_trellis.py` - load, device
  placement, `image_to_3d` returns a `Generation3DResult` with non-empty
  `glb_bytes`, capability flags, deferred-import behavior. ~10-13 tests.
- **Dispatch** (`tests/modalities/model_3d_generation/test_hf_per_family_dispatch.py`):
  assert `microsoft/TRELLIS.2-4B` routes to `TRELLIS2Runtime` (not the original
  `TRELLISRuntime`), and the `_family_for` ordering (trellis2 wins over trellis).
- **Resolver** (mirror existing): `trellis2-image` curated id resolves via the
  explicit `modality: 3d/generation` to `TRELLIS2Runtime`, image-to-3d caps.
- **Slow e2e** supervisor test (mocked backend) and **opt-in** real-GPU
  integration test, gated like the other heavy 3D models.
- **CI smoke matrix:** GPU-only => NOT added to the free-tier fresh-venv smoke
  matrix (documented, consistent with TRELLIS / Hunyuan3D / video models).
- **pip_extras audit** (`test_pip_extras_audit.py`): the runtime's direct
  imports must be covered by `_TRELLIS2_PIP_EXTRAS`.

## Release

v0.47.0: bump, full fast lane, build + **wheel smoke-install** (verify version +
curated still loads, per the v0.46.0 packaging lesson), `twine upload`, tag,
push, GitHub release.

## Out of scope

- Text-to-3D (TRELLIS.2-4B is image-to-3D only).
- Replacing `trellis-image` (explicitly kept alongside).
- The other three runtime upgrades (ACE-Step, Supertonic, Wan2.2) - separate
  sub-projects/specs.
- A shared standalone-3D-SDK base class (rejected: per-runtime files are the
  settled granularity).

## Open items resolved at implementation time (B1)

- trellis2 pipeline class/import path, load + run signatures, device placement.
- The `trellis2` git install URL.
- `trust_remote_code` applicability.
- `size_gb` / `memory_gb` (confirm via `muse models probe` on a GPU host).
