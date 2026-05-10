"""ShapERuntime: mocked-dep tests.

Module-level sentinels (torch, ShapEPipeline, trimesh) get patched
per-test; the autouse fixture restores them on teardown.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _reset_module_sentinels():
    import muse.modalities.model_3d_generation.runtimes.shape_e as mod
    orig = (mod.torch, mod.ShapEPipeline, mod.trimesh, mod._LAST_IMPORT_ERROR)
    yield
    (mod.torch, mod.ShapEPipeline, mod.trimesh, mod._LAST_IMPORT_ERROR) = orig


# ---------------- missing-deps errors ----------------


def test_raises_when_torch_not_installed(monkeypatch):
    import muse.modalities.model_3d_generation.runtimes.shape_e as mod
    monkeypatch.setattr(mod, "_ensure_deps", lambda: None)
    mod.torch = None
    mod.ShapEPipeline = MagicMock()
    mod.trimesh = MagicMock()
    with pytest.raises(RuntimeError, match="torch is not installed"):
        mod.ShapERuntime(model_id="m", hf_repo="x", device="cpu")


def test_raises_when_diffusers_not_installed(monkeypatch):
    """Error names diffusers>=0.27.0 so the operator can act."""
    import muse.modalities.model_3d_generation.runtimes.shape_e as mod
    monkeypatch.setattr(mod, "_ensure_deps", lambda: None)
    mod.torch = MagicMock()
    mod.ShapEPipeline = None
    mod.trimesh = MagicMock()
    with pytest.raises(RuntimeError, match=r"diffusers"):
        mod.ShapERuntime(model_id="m", hf_repo="x", device="cpu")


def test_raises_when_trimesh_not_installed(monkeypatch):
    import muse.modalities.model_3d_generation.runtimes.shape_e as mod
    monkeypatch.setattr(mod, "_ensure_deps", lambda: None)
    mod.torch = MagicMock()
    mod.ShapEPipeline = MagicMock()
    mod.trimesh = None
    with pytest.raises(RuntimeError, match=r"trimesh"):
        mod.ShapERuntime(model_id="m", hf_repo="x", device="cpu")


# ---------------- helper: wire a fully-mocked runtime ----------------


def _wire_runtime(mod):
    """Install fake torch + ShapEPipeline + trimesh.

    The pipeline returns a fake result with .meshes containing one mesh
    object whose .vertices / .faces are tensor-like. trimesh.Trimesh
    then takes the numpy arrays and produces a mesh whose .export()
    returns canned GLB bytes.
    """
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False
    fake_torch.backends = MagicMock(mps=None)
    mod.torch = fake_torch

    # Fake pipeline.
    pipeline = MagicMock()
    pipeline.to = MagicMock(return_value=pipeline)

    fake_mesh_data = MagicMock()
    fake_mesh_data.vertices = MagicMock()
    fake_mesh_data.vertices.cpu = MagicMock(return_value=fake_mesh_data.vertices)
    fake_mesh_data.vertices.numpy = MagicMock(return_value="vertices_array")
    fake_mesh_data.faces = MagicMock()
    fake_mesh_data.faces.cpu = MagicMock(return_value=fake_mesh_data.faces)
    fake_mesh_data.faces.numpy = MagicMock(return_value="faces_array")
    fake_result = MagicMock()
    fake_result.meshes = [fake_mesh_data]
    pipeline.return_value = fake_result

    pipe_factory = MagicMock()
    pipe_factory.from_pretrained = MagicMock(return_value=pipeline)
    mod.ShapEPipeline = pipe_factory

    # Fake trimesh.
    fake_mesh = MagicMock()
    fake_mesh.export = MagicMock(return_value=b"GLB_BYTES")
    fake_trimesh = MagicMock()
    fake_trimesh.Trimesh = MagicMock(return_value=fake_mesh)
    mod.trimesh = fake_trimesh

    return pipeline, fake_trimesh, fake_mesh


# ---------------- happy path ----------------


def test_text_to_3d_returns_glb_in_result():
    import muse.modalities.model_3d_generation.runtimes.shape_e as mod
    pipeline, fake_trimesh, fake_mesh = _wire_runtime(mod)
    runtime = mod.ShapERuntime(model_id="shap-e", hf_repo="openai/shap-e", device="cpu")
    results = runtime.text_to_3d("a chair shaped like an avocado")
    assert len(results) == 1
    assert results[0].model_id == "shap-e"
    assert results[0].glb_bytes == b"GLB_BYTES"
    fake_trimesh.Trimesh.assert_called_once_with(
        vertices="vertices_array", faces="faces_array",
    )
    fake_mesh.export.assert_called_once_with(file_type="glb")


def test_text_to_3d_forwards_kwargs_to_pipeline():
    """Per-call guidance_scale / num_inference_steps / frame_size override
    constructor defaults."""
    import muse.modalities.model_3d_generation.runtimes.shape_e as mod
    pipeline, _, _ = _wire_runtime(mod)
    runtime = mod.ShapERuntime(
        model_id="m", hf_repo="x", device="cpu",
        guidance_scale=15.0, num_inference_steps=64, frame_size=256,
    )
    runtime.text_to_3d(
        "test prompt",
        guidance_scale=10.0,
        num_inference_steps=32,
        frame_size=128,
    )
    call_kwargs = pipeline.call_args.kwargs
    assert call_kwargs["guidance_scale"] == 10.0
    assert call_kwargs["num_inference_steps"] == 32
    assert call_kwargs["frame_size"] == 128


def test_text_to_3d_uses_constructor_defaults_when_kwargs_absent():
    import muse.modalities.model_3d_generation.runtimes.shape_e as mod
    pipeline, _, _ = _wire_runtime(mod)
    runtime = mod.ShapERuntime(
        model_id="m", hf_repo="x", device="cpu",
        guidance_scale=20.0, num_inference_steps=128,
    )
    runtime.text_to_3d("test")
    call_kwargs = pipeline.call_args.kwargs
    assert call_kwargs["guidance_scale"] == 20.0
    assert call_kwargs["num_inference_steps"] == 128


def test_image_to_3d_raises_not_implemented():
    import muse.modalities.model_3d_generation.runtimes.shape_e as mod
    _wire_runtime(mod)
    runtime = mod.ShapERuntime(model_id="m", hf_repo="x", device="cpu")
    with pytest.raises(NotImplementedError, match="text-to-3D only"):
        runtime.image_to_3d("/tmp/fake.png")


def test_supports_capability_attrs():
    """Class-level capability attributes match the manifest declaration."""
    import muse.modalities.model_3d_generation.runtimes.shape_e as mod
    assert mod.ShapERuntime.supports_text_to_3d is True
    assert mod.ShapERuntime.supports_image_to_3d is False


def test_local_dir_preferred_over_hf_repo():
    import muse.modalities.model_3d_generation.runtimes.shape_e as mod
    _wire_runtime(mod)
    mod.ShapERuntime(
        model_id="m",
        hf_repo="openai/shap-e",
        local_dir="/tmp/local-shape-e",
        device="cpu",
    )
    src_arg = mod.ShapEPipeline.from_pretrained.call_args.args[0]
    assert src_arg == "/tmp/local-shape-e"


def test_dtype_resolution_passes_to_from_pretrained():
    """dtype string aliases reach ShapEPipeline.from_pretrained as torch_dtype."""
    import muse.modalities.model_3d_generation.runtimes.shape_e as mod
    _wire_runtime(mod)
    mod.ShapERuntime(model_id="m", hf_repo="x", device="cpu", dtype="fp16")
    call_kwargs = mod.ShapEPipeline.from_pretrained.call_args.kwargs
    assert "torch_dtype" in call_kwargs
