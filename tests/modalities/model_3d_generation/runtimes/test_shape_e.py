"""ShapERuntime: mocked-dep tests.

Module-level sentinels (torch, ShapEPipeline, trimesh) get patched
per-test; the autouse fixture restores them on teardown.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import muse.modalities.model_3d_generation.runtimes.shape_e as mod


@pytest.fixture(autouse=True)
def _reset_module_sentinels():
    orig = (mod.torch, mod.ShapEPipeline, mod.trimesh, mod._LAST_IMPORT_ERROR)
    yield
    (mod.torch, mod.ShapEPipeline, mod.trimesh, mod._LAST_IMPORT_ERROR) = orig


# ---------------- missing-deps errors ----------------


@pytest.mark.parametrize(
    "sentinel_name, match_str",
    [
        ("torch", "torch is not installed"),
        ("ShapEPipeline", "diffusers"),
        ("trimesh", "trimesh"),
    ],
)
def test_raises_when_dep_not_installed(monkeypatch, sentinel_name, match_str):
    """RuntimeError names the missing dep so the operator can act."""
    monkeypatch.setattr(mod, "_ensure_deps", lambda: None)
    # Wire all deps to valid mocks, then null out the one under test.
    mod.torch = MagicMock()
    mod.ShapEPipeline = MagicMock()
    mod.trimesh = MagicMock()
    setattr(mod, sentinel_name, None)
    with pytest.raises(RuntimeError, match=match_str):
        mod.ShapERuntime(model_id="m", hf_repo="x", device="cpu")


# ---------------- helper: wire a fully-mocked runtime ----------------


def _wire_runtime():
    """Install fake torch + ShapEPipeline + trimesh.

    The pipeline is called with output_type="mesh" and returns a fake
    ShapEPipelineOutput whose .images[0] is a MeshDecoderOutput-like
    object with .verts / .faces tensor attributes (matching the real
    diffusers API). trimesh.Trimesh then takes the numpy arrays and
    produces a mesh whose .export() returns canned GLB bytes.
    """
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False
    fake_torch.backends = MagicMock(mps=None)
    mod.torch = fake_torch

    # Fake pipeline.
    pipeline = MagicMock()
    pipeline.to = MagicMock(return_value=pipeline)

    fake_mesh_data = MagicMock()
    fake_mesh_data.verts = MagicMock()
    fake_mesh_data.verts.cpu = MagicMock(return_value=fake_mesh_data.verts)
    fake_mesh_data.verts.numpy = MagicMock(return_value="vertices_array")
    fake_mesh_data.faces = MagicMock()
    fake_mesh_data.faces.cpu = MagicMock(return_value=fake_mesh_data.faces)
    fake_mesh_data.faces.numpy = MagicMock(return_value="faces_array")
    fake_result = MagicMock()
    fake_result.images = [fake_mesh_data]
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
    pipeline, fake_trimesh, fake_mesh = _wire_runtime()
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
    """Per-call guidance_scale / num_inference_steps override constructor defaults.
    output_type="mesh" is always forwarded; frame_size is not a valid kwarg."""
    pipeline, _, _ = _wire_runtime()
    runtime = mod.ShapERuntime(
        model_id="m", hf_repo="x", device="cpu",
        guidance_scale=15.0, num_inference_steps=64,
    )
    runtime.text_to_3d(
        "test prompt",
        guidance_scale=10.0,
        num_inference_steps=32,
    )
    call_kwargs = pipeline.call_args.kwargs
    assert call_kwargs["guidance_scale"] == 10.0
    assert call_kwargs["num_inference_steps"] == 32
    assert call_kwargs["output_type"] == "mesh"
    assert "frame_size" not in call_kwargs


def test_text_to_3d_uses_constructor_defaults_when_kwargs_absent():
    pipeline, _, _ = _wire_runtime()
    runtime = mod.ShapERuntime(
        model_id="m", hf_repo="x", device="cpu",
        guidance_scale=20.0, num_inference_steps=128,
    )
    runtime.text_to_3d("test")
    call_kwargs = pipeline.call_args.kwargs
    assert call_kwargs["guidance_scale"] == 20.0
    assert call_kwargs["num_inference_steps"] == 128


def test_pipeline_always_called_with_output_type_mesh():
    """Regression guard: output_type="mesh" must be forwarded on every call."""
    pipeline, _, _ = _wire_runtime()
    runtime = mod.ShapERuntime(model_id="m", hf_repo="x", device="cpu")
    runtime.text_to_3d("a sphere")
    assert pipeline.call_args.kwargs["output_type"] == "mesh"


def test_image_to_3d_raises_not_implemented():
    _wire_runtime()
    runtime = mod.ShapERuntime(model_id="m", hf_repo="x", device="cpu")
    with pytest.raises(NotImplementedError, match="text-to-3D only"):
        runtime.image_to_3d("/tmp/fake.png")


def test_supports_capability_attrs():
    """Class-level capability attributes match the manifest declaration."""
    assert mod.ShapERuntime.supports_text_to_3d is True
    assert mod.ShapERuntime.supports_image_to_3d is False


def test_local_dir_preferred_over_hf_repo():
    _wire_runtime()
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
    _wire_runtime()
    mod.ShapERuntime(model_id="m", hf_repo="x", device="cpu", dtype="fp16")
    call_kwargs = mod.ShapEPipeline.from_pretrained.call_args.kwargs
    assert "torch_dtype" in call_kwargs
