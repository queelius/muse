"""TripoSRRuntime: mocked-dep tests.

Module-level sentinels (torch, TSR, trimesh, PIL_Image,
_LAST_IMPORT_ERROR) get patched per-test; the autouse fixture
restores them on teardown.

The runtime wraps Stability AI's TripoSR via the upstream `tsr`
package. We mock TSR.from_pretrained to return a fake model whose
forward returns scene_codes and whose extract_mesh returns a list
containing one fake trimesh.Trimesh whose .export(file_type='glb')
returns deterministic bytes.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _reset_module_sentinels():
    """Reset module-top sentinels between tests (deferred-imports pattern)."""
    import muse.modalities.model_3d_generation.runtimes.triposr as mod
    orig = (
        mod.torch, mod.TSR, mod.trimesh, mod.PIL_Image,
        mod._LAST_IMPORT_ERROR,
    )
    yield
    (
        mod.torch, mod.TSR, mod.trimesh, mod.PIL_Image,
        mod._LAST_IMPORT_ERROR,
    ) = orig


def _make_no_grad_cm():
    """Build a context-manager mock for torch.no_grad()."""
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=None)
    cm.__exit__ = MagicMock(return_value=None)
    return cm


def _wire_basic_runtime(mod, *, glb_payload=b"fake-glb-bytes"):
    """Install fake torch + tsr + trimesh + PIL + a working model.

    Returns (fake_model, fake_mesh, fake_pil) for assertions.
    """
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False
    fake_torch.backends = MagicMock(mps=None)
    # torch.no_grad() must return a context manager.
    fake_torch.no_grad = MagicMock(return_value=_make_no_grad_cm())
    # dtype names referenced by dtype_for_name.
    fake_torch.float16 = "float16"
    fake_torch.float32 = "float32"
    fake_torch.bfloat16 = "bfloat16"
    mod.torch = fake_torch

    fake_pil = MagicMock()
    fake_image = MagicMock()
    fake_image.convert = MagicMock(return_value=fake_image)
    fake_pil.open = MagicMock(return_value=fake_image)
    mod.PIL_Image = fake_pil

    # Build a fake mesh whose export(file_type='glb') returns bytes.
    fake_mesh = MagicMock()
    fake_mesh.export = MagicMock(return_value=glb_payload)

    # Build the fake TSR model. It needs:
    #   - .renderer.set_chunk_size(int)
    #   - .to(device) -> self
    #   - __call__(images, device=...) -> scene_codes
    #   - .extract_mesh(scene_codes, has_vertex_color, resolution=...)
    fake_model = MagicMock()
    fake_model.to = MagicMock(return_value=fake_model)
    fake_model.renderer = MagicMock()
    fake_scene_codes = MagicMock(name="scene_codes")
    fake_model.return_value = fake_scene_codes
    fake_model.extract_mesh = MagicMock(return_value=[fake_mesh])

    # tsr.system.TSR -> a class-shaped object with from_pretrained.
    fake_tsr = MagicMock()
    fake_tsr.from_pretrained = MagicMock(return_value=fake_model)
    mod.TSR = fake_tsr

    mod.trimesh = MagicMock()

    return fake_model, fake_mesh, fake_pil


# ---------------- happy path ----------------


def test_image_to_3d_returns_glb_bytes():
    """One image -> one GLB blob with format='glb' and the right model_id."""
    import muse.modalities.model_3d_generation.runtimes.triposr as mod
    fake_model, fake_mesh, _ = _wire_basic_runtime(mod, glb_payload=b"glb-payload")

    runtime = mod.TripoSRRuntime(
        model_id="triposr-test",
        hf_repo="stabilityai/TripoSR",
        device="cpu",
    )
    results = runtime.image_to_3d("/tmp/fake.png")

    assert len(results) == 1
    assert results[0].glb_bytes == b"glb-payload"
    assert results[0].model_id == "triposr-test"
    # format is the protocol's default, populated by the dataclass.
    assert results[0].format == "glb"
    # The mesh's export was called with file_type='glb'.
    fake_mesh.export.assert_called_with(file_type="glb")


def test_image_to_3d_n_equals_two_returns_two_results():
    """n=2 returns 2 results (TripoSR is deterministic, so they're identical bytes)."""
    import muse.modalities.model_3d_generation.runtimes.triposr as mod
    _wire_basic_runtime(mod, glb_payload=b"x")

    runtime = mod.TripoSRRuntime(
        model_id="triposr",
        hf_repo="x",
        device="cpu",
    )
    results = runtime.image_to_3d("/tmp/fake.png", n=2)

    assert len(results) == 2
    assert all(r.glb_bytes == b"x" for r in results)
    assert all(r.model_id == "triposr" for r in results)


def test_image_to_3d_n_zero_clamped_to_one():
    """n=0 (or negative) clamps up to 1 result; never returns an empty list."""
    import muse.modalities.model_3d_generation.runtimes.triposr as mod
    _wire_basic_runtime(mod)

    runtime = mod.TripoSRRuntime(
        model_id="triposr", hf_repo="x", device="cpu",
    )
    # n=0 should still yield 1 result (max(1, n)).
    results = runtime.image_to_3d("/tmp/fake.png", n=0)
    assert len(results) == 1


def test_image_to_3d_seed_kwarg_accepted_but_ignored():
    """seed is forwarded for protocol uniformity but not consumed.

    Regression: the kwargs should not cause a TypeError; TripoSR is
    deterministic so the seed is meaningless to the upstream model.
    """
    import muse.modalities.model_3d_generation.runtimes.triposr as mod
    _wire_basic_runtime(mod)

    runtime = mod.TripoSRRuntime(
        model_id="triposr", hf_repo="x", device="cpu",
    )
    # Pass seed; expect no error and one result.
    results = runtime.image_to_3d("/tmp/fake.png", seed=42)
    assert len(results) == 1


def test_image_path_is_opened_via_pil():
    """The runtime must open the path via PIL.Image.open(...).convert('RGB')."""
    import muse.modalities.model_3d_generation.runtimes.triposr as mod
    _, _, fake_pil = _wire_basic_runtime(mod)

    runtime = mod.TripoSRRuntime(
        model_id="triposr", hf_repo="x", device="cpu",
    )
    runtime.image_to_3d("/path/to/input.png")

    fake_pil.open.assert_called_once_with("/path/to/input.png")
    # The opened image got .convert('RGB') called on it.
    fake_pil.open.return_value.convert.assert_called_once_with("RGB")


def test_extract_mesh_called_with_resolution_and_vertex_color():
    """Constructor's mc_resolution + has_vertex_color flow into extract_mesh."""
    import muse.modalities.model_3d_generation.runtimes.triposr as mod
    fake_model, _, _ = _wire_basic_runtime(mod)

    runtime = mod.TripoSRRuntime(
        model_id="triposr", hf_repo="x", device="cpu",
        mc_resolution=128, has_vertex_color=True,
    )
    runtime.image_to_3d("/tmp/fake.png")

    args, kwargs = fake_model.extract_mesh.call_args
    # Second positional arg is has_vertex_color.
    assert args[1] is True
    assert kwargs.get("resolution") == 128


def test_renderer_chunk_size_applied_at_load():
    """Constructor must call renderer.set_chunk_size with the configured value."""
    import muse.modalities.model_3d_generation.runtimes.triposr as mod
    fake_model, _, _ = _wire_basic_runtime(mod)

    mod.TripoSRRuntime(
        model_id="triposr", hf_repo="x", device="cpu", chunk_size=4096,
    )
    fake_model.renderer.set_chunk_size.assert_called_once_with(4096)


def test_model_moved_to_device():
    """model.to(device) is called with the resolved device."""
    import muse.modalities.model_3d_generation.runtimes.triposr as mod
    fake_model, _, _ = _wire_basic_runtime(mod)

    mod.TripoSRRuntime(
        model_id="triposr", hf_repo="x", device="cpu",
    )
    fake_model.to.assert_called_once_with("cpu")


def test_local_dir_preferred_over_hf_repo():
    """When local_dir is set, TSR.from_pretrained gets the local snapshot path."""
    import muse.modalities.model_3d_generation.runtimes.triposr as mod
    _wire_basic_runtime(mod)

    mod.TripoSRRuntime(
        model_id="triposr",
        hf_repo="stabilityai/TripoSR",
        local_dir="/tmp/local-snapshot",
        device="cpu",
    )
    args, kwargs = mod.TSR.from_pretrained.call_args
    assert args[0] == "/tmp/local-snapshot"
    # config_name and weight_name must default to the upstream layout.
    assert kwargs["config_name"] == "config.yaml"
    assert kwargs["weight_name"] == "model.ckpt"


def test_config_and_weight_name_overrides_forwarded():
    """Manifest may override config_name + weight_name (safetensors variants)."""
    import muse.modalities.model_3d_generation.runtimes.triposr as mod
    _wire_basic_runtime(mod)

    mod.TripoSRRuntime(
        model_id="triposr-st",
        hf_repo="some/triposr-safetensors",
        device="cpu",
        config_name="cfg.yaml",
        weight_name="model.safetensors",
    )
    _, kwargs = mod.TSR.from_pretrained.call_args
    assert kwargs["config_name"] == "cfg.yaml"
    assert kwargs["weight_name"] == "model.safetensors"


def test_forward_call_passes_device():
    """The model forward call must be invoked with [image] and device=<resolved>."""
    import muse.modalities.model_3d_generation.runtimes.triposr as mod
    fake_model, _, _ = _wire_basic_runtime(mod)

    runtime = mod.TripoSRRuntime(
        model_id="triposr", hf_repo="x", device="cpu",
    )
    runtime.image_to_3d("/tmp/fake.png")

    # fake_model is callable; the runtime invokes it as
    # `self._model([image], device=self._device)`. Inspect the
    # last positional call_args.
    args, kwargs = fake_model.call_args
    assert kwargs.get("device") == "cpu"
    # First positional is the image-list.
    assert isinstance(args[0], list)
    assert len(args[0]) == 1


def test_empty_meshes_raises_clear_runtime_error():
    """If extract_mesh returns [], the runtime raises with a clear message."""
    import muse.modalities.model_3d_generation.runtimes.triposr as mod
    fake_model, _, _ = _wire_basic_runtime(mod)
    fake_model.extract_mesh = MagicMock(return_value=[])

    runtime = mod.TripoSRRuntime(
        model_id="triposr", hf_repo="x", device="cpu",
    )
    with pytest.raises(RuntimeError, match="produced no meshes"):
        runtime.image_to_3d("/tmp/empty.png")


def test_glb_bytes_field_is_concrete_bytes():
    """If trimesh export returns a bytearray-like, the runtime coerces to bytes
    so downstream codec.b64encode never fails on a non-bytes object."""
    import muse.modalities.model_3d_generation.runtimes.triposr as mod
    fake_model, fake_mesh, _ = _wire_basic_runtime(mod)
    # Some trimesh versions return memoryview / bytearray; coerce.
    fake_mesh.export = MagicMock(return_value=bytearray(b"ba"))

    runtime = mod.TripoSRRuntime(
        model_id="triposr", hf_repo="x", device="cpu",
    )
    results = runtime.image_to_3d("/tmp/fake.png")
    assert isinstance(results[0].glb_bytes, bytes)
    assert results[0].glb_bytes == b"ba"


# ---------------- error envelopes for missing deps ----------------


def test_raises_when_torch_not_installed(monkeypatch):
    import muse.modalities.model_3d_generation.runtimes.triposr as mod
    monkeypatch.setattr(mod, "_ensure_deps", lambda: None)
    mod.torch = None
    mod.TSR = MagicMock()
    mod.trimesh = MagicMock()
    mod.PIL_Image = MagicMock()
    with pytest.raises(RuntimeError, match="torch is not installed"):
        mod.TripoSRRuntime(
            model_id="triposr", hf_repo="x", device="cpu",
        )


def test_raises_when_tsr_not_installed_mentions_model_id(monkeypatch):
    """The error must surface the model_id so the user knows which venv
    to refresh."""
    import muse.modalities.model_3d_generation.runtimes.triposr as mod
    monkeypatch.setattr(mod, "_ensure_deps", lambda: None)
    mod.torch = MagicMock()
    mod.TSR = None
    mod.trimesh = MagicMock()
    mod.PIL_Image = MagicMock()
    with pytest.raises(RuntimeError, match="tsr.*triposr-x"):
        mod.TripoSRRuntime(
            model_id="triposr-x", hf_repo="x", device="cpu",
        )


def test_raises_when_trimesh_not_installed(monkeypatch):
    import muse.modalities.model_3d_generation.runtimes.triposr as mod
    monkeypatch.setattr(mod, "_ensure_deps", lambda: None)
    mod.torch = MagicMock()
    mod.TSR = MagicMock()
    mod.trimesh = None
    mod.PIL_Image = MagicMock()
    with pytest.raises(RuntimeError, match="trimesh.*not installed"):
        mod.TripoSRRuntime(
            model_id="triposr", hf_repo="x", device="cpu",
        )


def test_raises_when_pillow_not_installed(monkeypatch):
    import muse.modalities.model_3d_generation.runtimes.triposr as mod
    monkeypatch.setattr(mod, "_ensure_deps", lambda: None)
    mod.torch = MagicMock()
    mod.TSR = MagicMock()
    mod.trimesh = MagicMock()
    mod.PIL_Image = None
    with pytest.raises(RuntimeError, match="Pillow.*not installed"):
        mod.TripoSRRuntime(
            model_id="triposr", hf_repo="x", device="cpu",
        )


# ---------------- module-level helpers ----------------


def test_select_device_delegates_to_runtime_helper():
    """The thin delegator must call the canonical select_device helper."""
    import muse.modalities.model_3d_generation.runtimes.triposr as mod
    # The module-level _select_device is the thin delegator. With torch
    # set and cuda unavailable, "auto" should resolve to "cpu".
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False
    fake_torch.backends = MagicMock(mps=None)
    mod.torch = fake_torch
    assert mod._select_device("auto") == "cpu"
    assert mod._select_device("cuda") == "cuda"


def test_resolve_dtype_delegates_to_runtime_helper():
    """The thin delegator returns the right torch dtype attribute."""
    import muse.modalities.model_3d_generation.runtimes.triposr as mod
    fake_torch = MagicMock()
    fake_torch.float16 = "FP16-SENTINEL"
    fake_torch.bfloat16 = "BF16-SENTINEL"
    fake_torch.float32 = "FP32-SENTINEL"
    mod.torch = fake_torch
    assert mod._resolve_dtype("fp16") == "FP16-SENTINEL"
    assert mod._resolve_dtype("bf16") == "BF16-SENTINEL"
    assert mod._resolve_dtype("fp32") == "FP32-SENTINEL"
