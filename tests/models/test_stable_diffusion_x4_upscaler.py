"""Tests for the SD x4 Upscaler bundled model script (fully mocked).

The script aliases the generic `DiffusersUpscaleRuntime` as its `Model`;
runtime behavior is covered by tests/modalities/image_upscale/runtimes/.
These tests pin the MANIFEST and check that the runtime honors it when
constructed the way `load_backend` does (`model_id` + capabilities
splatted as kwargs).
"""
import importlib
from unittest.mock import MagicMock, patch

from PIL import Image

from muse.modalities.image_upscale.protocol import UpscaleResult

_RT = "muse.modalities.image_upscale.runtimes.diffusers_upscaler"


def _mod():
    return importlib.import_module("muse.models.stable_diffusion_x4_upscaler")


def test_manifest_required_fields():
    M = _mod().MANIFEST
    assert M["model_id"] == "stable-diffusion-x4-upscaler"
    assert M["modality"] == "image/upscale"
    assert "hf_repo" in M
    assert "pip_extras" in M


def test_manifest_pip_extras_declares_torch_diffusers_transformers():
    extras_str = " ".join(_mod().MANIFEST["pip_extras"])
    assert "torch" in extras_str
    assert "diffusers" in extras_str
    assert "transformers" in extras_str


def test_manifest_advertises_scale_steps_and_guidance():
    caps = _mod().MANIFEST["capabilities"]
    assert caps["supported_scales"] == [4]
    assert caps["default_scale"] == 4
    assert caps["default_steps"] == 20
    assert caps["default_guidance"] == 9.0


def test_model_is_diffusers_upscale_runtime():
    from muse.modalities.image_upscale.runtimes.diffusers_upscaler import (
        DiffusersUpscaleRuntime,
    )
    assert _mod().Model is DiffusersUpscaleRuntime


def test_manifest_capabilities_drive_runtime_construction_and_defaults():
    mod = _mod()
    manifest = mod.MANIFEST
    fake_pipe = MagicMock()
    fake_pipe.return_value.images = [Image.new("RGB", (512, 512))]
    fake_pipe.to = MagicMock(return_value=fake_pipe)
    pipe_cls = MagicMock()
    pipe_cls.from_pretrained.return_value = fake_pipe

    kwargs = {"model_id": manifest["model_id"], **manifest["capabilities"]}
    with patch(f"{_RT}.StableDiffusionUpscalePipeline", pipe_cls), \
            patch(f"{_RT}.torch", MagicMock()):
        m = mod.Model(hf_repo=manifest["hf_repo"], local_dir="/fake", **kwargs)
        r = m.upscale(Image.new("RGB", (128, 128)))

    assert pipe_cls.from_pretrained.call_args.args[0] == "/fake"
    fake_pipe.to.assert_called_once_with("cuda")
    assert m.supported_scales == [4]
    call = fake_pipe.call_args.kwargs
    assert call["num_inference_steps"] == 20
    assert call["guidance_scale"] == 9.0
    assert call["prompt"] == ""
    assert isinstance(r, UpscaleResult)
    assert r.scale == 4
    assert (r.original_width, r.upscaled_width) == (128, 512)
    assert r.metadata["model"] == "stable-diffusion-x4-upscaler"
