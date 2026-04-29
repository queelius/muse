"""Tests for the SD x4 Upscaler bundled model script (fully mocked)."""
import importlib
from unittest.mock import MagicMock, patch

from PIL import Image

from muse.modalities.image_upscale.protocol import UpscaleResult


def _patched_pipe(out_size=(512, 512)):
    """Fake StableDiffusionUpscalePipeline whose call returns one image."""
    fake_img = Image.new("RGB", out_size, (40, 40, 40))
    fake_pipe = MagicMock()
    fake_pipe.return_value.images = [fake_img]
    fake_pipe.to = MagicMock(return_value=fake_pipe)
    return fake_pipe


def test_manifest_required_fields():
    mod = importlib.import_module("muse.models.stable_diffusion_x4_upscaler")
    M = mod.MANIFEST
    assert M["model_id"] == "stable-diffusion-x4-upscaler"
    assert M["modality"] == "image/upscale"
    assert "hf_repo" in M
    assert "pip_extras" in M


def test_manifest_pip_extras_declares_torch_diffusers_transformers():
    mod = importlib.import_module("muse.models.stable_diffusion_x4_upscaler")
    extras_str = " ".join(mod.MANIFEST["pip_extras"])
    assert "torch" in extras_str
    assert "diffusers" in extras_str
    assert "transformers" in extras_str


def test_manifest_advertises_supported_scales_4():
    mod = importlib.import_module("muse.models.stable_diffusion_x4_upscaler")
    caps = mod.MANIFEST["capabilities"]
    assert caps["supported_scales"] == [4]
    assert caps["default_scale"] == 4


def test_manifest_advertises_default_steps_and_guidance():
    mod = importlib.import_module("muse.models.stable_diffusion_x4_upscaler")
    caps = mod.MANIFEST["capabilities"]
    assert caps["default_steps"] == 20
    assert caps["default_guidance"] == 9.0


def test_model_class_supported_scales_attr():
    mod = importlib.import_module("muse.models.stable_diffusion_x4_upscaler")
    assert mod.Model.supported_scales == [4]


def test_model_loads_via_patched_pipeline():
    fake_cls = MagicMock()
    fake_cls.from_pretrained.return_value = _patched_pipe()
    mod = importlib.import_module("muse.models.stable_diffusion_x4_upscaler")
    with patch.object(mod, "StableDiffusionUpscalePipeline", fake_cls), \
         patch.object(mod, "torch", MagicMock()):
        m = mod.Model(
            hf_repo="stabilityai/stable-diffusion-x4-upscaler",
            local_dir="/fake",
            device="cpu",
        )
    assert m.model_id == "stable-diffusion-x4-upscaler"
    assert m.supported_scales == [4]
    fake_cls.from_pretrained.assert_called_once()


def test_model_uses_local_dir_over_hf_repo():
    fake_cls = MagicMock()
    fake_cls.from_pretrained.return_value = _patched_pipe()
    mod = importlib.import_module("muse.models.stable_diffusion_x4_upscaler")
    with patch.object(mod, "StableDiffusionUpscalePipeline", fake_cls), \
         patch.object(mod, "torch", MagicMock()):
        mod.Model(
            hf_repo="stabilityai/stable-diffusion-x4-upscaler",
            local_dir="/real/local",
            device="cpu",
        )
    assert fake_cls.from_pretrained.call_args.args[0] == "/real/local"


def test_model_falls_back_to_hf_repo_when_no_local_dir():
    fake_cls = MagicMock()
    fake_cls.from_pretrained.return_value = _patched_pipe()
    mod = importlib.import_module("muse.models.stable_diffusion_x4_upscaler")
    with patch.object(mod, "StableDiffusionUpscalePipeline", fake_cls), \
         patch.object(mod, "torch", MagicMock()):
        mod.Model(
            hf_repo="stabilityai/stable-diffusion-x4-upscaler",
            local_dir=None,
            device="cpu",
        )
    assert (
        fake_cls.from_pretrained.call_args.args[0]
        == "stabilityai/stable-diffusion-x4-upscaler"
    )


def test_model_upscale_returns_upscale_result():
    fake_cls = MagicMock()
    fake_cls.from_pretrained.return_value = _patched_pipe(out_size=(512, 512))
    mod = importlib.import_module("muse.models.stable_diffusion_x4_upscaler")
    with patch.object(mod, "StableDiffusionUpscalePipeline", fake_cls), \
         patch.object(mod, "torch", MagicMock()):
        m = mod.Model(
            hf_repo="stabilityai/stable-diffusion-x4-upscaler",
            local_dir="/fake",
            device="cpu",
        )
        src = Image.new("RGB", (128, 128))
        result = m.upscale(src, scale=4)
    assert isinstance(result, UpscaleResult)
    assert result.original_width == 128
    assert result.original_height == 128
    assert result.upscaled_width == 512
    assert result.upscaled_height == 512
    assert result.scale == 4


def test_model_passes_prompt_to_pipeline():
    fake_pipe = _patched_pipe()
    fake_cls = MagicMock()
    fake_cls.from_pretrained.return_value = fake_pipe
    mod = importlib.import_module("muse.models.stable_diffusion_x4_upscaler")
    with patch.object(mod, "StableDiffusionUpscalePipeline", fake_cls), \
         patch.object(mod, "torch", MagicMock()):
        m = mod.Model(
            hf_repo="x", local_dir="/fake", device="cpu",
        )
        src = Image.new("RGB", (128, 128))
        m.upscale(src, scale=4, prompt="razor sharp")
    assert fake_pipe.call_args.kwargs["prompt"] == "razor sharp"


def test_model_defaults_empty_prompt():
    fake_pipe = _patched_pipe()
    fake_cls = MagicMock()
    fake_cls.from_pretrained.return_value = fake_pipe
    mod = importlib.import_module("muse.models.stable_diffusion_x4_upscaler")
    with patch.object(mod, "StableDiffusionUpscalePipeline", fake_cls), \
         patch.object(mod, "torch", MagicMock()):
        m = mod.Model(
            hf_repo="x", local_dir="/fake", device="cpu",
        )
        src = Image.new("RGB", (128, 128))
        m.upscale(src, scale=4)
    assert fake_pipe.call_args.kwargs["prompt"] == ""


def test_model_uses_seeded_generator():
    fake_pipe = _patched_pipe()
    fake_cls = MagicMock()
    fake_cls.from_pretrained.return_value = fake_pipe
    fake_torch = MagicMock()
    mod = importlib.import_module("muse.models.stable_diffusion_x4_upscaler")
    with patch.object(mod, "StableDiffusionUpscalePipeline", fake_cls), \
         patch.object(mod, "torch", fake_torch):
        m = mod.Model(hf_repo="x", local_dir="/fake", device="cpu")
        src = Image.new("RGB", (128, 128))
        m.upscale(src, scale=4, seed=7)
    fake_torch.Generator.return_value.manual_seed.assert_called_with(7)


def test_model_honors_custom_steps_and_guidance():
    fake_pipe = _patched_pipe()
    fake_cls = MagicMock()
    fake_cls.from_pretrained.return_value = fake_pipe
    mod = importlib.import_module("muse.models.stable_diffusion_x4_upscaler")
    with patch.object(mod, "StableDiffusionUpscalePipeline", fake_cls), \
         patch.object(mod, "torch", MagicMock()):
        m = mod.Model(hf_repo="x", local_dir="/fake", device="cpu")
        src = Image.new("RGB", (128, 128))
        m.upscale(src, scale=4, steps=10, guidance=3.0)
    assert fake_pipe.call_args.kwargs["num_inference_steps"] == 10
    assert fake_pipe.call_args.kwargs["guidance_scale"] == 3.0


def test_model_default_steps_is_20():
    fake_pipe = _patched_pipe()
    fake_cls = MagicMock()
    fake_cls.from_pretrained.return_value = fake_pipe
    mod = importlib.import_module("muse.models.stable_diffusion_x4_upscaler")
    with patch.object(mod, "StableDiffusionUpscalePipeline", fake_cls), \
         patch.object(mod, "torch", MagicMock()):
        m = mod.Model(hf_repo="x", local_dir="/fake", device="cpu")
        src = Image.new("RGB", (128, 128))
        m.upscale(src, scale=4)
    assert fake_pipe.call_args.kwargs["num_inference_steps"] == 20


def test_model_default_guidance_is_9():
    fake_pipe = _patched_pipe()
    fake_cls = MagicMock()
    fake_cls.from_pretrained.return_value = fake_pipe
    mod = importlib.import_module("muse.models.stable_diffusion_x4_upscaler")
    with patch.object(mod, "StableDiffusionUpscalePipeline", fake_cls), \
         patch.object(mod, "torch", MagicMock()):
        m = mod.Model(hf_repo="x", local_dir="/fake", device="cpu")
        src = Image.new("RGB", (128, 128))
        m.upscale(src, scale=4)
    assert fake_pipe.call_args.kwargs["guidance_scale"] == 9.0


def test_model_passes_negative_prompt():
    fake_pipe = _patched_pipe()
    fake_cls = MagicMock()
    fake_cls.from_pretrained.return_value = fake_pipe
    mod = importlib.import_module("muse.models.stable_diffusion_x4_upscaler")
    with patch.object(mod, "StableDiffusionUpscalePipeline", fake_cls), \
         patch.object(mod, "torch", MagicMock()):
        m = mod.Model(hf_repo="x", local_dir="/fake", device="cpu")
        src = Image.new("RGB", (128, 128))
        m.upscale(src, scale=4, negative_prompt="blurry")
    assert fake_pipe.call_args.kwargs["negative_prompt"] == "blurry"


def test_model_metadata_records_model_id():
    fake_pipe = _patched_pipe()
    fake_cls = MagicMock()
    fake_cls.from_pretrained.return_value = fake_pipe
    mod = importlib.import_module("muse.models.stable_diffusion_x4_upscaler")
    with patch.object(mod, "StableDiffusionUpscalePipeline", fake_cls), \
         patch.object(mod, "torch", MagicMock()):
        m = mod.Model(hf_repo="x", local_dir="/fake", device="cpu")
        src = Image.new("RGB", (128, 128))
        result = m.upscale(src, scale=4, prompt="hi")
    assert result.metadata["model"] == "stable-diffusion-x4-upscaler"
    assert result.metadata["prompt"] == "hi"


def test_model_accepts_unknown_kwargs():
    """Future catalog kwargs should be absorbed by **_."""
    fake_cls = MagicMock()
    fake_cls.from_pretrained.return_value = _patched_pipe()
    mod = importlib.import_module("muse.models.stable_diffusion_x4_upscaler")
    with patch.object(mod, "StableDiffusionUpscalePipeline", fake_cls), \
         patch.object(mod, "torch", MagicMock()):
        # Should not TypeError
        mod.Model(
            hf_repo="x", local_dir="/fake", device="cpu",
            extra_future_param="ignored",
        )
