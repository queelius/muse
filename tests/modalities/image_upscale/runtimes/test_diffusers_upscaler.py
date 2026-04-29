"""Tests for DiffusersUpscaleRuntime (fully patched; no real weights)."""
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


def test_runtime_constructor_loads_pipeline():
    fake_cls = MagicMock()
    fake_cls.from_pretrained.return_value = _patched_pipe()
    with patch(
        "muse.modalities.image_upscale.runtimes.diffusers_upscaler"
        ".StableDiffusionUpscalePipeline",
        fake_cls,
    ), patch(
        "muse.modalities.image_upscale.runtimes.diffusers_upscaler.torch",
        MagicMock(),
    ):
        from muse.modalities.image_upscale.runtimes.diffusers_upscaler import (
            DiffusersUpscaleRuntime,
        )
        m = DiffusersUpscaleRuntime(
            hf_repo="stabilityai/stable-diffusion-x4-upscaler",
            local_dir="/fake",
            device="cpu",
            model_id="stable-diffusion-x4-upscaler",
        )
        assert m.model_id == "stable-diffusion-x4-upscaler"
        assert m.supported_scales == [4]
    fake_cls.from_pretrained.assert_called_once()


def test_runtime_uses_local_dir_over_hf_repo():
    fake_cls = MagicMock()
    fake_cls.from_pretrained.return_value = _patched_pipe()
    with patch(
        "muse.modalities.image_upscale.runtimes.diffusers_upscaler"
        ".StableDiffusionUpscalePipeline",
        fake_cls,
    ), patch(
        "muse.modalities.image_upscale.runtimes.diffusers_upscaler.torch",
        MagicMock(),
    ):
        from muse.modalities.image_upscale.runtimes.diffusers_upscaler import (
            DiffusersUpscaleRuntime,
        )
        DiffusersUpscaleRuntime(
            hf_repo="stabilityai/stable-diffusion-x4-upscaler",
            local_dir="/real/local",
            device="cpu",
            model_id="x",
        )
    assert fake_cls.from_pretrained.call_args.args[0] == "/real/local"


def test_runtime_falls_back_to_hf_repo_when_no_local_dir():
    fake_cls = MagicMock()
    fake_cls.from_pretrained.return_value = _patched_pipe()
    with patch(
        "muse.modalities.image_upscale.runtimes.diffusers_upscaler"
        ".StableDiffusionUpscalePipeline",
        fake_cls,
    ), patch(
        "muse.modalities.image_upscale.runtimes.diffusers_upscaler.torch",
        MagicMock(),
    ):
        from muse.modalities.image_upscale.runtimes.diffusers_upscaler import (
            DiffusersUpscaleRuntime,
        )
        DiffusersUpscaleRuntime(
            hf_repo="stabilityai/stable-diffusion-x4-upscaler",
            local_dir=None,
            device="cpu",
            model_id="x",
        )
    assert (
        fake_cls.from_pretrained.call_args.args[0]
        == "stabilityai/stable-diffusion-x4-upscaler"
    )


def test_runtime_upscale_returns_upscale_result_with_correct_dims():
    fake_cls = MagicMock()
    fake_cls.from_pretrained.return_value = _patched_pipe(out_size=(512, 512))
    with patch(
        "muse.modalities.image_upscale.runtimes.diffusers_upscaler"
        ".StableDiffusionUpscalePipeline",
        fake_cls,
    ), patch(
        "muse.modalities.image_upscale.runtimes.diffusers_upscaler.torch",
        MagicMock(),
    ):
        from muse.modalities.image_upscale.runtimes.diffusers_upscaler import (
            DiffusersUpscaleRuntime,
        )
        m = DiffusersUpscaleRuntime(
            hf_repo="stabilityai/stable-diffusion-x4-upscaler",
            local_dir="/fake",
            device="cpu",
            model_id="stable-diffusion-x4-upscaler",
        )
        src = Image.new("RGB", (128, 128), (10, 10, 10))
        result = m.upscale(src, scale=4)
        assert isinstance(result, UpscaleResult)
        assert result.original_width == 128
        assert result.original_height == 128
        assert result.upscaled_width == 512
        assert result.upscaled_height == 512
        assert result.scale == 4


def test_runtime_passes_prompt_to_pipeline():
    fake_pipe = _patched_pipe()
    fake_cls = MagicMock()
    fake_cls.from_pretrained.return_value = fake_pipe
    with patch(
        "muse.modalities.image_upscale.runtimes.diffusers_upscaler"
        ".StableDiffusionUpscalePipeline",
        fake_cls,
    ), patch(
        "muse.modalities.image_upscale.runtimes.diffusers_upscaler.torch",
        MagicMock(),
    ):
        from muse.modalities.image_upscale.runtimes.diffusers_upscaler import (
            DiffusersUpscaleRuntime,
        )
        m = DiffusersUpscaleRuntime(
            hf_repo="x", local_dir="/fake", device="cpu", model_id="x",
        )
        src = Image.new("RGB", (128, 128), (0, 0, 0))
        m.upscale(src, scale=4, prompt="sharper than the original")
    assert fake_pipe.call_args.kwargs["prompt"] == "sharper than the original"


def test_runtime_defaults_empty_prompt():
    fake_pipe = _patched_pipe()
    fake_cls = MagicMock()
    fake_cls.from_pretrained.return_value = fake_pipe
    with patch(
        "muse.modalities.image_upscale.runtimes.diffusers_upscaler"
        ".StableDiffusionUpscalePipeline",
        fake_cls,
    ), patch(
        "muse.modalities.image_upscale.runtimes.diffusers_upscaler.torch",
        MagicMock(),
    ):
        from muse.modalities.image_upscale.runtimes.diffusers_upscaler import (
            DiffusersUpscaleRuntime,
        )
        m = DiffusersUpscaleRuntime(
            hf_repo="x", local_dir="/fake", device="cpu", model_id="x",
        )
        src = Image.new("RGB", (128, 128))
        m.upscale(src, scale=4)
    assert fake_pipe.call_args.kwargs["prompt"] == ""


def test_runtime_uses_seeded_generator():
    fake_pipe = _patched_pipe()
    fake_cls = MagicMock()
    fake_cls.from_pretrained.return_value = fake_pipe
    fake_torch = MagicMock()
    with patch(
        "muse.modalities.image_upscale.runtimes.diffusers_upscaler"
        ".StableDiffusionUpscalePipeline",
        fake_cls,
    ), patch(
        "muse.modalities.image_upscale.runtimes.diffusers_upscaler.torch",
        fake_torch,
    ):
        from muse.modalities.image_upscale.runtimes.diffusers_upscaler import (
            DiffusersUpscaleRuntime,
        )
        m = DiffusersUpscaleRuntime(
            hf_repo="x", local_dir="/fake", device="cpu", model_id="x",
        )
        src = Image.new("RGB", (128, 128))
        m.upscale(src, scale=4, seed=42)
    fake_torch.Generator.return_value.manual_seed.assert_called_with(42)


def test_runtime_no_generator_when_seed_omitted():
    fake_pipe = _patched_pipe()
    fake_cls = MagicMock()
    fake_cls.from_pretrained.return_value = fake_pipe
    fake_torch = MagicMock()
    with patch(
        "muse.modalities.image_upscale.runtimes.diffusers_upscaler"
        ".StableDiffusionUpscalePipeline",
        fake_cls,
    ), patch(
        "muse.modalities.image_upscale.runtimes.diffusers_upscaler.torch",
        fake_torch,
    ):
        from muse.modalities.image_upscale.runtimes.diffusers_upscaler import (
            DiffusersUpscaleRuntime,
        )
        m = DiffusersUpscaleRuntime(
            hf_repo="x", local_dir="/fake", device="cpu", model_id="x",
        )
        src = Image.new("RGB", (128, 128))
        m.upscale(src, scale=4)
    # Generator was not requested
    fake_torch.Generator.assert_not_called()
    # And no `generator` kwarg was passed to the pipe
    assert "generator" not in fake_pipe.call_args.kwargs


def test_runtime_honors_custom_steps_and_guidance():
    fake_pipe = _patched_pipe()
    fake_cls = MagicMock()
    fake_cls.from_pretrained.return_value = fake_pipe
    with patch(
        "muse.modalities.image_upscale.runtimes.diffusers_upscaler"
        ".StableDiffusionUpscalePipeline",
        fake_cls,
    ), patch(
        "muse.modalities.image_upscale.runtimes.diffusers_upscaler.torch",
        MagicMock(),
    ):
        from muse.modalities.image_upscale.runtimes.diffusers_upscaler import (
            DiffusersUpscaleRuntime,
        )
        m = DiffusersUpscaleRuntime(
            hf_repo="x", local_dir="/fake", device="cpu", model_id="x",
            default_steps=20, default_guidance=9.0,
        )
        src = Image.new("RGB", (128, 128))
        m.upscale(src, scale=4, steps=5, guidance=3.0)
    assert fake_pipe.call_args.kwargs["num_inference_steps"] == 5
    assert fake_pipe.call_args.kwargs["guidance_scale"] == 3.0


def test_runtime_uses_default_steps_and_guidance_when_omitted():
    fake_pipe = _patched_pipe()
    fake_cls = MagicMock()
    fake_cls.from_pretrained.return_value = fake_pipe
    with patch(
        "muse.modalities.image_upscale.runtimes.diffusers_upscaler"
        ".StableDiffusionUpscalePipeline",
        fake_cls,
    ), patch(
        "muse.modalities.image_upscale.runtimes.diffusers_upscaler.torch",
        MagicMock(),
    ):
        from muse.modalities.image_upscale.runtimes.diffusers_upscaler import (
            DiffusersUpscaleRuntime,
        )
        m = DiffusersUpscaleRuntime(
            hf_repo="x", local_dir="/fake", device="cpu", model_id="x",
            default_steps=20, default_guidance=9.0,
        )
        src = Image.new("RGB", (128, 128))
        m.upscale(src, scale=4)
    assert fake_pipe.call_args.kwargs["num_inference_steps"] == 20
    assert fake_pipe.call_args.kwargs["guidance_scale"] == 9.0


def test_runtime_supported_scales_from_capability():
    fake_cls = MagicMock()
    fake_cls.from_pretrained.return_value = _patched_pipe()
    with patch(
        "muse.modalities.image_upscale.runtimes.diffusers_upscaler"
        ".StableDiffusionUpscalePipeline",
        fake_cls,
    ), patch(
        "muse.modalities.image_upscale.runtimes.diffusers_upscaler.torch",
        MagicMock(),
    ):
        from muse.modalities.image_upscale.runtimes.diffusers_upscaler import (
            DiffusersUpscaleRuntime,
        )
        m = DiffusersUpscaleRuntime(
            hf_repo="x", local_dir="/fake", device="cpu", model_id="x",
            supported_scales=[2, 4],
        )
    assert m.supported_scales == [2, 4]


def test_runtime_supported_scales_defaults_to_default_scale():
    fake_cls = MagicMock()
    fake_cls.from_pretrained.return_value = _patched_pipe()
    with patch(
        "muse.modalities.image_upscale.runtimes.diffusers_upscaler"
        ".StableDiffusionUpscalePipeline",
        fake_cls,
    ), patch(
        "muse.modalities.image_upscale.runtimes.diffusers_upscaler.torch",
        MagicMock(),
    ):
        from muse.modalities.image_upscale.runtimes.diffusers_upscaler import (
            DiffusersUpscaleRuntime,
        )
        m = DiffusersUpscaleRuntime(
            hf_repo="x", local_dir="/fake", device="cpu", model_id="x",
            default_scale=4,
        )
    assert m.supported_scales == [4]


def test_runtime_passes_negative_prompt():
    fake_pipe = _patched_pipe()
    fake_cls = MagicMock()
    fake_cls.from_pretrained.return_value = fake_pipe
    with patch(
        "muse.modalities.image_upscale.runtimes.diffusers_upscaler"
        ".StableDiffusionUpscalePipeline",
        fake_cls,
    ), patch(
        "muse.modalities.image_upscale.runtimes.diffusers_upscaler.torch",
        MagicMock(),
    ):
        from muse.modalities.image_upscale.runtimes.diffusers_upscaler import (
            DiffusersUpscaleRuntime,
        )
        m = DiffusersUpscaleRuntime(
            hf_repo="x", local_dir="/fake", device="cpu", model_id="x",
        )
        src = Image.new("RGB", (128, 128))
        m.upscale(src, scale=4, negative_prompt="blurry, low quality")
    assert (
        fake_pipe.call_args.kwargs["negative_prompt"] == "blurry, low quality"
    )


def test_runtime_omits_negative_prompt_when_none():
    fake_pipe = _patched_pipe()
    fake_cls = MagicMock()
    fake_cls.from_pretrained.return_value = fake_pipe
    with patch(
        "muse.modalities.image_upscale.runtimes.diffusers_upscaler"
        ".StableDiffusionUpscalePipeline",
        fake_cls,
    ), patch(
        "muse.modalities.image_upscale.runtimes.diffusers_upscaler.torch",
        MagicMock(),
    ):
        from muse.modalities.image_upscale.runtimes.diffusers_upscaler import (
            DiffusersUpscaleRuntime,
        )
        m = DiffusersUpscaleRuntime(
            hf_repo="x", local_dir="/fake", device="cpu", model_id="x",
        )
        src = Image.new("RGB", (128, 128))
        m.upscale(src, scale=4)
    assert "negative_prompt" not in fake_pipe.call_args.kwargs


def test_runtime_accepts_unknown_kwargs():
    """Future catalog kwargs must be absorbed by **_."""
    fake_cls = MagicMock()
    fake_cls.from_pretrained.return_value = _patched_pipe()
    with patch(
        "muse.modalities.image_upscale.runtimes.diffusers_upscaler"
        ".StableDiffusionUpscalePipeline",
        fake_cls,
    ), patch(
        "muse.modalities.image_upscale.runtimes.diffusers_upscaler.torch",
        MagicMock(),
    ):
        from muse.modalities.image_upscale.runtimes.diffusers_upscaler import (
            DiffusersUpscaleRuntime,
        )
        # Should not TypeError on unknown future kwarg
        DiffusersUpscaleRuntime(
            hf_repo="x", local_dir="/fake", device="cpu", model_id="x",
            future_param="ignored",
        )


def test_runtime_metadata_records_prompt_steps_guidance_model():
    fake_cls = MagicMock()
    fake_cls.from_pretrained.return_value = _patched_pipe()
    with patch(
        "muse.modalities.image_upscale.runtimes.diffusers_upscaler"
        ".StableDiffusionUpscalePipeline",
        fake_cls,
    ), patch(
        "muse.modalities.image_upscale.runtimes.diffusers_upscaler.torch",
        MagicMock(),
    ):
        from muse.modalities.image_upscale.runtimes.diffusers_upscaler import (
            DiffusersUpscaleRuntime,
        )
        m = DiffusersUpscaleRuntime(
            hf_repo="x", local_dir="/fake", device="cpu", model_id="my-model",
        )
        src = Image.new("RGB", (128, 128))
        result = m.upscale(src, scale=4, prompt="hi", steps=10, guidance=5.0)
    assert result.metadata["model"] == "my-model"
    assert result.metadata["prompt"] == "hi"
    assert result.metadata["steps"] == 10
    assert result.metadata["guidance"] == 5.0
