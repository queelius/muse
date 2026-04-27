"""Tests for DiffusersText2ImageModel generic runtime.

The runtime wraps diffusers.AutoPipelineForText2Image; tests stub it
out so no real diffusion happens. Mirrors the patching pattern used
in tests/models/test_sd_turbo.py.
"""
from unittest.mock import MagicMock, patch

import pytest

from muse.modalities.image_generation.protocol import ImageResult
from muse.modalities.image_generation.runtimes.diffusers import (
    DiffusersText2ImageModel,
)


def _patched_pipe():
    """Return a fake pipeline whose .from_pretrained yields a mock that
    returns one PIL-shaped image when called."""
    fake_pipe = MagicMock()
    fake_image = MagicMock()
    fake_image.size = (512, 512)
    fake_pipe.return_value.images = [fake_image]
    return fake_pipe


def test_construction_uses_local_dir_when_provided(tmp_path):
    """Constructor passes local_dir as the first positional to from_pretrained."""
    fake_class = MagicMock()
    fake_class.from_pretrained.return_value = _patched_pipe()
    with patch(
        "muse.modalities.image_generation.runtimes.diffusers.AutoPipelineForText2Image",
        fake_class,
    ), patch(
        "muse.modalities.image_generation.runtimes.diffusers.torch",
        MagicMock(),
    ):
        m = DiffusersText2ImageModel(
            hf_repo="org/repo",
            local_dir=str(tmp_path),
            device="cpu",
            dtype="float32",
            model_id="org-repo",
        )
    fake_class.from_pretrained.assert_called_once()
    assert fake_class.from_pretrained.call_args.args[0] == str(tmp_path)
    assert m.model_id == "org-repo"


def test_construction_falls_back_to_hf_repo_when_no_local_dir():
    """When local_dir is None, from_pretrained gets the hf_repo as first arg."""
    fake_class = MagicMock()
    fake_class.from_pretrained.return_value = _patched_pipe()
    with patch(
        "muse.modalities.image_generation.runtimes.diffusers.AutoPipelineForText2Image",
        fake_class,
    ), patch(
        "muse.modalities.image_generation.runtimes.diffusers.torch",
        MagicMock(),
    ):
        DiffusersText2ImageModel(
            hf_repo="org/repo",
            local_dir=None,
            device="cpu",
            dtype="float32",
            model_id="m",
        )
    assert fake_class.from_pretrained.call_args.args[0] == "org/repo"


def test_default_size_steps_guidance_from_kwargs():
    """Constructor reads default_size/steps/guidance from kwargs (manifest capabilities)."""
    fake_class = MagicMock()
    fake_class.from_pretrained.return_value = _patched_pipe()
    with patch(
        "muse.modalities.image_generation.runtimes.diffusers.AutoPipelineForText2Image",
        fake_class,
    ), patch(
        "muse.modalities.image_generation.runtimes.diffusers.torch",
        MagicMock(),
    ):
        m = DiffusersText2ImageModel(
            hf_repo="org/repo",
            local_dir="/tmp/fake",
            device="cpu",
            model_id="m",
            default_size=(1024, 1024),
            default_steps=4,
            default_guidance=3.5,
        )
    assert m.default_size == (1024, 1024)
    assert m._default_steps == 4
    assert m._default_guidance == 3.5


def test_generate_uses_request_overrides_when_provided():
    """When generate() is called with steps/guidance/size, those override defaults."""
    pipe = _patched_pipe()
    fake_class = MagicMock()
    fake_class.from_pretrained.return_value = pipe
    with patch(
        "muse.modalities.image_generation.runtimes.diffusers.AutoPipelineForText2Image",
        fake_class,
    ), patch(
        "muse.modalities.image_generation.runtimes.diffusers.torch",
        MagicMock(),
    ):
        m = DiffusersText2ImageModel(
            hf_repo="org/repo", local_dir="/tmp/fake", device="cpu",
            model_id="m", default_steps=1, default_guidance=0.0,
        )
        result = m.generate(
            "a fox", steps=25, guidance=7.5, width=768, height=768,
        )
    call_kwargs = pipe.call_args.kwargs
    assert call_kwargs["num_inference_steps"] == 25
    assert call_kwargs["guidance_scale"] == 7.5
    assert call_kwargs["width"] == 768
    assert call_kwargs["height"] == 768
    assert isinstance(result, ImageResult)


def test_generate_uses_defaults_when_request_omits_them():
    """When generate() omits steps/guidance, defaults from constructor are used."""
    pipe = _patched_pipe()
    fake_class = MagicMock()
    fake_class.from_pretrained.return_value = pipe
    with patch(
        "muse.modalities.image_generation.runtimes.diffusers.AutoPipelineForText2Image",
        fake_class,
    ), patch(
        "muse.modalities.image_generation.runtimes.diffusers.torch",
        MagicMock(),
    ):
        m = DiffusersText2ImageModel(
            hf_repo="org/repo", local_dir="/tmp/fake", device="cpu",
            model_id="m", default_size=(1024, 1024),
            default_steps=4, default_guidance=0.0,
        )
        m.generate("a fox")
    call_kwargs = pipe.call_args.kwargs
    assert call_kwargs["num_inference_steps"] == 4
    assert call_kwargs["guidance_scale"] == 0.0
    assert call_kwargs["width"] == 1024
    assert call_kwargs["height"] == 1024


def test_generate_passes_negative_prompt_when_set():
    pipe = _patched_pipe()
    fake_class = MagicMock()
    fake_class.from_pretrained.return_value = pipe
    with patch(
        "muse.modalities.image_generation.runtimes.diffusers.AutoPipelineForText2Image",
        fake_class,
    ), patch(
        "muse.modalities.image_generation.runtimes.diffusers.torch",
        MagicMock(),
    ):
        m = DiffusersText2ImageModel(
            hf_repo="org/repo", local_dir="/tmp/fake", device="cpu", model_id="m",
        )
        m.generate("a fox", negative_prompt="blurry, ugly")
    assert pipe.call_args.kwargs.get("negative_prompt") == "blurry, ugly"


def test_generate_omits_negative_prompt_when_none():
    """negative_prompt=None should NOT be passed to the pipe (some models reject it)."""
    pipe = _patched_pipe()
    fake_class = MagicMock()
    fake_class.from_pretrained.return_value = pipe
    with patch(
        "muse.modalities.image_generation.runtimes.diffusers.AutoPipelineForText2Image",
        fake_class,
    ), patch(
        "muse.modalities.image_generation.runtimes.diffusers.torch",
        MagicMock(),
    ):
        m = DiffusersText2ImageModel(
            hf_repo="org/repo", local_dir="/tmp/fake", device="cpu", model_id="m",
        )
        m.generate("a fox")
    assert "negative_prompt" not in pipe.call_args.kwargs


def test_generate_returns_image_result_with_seed():
    pipe = _patched_pipe()
    fake_class = MagicMock()
    fake_class.from_pretrained.return_value = pipe
    fake_torch = MagicMock()
    with patch(
        "muse.modalities.image_generation.runtimes.diffusers.AutoPipelineForText2Image",
        fake_class,
    ), patch(
        "muse.modalities.image_generation.runtimes.diffusers.torch",
        fake_torch,
    ):
        m = DiffusersText2ImageModel(
            hf_repo="org/repo", local_dir="/tmp/fake", device="cpu", model_id="m",
        )
        result = m.generate("a fox", seed=42)
    assert result.seed == 42
    assert result.metadata["model"] == "m"


def test_construction_absorbs_unknown_kwargs():
    """Future capability keys must not crash the constructor.

    Catalog's load_backend splats manifest.capabilities into the constructor
    call. As muse adds new capability flags (e.g., supports_negative_prompt,
    supports_seeded_generation, supports_img2img), older runtimes must accept
    them gracefully via **_.
    """
    fake_class = MagicMock()
    fake_class.from_pretrained.return_value = _patched_pipe()
    with patch(
        "muse.modalities.image_generation.runtimes.diffusers.AutoPipelineForText2Image",
        fake_class,
    ), patch(
        "muse.modalities.image_generation.runtimes.diffusers.torch",
        MagicMock(),
    ):
        DiffusersText2ImageModel(
            hf_repo="org/repo",
            local_dir="/tmp/fake",
            device="cpu",
            model_id="m",
            supports_negative_prompt=True,
            supports_seeded_generation=True,
            future_unrecognized_flag="whatever",
        )
    fake_class.from_pretrained.assert_called_once()


def test_default_size_normalized_to_tuple_when_passed_list():
    """JSON-loaded manifests produce default_size as list[int]; coerce to tuple."""
    fake_class = MagicMock()
    fake_class.from_pretrained.return_value = _patched_pipe()
    with patch(
        "muse.modalities.image_generation.runtimes.diffusers.AutoPipelineForText2Image",
        fake_class,
    ), patch(
        "muse.modalities.image_generation.runtimes.diffusers.torch",
        MagicMock(),
    ):
        m = DiffusersText2ImageModel(
            hf_repo="org/repo", local_dir="/tmp/fake", device="cpu",
            model_id="m", default_size=[1024, 1024],  # list, not tuple
        )
    assert m.default_size == (1024, 1024)
    assert isinstance(m.default_size, tuple)


def test_generate_with_init_image_uses_img2img_pipeline():
    """When init_image is set, runtime calls AutoPipelineForImage2Image."""
    from PIL import Image

    fake_t2i_class = MagicMock()
    fake_t2i_class.from_pretrained.return_value = _patched_pipe()
    fake_i2i_class = MagicMock()
    fake_i2i_class.from_pretrained.return_value = _patched_pipe()

    with patch(
        "muse.modalities.image_generation.runtimes.diffusers.AutoPipelineForText2Image",
        fake_t2i_class,
    ), patch(
        "muse.modalities.image_generation.runtimes.diffusers.AutoPipelineForImage2Image",
        fake_i2i_class,
    ), patch(
        "muse.modalities.image_generation.runtimes.diffusers.torch",
        MagicMock(),
    ):
        m = DiffusersText2ImageModel(
            hf_repo="org/repo", local_dir="/tmp/fake", device="cpu",
            model_id="m",
        )
        init_img = Image.new("RGB", (64, 64))
        m.generate("repaint", init_image=init_img, strength=0.6)

    fake_i2i_class.from_pretrained.assert_called_once()
    # The img2img pipeline (not the t2i one) was called for inference
    fake_i2i_class.from_pretrained.return_value.assert_called()


def test_generate_without_init_image_uses_text2image_pipeline():
    """init_image=None keeps the existing text-to-image path (no regression)."""
    fake_t2i_class = MagicMock()
    fake_t2i_class.from_pretrained.return_value = _patched_pipe()
    fake_i2i_class = MagicMock()

    with patch(
        "muse.modalities.image_generation.runtimes.diffusers.AutoPipelineForText2Image",
        fake_t2i_class,
    ), patch(
        "muse.modalities.image_generation.runtimes.diffusers.AutoPipelineForImage2Image",
        fake_i2i_class,
    ), patch(
        "muse.modalities.image_generation.runtimes.diffusers.torch",
        MagicMock(),
    ):
        m = DiffusersText2ImageModel(
            hf_repo="org/repo", local_dir="/tmp/fake", device="cpu",
            model_id="m",
        )
        m.generate("a fox")

    # img2img was NEVER loaded
    fake_i2i_class.from_pretrained.assert_not_called()


def test_generate_img2img_default_strength_when_omitted():
    """When strength is None on an img2img call, defaults to 0.5."""
    from PIL import Image

    fake_t2i_class = MagicMock()
    fake_t2i_class.from_pretrained.return_value = _patched_pipe()
    fake_i2i_class = MagicMock()
    fake_i2i_pipe = _patched_pipe()
    fake_i2i_class.from_pretrained.return_value = fake_i2i_pipe

    with patch(
        "muse.modalities.image_generation.runtimes.diffusers.AutoPipelineForText2Image",
        fake_t2i_class,
    ), patch(
        "muse.modalities.image_generation.runtimes.diffusers.AutoPipelineForImage2Image",
        fake_i2i_class,
    ), patch(
        "muse.modalities.image_generation.runtimes.diffusers.torch",
        MagicMock(),
    ):
        m = DiffusersText2ImageModel(
            hf_repo="org/repo", local_dir="/tmp/fake", device="cpu",
            model_id="m",
        )
        init_img = Image.new("RGB", (64, 64))
        m.generate("repaint", init_image=init_img)  # no strength

    assert fake_i2i_pipe.call_args.kwargs["strength"] == 0.5


def test_generate_img2img_caches_pipeline():
    """Second img2img call reuses the cached pipeline (no second from_pretrained)."""
    from PIL import Image

    fake_t2i_class = MagicMock()
    fake_t2i_class.from_pretrained.return_value = _patched_pipe()
    fake_i2i_class = MagicMock()
    fake_i2i_class.from_pretrained.return_value = _patched_pipe()

    with patch(
        "muse.modalities.image_generation.runtimes.diffusers.AutoPipelineForText2Image",
        fake_t2i_class,
    ), patch(
        "muse.modalities.image_generation.runtimes.diffusers.AutoPipelineForImage2Image",
        fake_i2i_class,
    ), patch(
        "muse.modalities.image_generation.runtimes.diffusers.torch",
        MagicMock(),
    ):
        m = DiffusersText2ImageModel(
            hf_repo="org/repo", local_dir="/tmp/fake", device="cpu",
            model_id="m",
        )
        init_img = Image.new("RGB", (64, 64))
        m.generate("a", init_image=init_img)
        m.generate("b", init_image=init_img)

    assert fake_i2i_class.from_pretrained.call_count == 1
