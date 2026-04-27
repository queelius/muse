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


def test_construction_loads_from_local_dir(tmp_path):
    """Constructor passes local_dir to AutoPipelineForText2Image.from_pretrained."""
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
    assert m.model_id == "org-repo"


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
