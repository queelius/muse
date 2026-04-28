"""Tests for AnimateDiffRuntime: generic runtime over diffusers AnimateDiff.

Mirrors the lazy-import sentinel pattern from sd_turbo and runtimes/diffusers.
Stubs AnimateDiffPipeline + MotionAdapter so no real diffusion runs.
"""
from unittest.mock import MagicMock, patch

import pytest

from muse.modalities.image_animation.protocol import AnimationResult
from muse.modalities.image_animation.runtimes.animatediff import (
    AnimateDiffRuntime,
)


def _patched_pipe():
    """Fake pipeline whose .from_pretrained yields a callable returning
    a fake output object with .frames[0] = list[PIL-shaped images]."""
    fake_pipe = MagicMock()
    fake_frame = MagicMock()
    fake_frame.size = (512, 512)
    # AnimateDiffPipeline output: out.frames is list of lists (per video, per frame)
    fake_pipe.return_value.frames = [[fake_frame, fake_frame, fake_frame]]
    return fake_pipe


def test_construction_loads_pipeline_and_adapter():
    fake_pipe_class = MagicMock()
    fake_pipe_class.from_pretrained.return_value = _patched_pipe()
    fake_adapter_class = MagicMock()
    fake_adapter_class.from_pretrained.return_value = MagicMock()

    with patch(
        "muse.modalities.image_animation.runtimes.animatediff.AnimateDiffPipeline",
        fake_pipe_class,
    ), patch(
        "muse.modalities.image_animation.runtimes.animatediff.MotionAdapter",
        fake_adapter_class,
    ), patch(
        "muse.modalities.image_animation.runtimes.animatediff.torch",
        MagicMock(),
    ):
        m = AnimateDiffRuntime(
            hf_repo="guoyww/animatediff-motion-adapter-v1-5-3",
            local_dir="/fake/adapter",
            device="cpu",
            model_id="adv3",
            base_model="emilianJR/epiCRealism",
        )
    fake_adapter_class.from_pretrained.assert_called_once()
    fake_pipe_class.from_pretrained.assert_called_once()
    assert m.model_id == "adv3"


def test_generate_returns_animation_result():
    fake_pipe = _patched_pipe()
    fake_pipe_class = MagicMock()
    fake_pipe_class.from_pretrained.return_value = fake_pipe
    fake_adapter_class = MagicMock()
    fake_adapter_class.from_pretrained.return_value = MagicMock()

    with patch(
        "muse.modalities.image_animation.runtimes.animatediff.AnimateDiffPipeline",
        fake_pipe_class,
    ), patch(
        "muse.modalities.image_animation.runtimes.animatediff.MotionAdapter",
        fake_adapter_class,
    ), patch(
        "muse.modalities.image_animation.runtimes.animatediff.torch",
        MagicMock(),
    ):
        m = AnimateDiffRuntime(
            hf_repo="guoyww/animatediff-motion-adapter-v1-5-3",
            local_dir="/fake/adapter",
            device="cpu",
            model_id="adv3",
            base_model="emilianJR/epiCRealism",
            default_frames=16, default_fps=8,
            default_size=(512, 512), default_steps=25, default_guidance=7.5,
        )
        r = m.generate("a cat")
    assert isinstance(r, AnimationResult)
    assert len(r.frames) == 3  # the fake pipe returned 3 frames
    assert r.fps == 8
    assert r.width == 512


def test_generate_request_overrides_defaults():
    fake_pipe = _patched_pipe()
    fake_pipe_class = MagicMock()
    fake_pipe_class.from_pretrained.return_value = fake_pipe
    fake_adapter_class = MagicMock()
    fake_adapter_class.from_pretrained.return_value = MagicMock()

    with patch(
        "muse.modalities.image_animation.runtimes.animatediff.AnimateDiffPipeline",
        fake_pipe_class,
    ), patch(
        "muse.modalities.image_animation.runtimes.animatediff.MotionAdapter",
        fake_adapter_class,
    ), patch(
        "muse.modalities.image_animation.runtimes.animatediff.torch",
        MagicMock(),
    ):
        m = AnimateDiffRuntime(
            hf_repo="x", local_dir="/fake", device="cpu",
            model_id="m", base_model="b",
            default_frames=16, default_fps=8, default_steps=25, default_guidance=7.5,
        )
        m.generate("a fox", frames=8, fps=12, steps=50, guidance=9.0)
    kwargs = fake_pipe.call_args.kwargs
    assert kwargs["num_frames"] == 8
    assert kwargs["num_inference_steps"] == 50
    assert kwargs["guidance_scale"] == 9.0


def test_generate_passes_negative_prompt_when_set():
    fake_pipe = _patched_pipe()
    fake_pipe_class = MagicMock()
    fake_pipe_class.from_pretrained.return_value = fake_pipe
    fake_adapter_class = MagicMock()
    fake_adapter_class.from_pretrained.return_value = MagicMock()

    with patch(
        "muse.modalities.image_animation.runtimes.animatediff.AnimateDiffPipeline",
        fake_pipe_class,
    ), patch(
        "muse.modalities.image_animation.runtimes.animatediff.MotionAdapter",
        fake_adapter_class,
    ), patch(
        "muse.modalities.image_animation.runtimes.animatediff.torch",
        MagicMock(),
    ):
        m = AnimateDiffRuntime(
            hf_repo="x", local_dir="/fake", device="cpu",
            model_id="m", base_model="b",
        )
        m.generate("a fox", negative_prompt="blurry, ugly")
    assert fake_pipe.call_args.kwargs.get("negative_prompt") == "blurry, ugly"


def test_generate_omits_negative_prompt_when_none():
    fake_pipe = _patched_pipe()
    fake_pipe_class = MagicMock()
    fake_pipe_class.from_pretrained.return_value = fake_pipe
    fake_adapter_class = MagicMock()
    fake_adapter_class.from_pretrained.return_value = MagicMock()

    with patch(
        "muse.modalities.image_animation.runtimes.animatediff.AnimateDiffPipeline",
        fake_pipe_class,
    ), patch(
        "muse.modalities.image_animation.runtimes.animatediff.MotionAdapter",
        fake_adapter_class,
    ), patch(
        "muse.modalities.image_animation.runtimes.animatediff.torch",
        MagicMock(),
    ):
        m = AnimateDiffRuntime(
            hf_repo="x", local_dir="/fake", device="cpu",
            model_id="m", base_model="b",
        )
        m.generate("a fox")
    assert "negative_prompt" not in fake_pipe.call_args.kwargs


def test_construction_absorbs_unknown_kwargs():
    """Future capability flags must not crash the constructor."""
    fake_pipe_class = MagicMock()
    fake_pipe_class.from_pretrained.return_value = _patched_pipe()
    fake_adapter_class = MagicMock()
    fake_adapter_class.from_pretrained.return_value = MagicMock()

    with patch(
        "muse.modalities.image_animation.runtimes.animatediff.AnimateDiffPipeline",
        fake_pipe_class,
    ), patch(
        "muse.modalities.image_animation.runtimes.animatediff.MotionAdapter",
        fake_adapter_class,
    ), patch(
        "muse.modalities.image_animation.runtimes.animatediff.torch",
        MagicMock(),
    ):
        AnimateDiffRuntime(
            hf_repo="x", local_dir="/fake", device="cpu", model_id="m",
            base_model="b",
            future_unrecognized_flag="whatever",
            supports_text_to_animation=True,
        )
