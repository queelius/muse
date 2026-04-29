"""Tests for WanRuntime: generic runtime over diffusers WanPipeline.

Mirrors the lazy-import sentinel pattern from sd_turbo and
animatediff. Stubs WanPipeline and DiffusionPipeline so no real
diffusion runs.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from muse.modalities.video_generation.protocol import VideoResult
from muse.modalities.video_generation.runtimes.wan_runtime import (
    WanRuntime,
)


def _patched_pipe(n_frames=5):
    """Fake pipeline whose .from_pretrained yields a callable returning
    a fake output with .frames[0] = list[PIL-shaped objects]."""
    fake_pipe = MagicMock()
    fake_frame = MagicMock()
    fake_frame.size = (832, 480)
    fake_pipe.return_value.frames = [[fake_frame] * n_frames]
    return fake_pipe


def _runtime_with_patched_pipeline(*, wan_present=True, **kwargs):
    """Helper: construct a WanRuntime under patched diffusers + torch."""
    fake_wan_pipeline = MagicMock() if wan_present else None
    fake_diffusion = MagicMock()
    if fake_wan_pipeline is not None:
        fake_wan_pipeline.from_pretrained.return_value = _patched_pipe()
    fake_diffusion.from_pretrained.return_value = _patched_pipe()

    patches = [
        patch(
            "muse.modalities.video_generation.runtimes.wan_runtime.torch",
            MagicMock(),
        ),
        patch(
            "muse.modalities.video_generation.runtimes.wan_runtime.DiffusionPipeline",
            fake_diffusion,
        ),
    ]
    if wan_present:
        patches.append(
            patch(
                "muse.modalities.video_generation.runtimes.wan_runtime.WanPipeline",
                fake_wan_pipeline,
            )
        )
    else:
        patches.append(
            patch(
                "muse.modalities.video_generation.runtimes.wan_runtime.WanPipeline",
                None,
            )
        )

    with patches[0], patches[1], patches[2]:
        construction_kwargs = {
            "hf_repo": "Wan-AI/Wan2.1-T2V-1.3B",
            "local_dir": "/fake",
            "device": "cpu",
            "model_id": "wan2-1-t2v-1-3b",
        }
        construction_kwargs.update(kwargs)
        rt = WanRuntime(**construction_kwargs)
    return rt, fake_wan_pipeline, fake_diffusion


def test_construction_uses_wan_pipeline_when_available():
    fake_wan_pipeline = MagicMock()
    fake_wan_pipeline.from_pretrained.return_value = _patched_pipe()
    fake_diffusion = MagicMock()
    fake_diffusion.from_pretrained.return_value = _patched_pipe()

    with patch(
        "muse.modalities.video_generation.runtimes.wan_runtime.WanPipeline",
        fake_wan_pipeline,
    ), patch(
        "muse.modalities.video_generation.runtimes.wan_runtime.DiffusionPipeline",
        fake_diffusion,
    ), patch(
        "muse.modalities.video_generation.runtimes.wan_runtime.torch",
        MagicMock(),
    ):
        m = WanRuntime(
            hf_repo="Wan-AI/Wan2.1-T2V-1.3B",
            local_dir="/fake",
            device="cpu",
            model_id="wan2-1-t2v-1-3b",
        )
    fake_wan_pipeline.from_pretrained.assert_called_once()
    fake_diffusion.from_pretrained.assert_not_called()
    assert m.model_id == "wan2-1-t2v-1-3b"


def test_construction_falls_back_to_diffusion_pipeline_when_wan_absent():
    fake_diffusion = MagicMock()
    fake_diffusion.from_pretrained.return_value = _patched_pipe()

    with patch(
        "muse.modalities.video_generation.runtimes.wan_runtime.WanPipeline",
        None,
    ), patch(
        "muse.modalities.video_generation.runtimes.wan_runtime.DiffusionPipeline",
        fake_diffusion,
    ), patch(
        "muse.modalities.video_generation.runtimes.wan_runtime.torch",
        MagicMock(),
    ):
        m = WanRuntime(
            hf_repo="Wan-AI/Wan2.1-T2V-1.3B",
            local_dir="/fake",
            device="cpu",
            model_id="wan2-1-t2v-1-3b",
        )
    fake_diffusion.from_pretrained.assert_called_once()
    assert m.model_id == "wan2-1-t2v-1-3b"


def test_construction_raises_when_neither_pipeline_class_available():
    with patch(
        "muse.modalities.video_generation.runtimes.wan_runtime.WanPipeline",
        None,
    ), patch(
        "muse.modalities.video_generation.runtimes.wan_runtime.DiffusionPipeline",
        None,
    ), patch(
        "muse.modalities.video_generation.runtimes.wan_runtime.torch",
        MagicMock(),
    ), patch(
        "muse.modalities.video_generation.runtimes.wan_runtime._ensure_deps",
        lambda: None,
    ):
        with pytest.raises(RuntimeError, match="diffusers is not installed"):
            WanRuntime(
                hf_repo="x", local_dir="/fake", device="cpu", model_id="m",
            )


def test_generate_returns_video_result():
    rt, _wan, _diff = _runtime_with_patched_pipeline(
        default_duration_seconds=2.0,
        default_fps=5,
        default_size=(832, 480),
        default_steps=10,
        default_guidance=5.0,
    )
    r = rt.generate("a cat in a field")
    assert isinstance(r, VideoResult)
    assert r.fps == 5
    assert r.width == 832
    assert r.height == 480
    assert len(r.frames) == 5
    assert r.metadata["model"] == "wan2-1-t2v-1-3b"


def test_generate_forwards_call_kwargs_to_pipeline():
    fake_wan_pipeline = MagicMock()
    inner_pipe = _patched_pipe()
    fake_wan_pipeline.from_pretrained.return_value = inner_pipe
    fake_diffusion = MagicMock()

    with patch(
        "muse.modalities.video_generation.runtimes.wan_runtime.WanPipeline",
        fake_wan_pipeline,
    ), patch(
        "muse.modalities.video_generation.runtimes.wan_runtime.DiffusionPipeline",
        fake_diffusion,
    ), patch(
        "muse.modalities.video_generation.runtimes.wan_runtime.torch",
        MagicMock(),
    ):
        rt = WanRuntime(
            hf_repo="x",
            local_dir="/fake",
            device="cpu",
            model_id="m",
            default_duration_seconds=5.0,
            default_fps=5,
            default_size=(832, 480),
            default_steps=30,
            default_guidance=5.0,
        )
        rt.generate(
            "test",
            duration_seconds=3.0,
            fps=8,
            width=512,
            height=288,
            steps=20,
            guidance=4.0,
        )
    kwargs = inner_pipe.call_args.kwargs
    assert kwargs["num_frames"] == 24  # 3.0 * 8
    assert kwargs["num_inference_steps"] == 20
    assert kwargs["guidance_scale"] == 4.0
    assert kwargs["width"] == 512
    assert kwargs["height"] == 288


def test_generate_forwards_negative_prompt_when_set():
    fake_wan_pipeline = MagicMock()
    inner_pipe = _patched_pipe()
    fake_wan_pipeline.from_pretrained.return_value = inner_pipe

    with patch(
        "muse.modalities.video_generation.runtimes.wan_runtime.WanPipeline",
        fake_wan_pipeline,
    ), patch(
        "muse.modalities.video_generation.runtimes.wan_runtime.DiffusionPipeline",
        MagicMock(),
    ), patch(
        "muse.modalities.video_generation.runtimes.wan_runtime.torch",
        MagicMock(),
    ):
        rt = WanRuntime(
            hf_repo="x", local_dir="/fake", device="cpu", model_id="m",
        )
        rt.generate("test", negative_prompt="blurry, low-quality")
    assert inner_pipe.call_args.kwargs.get(
        "negative_prompt"
    ) == "blurry, low-quality"


def test_generate_omits_negative_prompt_when_none():
    fake_wan_pipeline = MagicMock()
    inner_pipe = _patched_pipe()
    fake_wan_pipeline.from_pretrained.return_value = inner_pipe

    with patch(
        "muse.modalities.video_generation.runtimes.wan_runtime.WanPipeline",
        fake_wan_pipeline,
    ), patch(
        "muse.modalities.video_generation.runtimes.wan_runtime.DiffusionPipeline",
        MagicMock(),
    ), patch(
        "muse.modalities.video_generation.runtimes.wan_runtime.torch",
        MagicMock(),
    ):
        rt = WanRuntime(
            hf_repo="x", local_dir="/fake", device="cpu", model_id="m",
        )
        rt.generate("test")
    assert "negative_prompt" not in inner_pipe.call_args.kwargs


def test_construction_absorbs_unknown_kwargs():
    """Future capability flags must not crash the constructor."""
    fake_wan_pipeline = MagicMock()
    fake_wan_pipeline.from_pretrained.return_value = _patched_pipe()

    with patch(
        "muse.modalities.video_generation.runtimes.wan_runtime.WanPipeline",
        fake_wan_pipeline,
    ), patch(
        "muse.modalities.video_generation.runtimes.wan_runtime.DiffusionPipeline",
        MagicMock(),
    ), patch(
        "muse.modalities.video_generation.runtimes.wan_runtime.torch",
        MagicMock(),
    ):
        WanRuntime(
            hf_repo="x", local_dir="/fake", device="cpu", model_id="m",
            future_unrecognized_flag="whatever",
            supports_image_to_video=False,
            min_duration_seconds=1.0,
            max_duration_seconds=10.0,
            memory_gb=6.0,
        )


def test_generate_records_actual_duration_from_returned_frames():
    """Pipelines may align num_frames to model-native count; runtime
    records actual_frames / fps as duration_seconds."""
    fake_wan_pipeline = MagicMock()
    inner_pipe = _patched_pipe(n_frames=25)  # different from request
    fake_wan_pipeline.from_pretrained.return_value = inner_pipe

    with patch(
        "muse.modalities.video_generation.runtimes.wan_runtime.WanPipeline",
        fake_wan_pipeline,
    ), patch(
        "muse.modalities.video_generation.runtimes.wan_runtime.DiffusionPipeline",
        MagicMock(),
    ), patch(
        "muse.modalities.video_generation.runtimes.wan_runtime.torch",
        MagicMock(),
    ):
        rt = WanRuntime(
            hf_repo="x",
            local_dir="/fake",
            device="cpu",
            model_id="m",
            default_duration_seconds=5.0,
            default_fps=5,
        )
        r = rt.generate("test", duration_seconds=5.0, fps=5)
    assert len(r.frames) == 25
    assert r.duration_seconds == 5.0  # 25 / 5


def test_generate_forwards_seed_via_generator():
    """When seed is set, runtime builds a torch.Generator and forwards it."""
    fake_wan_pipeline = MagicMock()
    inner_pipe = _patched_pipe()
    fake_wan_pipeline.from_pretrained.return_value = inner_pipe

    fake_torch = MagicMock()

    with patch(
        "muse.modalities.video_generation.runtimes.wan_runtime.WanPipeline",
        fake_wan_pipeline,
    ), patch(
        "muse.modalities.video_generation.runtimes.wan_runtime.DiffusionPipeline",
        MagicMock(),
    ), patch(
        "muse.modalities.video_generation.runtimes.wan_runtime.torch",
        fake_torch,
    ):
        rt = WanRuntime(
            hf_repo="x", local_dir="/fake", device="cpu", model_id="m",
        )
        rt.generate("test", seed=42)
    assert "generator" in inner_pipe.call_args.kwargs
