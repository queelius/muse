"""Tests for CogVideoXRuntime: generic runtime over diffusers
CogVideoXPipeline.

Mirrors the lazy-import sentinel pattern from sd_turbo and animatediff.
Stubs CogVideoXPipeline so no real diffusion runs.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from muse.modalities.video_generation.protocol import VideoResult
from muse.modalities.video_generation.runtimes.cogvideox_runtime import (
    CogVideoXRuntime,
)


def _patched_pipe(n_frames=49):
    fake_pipe = MagicMock()
    fake_frame = MagicMock()
    fake_frame.size = (720, 480)
    fake_pipe.return_value.frames = [[fake_frame] * n_frames]
    return fake_pipe


def test_construction_loads_pipeline():
    fake_cogvideox = MagicMock()
    fake_cogvideox.from_pretrained.return_value = _patched_pipe()

    with patch(
        "muse.modalities.video_generation.runtimes.cogvideox_runtime.CogVideoXPipeline",
        fake_cogvideox,
    ), patch(
        "muse.modalities.video_generation.runtimes.cogvideox_runtime.torch",
        MagicMock(),
    ):
        m = CogVideoXRuntime(
            hf_repo="THUDM/CogVideoX-2b",
            local_dir="/fake",
            device="cpu",
            model_id="cogvideox-2b",
        )
    fake_cogvideox.from_pretrained.assert_called_once()
    assert m.model_id == "cogvideox-2b"


def test_construction_raises_when_pipeline_unavailable():
    with patch(
        "muse.modalities.video_generation.runtimes.cogvideox_runtime.CogVideoXPipeline",
        None,
    ), patch(
        "muse.modalities.video_generation.runtimes.cogvideox_runtime.torch",
        MagicMock(),
    ), patch(
        "muse.modalities.video_generation.runtimes.cogvideox_runtime._ensure_deps",
        lambda: None,
    ):
        with pytest.raises(RuntimeError, match="CogVideoXPipeline"):
            CogVideoXRuntime(
                hf_repo="x", local_dir="/fake", device="cpu", model_id="m",
            )


def test_generate_returns_video_result():
    fake_cogvideox = MagicMock()
    fake_cogvideox.from_pretrained.return_value = _patched_pipe(n_frames=10)

    with patch(
        "muse.modalities.video_generation.runtimes.cogvideox_runtime.CogVideoXPipeline",
        fake_cogvideox,
    ), patch(
        "muse.modalities.video_generation.runtimes.cogvideox_runtime.torch",
        MagicMock(),
    ):
        rt = CogVideoXRuntime(
            hf_repo="x",
            local_dir="/fake",
            device="cpu",
            model_id="cogvideox-2b",
            default_duration_seconds=2.0,
            default_fps=5,
            default_size=(720, 480),
            default_steps=10,
            default_guidance=6.0,
        )
        r = rt.generate("a cat in a field")
    assert isinstance(r, VideoResult)
    assert r.fps == 5
    assert r.width == 720
    assert r.height == 480
    assert len(r.frames) == 10
    assert r.metadata["model"] == "cogvideox-2b"


def test_generate_forwards_call_kwargs_to_pipeline():
    fake_cogvideox = MagicMock()
    inner_pipe = _patched_pipe()
    fake_cogvideox.from_pretrained.return_value = inner_pipe

    with patch(
        "muse.modalities.video_generation.runtimes.cogvideox_runtime.CogVideoXPipeline",
        fake_cogvideox,
    ), patch(
        "muse.modalities.video_generation.runtimes.cogvideox_runtime.torch",
        MagicMock(),
    ):
        rt = CogVideoXRuntime(
            hf_repo="x",
            local_dir="/fake",
            device="cpu",
            model_id="m",
            default_duration_seconds=6.0,
            default_fps=8,
            default_size=(720, 480),
            default_steps=50,
            default_guidance=6.0,
        )
        rt.generate(
            "test",
            duration_seconds=4.0,
            fps=8,
            width=512,
            height=320,
            steps=20,
            guidance=4.0,
        )
    kwargs = inner_pipe.call_args.kwargs
    assert kwargs["num_frames"] == 32  # 4.0 * 8
    assert kwargs["num_inference_steps"] == 20
    assert kwargs["guidance_scale"] == 4.0
    assert kwargs["width"] == 512
    assert kwargs["height"] == 320


def test_generate_forwards_negative_prompt_when_set():
    fake_cogvideox = MagicMock()
    inner_pipe = _patched_pipe()
    fake_cogvideox.from_pretrained.return_value = inner_pipe

    with patch(
        "muse.modalities.video_generation.runtimes.cogvideox_runtime.CogVideoXPipeline",
        fake_cogvideox,
    ), patch(
        "muse.modalities.video_generation.runtimes.cogvideox_runtime.torch",
        MagicMock(),
    ):
        rt = CogVideoXRuntime(
            hf_repo="x", local_dir="/fake", device="cpu", model_id="m",
        )
        rt.generate("test", negative_prompt="blurry")
    assert inner_pipe.call_args.kwargs.get("negative_prompt") == "blurry"


def test_generate_omits_negative_prompt_when_none():
    fake_cogvideox = MagicMock()
    inner_pipe = _patched_pipe()
    fake_cogvideox.from_pretrained.return_value = inner_pipe

    with patch(
        "muse.modalities.video_generation.runtimes.cogvideox_runtime.CogVideoXPipeline",
        fake_cogvideox,
    ), patch(
        "muse.modalities.video_generation.runtimes.cogvideox_runtime.torch",
        MagicMock(),
    ):
        rt = CogVideoXRuntime(
            hf_repo="x", local_dir="/fake", device="cpu", model_id="m",
        )
        rt.generate("test")
    assert "negative_prompt" not in inner_pipe.call_args.kwargs


def test_construction_absorbs_unknown_kwargs():
    fake_cogvideox = MagicMock()
    fake_cogvideox.from_pretrained.return_value = _patched_pipe()

    with patch(
        "muse.modalities.video_generation.runtimes.cogvideox_runtime.CogVideoXPipeline",
        fake_cogvideox,
    ), patch(
        "muse.modalities.video_generation.runtimes.cogvideox_runtime.torch",
        MagicMock(),
    ):
        CogVideoXRuntime(
            hf_repo="x", local_dir="/fake", device="cpu", model_id="m",
            future_unrecognized_flag="whatever",
            supports_image_to_video=False,
            min_duration_seconds=1.0,
            max_duration_seconds=10.0,
            memory_gb=9.0,
        )


def test_generate_forwards_seed_via_generator():
    fake_cogvideox = MagicMock()
    inner_pipe = _patched_pipe()
    fake_cogvideox.from_pretrained.return_value = inner_pipe

    with patch(
        "muse.modalities.video_generation.runtimes.cogvideox_runtime.CogVideoXPipeline",
        fake_cogvideox,
    ), patch(
        "muse.modalities.video_generation.runtimes.cogvideox_runtime.torch",
        MagicMock(),
    ):
        rt = CogVideoXRuntime(
            hf_repo="x", local_dir="/fake", device="cpu", model_id="m",
        )
        rt.generate("test", seed=42)
    assert "generator" in inner_pipe.call_args.kwargs


def test_generate_records_actual_duration_from_returned_frames():
    """Pipelines may align num_frames to model-native count."""
    fake_cogvideox = MagicMock()
    inner_pipe = _patched_pipe(n_frames=49)
    fake_cogvideox.from_pretrained.return_value = inner_pipe

    with patch(
        "muse.modalities.video_generation.runtimes.cogvideox_runtime.CogVideoXPipeline",
        fake_cogvideox,
    ), patch(
        "muse.modalities.video_generation.runtimes.cogvideox_runtime.torch",
        MagicMock(),
    ):
        rt = CogVideoXRuntime(
            hf_repo="x", local_dir="/fake", device="cpu", model_id="m",
            default_duration_seconds=6.0, default_fps=8,
        )
        r = rt.generate("test", duration_seconds=6.0, fps=8)
    assert len(r.frames) == 49
    assert r.duration_seconds == round(49 / 8, 3)
