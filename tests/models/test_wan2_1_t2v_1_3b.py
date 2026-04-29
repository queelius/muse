"""Tests for the bundled wan2_1_t2v_1_3b script.

Imports of MANIFEST and Model are done lazily inside each test
function (matches the sd_turbo / animatediff_motion_v3 pattern). The
discovery edge-case test in tests/core/test_discovery.py pops
bundled-model modules out of sys.modules and re-imports them, which
would leave a top-level `from muse.models.wan2_1_t2v_1_3b import Model`
in this file pointing at the stale (popped) module object. Lazy
imports sidestep that.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def test_manifest_required_fields():
    from muse.models.wan2_1_t2v_1_3b import MANIFEST
    assert MANIFEST["model_id"] == "wan2-1-t2v-1-3b"
    assert MANIFEST["modality"] == "video/generation"
    assert MANIFEST["hf_repo"] == "Wan-AI/Wan2.1-T2V-1.3B"
    assert MANIFEST["license"] == "Apache 2.0"


def test_manifest_pip_extras_pin_diffusers_and_imageio():
    from muse.models.wan2_1_t2v_1_3b import MANIFEST
    extras_text = "\n".join(MANIFEST["pip_extras"])
    assert "torch" in extras_text
    assert "diffusers>=0.32" in extras_text
    assert "imageio" in extras_text


def test_manifest_capabilities_advertise_defaults():
    from muse.models.wan2_1_t2v_1_3b import MANIFEST
    caps = MANIFEST["capabilities"]
    assert caps["device"] == "cuda"  # video models are GPU-required
    assert caps["default_duration_seconds"] == 5.0
    assert caps["default_fps"] == 5
    assert tuple(caps["default_size"]) == (832, 480)
    assert caps["default_steps"] == 30
    assert caps["supports_image_to_video"] is False
    assert caps["memory_gb"] == 6.0


def _patched_pipe(n_frames=25):
    fake_pipe = MagicMock()
    fake_frame = MagicMock()
    fake_frame.size = (832, 480)
    fake_pipe.return_value.frames = [[fake_frame] * n_frames]
    return fake_pipe


def test_construction_uses_wan_pipeline_when_present():
    fake_wan_pipeline = MagicMock()
    fake_wan_pipeline.from_pretrained.return_value = _patched_pipe()
    fake_diffusion = MagicMock()
    fake_diffusion.from_pretrained.return_value = _patched_pipe()

    with patch(
        "muse.models.wan2_1_t2v_1_3b.WanPipeline", fake_wan_pipeline,
    ), patch(
        "muse.models.wan2_1_t2v_1_3b.DiffusionPipeline", fake_diffusion,
    ), patch(
        "muse.models.wan2_1_t2v_1_3b.torch", MagicMock(),
    ):
        from muse.models.wan2_1_t2v_1_3b import MANIFEST, Model
        m = Model(
            hf_repo=MANIFEST["hf_repo"],
            local_dir="/fake/cache",
            device="cpu",
        )
    fake_wan_pipeline.from_pretrained.assert_called_once()
    fake_diffusion.from_pretrained.assert_not_called()
    assert m.model_id == MANIFEST["model_id"]


def test_construction_falls_back_to_diffusion_pipeline_when_wan_absent():
    fake_diffusion = MagicMock()
    fake_diffusion.from_pretrained.return_value = _patched_pipe()

    with patch(
        "muse.models.wan2_1_t2v_1_3b.WanPipeline", None,
    ), patch(
        "muse.models.wan2_1_t2v_1_3b.DiffusionPipeline", fake_diffusion,
    ), patch(
        "muse.models.wan2_1_t2v_1_3b.torch", MagicMock(),
    ):
        from muse.models.wan2_1_t2v_1_3b import MANIFEST, Model
        m = Model(hf_repo="x", local_dir="/fake", device="cpu")
    fake_diffusion.from_pretrained.assert_called_once()
    assert m.model_id == MANIFEST["model_id"]


def test_construction_raises_when_no_pipeline_class():
    with patch(
        "muse.models.wan2_1_t2v_1_3b.WanPipeline", None,
    ), patch(
        "muse.models.wan2_1_t2v_1_3b.DiffusionPipeline", None,
    ), patch(
        "muse.models.wan2_1_t2v_1_3b.torch", MagicMock(),
    ), patch(
        "muse.models.wan2_1_t2v_1_3b._ensure_deps", lambda: None,
    ):
        from muse.models.wan2_1_t2v_1_3b import Model
        with pytest.raises(RuntimeError, match="diffusers is not installed"):
            Model(hf_repo="x", local_dir="/fake", device="cpu")


def test_generate_returns_video_result():
    from muse.modalities.video_generation.protocol import VideoResult

    fake_wan_pipeline = MagicMock()
    fake_wan_pipeline.from_pretrained.return_value = _patched_pipe(n_frames=25)

    with patch(
        "muse.models.wan2_1_t2v_1_3b.WanPipeline", fake_wan_pipeline,
    ), patch(
        "muse.models.wan2_1_t2v_1_3b.DiffusionPipeline", MagicMock(),
    ), patch(
        "muse.models.wan2_1_t2v_1_3b.torch", MagicMock(),
    ):
        from muse.models.wan2_1_t2v_1_3b import Model
        m = Model(hf_repo="x", local_dir="/fake", device="cpu")
        r = m.generate("a flag waving in the wind")
    assert isinstance(r, VideoResult)
    assert len(r.frames) == 25
    assert r.fps == 5  # MANIFEST default
    assert r.metadata["model"] == "wan2-1-t2v-1-3b"


def test_generate_forwards_request_overrides_to_pipeline():
    fake_wan_pipeline = MagicMock()
    inner_pipe = _patched_pipe()
    fake_wan_pipeline.from_pretrained.return_value = inner_pipe

    with patch(
        "muse.models.wan2_1_t2v_1_3b.WanPipeline", fake_wan_pipeline,
    ), patch(
        "muse.models.wan2_1_t2v_1_3b.DiffusionPipeline", MagicMock(),
    ), patch(
        "muse.models.wan2_1_t2v_1_3b.torch", MagicMock(),
    ):
        from muse.models.wan2_1_t2v_1_3b import Model
        m = Model(hf_repo="x", local_dir="/fake", device="cpu")
        m.generate(
            "test",
            duration_seconds=3.0,
            fps=8,
            width=640,
            height=384,
            steps=20,
            guidance=4.0,
            negative_prompt="blurry",
        )
    kwargs = inner_pipe.call_args.kwargs
    assert kwargs["num_frames"] == 24
    assert kwargs["num_inference_steps"] == 20
    assert kwargs["guidance_scale"] == 4.0
    assert kwargs["width"] == 640
    assert kwargs["height"] == 384
    assert kwargs["negative_prompt"] == "blurry"
