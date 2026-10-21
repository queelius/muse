"""Tests for the bundled wan2_1_t2v_1_3b script.

The script aliases the generic `WanRuntime` as its `Model`; runtime
behavior is covered by tests/modalities/video_generation/runtimes/
test_wan_runtime.py. These tests pin the MANIFEST and check that the
runtime honors it when constructed the way `load_backend` does
(`model_id` + capabilities splatted as kwargs).

Imports are done lazily inside each test function: the discovery
edge-case test in tests/core/test_discovery.py pops bundled-model
modules out of sys.modules and re-imports them, which would leave a
top-level import here pointing at a stale module object.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

_RT = "muse.modalities.video_generation.runtimes.wan_runtime"


def test_manifest_required_fields():
    from muse.models.wan2_1_t2v_1_3b import MANIFEST
    assert MANIFEST["model_id"] == "wan2-1-t2v-1-3b"
    assert MANIFEST["modality"] == "video/generation"
    assert MANIFEST["hf_repo"] == "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
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
    assert caps["memory_gb"] == 5.0
    assert caps["cpu_offload"] == "sequential"
    assert caps["vae_tiling"] is True


def test_model_is_wan_runtime():
    from muse.models.wan2_1_t2v_1_3b import Model
    from muse.modalities.video_generation.runtimes.wan_runtime import (
        WanRuntime,
    )
    assert Model is WanRuntime


def _fake_pipe(n_frames=25):
    fake_pipe = MagicMock()
    fake_frame = np.zeros((480, 832, 3), dtype=np.uint8)  # H=480, W=832
    fake_pipe.return_value.frames = [[fake_frame] * n_frames]
    fake_pipe.to.return_value = fake_pipe
    return fake_pipe


def test_manifest_capabilities_drive_runtime_construction_and_defaults():
    """load_backend-shaped construction: sequential offload + VAE tiling
    from the MANIFEST, and the MANIFEST generation defaults."""
    from muse.models.wan2_1_t2v_1_3b import MANIFEST, Model

    inner_pipe = _fake_pipe()
    fake_wan = MagicMock()
    fake_wan.from_pretrained.return_value = inner_pipe
    kwargs = {"model_id": MANIFEST["model_id"], **MANIFEST["capabilities"]}
    with patch(f"{_RT}.WanPipeline", fake_wan), \
            patch(f"{_RT}.DiffusionPipeline", MagicMock()), \
            patch(f"{_RT}.torch", MagicMock()):
        m = Model(hf_repo=MANIFEST["hf_repo"], local_dir="/fake", **kwargs)
        r = m.generate("a flag waving in the wind")

    inner_pipe.enable_sequential_cpu_offload.assert_called_once_with(
        device="cuda",
    )
    inner_pipe.to.assert_not_called()
    inner_pipe.enable_vae_tiling.assert_called_once()
    call = inner_pipe.call_args.kwargs
    assert call["num_frames"] == 25  # 5.0s * 5fps
    assert call["num_inference_steps"] == 30
    assert call["guidance_scale"] == 5.0
    assert (call["width"], call["height"]) == (832, 480)
    assert r.fps == 5
    assert r.metadata["model"] == "wan2-1-t2v-1-3b"
