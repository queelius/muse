"""Tests for the bundled animatediff_motion_v3 script.

Imports of `MANIFEST` and `Model` are done lazily inside each test
function (matches the sd_turbo / nv_embed_v2 pattern). The discovery
edge-case test in tests/core/test_discovery.py pops bundled-model
modules out of sys.modules and re-imports them, which would leave a
top-level `from muse.models.animatediff_motion_v3 import Model` in
this file pointing at the stale (popped) module object. Lazy imports
sidestep that.
"""
from unittest.mock import MagicMock, patch

import pytest


def test_manifest_shape():
    from muse.models.animatediff_motion_v3 import MANIFEST
    assert MANIFEST["model_id"] == "animatediff-motion-v3"
    assert MANIFEST["modality"] == "image/animation"
    assert MANIFEST["hf_repo"] == "guoyww/animatediff-motion-adapter-v1-5-3"
    caps = MANIFEST["capabilities"]
    assert caps["supports_text_to_animation"] is True
    assert caps["supports_image_to_animation"] is False
    assert caps["default_frames"] == 16
    assert caps["default_fps"] == 8
    assert caps["device"] == "cuda"
    # Base model is referenced
    assert "base_model" in caps


def _patched_pipe():
    fake_pipe = MagicMock()
    fake_frame = MagicMock()
    fake_frame.size = (512, 512)
    fake_pipe.return_value.frames = [[fake_frame] * 16]
    return fake_pipe


def test_construction_loads_adapter_and_base():
    fake_pipe_class = MagicMock()
    fake_pipe_class.from_pretrained.return_value = _patched_pipe()
    fake_adapter_class = MagicMock()
    fake_adapter_class.from_pretrained.return_value = MagicMock()

    with patch(
        "muse.models.animatediff_motion_v3.AnimateDiffPipeline",
        fake_pipe_class,
    ), patch(
        "muse.models.animatediff_motion_v3.MotionAdapter",
        fake_adapter_class,
    ), patch(
        "muse.models.animatediff_motion_v3.torch",
        MagicMock(),
    ):
        from muse.models.animatediff_motion_v3 import MANIFEST, Model
        m = Model(
            hf_repo="guoyww/animatediff-motion-adapter-v1-5-3",
            local_dir="/fake/adapter",
            device="cpu",
        )
    fake_adapter_class.from_pretrained.assert_called_once()
    fake_pipe_class.from_pretrained.assert_called_once()
    # The pipe is loaded with base_model from MANIFEST capabilities
    pipe_call = fake_pipe_class.from_pretrained.call_args
    assert pipe_call.args[0] == MANIFEST["capabilities"]["base_model"]
    assert m.model_id == MANIFEST["model_id"]


def test_generate_returns_animation_result():
    from muse.modalities.image_animation.protocol import AnimationResult

    fake_pipe = _patched_pipe()
    fake_pipe_class = MagicMock()
    fake_pipe_class.from_pretrained.return_value = fake_pipe
    fake_adapter_class = MagicMock()
    fake_adapter_class.from_pretrained.return_value = MagicMock()

    with patch(
        "muse.models.animatediff_motion_v3.AnimateDiffPipeline",
        fake_pipe_class,
    ), patch(
        "muse.models.animatediff_motion_v3.MotionAdapter",
        fake_adapter_class,
    ), patch(
        "muse.models.animatediff_motion_v3.torch",
        MagicMock(),
    ):
        from muse.models.animatediff_motion_v3 import MANIFEST, Model
        m = Model(hf_repo="x", local_dir="/fake", device="cpu")
        r = m.generate("a cat")

    assert isinstance(r, AnimationResult)
    assert len(r.frames) == 16
    assert r.fps == MANIFEST["capabilities"]["default_fps"]
