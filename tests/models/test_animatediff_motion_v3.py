"""Tests for the bundled animatediff_motion_v3 script.

The script aliases the generic `AnimateDiffRuntime` as its `Model`;
runtime behavior is covered by tests/modalities/image_animation/
runtimes/test_animatediff.py. These tests pin the MANIFEST and check
that the runtime honors it when constructed the way `load_backend` does
(`model_id` + capabilities splatted as kwargs).

Imports are done lazily inside each test function: the discovery
edge-case test in tests/core/test_discovery.py pops bundled-model
modules out of sys.modules and re-imports them.
"""
from unittest.mock import MagicMock, patch

_RT = "muse.modalities.image_animation.runtimes.animatediff"


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
    assert caps["base_model"] == "emilianJR/epiCRealism"
    assert caps["adapter_model_subdir"] == "motion_adapter"
    assert caps["base_model_subdir"] == "base_model"
    assert MANIFEST["revision"] == "2e8139b1d1269fd8a21deb96ad19455e187692eb"
    assert len(MANIFEST["hf_artifacts"]) == 2
    assert MANIFEST["hf_artifacts"][0]["required_patterns"] == [
        "config.json",
        "*.safetensors",
    ]
    assert "unet/*.safetensors" in (
        MANIFEST["hf_artifacts"][1]["required_patterns"]
    )


def test_model_is_animatediff_runtime():
    from muse.models.animatediff_motion_v3 import Model
    from muse.modalities.image_animation.runtimes.animatediff import (
        AnimateDiffRuntime,
    )
    assert Model is AnimateDiffRuntime


def test_manifest_capabilities_drive_bundle_load_and_defaults(tmp_path):
    from muse.models.animatediff_motion_v3 import MANIFEST, Model

    fake_frame = MagicMock()
    fake_frame.size = (512, 512)
    fake_pipe = MagicMock()
    fake_pipe.return_value.frames = [[fake_frame] * 16]
    pipe_cls = MagicMock()
    pipe_cls.from_pretrained.return_value = fake_pipe
    adapter_cls = MagicMock()
    (tmp_path / "motion_adapter").mkdir()
    (tmp_path / "base_model").mkdir()

    kwargs = {"model_id": MANIFEST["model_id"], **MANIFEST["capabilities"]}
    with patch(f"{_RT}.AnimateDiffPipeline", pipe_cls), \
            patch(f"{_RT}.MotionAdapter", adapter_cls), \
            patch(f"{_RT}.torch", MagicMock()):
        m = Model(hf_repo=MANIFEST["hf_repo"], local_dir=str(tmp_path), **kwargs)
        r = m.generate("a cat")

    assert adapter_cls.from_pretrained.call_args.args[0] == str(
        tmp_path / "motion_adapter"
    )
    assert pipe_cls.from_pretrained.call_args.args[0] == str(
        tmp_path / "base_model"
    )
    fake_pipe.to.assert_called_once_with("cuda")
    call = fake_pipe.call_args.kwargs
    assert call["num_frames"] == 16
    assert call["num_inference_steps"] == 25
    assert call["guidance_scale"] == 7.5
    assert (call["width"], call["height"]) == (512, 512)
    assert r.fps == 8
    assert r.metadata["model"] == "animatediff-motion-v3"
    assert r.metadata["base_model"] == "emilianJR/epiCRealism"
