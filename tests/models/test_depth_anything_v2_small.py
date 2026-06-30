"""Bundled depth-anything-v2-small discovery + manifest tests."""
from muse.models.depth_anything_v2_small import MANIFEST, Model


def test_manifest_required_keys():
    assert MANIFEST["model_id"] == "depth-anything-v2-small"
    assert MANIFEST["modality"] == "image/cv"
    assert MANIFEST["hf_repo"] == "depth-anything/Depth-Anything-V2-Small-hf"


def test_manifest_capabilities():
    caps = MANIFEST["capabilities"]
    assert caps["supports_depth"] is True
    assert caps["supports_keypoints"] is False
    assert caps["supports_detection"] is False
    assert caps["metric_depth"] is False  # relative depth
    assert caps["device"] == "auto"


def test_model_inherits_runtime():
    from muse.modalities.image_cv.runtimes import HFDepthRuntime
    assert issubclass(Model, HFDepthRuntime)


def test_pip_extras():
    extras = MANIFEST["pip_extras"]
    assert any(e.startswith("torch") for e in extras)
    assert any(e.startswith("transformers") for e in extras)
    assert any("Pillow" in e for e in extras)


def test_license():
    assert MANIFEST["license"] == "Apache 2.0"
