"""Bundled detr-resnet-50 discovery + manifest tests."""
from muse.models.detr_resnet_50 import MANIFEST, Model


def test_manifest_required_keys():
    assert MANIFEST["model_id"] == "detr-resnet-50"
    assert MANIFEST["modality"] == "image/cv"
    assert MANIFEST["hf_repo"] == "facebook/detr-resnet-50"


def test_manifest_capabilities():
    caps = MANIFEST["capabilities"]
    assert caps["supports_depth"] is False
    assert caps["supports_keypoints"] is False
    assert caps["supports_detection"] is True
    assert caps["device"] == "auto"


def test_model_inherits_runtime():
    from muse.modalities.image_cv.runtimes import HFObjectDetectionRuntime
    assert issubclass(Model, HFObjectDetectionRuntime)


def test_pip_extras_includes_timm():
    """DETR's ResNet backbone is loaded via timm; missing it would
    surface as an ImportError on first inference."""
    extras = MANIFEST["pip_extras"]
    assert "timm" in extras


def test_license():
    assert MANIFEST["license"] == "Apache 2.0"
