"""Bundled deberta-v3-base-zeroshot-v2.0 discovery + manifest tests."""
from muse.models.deberta_v3_base_zeroshot_v2_0 import MANIFEST, Model


def test_manifest_required_keys():
    assert MANIFEST["model_id"] == "deberta-v3-base-zeroshot-v2.0"
    assert MANIFEST["modality"] == "text/classification"
    assert MANIFEST["hf_repo"] == "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"


def test_manifest_capabilities():
    """Zero-shot only; supports_classification=False so the route
    properly gates candidate-labels-required errors when a user
    omits them."""
    caps = MANIFEST["capabilities"]
    assert caps["supports_classification"] is False
    assert caps["supports_zero_shot"] is True
    assert caps["device"] == "cpu"
    assert caps["memory_gb"] == 1.2


def test_model_is_hf_zero_shot_pipeline_subclass():
    from muse.modalities.text_classification.runtimes import HFZeroShotPipeline
    assert issubclass(Model, HFZeroShotPipeline)


def test_pip_extras_listed():
    extras = MANIFEST["pip_extras"]
    assert any(e.startswith("torch") for e in extras)
    assert any(e.startswith("transformers") for e in extras)


def test_license_is_mit():
    assert MANIFEST["license"] == "MIT"
