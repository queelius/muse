"""Bundled twitter-roberta-base-sentiment-latest discovery + manifest tests."""
from muse.models.twitter_roberta_base_sentiment_latest import MANIFEST, Model


def test_manifest_required_keys():
    assert MANIFEST["model_id"] == "twitter-roberta-base-sentiment-latest"
    assert MANIFEST["modality"] == "text/classification"
    assert MANIFEST["hf_repo"] == "cardiffnlp/twitter-roberta-base-sentiment-latest"


def test_manifest_capabilities():
    caps = MANIFEST["capabilities"]
    assert caps["supports_classification"] is True
    assert caps["supports_zero_shot"] is False
    assert caps["device"] == "cpu"
    assert caps["memory_gb"] == 0.6


def test_model_is_hf_text_classifier_subclass():
    """The bundled Model class delegates everything to HFTextClassifier;
    discovery only requires the class name to be `Model`."""
    from muse.modalities.text_classification.runtimes import HFTextClassifier
    assert issubclass(Model, HFTextClassifier)


def test_pip_extras_listed():
    extras = MANIFEST["pip_extras"]
    assert any(e.startswith("torch") for e in extras)
    assert any(e.startswith("transformers") for e in extras)


def test_license_is_mit():
    assert MANIFEST["license"] == "MIT"
