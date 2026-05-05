"""Bundled ast-audioset discovery + manifest tests."""
from muse.models.ast_audioset import MANIFEST, Model


def test_manifest_required_keys():
    assert MANIFEST["model_id"] == "ast-audioset"
    assert MANIFEST["modality"] == "audio/classification"
    assert MANIFEST["hf_repo"] == "MIT/ast-finetuned-audioset-10-10-0.4593"


def test_manifest_capabilities():
    caps = MANIFEST["capabilities"]
    assert caps["device"] == "cpu"
    assert caps["multi_label"] is True
    assert caps["num_labels"] == 527


def test_model_inherits_runtime():
    from muse.modalities.audio_classification.runtimes import HFAudioClassifier
    assert issubclass(Model, HFAudioClassifier)


def test_pip_extras_includes_librosa_and_transformers():
    extras = MANIFEST["pip_extras"]
    assert any("librosa" in e for e in extras)
    assert any(e.startswith("transformers") for e in extras)
    assert any(e.startswith("torch") for e in extras)


def test_license():
    assert MANIFEST["license"] == "BSD-3-Clause"
