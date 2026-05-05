"""HF plugin tests for audio/classification."""
from unittest.mock import MagicMock

from muse.core.discovery import REQUIRED_HF_PLUGIN_KEYS
from muse.core.resolvers import ResolvedModel
from muse.modalities.audio_classification.hf import HF_PLUGIN


def _fake_info(siblings=None, tags=None, repo_id="org/repo"):
    info = MagicMock()
    info.siblings = [MagicMock(rfilename=f) for f in (siblings or [])]
    info.tags = tags or []
    info.card_data = MagicMock(license=None)
    info.id = repo_id
    return info


def test_required_keys():
    for k in REQUIRED_HF_PLUGIN_KEYS:
        assert k in HF_PLUGIN


def test_metadata():
    assert HF_PLUGIN["modality"] == "audio/classification"
    assert HF_PLUGIN["priority"] == 110
    assert HF_PLUGIN["runtime_path"].endswith(":HFAudioClassifier")


def test_sniff_audio_classification_tag():
    assert HF_PLUGIN["sniff"](_fake_info(tags=["audio-classification"])) is True


def test_sniff_repo_name_fallback_ast():
    info = _fake_info(tags=[], repo_id="MIT/ast-finetuned-audioset")
    assert HF_PLUGIN["sniff"](info) is True


def test_sniff_repo_name_fallback_emotion():
    info = _fake_info(
        tags=[], repo_id="ehcalabres/wav2vec2-emotion-foo",
    )
    assert HF_PLUGIN["sniff"](info) is True


def test_sniff_repo_name_fallback_lid():
    info = _fake_info(tags=[], repo_id="facebook/mms-lid-126")
    assert HF_PLUGIN["sniff"](info) is True


def test_sniff_false_random_repo():
    info = _fake_info(tags=["text-generation"], repo_id="org/llm")
    assert HF_PLUGIN["sniff"](info) is False


def test_resolve_returns_runtime_path_and_capabilities():
    info = _fake_info(
        tags=["audio-classification"],
        repo_id="MIT/ast-finetuned-audioset-10-10-0.4593",
    )
    result = HF_PLUGIN["resolve"](
        "MIT/ast-finetuned-audioset-10-10-0.4593", None, info,
    )
    assert isinstance(result, ResolvedModel)
    assert "HFAudioClassifier" in result.backend_path
    assert result.manifest["modality"] == "audio/classification"
    assert "device" in result.manifest["capabilities"]


def test_resolve_kebab_case_model_id():
    info = _fake_info(
        tags=["audio-classification"], repo_id="MIT/AST-FOO-BAR",
    )
    result = HF_PLUGIN["resolve"]("MIT/AST-FOO-BAR", None, info)
    assert result.manifest["model_id"] == "ast-foo-bar"


def test_pip_extras_includes_librosa():
    """librosa is the audio decoder; missing it is a load-time
    ImportError on first inference."""
    extras = HF_PLUGIN["pip_extras"]
    assert any("librosa" in e for e in extras)


def test_search_yields_results():
    fake_api = MagicMock()
    fake_repo = MagicMock(id="org/audio-cls", downloads=42)
    fake_api.list_models.return_value = [fake_repo]
    rows = list(HF_PLUGIN["search"](
        fake_api, "audio", sort="downloads", limit=20,
    ))
    assert len(rows) == 1
    assert rows[0].modality == "audio/classification"
