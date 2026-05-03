"""Tests for the text_classification HF plugin."""
from unittest.mock import MagicMock

from muse.modalities.text_classification.hf import HF_PLUGIN
from muse.core.discovery import REQUIRED_HF_PLUGIN_KEYS
from muse.core.resolvers import ResolvedModel


def _fake_info(siblings=None, tags=None, repo_id="org/repo"):
    info = MagicMock()
    info.siblings = [MagicMock(rfilename=f) for f in (siblings or [])]
    info.tags = tags or []
    info.card_data = MagicMock(license=None)
    # info.id is the repo path ("owner/name"); the v0.35.0 zero-shot
    # dispatch reads it for fallback name-based matching, so the
    # default must be a real string (not a MagicMock auto-attr).
    info.id = repo_id
    return info


def test_plugin_has_all_required_keys():
    for key in REQUIRED_HF_PLUGIN_KEYS:
        assert key in HF_PLUGIN


def test_plugin_metadata_correct():
    assert HF_PLUGIN["modality"] == "text/classification"
    assert HF_PLUGIN["runtime_path"].endswith(":HFTextClassifier")
    # priority 200: tag-only catch-all, must lose to specific plugins
    assert HF_PLUGIN["priority"] == 200


def test_sniff_true_on_text_classification_tag():
    assert HF_PLUGIN["sniff"](_fake_info(tags=["text-classification"])) is True


def test_sniff_false_on_random_repo():
    assert HF_PLUGIN["sniff"](_fake_info(tags=["text-generation"])) is False


def test_resolve_returns_resolved_model():
    info = _fake_info(tags=["text-classification"])
    result = HF_PLUGIN["resolve"]("KoalaAI/Text-Moderation", None, info)
    assert isinstance(result, ResolvedModel)
    assert result.manifest["modality"] == "text/classification"
    assert result.manifest["model_id"] == "text-moderation"


def test_search_yields_results():
    fake_api = MagicMock()
    fake_repo = MagicMock(id="org/classifier", downloads=80)
    fake_api.list_models.return_value = [fake_repo]
    rows = list(HF_PLUGIN["search"](fake_api, "moderation", sort="downloads", limit=20))
    assert len(rows) == 1
    assert rows[0].modality == "text/classification"


# ---- v0.35.0: zero-shot dispatch ----


def test_sniff_true_on_zero_shot_classification_tag():
    """Repos tagged zero-shot-classification (without text-classification)
    are claimed by this plugin so they resolve to HFZeroShotPipeline."""
    info = _fake_info(tags=["zero-shot-classification"])
    assert HF_PLUGIN["sniff"](info) is True


def test_sniff_true_on_nli_repo_name_fallback():
    """Some NLI checkpoints ship without our preferred tags but the
    repo name makes intent clear."""
    info = _fake_info(tags=[], repo_id="MoritzLaurer/some-zero-shot-model")
    assert HF_PLUGIN["sniff"](info) is True
    info = _fake_info(tags=[], repo_id="some-org/mnli-roberta")
    assert HF_PLUGIN["sniff"](info) is True


def test_resolve_zero_shot_dispatches_to_zero_shot_pipeline():
    """A zero-shot-classification-tagged repo resolves to the NLI
    runtime with supports_zero_shot=True, supports_classification=False."""
    info = _fake_info(
        tags=["zero-shot-classification", "text-classification"],
        repo_id="MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
    )
    result = HF_PLUGIN["resolve"](
        "MoritzLaurer/deberta-v3-base-zeroshot-v2.0", None, info,
    )
    assert "HFZeroShotPipeline" in result.backend_path
    caps = result.manifest["capabilities"]
    assert caps["supports_zero_shot"] is True
    assert caps["supports_classification"] is False
    assert "Zero-shot" in result.manifest["description"]


def test_resolve_classifier_dispatches_to_text_classifier():
    """A repo with only text-classification (no zero-shot) resolves
    to HFTextClassifier with supports_classification=True."""
    info = _fake_info(
        tags=["text-classification"],
        repo_id="cardiffnlp/twitter-roberta-base-sentiment-latest",
    )
    result = HF_PLUGIN["resolve"](
        "cardiffnlp/twitter-roberta-base-sentiment-latest", None, info,
    )
    assert "HFTextClassifier" in result.backend_path
    caps = result.manifest["capabilities"]
    assert caps["supports_classification"] is True
    assert caps["supports_zero_shot"] is False


def test_resolve_zero_shot_via_repo_name_fallback():
    """A repo without the zero-shot-classification tag but with NLI
    keywords in its id still dispatches to the NLI runtime."""
    info = _fake_info(
        tags=["text-classification"],  # only the generic tag
        repo_id="cross-encoder/nli-deberta-v3-large",
    )
    result = HF_PLUGIN["resolve"](
        "cross-encoder/nli-deberta-v3-large", None, info,
    )
    assert "HFZeroShotPipeline" in result.backend_path
    assert result.manifest["capabilities"]["supports_zero_shot"] is True


def test_existing_text_moderation_resolves_to_classifier():
    """Regression: KoalaAI/Text-Moderation has 'text-classification'
    tag and a name with no NLI hint, so it must keep resolving to
    HFTextClassifier (not HFZeroShotPipeline) so existing curated
    moderation entries continue to work."""
    info = _fake_info(
        tags=["text-classification"],
        repo_id="KoalaAI/Text-Moderation",
    )
    result = HF_PLUGIN["resolve"]("KoalaAI/Text-Moderation", None, info)
    assert "HFTextClassifier" in result.backend_path
    assert result.manifest["capabilities"]["supports_classification"] is True
