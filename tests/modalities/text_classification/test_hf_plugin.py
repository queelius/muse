"""Tests for the text_classification HF plugin."""
from unittest.mock import MagicMock

from muse.modalities.text_classification.hf import HF_PLUGIN
from muse.core.discovery import REQUIRED_HF_PLUGIN_KEYS
from muse.core.resolvers import ResolvedModel


def _fake_info(siblings=None, tags=None):
    info = MagicMock()
    info.siblings = [MagicMock(rfilename=f) for f in (siblings or [])]
    info.tags = tags or []
    info.card_data = MagicMock(license=None)
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
