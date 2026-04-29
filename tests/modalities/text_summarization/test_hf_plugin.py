"""Tests for the text/summarization HF resolver plugin."""
from types import SimpleNamespace
from unittest.mock import MagicMock

from muse.modalities.text_summarization.hf import HF_PLUGIN


def _fake_info(repo_id, tags=(), siblings=(), card=None):
    return SimpleNamespace(
        id=repo_id,
        tags=list(tags),
        siblings=[SimpleNamespace(rfilename=s) for s in siblings],
        card_data=card,
    )


def test_plugin_keys_present():
    keys = {"modality", "runtime_path", "pip_extras", "system_packages",
            "priority", "sniff", "resolve", "search"}
    assert keys.issubset(HF_PLUGIN.keys())


def test_plugin_modality_and_priority():
    assert HF_PLUGIN["modality"] == "text/summarization"
    # 110: matches embedding/text. Wins over text/classification (200,
    # catch-all). Loses to file-pattern plugins at 100.
    assert HF_PLUGIN["priority"] == 110


def test_plugin_runtime_path():
    assert HF_PLUGIN["runtime_path"] == (
        "muse.modalities.text_summarization.runtimes.bart_seq2seq"
        ":BartSeq2SeqRuntime"
    )


def test_plugin_pip_extras_includes_torch_and_transformers():
    extras = HF_PLUGIN["pip_extras"]
    assert any("torch" in e for e in extras)
    assert any("transformers" in e for e in extras)


def test_sniff_true_for_summarization_tag():
    info = _fake_info("facebook/bart-large-cnn", tags=["summarization"])
    assert HF_PLUGIN["sniff"](info) is True


def test_sniff_true_for_summarization_tag_with_others():
    """Multiple task tags shouldn't break the sniff."""
    info = _fake_info(
        "philschmid/bart-large-cnn-samsum",
        tags=["summarization", "text2text-generation", "transformers"],
    )
    assert HF_PLUGIN["sniff"](info) is True


def test_sniff_false_for_text_classification_only():
    info = _fake_info("KoalaAI/Text-Moderation",
                     tags=["text-classification"])
    assert HF_PLUGIN["sniff"](info) is False


def test_sniff_false_for_chat_completion_repo():
    info = _fake_info("Qwen/Qwen3-8B-GGUF",
                     tags=["chat", "text-generation"])
    assert HF_PLUGIN["sniff"](info) is False


def test_sniff_false_for_repo_without_tags():
    info = _fake_info("acme/empty", tags=[])
    assert HF_PLUGIN["sniff"](info) is False


def test_sniff_false_for_no_tags_attribute():
    info = SimpleNamespace(id="x/y", siblings=[], card_data=None)
    # tags attribute is missing; sniff should default to "no tags".
    assert HF_PLUGIN["sniff"](info) is False


def test_resolve_synthesizes_manifest_for_bart_cnn():
    info = _fake_info(
        "facebook/bart-large-cnn",
        tags=["summarization"],
        card=SimpleNamespace(license="apache-2.0"),
    )
    resolved = HF_PLUGIN["resolve"]("facebook/bart-large-cnn", None, info)
    m = resolved.manifest
    assert m["model_id"] == "bart-large-cnn"
    assert m["modality"] == "text/summarization"
    assert m["hf_repo"] == "facebook/bart-large-cnn"
    assert m["license"] == "apache-2.0"
    caps = m["capabilities"]
    assert caps["device"] == "auto"
    assert caps["default_length"] == "medium"
    assert caps["default_format"] == "paragraph"
    assert caps["max_input_tokens"] == 1024
    # Generic news repo: not dialog-tuned.
    assert caps["supports_dialog_summarization"] is False


def test_resolve_dialog_heuristic_samsum():
    info = _fake_info(
        "philschmid/bart-large-cnn-samsum",
        tags=["summarization"],
    )
    resolved = HF_PLUGIN["resolve"](
        "philschmid/bart-large-cnn-samsum", None, info,
    )
    assert resolved.manifest["capabilities"]["supports_dialog_summarization"] is True


def test_resolve_dialog_heuristic_dialog():
    info = _fake_info(
        "acme/dialog-summarizer",
        tags=["summarization"],
    )
    resolved = HF_PLUGIN["resolve"]("acme/dialog-summarizer", None, info)
    assert resolved.manifest["capabilities"]["supports_dialog_summarization"] is True


def test_resolve_dialog_heuristic_chat():
    info = _fake_info(
        "acme/chat-summarizer",
        tags=["summarization"],
    )
    resolved = HF_PLUGIN["resolve"]("acme/chat-summarizer", None, info)
    assert resolved.manifest["capabilities"]["supports_dialog_summarization"] is True


def test_resolve_dialog_heuristic_meeting():
    info = _fake_info(
        "acme/meeting-notes-summarizer",
        tags=["summarization"],
    )
    resolved = HF_PLUGIN["resolve"](
        "acme/meeting-notes-summarizer", None, info,
    )
    assert resolved.manifest["capabilities"]["supports_dialog_summarization"] is True


def test_resolve_dialog_heuristic_default_false_for_news_summarizer():
    info = _fake_info(
        "google/pegasus-xsum",
        tags=["summarization"],
    )
    resolved = HF_PLUGIN["resolve"]("google/pegasus-xsum", None, info)
    assert resolved.manifest["capabilities"]["supports_dialog_summarization"] is False


def test_resolve_backend_path_points_to_bart_seq2seq_runtime():
    info = _fake_info("acme/x", tags=["summarization"])
    resolved = HF_PLUGIN["resolve"]("acme/x", None, info)
    assert resolved.backend_path == HF_PLUGIN["runtime_path"]


def test_resolve_includes_pip_extras_in_manifest():
    info = _fake_info("acme/x", tags=["summarization"])
    resolved = HF_PLUGIN["resolve"]("acme/x", None, info)
    extras = resolved.manifest["pip_extras"]
    assert any("torch" in e for e in extras)
    assert any("transformers" in e for e in extras)


def test_resolve_when_card_data_missing_license_is_none():
    info = _fake_info("acme/x", tags=["summarization"], card=None)
    resolved = HF_PLUGIN["resolve"]("acme/x", None, info)
    assert resolved.manifest["license"] is None


def test_search_yields_results():
    api = MagicMock()
    repo1 = SimpleNamespace(id="facebook/bart-large-cnn", downloads=1_000_000)
    repo2 = SimpleNamespace(id="philschmid/bart-large-cnn-samsum",
                            downloads=500_000)
    api.list_models.return_value = [repo1, repo2]
    out = list(HF_PLUGIN["search"](
        api, "summariz", sort="downloads", limit=10,
    ))
    assert len(out) == 2
    assert out[0].uri == "hf://facebook/bart-large-cnn"
    assert out[0].modality == "text/summarization"
    assert out[0].downloads == 1_000_000
    assert out[1].model_id == "bart-large-cnn-samsum"


def test_search_calls_list_models_with_summarization_filter():
    api = MagicMock()
    api.list_models.return_value = []
    list(HF_PLUGIN["search"](api, "x", sort="downloads", limit=5))
    _, kwargs = api.list_models.call_args
    assert kwargs["filter"] == "summarization"
    assert kwargs["sort"] == "downloads"
    assert kwargs["limit"] == 5
    assert kwargs["search"] == "x"
