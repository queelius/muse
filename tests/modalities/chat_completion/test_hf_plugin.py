"""Tests for the chat_completion HF plugin (GGUF)."""
from unittest.mock import MagicMock, patch

import pytest

from muse.modalities.chat_completion.hf import HF_PLUGIN
from muse.core.discovery import REQUIRED_HF_PLUGIN_KEYS
from muse.core.resolvers import ResolvedModel, SearchResult


def _fake_info(siblings=None, tags=None):
    info = MagicMock()
    info.siblings = [MagicMock(rfilename=f) for f in (siblings or [])]
    info.tags = tags or []
    info.card_data = MagicMock(license=None)
    return info


def test_plugin_has_all_required_keys():
    for key in REQUIRED_HF_PLUGIN_KEYS:
        assert key in HF_PLUGIN, f"missing {key!r}"


def test_plugin_metadata_correct():
    assert HF_PLUGIN["modality"] == "chat/completion"
    assert HF_PLUGIN["runtime_path"] == (
        "muse.modalities.chat_completion.runtimes.llama_cpp:LlamaCppModel"
    )
    assert HF_PLUGIN["priority"] == 100


def test_sniff_true_on_repo_with_gguf_files():
    info = _fake_info(siblings=["model-q4_k_m.gguf"])
    assert HF_PLUGIN["sniff"](info) is True


def test_sniff_false_when_no_gguf():
    info = _fake_info(siblings=["model.bin", "config.json"])
    assert HF_PLUGIN["sniff"](info) is False


def test_resolve_requires_variant_for_gguf():
    info = _fake_info(siblings=["model-q4_k_m.gguf", "model-q8_0.gguf"])
    from muse.core.resolvers import ResolverError
    with pytest.raises(ResolverError, match="variant required"):
        HF_PLUGIN["resolve"]("org/Model-GGUF", None, info)


def test_resolve_returns_resolved_model_with_correct_manifest():
    info = _fake_info(siblings=["model-q4_k_m.gguf"])
    with patch("muse.modalities.chat_completion.hf._try_sniff_tools_from_repo", return_value=False), \
         patch("muse.modalities.chat_completion.hf._try_sniff_context_length_from_repo", return_value=None), \
         patch("muse.modalities.chat_completion.hf.lookup_chat_format", return_value={}):
        result = HF_PLUGIN["resolve"]("org/Model-GGUF", "q4_k_m", info)
    assert isinstance(result, ResolvedModel)
    assert result.manifest["modality"] == "chat/completion"
    assert result.manifest["hf_repo"] == "org/Model-GGUF"
    assert result.manifest["capabilities"]["gguf_file"] == "model-q4_k_m.gguf"
    assert result.backend_path.endswith(":LlamaCppModel")


def test_resolve_applies_chat_format_hints_from_yaml():
    """When chat_formats.yaml has both chat_format and supports_tools, both land in capabilities."""
    info = _fake_info(siblings=["model-q4_k_m.gguf"])
    with patch("muse.modalities.chat_completion.hf._try_sniff_tools_from_repo", return_value=None), \
         patch("muse.modalities.chat_completion.hf._try_sniff_context_length_from_repo", return_value=None), \
         patch("muse.modalities.chat_completion.hf.lookup_chat_format", return_value={
             "chat_format": "chatml-function-calling",
             "supports_tools": True,
         }):
        result = HF_PLUGIN["resolve"]("unsloth/Qwen3.5-4B-GGUF", "q4_k_m", info)
    caps = result.manifest["capabilities"]
    assert caps["chat_format"] == "chatml-function-calling"
    assert caps["supports_tools"] is True


def test_resolve_yaml_supports_tools_overrides_sniff_result():
    """YAML's supports_tools wins over the sniff helper's result."""
    info = _fake_info(siblings=["model-q4_k_m.gguf"])
    with patch("muse.modalities.chat_completion.hf._try_sniff_tools_from_repo", return_value=False), \
         patch("muse.modalities.chat_completion.hf._try_sniff_context_length_from_repo", return_value=None), \
         patch("muse.modalities.chat_completion.hf.lookup_chat_format", return_value={
             "supports_tools": True,
         }):
        result = HF_PLUGIN["resolve"]("org/whatever", "q4_k_m", info)
    assert result.manifest["capabilities"]["supports_tools"] is True


def test_resolve_no_yaml_match_preserves_sniff_supports_tools():
    """When YAML has no entry, capabilities.supports_tools mirrors the sniff result."""
    info = _fake_info(siblings=["model-q4_k_m.gguf"])
    with patch("muse.modalities.chat_completion.hf._try_sniff_tools_from_repo", return_value=True), \
         patch("muse.modalities.chat_completion.hf._try_sniff_context_length_from_repo", return_value=None), \
         patch("muse.modalities.chat_completion.hf.lookup_chat_format", return_value={}):
        result = HF_PLUGIN["resolve"]("org/whatever", "q4_k_m", info)
    caps = result.manifest["capabilities"]
    assert caps["supports_tools"] is True
    assert "chat_format" not in caps


def test_resolve_variant_match_is_case_insensitive():
    """`Q4_K_M` should match `model-q4_k_m.gguf` (variant matching lowercases input)."""
    info = _fake_info(siblings=["model-q4_k_m.gguf"])
    with patch("muse.modalities.chat_completion.hf._try_sniff_tools_from_repo", return_value=None), \
         patch("muse.modalities.chat_completion.hf._try_sniff_context_length_from_repo", return_value=None), \
         patch("muse.modalities.chat_completion.hf.lookup_chat_format", return_value={}):
        result = HF_PLUGIN["resolve"]("org/Model-GGUF", "Q4_K_M", info)
    assert result.manifest["capabilities"]["gguf_file"] == "model-q4_k_m.gguf"


def test_search_yields_search_results_with_modality_tag():
    fake_api = MagicMock()
    fake_repo = MagicMock(id="org/repo", downloads=100)
    fake_repo.siblings = [
        MagicMock(rfilename="model-q4_k_m.gguf", size=2_500_000_000),
    ]
    fake_api.list_models.return_value = [fake_repo]
    rows = list(HF_PLUGIN["search"](fake_api, "qwen", sort="downloads", limit=20))
    assert len(rows) >= 1
    assert all(r.modality == "chat/completion" for r in rows)
    assert all(r.uri.startswith("hf://") for r in rows)
