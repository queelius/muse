"""Tests for VLM-aware sniff/resolve in the chat_completion HF plugin.

Five tests:
1. _sniff claims an `image-text-to-text` tag.
2. _resolve for `idefics3` model_type returns `supports_multi_image: True`.
3. _resolve for `llava` model_type returns `supports_multi_image: False`.
4. _sniff claims a `.gguf` repo (regression watchdog); _resolve for a GGUF
   returns the LlamaCppModel backend path.
5. _sniff does NOT claim a generic text-only chat repo (no VLM tag, no VLM
   model_type, no .gguf files).
"""
from unittest.mock import MagicMock, patch

import pytest

from muse.modalities.chat_completion.hf import HF_PLUGIN
from muse.core.resolvers import ResolvedModel


def _fake_repo_info(*, tags=None, files=None, model_type=None, siblings=None):
    """Build a minimal fake HF repo info object.

    `files` populates `siblings[].rfilename`.
    `model_type` is placed into `info.config["model_type"]`.
    """
    info = MagicMock()
    info.tags = tags or []
    info.card_data = MagicMock(license=None)

    # siblings: used by GGUF detection and download patterns
    sib_names = files or siblings or []
    info.siblings = [MagicMock(rfilename=f) for f in sib_names]

    # config dict: used by _is_vlm for model_type
    if model_type is not None:
        info.config = {"model_type": model_type}
    else:
        info.config = {}

    return info


# ---------------------------------------------------------------------------
# Test 1: _sniff claims image-text-to-text tag
# ---------------------------------------------------------------------------

def test_sniff_image_text_to_text_tag_claims_repo():
    info = _fake_repo_info(
        tags=["image-text-to-text", "transformers"],
        files=["config.json", "preprocessor_config.json", "model.safetensors"],
        model_type="idefics3",
    )
    assert HF_PLUGIN["sniff"](info) is True


# ---------------------------------------------------------------------------
# Test 2: _resolve for idefics3 model_type returns supports_multi_image: True
# ---------------------------------------------------------------------------

def test_resolve_idefics3_model_type_returns_multi_image():
    info = _fake_repo_info(
        tags=["image-text-to-text", "transformers"],
        files=["config.json", "preprocessor_config.json", "model.safetensors"],
        model_type="idefics3",
    )
    resolved = HF_PLUGIN["resolve"]("HuggingFaceTB/SmolVLM-Test", None, info)
    assert isinstance(resolved, ResolvedModel)
    assert resolved.backend_path.endswith(":HFVisionLanguageModel")
    assert resolved.manifest["modality"] == "chat/completion"
    assert resolved.manifest["capabilities"]["supports_vision"] is True
    assert resolved.manifest["capabilities"]["supports_multi_image"] is True


# ---------------------------------------------------------------------------
# Test 3: _resolve for llava model_type returns supports_multi_image: False
# ---------------------------------------------------------------------------

def test_resolve_llava_model_type_returns_single_image():
    info = _fake_repo_info(
        tags=["image-text-to-text", "transformers"],
        files=["config.json", "preprocessor_config.json", "model.safetensors"],
        model_type="llava",
    )
    resolved = HF_PLUGIN["resolve"]("llava-hf/llava-1.5-7b-hf", None, info)
    assert isinstance(resolved, ResolvedModel)
    assert resolved.backend_path.endswith(":HFVisionLanguageModel")
    assert resolved.manifest["capabilities"]["supports_vision"] is True
    assert resolved.manifest["capabilities"]["supports_multi_image"] is False


# ---------------------------------------------------------------------------
# Test 4: _sniff claims .gguf repos; _resolve for GGUF -> LlamaCppModel
# ---------------------------------------------------------------------------

def test_sniff_gguf_repo_claims_it():
    info = _fake_repo_info(
        tags=[],
        files=["model-q4_k_m.gguf"],
    )
    assert HF_PLUGIN["sniff"](info) is True


def test_resolve_gguf_returns_llama_cpp_backend():
    info = _fake_repo_info(
        tags=[],
        files=["model-q4_k_m.gguf"],
    )
    with patch("muse.modalities.chat_completion.hf._try_sniff_tools_from_repo", return_value=False), \
         patch("muse.modalities.chat_completion.hf._try_sniff_context_length_from_repo", return_value=None), \
         patch("muse.modalities.chat_completion.hf.lookup_chat_format", return_value={}):
        resolved = HF_PLUGIN["resolve"]("org/SomeModel-GGUF", "q4_k_m", info)
    assert isinstance(resolved, ResolvedModel)
    assert resolved.backend_path.endswith(":LlamaCppModel")
    assert resolved.manifest["capabilities"]["gguf_file"] == "model-q4_k_m.gguf"


# ---------------------------------------------------------------------------
# Test 5: _sniff does NOT claim a generic text-only chat repo
# ---------------------------------------------------------------------------

def test_sniff_does_not_claim_text_only_repo():
    """A repo with model_type=llama and no .gguf files and no VLM tags
    should NOT be claimed by this plugin."""
    info = _fake_repo_info(
        tags=["text-generation", "transformers"],
        files=["config.json", "model.safetensors"],
        model_type="llama",
    )
    assert HF_PLUGIN["sniff"](info) is False
