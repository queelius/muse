"""Resolver downloads must keep SentencePiece tokenizer files.

Regression for the 2026-10-18 install audit: explicit-glob allow-lists
(`*.json`, `*.txt`, `*.safetensors`) silently dropped `*.model` files, so
XLM-R (`sentencepiece.bpe.model`), DeBERTa-v3 (`spm.model`), Llama-style
(`tokenizer.model`) and T5 (`spiece.model`) tokenizers arrived incomplete.
Patterns are checked with huggingface_hub's own filter.
"""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from huggingface_hub.utils import filter_repo_objects

from muse.modalities.embedding_text.hf import HF_PLUGIN as EMBEDDING_TEXT
from muse.modalities.image_generation.hf import HF_PLUGIN as IMAGE_GENERATION
from muse.modalities.image_upscale.hf import HF_PLUGIN as IMAGE_UPSCALE
from muse.modalities.text_classification.hf import HF_PLUGIN as TEXT_CLASSIFICATION
from muse.modalities.text_rerank.hf import HF_PLUGIN as TEXT_RERANK


def _info(siblings, tags):
    info = MagicMock()
    info.siblings = [MagicMock(rfilename=f) for f in siblings]
    info.tags = tags
    info.card_data = MagicMock(license="mit")
    info.sha = "b" * 40
    info.config = {}
    return info


def _downloaded(plugin, module, repo_id, siblings, tags):
    resolved = plugin["resolve"](repo_id, None, _info(siblings, tags))
    with patch(f"{module}.snapshot_download", return_value="/snap") as dl:
        resolved.download(Path("/cache"))
    patterns = dl.call_args.kwargs["allow_patterns"]
    return set(filter_repo_objects(siblings, allow_patterns=patterns))


_TEXT_FILES = [
    "config.json", "model.safetensors", "tokenizer_config.json",
    "special_tokens_map.json",
]


@pytest.mark.parametrize("spm_file", [
    "sentencepiece.bpe.model", "spm.model", "tokenizer.model", "spiece.model",
])
@pytest.mark.parametrize("plugin, module, tags", [
    (EMBEDDING_TEXT, "muse.modalities.embedding_text.hf", ["sentence-transformers"]),
    (TEXT_CLASSIFICATION, "muse.modalities.text_classification.hf", ["text-classification"]),
    (TEXT_RERANK, "muse.modalities.text_rerank.hf", ["text-classification", "reranker"]),
])
def test_text_plugins_keep_sentencepiece_files(plugin, module, tags, spm_file):
    siblings = [*_TEXT_FILES, spm_file]
    got = _downloaded(plugin, module, "org/xlmr-reranker", siblings, tags)
    assert spm_file in got


_DIFFUSERS_FILES = [
    "model_index.json",
    "text_encoder_2/config.json",
    "text_encoder_2/model.safetensors",
    "tokenizer_2/tokenizer_config.json",
    "tokenizer_2/spiece.model",
]


@pytest.mark.parametrize("plugin, module, tags, repo", [
    (IMAGE_GENERATION, "muse.modalities.image_generation.hf",
     ["text-to-image", "diffusers"], "org/flux-like"),
    (IMAGE_UPSCALE, "muse.modalities.image_upscale.hf",
     ["image-to-image", "diffusers"], "org/sd-x4-upscaler"),
])
def test_diffusers_plugins_keep_t5_sentencepiece(plugin, module, tags, repo):
    got = _downloaded(plugin, module, repo, list(_DIFFUSERS_FILES), tags)
    assert "tokenizer_2/spiece.model" in got
