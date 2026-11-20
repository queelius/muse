"""Real (priority, modality) sniff dispatch across every bundled HF plugin.

Guards name hints and tag overlaps that let one plugin claim another
modality's repos (review findings I3, I4 and R5).
"""
from unittest.mock import MagicMock

import pytest

from muse.core.discovery import modality_dirs, discover_hf_plugins


def _info(repo_id, tags=(), files=()):
    info = MagicMock()
    info.id = repo_id
    info.tags = list(tags)
    info.siblings = [MagicMock(rfilename=name) for name in files]
    info.card_data = MagicMock(license=None)
    return info


def _first_claim(info):
    for plugin in discover_hf_plugins(modality_dirs()):
        if plugin["sniff"](info):
            return plugin["modality"]
    return None


@pytest.mark.parametrize("repo_id, tags, expected", [
    # I3: the "ast-" hint must not match fast-/podcast-/contrast-.
    ("acme/fast-sentence-embedder", ["sentence-transformers"], "embedding/text"),
    ("acme/podcast-summarizer", ["summarization"], "text/summarization"),
    ("acme/contrast-detector", ["object-detection"], "image/cv"),
    ("MIT/ast-finetuned-audioset-10-10-0.4593", [], "audio/classification"),
    # I4: cross-encoders tagged sentence-transformers are rerankers.
    ("cross-encoder/ms-marco-MiniLM-L6-v2",
     ["sentence-transformers", "cross-encoder", "text-classification"],
     "text/rerank"),
    ("BAAI/bge-reranker-v2-m3",
     ["sentence-transformers", "text-classification"], "text/rerank"),
    ("acme/some-ranker", ["sentence-transformers", "text-ranking"], "text/rerank"),
    ("sentence-transformers/all-MiniLM-L6-v2",
     ["sentence-transformers"], "embedding/text"),
])
def test_dispatch(repo_id, tags, expected):
    assert _first_claim(_info(repo_id, tags)) == expected


@pytest.mark.parametrize("repo_id", ["acme/unlimited-thing", "acme/online-tool"])
def test_bare_nli_substring_does_not_claim_tagless_repos(repo_id):
    """R5: "nli" inside "online"/"unlimited" is not an NLI checkpoint."""
    assert _first_claim(_info(repo_id)) is None


@pytest.mark.parametrize("repo_id", [
    "facebook/bart-large-mnli",
    "joeddav/xlm-roberta-large-xnli",
    "cross-encoder/nli-deberta-v3-base",
    "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
])
def test_nli_names_still_resolve_zero_shot(repo_id):
    import importlib.util
    from pathlib import Path

    import muse.modalities.text_classification as pkg
    spec = importlib.util.spec_from_file_location(
        "_tc_hf_for_test", Path(pkg.__file__).parent / "hf.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    info = _info(repo_id)
    assert module._sniff(info)
    assert module._is_zero_shot(info)
