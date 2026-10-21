"""Tests for the shared HF plugin scaffolding."""
from types import SimpleNamespace

from muse.core.hf_plugin_helpers import (
    hf_model_id,
    hf_repo_license,
    hf_tag_search,
    transformers_allow_patterns,
)


def _info(*files, card=None):
    return SimpleNamespace(
        siblings=[SimpleNamespace(rfilename=f) for f in files],
        card_data=card,
    )


def test_hf_model_id_is_lowercased_basename():
    assert hf_model_id("BAAI/BGE-Reranker-v2-M3") == "bge-reranker-v2-m3"
    assert hf_model_id("no-org") == "no-org"


def test_hf_repo_license_reads_card_or_none():
    assert hf_repo_license(_info(card=SimpleNamespace(license="mit"))) == "mit"
    assert hf_repo_license(_info()) is None
    assert hf_repo_license(SimpleNamespace()) is None


def test_allow_patterns_prefers_fp16_safetensors():
    patterns = transformers_allow_patterns(
        _info("model.safetensors", "model.fp16.safetensors", "pytorch_model.bin"),
    )
    assert "*.fp16.safetensors" in patterns
    assert "*.safetensors" not in patterns


def test_allow_patterns_skip_pickle_when_safetensors_exist():
    patterns = transformers_allow_patterns(
        _info("model.safetensors", "pytorch_model.bin", "config.json"),
    )
    assert "*.safetensors" in patterns
    assert not any(p.endswith(".bin") for p in patterns)


def test_allow_patterns_fall_back_to_pickle_without_safetensors():
    patterns = transformers_allow_patterns(
        _info("pytorch_model-00001-of-00002.bin", "config.json"),
    )
    assert "pytorch_model*.bin" in patterns
    # No siblings metadata at all: keep the fallback rather than risk
    # downloading no weights.
    assert "pytorch_model*.bin" in transformers_allow_patterns(SimpleNamespace())


def test_allow_patterns_returns_fresh_list():
    info = _info("model.safetensors")
    transformers_allow_patterns(info).append("x")
    assert "x" not in transformers_allow_patterns(info)


class _Api:
    def __init__(self, ids):
        self.ids = ids
        self.calls = []

    def list_models(self, **kwargs):
        self.calls.append(kwargs)
        return [SimpleNamespace(id=i, downloads=7) for i in self.ids]


def test_hf_tag_search_builds_rows_and_filters():
    api = _Api(["org/Keep-Me", "org/drop"])
    rows = list(hf_tag_search(
        api, "q", sort="downloads", limit=5,
        tag="summarization", modality="text/summarization",
        keep=lambda rid: "keep" in rid.lower(),
    ))
    assert api.calls == [
        {"search": "q", "filter": "summarization", "sort": "downloads", "limit": 5},
    ]
    assert len(rows) == 1
    row = rows[0]
    assert row.uri == "hf://org/Keep-Me"
    assert row.model_id == "keep-me"
    assert row.modality == "text/summarization"
    assert row.downloads == 7
    assert row.license is None and row.size_gb is None
    assert row.description == "org/Keep-Me"
