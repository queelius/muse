"""Split (sharded) GGUF variants must download and size every shard.

Regression for the 2026-10-18 install audit: variant matching returned the
first matching file and the download allow-listed only that file, so a
`*-00001-of-00002.gguf` variant (e.g. Qwen/Qwen3-32B-GGUF's Q8_0) pulled
half its weights.
"""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from muse.core.resolvers import ResolverError
from muse.modalities.chat_completion.hf import HF_PLUGIN

_Q8 = ("Qwen3-32B-Q8_0-00001-of-00002.gguf", "Qwen3-32B-Q8_0-00002-of-00002.gguf")
_SIBLINGS = [
    "README.md",
    "config.json",
    "Qwen3-32B-Q4_K_M.gguf",
    *_Q8,
    "params",
]


def _info(siblings):
    info = MagicMock()
    info.siblings = [MagicMock(rfilename=f) for f in siblings]
    info.tags = []
    info.card_data = MagicMock(license="apache-2.0")
    info.sha = "a" * 40
    return info


def _resolve(siblings, variant):
    with patch("muse.modalities.chat_completion.hf._try_sniff_tools_from_repo", return_value=False), \
         patch("muse.modalities.chat_completion.hf._try_sniff_context_length_from_repo", return_value=None), \
         patch("muse.modalities.chat_completion.hf.lookup_chat_format", return_value={}):
        return HF_PLUGIN["resolve"]("Qwen/Qwen3-32B-GGUF", variant, _info(siblings))


def _allow_patterns(resolved):
    with patch(
        "muse.modalities.chat_completion.hf.snapshot_download",
        return_value="/snap",
    ) as dl:
        resolved.download(Path("/cache"))
    return dl.call_args.kwargs["allow_patterns"]


def test_split_variant_downloads_every_shard_and_loads_the_first():
    resolved = _resolve(_SIBLINGS, "q8_0")

    assert resolved.manifest["capabilities"]["gguf_file"] == _Q8[0]
    patterns = _allow_patterns(resolved)
    assert set(_Q8) <= set(patterns)
    assert "Qwen3-32B-Q4_K_M.gguf" not in patterns


def test_split_variant_in_a_subdirectory():
    shards = ["Q8_0/" + f for f in _Q8]
    resolved = _resolve(["Q4_K_M/Qwen3-32B-Q4_K_M.gguf", *reversed(shards)], "q8_0")

    assert resolved.manifest["capabilities"]["gguf_file"] == shards[0]
    assert set(shards) <= set(_allow_patterns(resolved))


def test_missing_shard_is_a_resolve_error():
    with pytest.raises(ResolverError, match="00002"):
        _resolve(["Qwen3-32B-Q8_0-00001-of-00002.gguf"], "q8_0")


def test_single_file_variant_is_unchanged():
    resolved = _resolve(_SIBLINGS, "q4_k_m")

    assert resolved.manifest["capabilities"]["gguf_file"] == "Qwen3-32B-Q4_K_M.gguf"
    patterns = _allow_patterns(resolved)
    assert "Qwen3-32B-Q4_K_M.gguf" in patterns
    assert not set(_Q8) & set(patterns)


def test_search_still_emits_one_row_per_split_variant():
    api = MagicMock()
    repo = MagicMock(id="Qwen/Qwen3-32B-GGUF", downloads=1)
    repo.siblings = [
        MagicMock(rfilename=_Q8[0], size=20_000_000_000),
        MagicMock(rfilename=_Q8[1], size=14_000_000_000),
    ]
    api.list_models.return_value = [repo]

    rows = list(HF_PLUGIN["search"](api, "qwen3", sort="downloads", limit=5))

    assert [r.uri for r in rows] == ["hf://Qwen/Qwen3-32B-GGUF@q8_0"]
    assert rows[0].size_gb == pytest.approx(34.0)
