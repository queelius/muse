"""Tests for the ``music/midi`` HuggingFace resolver plugin (``hf.py``)."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

from muse.core.discovery import REQUIRED_HF_PLUGIN_KEYS
from muse.core.resolvers import ResolvedModel, SearchResult


_HF_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "muse"
    / "modalities"
    / "music_midi"
    / "hf.py"
)


def _load_single_file_plugin():
    """Load ``hf.py`` via single-file spec import, matching Muse's resolver."""
    spec = importlib.util.spec_from_file_location("music_midi_hf_isolated", _HF_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.HF_PLUGIN


def _fake_info(
    repo_id: str,
    *,
    tags: list[str] | None = None,
    files: list[str] | None = None,
    sha: str = "abc1234",
):
    info = MagicMock()
    info.id = repo_id
    info.tags = tags or []
    info.sha = sha
    info.card_data = None
    siblings = []
    for fname in files if files is not None else ["config.json", "model.safetensors"]:
        s = MagicMock()
        s.rfilename = fname
        siblings.append(s)
    info.siblings = siblings
    return info


def test_hf_plugin_contract_and_priority():
    plugin = _load_single_file_plugin()
    assert set(REQUIRED_HF_PLUGIN_KEYS) <= set(plugin)
    assert plugin["modality"] == "music/midi"
    assert plugin["priority"] == 95
    assert plugin["runtime_path"] == (
        "muse.modalities.music_midi.runtimes.abc_lm:AbcMidiLMRuntime"
    )
    assert any(p.startswith("transformers") for p in plugin["pip_extras"])
    assert any(p.startswith("torch") for p in plugin["pip_extras"])


def test_hf_plugin_sniff_positive_and_negative():
    plugin = _load_single_file_plugin()
    sniff = plugin["sniff"]

    # Positive: known ABC/MIDI LM repo names
    assert sniff(_fake_info("sander-wood/text-to-music", tags=["text-generation"]))
    assert sniff(_fake_info("ElectricAlexis/NotaGen", tags=["music"]))
    assert sniff(_fake_info("org/abc-gpt-small", tags=["text-generation"]))

    # Positive: explicit symbolic-music tag
    assert sniff(_fake_info("someone/my-composer", tags=["symbolic-music"]))
    assert sniff(_fake_info("someone/abc-model", tags=["abc-notation"]))

    # Negative: missing config.json
    assert not sniff(_fake_info("sander-wood/text-to-music", files=["README.md"]))
    # Negative: generic LLM
    assert not sniff(_fake_info("meta-llama/Llama-3.2-3B", tags=["text-generation"]))
    # Negative: audio diffusion model
    assert not sniff(
        _fake_info("stabilityai/stable-audio-open-1.0", tags=["text-to-audio"])
    )


def test_hf_plugin_resolve_builds_manifest_and_download_closure(tmp_path):
    plugin = _load_single_file_plugin()
    commit_sha = "a" * 40
    info = _fake_info(
        "sander-wood/text-to-music",
        tags=["text-generation", "symbolic-music"],
        files=["config.json", "tokenizer.json", "model.safetensors"],
        sha=commit_sha,
    )
    resolved = plugin["resolve"]("sander-wood/text-to-music", None, info)
    assert isinstance(resolved, ResolvedModel)
    assert resolved.backend_path == plugin["runtime_path"]
    m = resolved.manifest
    assert m["modality"] == "music/midi"
    assert m["hf_repo"] == "sander-wood/text-to-music"
    assert m["revision"] == commit_sha
    assert m["capabilities"]["device"] == "auto"
    assert m["capabilities"]["format"] == "abc"

    spec = importlib.util.spec_from_file_location("music_midi_hf_dl", _HF_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with patch.object(mod, "snapshot_download", return_value=str(tmp_path)) as dl:
        res2 = mod.HF_PLUGIN["resolve"]("sander-wood/text-to-music", None, info)
        out_path = res2.download(tmp_path)
        assert out_path == tmp_path
        assert dl.called


def test_hf_plugin_search_yields_music_midi_results():
    plugin = _load_single_file_plugin()
    api = MagicMock()
    m1 = MagicMock(id="sander-wood/text-to-music", downloads=500, likes=50, tags=["text-generation"])
    m2 = MagicMock(id="meta-llama/Llama-3-8B", downloads=9999, likes=999, tags=["text-generation"])
    api.list_models.return_value = [m1, m2]

    results = list(plugin["search"](api, "music", sort="downloads", limit=10))
    assert len(results) == 1
    assert isinstance(results[0], SearchResult)
    assert results[0].modality == "music/midi"
    assert results[0].uri == "hf://sander-wood/text-to-music"
