from unittest.mock import MagicMock, patch

import pytest

from muse.core.discovery import (
    REQUIRED_HF_PLUGIN_KEYS,
    _default_hf_plugin_dirs,
    discover_hf_plugins,
)
from muse.core.resolvers import ResolvedModel, ResolverError
from muse.modalities.audio_alignment.hf import HF_PLUGIN


FILES = [
    "config.json", "model.safetensors", "processor_config.json",
    "tokenizer.json", "tokenizer_config.json",
]


def _info(repo_id, files=FILES, *, license="apache-2.0", downloads=0):
    info = MagicMock()
    info.id = repo_id
    info.siblings = [MagicMock(rfilename=name) for name in files]
    info.card_data = MagicMock(license=license)
    info.downloads = downloads
    return info


def test_required_keys_and_metadata():
    assert all(key in HF_PLUGIN for key in REQUIRED_HF_PLUGIN_KEYS)
    assert HF_PLUGIN["modality"] == "audio/alignment"
    assert HF_PLUGIN["priority"] == 90
    assert HF_PLUGIN["runtime_path"].endswith(":Qwen3ForcedAlignerRuntime")
    assert "ffmpeg" in HF_PLUGIN["system_packages"]


def test_plugin_precedes_broad_audio_plugins():
    modalities = [
        plugin["modality"]
        for plugin in discover_hf_plugins(_default_hf_plugin_dirs())
    ]
    assert modalities.index("audio/alignment") < modalities.index(
        "audio/classification"
    )


def test_sniff_requires_exact_repo_and_checkpoint_shape():
    assert HF_PLUGIN["sniff"](
        _info("Qwen/Qwen3-ForcedAligner-0.6B-hf")
    ) is True
    assert HF_PLUGIN["sniff"](
        _info("someone/Qwen3-ForcedAligner-0.6B-hf")
    ) is False
    assert HF_PLUGIN["sniff"](
        _info("Qwen/Qwen3-ForcedAligner-0.6B-hf", FILES[:-1])
    ) is False


def test_resolve_manifest_and_download_filter(tmp_path):
    info = _info("Qwen/Qwen3-ForcedAligner-0.6B-hf")
    resolved = HF_PLUGIN["resolve"](info.id, None, info)
    assert isinstance(resolved, ResolvedModel)
    assert resolved.manifest["modality"] == "audio/alignment"
    caps = resolved.manifest["capabilities"]
    assert caps["max_duration_seconds"] == 300
    assert caps["max_input_tokens"] == 8192
    assert caps["max_reference_words"] == 2048
    assert caps["word_timestamps"] is True
    assert len(caps["supported_languages"]) == 11
    assert any(
        extra.startswith("transformers>=5.13")
        for extra in resolved.manifest["pip_extras"]
    )
    with patch(
        "muse.modalities.audio_alignment.hf.snapshot_download",
        return_value=str(tmp_path),
    ) as download:
        assert resolved.download(tmp_path) == tmp_path
    patterns = download.call_args.kwargs["allow_patterns"]
    assert "model.safetensors" in patterns
    assert "*.bin" not in patterns


def test_resolve_rejects_unknown_shape():
    info = _info("org/random")
    with pytest.raises(ResolverError, match="unsupported"):
        HF_PLUGIN["resolve"]("org/random", None, info)


def test_search_yields_supported_model():
    api = MagicMock()
    api.list_models.return_value = [
        _info(
            "Qwen/Qwen3-ForcedAligner-0.6B-hf", downloads=123,
        ),
    ]
    rows = list(HF_PLUGIN["search"](
        api, "audiobook alignment", sort="downloads", limit=5,
    ))
    assert len(rows) == 1
    assert rows[0].modality == "audio/alignment"
    assert rows[0].size_gb == 1.84
    assert rows[0].downloads == 123
    assert api.list_models.call_args.kwargs["search"] == "Qwen3-ForcedAligner"
