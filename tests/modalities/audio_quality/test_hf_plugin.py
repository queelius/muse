from unittest.mock import MagicMock, patch

import pytest

from muse.core.discovery import (
    REQUIRED_HF_PLUGIN_KEYS,
    _default_hf_plugin_dirs,
    discover_hf_plugins,
)
from muse.core.resolvers import ResolvedModel, ResolverError
from muse.modalities.audio_quality.hf import HF_PLUGIN


def _info(repo_id, files, *, tags=(), license="mit", downloads=0):
    info = MagicMock()
    info.id = repo_id
    info.tags = list(tags)
    info.siblings = [MagicMock(rfilename=name) for name in files]
    info.card_data = MagicMock(license=license)
    info.downloads = downloads
    return info


def test_required_keys_and_metadata():
    assert all(key in HF_PLUGIN for key in REQUIRED_HF_PLUGIN_KEYS)
    assert HF_PLUGIN["modality"] == "audio/quality"
    assert HF_PLUGIN["priority"] == 100
    assert HF_PLUGIN["runtime_path"].endswith(":UTMOSRuntime")
    assert "ffmpeg" in HF_PLUGIN["system_packages"]


def test_plugin_precedes_broad_audio_classification_plugin():
    modalities = [
        plugin["modality"]
        for plugin in discover_hf_plugins(_default_hf_plugin_dirs())
    ]
    assert modalities.index("audio/quality") < modalities.index(
        "audio/classification"
    )


def test_sniff_utmos_shape():
    info = _info(
        "Blinorot/UTMOS-PyTorch",
        ["utmos_scripted.pt", "utmos_state_dict.pt"],
    )
    assert HF_PLUGIN["sniff"](info) is True


def test_sniff_audiobox_before_audio_classification_catchall():
    info = _info(
        "facebook/audiobox-aesthetics",
        ["checkpoint.pt", "config.json", "model.safetensors"],
        tags=["audio-classification"],
    )
    assert HF_PLUGIN["sniff"](info) is True


def test_sniff_requires_known_name_and_files():
    incomplete = _info("Blinorot/UTMOS-PyTorch", ["README.md"])
    unrelated = _info("org/random", ["checkpoint.pt", "config.json"])
    assert HF_PLUGIN["sniff"](incomplete) is False
    assert HF_PLUGIN["sniff"](unrelated) is False


def test_resolve_utmos_manifest():
    info = _info("Blinorot/UTMOS-PyTorch", ["utmos_scripted.pt"])
    resolved = HF_PLUGIN["resolve"](
        "Blinorot/UTMOS-PyTorch", None, info,
    )
    assert isinstance(resolved, ResolvedModel)
    assert resolved.manifest["modality"] == "audio/quality"
    assert resolved.manifest["capabilities"]["primary_score"] == "naturalness"
    assert resolved.manifest["capabilities"]["max_duration_seconds"] == 600
    assert resolved.backend_path.endswith(":UTMOSRuntime")
    extras = resolved.manifest["pip_extras"]
    assert any(extra.startswith("torch>=2.11") for extra in extras)
    assert any(extra.startswith("torchcodec") for extra in extras)
    assert "ffmpeg" in resolved.manifest["system_packages"]


def test_resolve_audiobox_manifest_and_download_filter(tmp_path):
    info = _info(
        "facebook/audiobox-aesthetics",
        ["checkpoint.pt", "config.json", "model.safetensors"],
        license="cc-by-4.0",
    )
    resolved = HF_PLUGIN["resolve"](
        "facebook/audiobox-aesthetics", None, info,
    )
    assert resolved.backend_path.endswith(":AudioboxAestheticsRuntime")
    assert resolved.manifest["capabilities"]["primary_score"] == "production_quality"
    assert resolved.manifest["license"] == "cc-by-4.0"
    with patch(
        "muse.modalities.audio_quality.hf.snapshot_download",
        return_value=str(tmp_path),
    ) as download:
        assert resolved.download(tmp_path) == tmp_path
    patterns = download.call_args.kwargs["allow_patterns"]
    assert "model.safetensors" in patterns
    assert "checkpoint.pt" not in patterns


def test_resolve_rejects_unknown_shape():
    info = _info("org/random", ["checkpoint.pt"])
    with pytest.raises(ResolverError, match="unsupported"):
        HF_PLUGIN["resolve"]("org/random", None, info)


def test_search_yields_only_supported_families():
    api = MagicMock()
    api.list_models.side_effect = [
        [
            MagicMock(id="mosmodels/utmos", downloads=999),
            MagicMock(id="Blinorot/UTMOS-PyTorch", downloads=12),
        ],
        [
            MagicMock(id="thunnai/audiobox-aesthetics", downloads=100),
            MagicMock(id="facebook/audiobox-aesthetics", downloads=34),
        ],
    ]
    infos = {
        "mosmodels/utmos": _info("mosmodels/utmos", ["wav2vec_small.pt"]),
        "Blinorot/UTMOS-PyTorch": _info(
            "Blinorot/UTMOS-PyTorch", ["utmos_scripted.pt"], downloads=12,
        ),
        "thunnai/audiobox-aesthetics": _info(
            "thunnai/audiobox-aesthetics", ["config.json"], downloads=100,
        ),
        "facebook/audiobox-aesthetics": _info(
            "facebook/audiobox-aesthetics",
            ["config.json", "model.safetensors"],
            downloads=34,
        ),
    }
    api.model_info.side_effect = lambda repo_id: infos[repo_id]
    rows = list(HF_PLUGIN["search"](
        api, "quality", sort="downloads", limit=20,
    ))
    assert [row.model_id for row in rows] == [
        "audiobox-aesthetics", "utmos-pytorch",
    ]
    assert all(row.modality == "audio/quality" for row in rows)
    searches = {
        call.kwargs["search"] for call in api.list_models.call_args_list
    }
    assert searches == {"utmos", "audiobox-aesthetics"}
    assert all(
        call.kwargs["limit"] is None
        for call in api.list_models.call_args_list
    )
    assert all(
        call.kwargs["full"] is True
        for call in api.list_models.call_args_list
    )


def test_search_uses_family_queries_and_honors_specific_query():
    api = MagicMock()
    api.list_models.side_effect = [
        [MagicMock(id="Blinorot/UTMOS-PyTorch", downloads=12)],
        [MagicMock(id="facebook/audiobox-aesthetics", downloads=34)],
    ]
    api.model_info.side_effect = lambda repo_id: {
        "Blinorot/UTMOS-PyTorch": _info(
            "Blinorot/UTMOS-PyTorch", ["utmos_scripted.pt"], downloads=12,
        ),
        "facebook/audiobox-aesthetics": _info(
            "facebook/audiobox-aesthetics",
            ["config.json", "model.safetensors"],
            downloads=34,
        ),
    }[repo_id]
    rows = list(HF_PLUGIN["search"](
        api, "utmos", sort="downloads", limit=20,
    ))
    assert [row.model_id for row in rows] == ["utmos-pytorch"]
    searches = {
        call.kwargs["search"] for call in api.list_models.call_args_list
    }
    assert searches == {"utmos", "audiobox-aesthetics"}
