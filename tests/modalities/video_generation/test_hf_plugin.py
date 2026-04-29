"""Tests for the video_generation HF plugin (text-to-video repos)."""
from __future__ import annotations

from unittest.mock import MagicMock

from muse.core.discovery import REQUIRED_HF_PLUGIN_KEYS
from muse.core.resolvers import ResolvedModel
from muse.modalities.video_generation.hf import HF_PLUGIN


def _fake_info(repo_id="org/repo", siblings=None, tags=None, license=None):
    info = MagicMock()
    info.id = repo_id
    info.siblings = [MagicMock(rfilename=f) for f in (siblings or [])]
    info.tags = tags or []
    info.card_data = MagicMock(license=license)
    return info


def test_plugin_has_all_required_keys():
    for key in REQUIRED_HF_PLUGIN_KEYS:
        assert key in HF_PLUGIN


def test_plugin_metadata_correct():
    assert HF_PLUGIN["modality"] == "video/generation"
    assert HF_PLUGIN["runtime_path"].endswith(":WanRuntime")
    assert HF_PLUGIN["priority"] == 105


def test_sniff_true_on_wan_repo():
    info = _fake_info(
        repo_id="Wan-AI/Wan2.1-T2V-1.3B",
        tags=["text-to-video", "diffusers"],
    )
    assert HF_PLUGIN["sniff"](info) is True


def test_sniff_true_on_cogvideox_repo():
    info = _fake_info(
        repo_id="THUDM/CogVideoX-2b",
        tags=["text-to-video", "diffusers"],
    )
    assert HF_PLUGIN["sniff"](info) is True


def test_sniff_true_on_ltx_video_repo():
    info = _fake_info(
        repo_id="Lightricks/LTX-Video",
        tags=["text-to-video"],
    )
    assert HF_PLUGIN["sniff"](info) is True


def test_sniff_true_on_mochi_repo():
    info = _fake_info(
        repo_id="genmo/mochi-1-preview",
        tags=["text-to-video"],
    )
    assert HF_PLUGIN["sniff"](info) is True


def test_sniff_true_on_hunyuan_repo():
    info = _fake_info(
        repo_id="tencent/HunyuanVideo",
        tags=["text-to-video"],
    )
    assert HF_PLUGIN["sniff"](info) is True


def test_sniff_false_without_text_to_video_tag():
    info = _fake_info(
        repo_id="Wan-AI/Wan2.1-T2V-1.3B",
        tags=["text-to-image"],
    )
    assert HF_PLUGIN["sniff"](info) is False


def test_sniff_false_for_unrelated_repo():
    info = _fake_info(
        repo_id="stabilityai/stable-diffusion-2-1",
        tags=["text-to-image"],
    )
    assert HF_PLUGIN["sniff"](info) is False


def test_sniff_false_for_text_to_video_without_known_arch():
    info = _fake_info(
        repo_id="someorg/random-video-model",
        tags=["text-to-video"],
    )
    assert HF_PLUGIN["sniff"](info) is False


def test_resolve_wan_dispatches_to_wan_runtime():
    info = _fake_info(
        repo_id="Wan-AI/Wan2.1-T2V-1.3B",
        tags=["text-to-video"],
        license="apache-2.0",
    )
    result = HF_PLUGIN["resolve"]("Wan-AI/Wan2.1-T2V-1.3B", None, info)
    assert isinstance(result, ResolvedModel)
    assert result.backend_path.endswith(":WanRuntime")
    caps = result.manifest["capabilities"]
    assert caps["device"] == "cuda"
    assert caps["default_duration_seconds"] == 5.0
    assert caps["default_fps"] == 5
    assert caps["memory_gb"] == 6.0
    assert result.manifest["modality"] == "video/generation"
    assert result.manifest["license"] == "apache-2.0"
    assert "diffusers>=0.32" in "\n".join(result.manifest["pip_extras"])


def test_resolve_cogvideox_dispatches_to_cogvideox_runtime():
    info = _fake_info(
        repo_id="THUDM/CogVideoX-2b",
        tags=["text-to-video"],
    )
    result = HF_PLUGIN["resolve"]("THUDM/CogVideoX-2b", None, info)
    assert result.backend_path.endswith(":CogVideoXRuntime")
    caps = result.manifest["capabilities"]
    assert caps["default_duration_seconds"] == 6.0
    assert caps["default_fps"] == 8
    assert caps["memory_gb"] == 9.0


def test_resolve_ltx_dispatches_to_wan_fallback():
    info = _fake_info(
        repo_id="Lightricks/LTX-Video",
        tags=["text-to-video"],
    )
    result = HF_PLUGIN["resolve"]("Lightricks/LTX-Video", None, info)
    # Fallback to WanRuntime; v1.next will add LTXVideoRuntime.
    assert result.backend_path.endswith(":WanRuntime")
    caps = result.manifest["capabilities"]
    assert caps["default_fps"] == 30
    assert tuple(caps["default_size"]) == (1216, 704)
    assert caps["memory_gb"] == 16.0


def test_resolve_mochi_uses_mochi_capabilities():
    info = _fake_info(
        repo_id="genmo/mochi-1-preview",
        tags=["text-to-video"],
    )
    result = HF_PLUGIN["resolve"]("genmo/mochi-1-preview", None, info)
    caps = result.manifest["capabilities"]
    assert caps["memory_gb"] == 24.0
    assert tuple(caps["default_size"]) == (848, 480)


def test_resolve_hunyuan_uses_hunyuan_capabilities():
    info = _fake_info(
        repo_id="tencent/HunyuanVideo",
        tags=["text-to-video"],
    )
    result = HF_PLUGIN["resolve"]("tencent/HunyuanVideo", None, info)
    caps = result.manifest["capabilities"]
    assert caps["memory_gb"] == 60.0
    assert tuple(caps["default_size"]) == (1280, 720)


def test_resolve_model_id_lowercased():
    info = _fake_info(
        repo_id="Wan-AI/Wan2.1-T2V-1.3B",
        tags=["text-to-video"],
    )
    result = HF_PLUGIN["resolve"]("Wan-AI/Wan2.1-T2V-1.3B", None, info)
    assert result.manifest["model_id"] == "wan2.1-t2v-1.3b"


def test_search_yields_results_with_modality_tag():
    fake_api = MagicMock()
    fake_repo = MagicMock(id="Wan-AI/Wan2.1-T2V-14B", downloads=1234)
    fake_api.list_models.return_value = [fake_repo]
    rows = list(
        HF_PLUGIN["search"](fake_api, "wan", sort="downloads", limit=20)
    )
    assert len(rows) == 1
    assert rows[0].modality == "video/generation"
    assert rows[0].uri == "hf://Wan-AI/Wan2.1-T2V-14B"
    fake_api.list_models.assert_called_once()
    call_kwargs = fake_api.list_models.call_args.kwargs
    assert call_kwargs["filter"] == "text-to-video"
