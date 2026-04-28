"""Tests for the audio/generation HF resolver plugin."""
from types import SimpleNamespace
from unittest.mock import MagicMock

from muse.modalities.audio_generation.hf import HF_PLUGIN


def _fake_info(repo_id, tags=(), siblings=(), card=None):
    return SimpleNamespace(
        id=repo_id,
        tags=list(tags),
        siblings=[SimpleNamespace(rfilename=s) for s in siblings],
        card_data=card,
    )


def test_plugin_keys_present():
    keys = {"modality", "runtime_path", "pip_extras", "system_packages",
            "priority", "sniff", "resolve", "search"}
    assert keys.issubset(HF_PLUGIN.keys())


def test_plugin_modality_and_priority():
    assert HF_PLUGIN["modality"] == "audio/generation"
    # 105: more specific than embedding/text (110); less specific than
    # file-pattern plugins at 100. Triple-guarded sniff narrows the
    # match to Stable Audio Open shaped repos.
    assert HF_PLUGIN["priority"] == 105


def test_plugin_runtime_path_points_to_stable_audio_runtime():
    assert HF_PLUGIN["runtime_path"] == (
        "muse.modalities.audio_generation.runtimes.stable_audio"
        ":StableAudioRuntime"
    )


def test_plugin_pip_extras_declare_torch_diffusers_transformers():
    extras = " ".join(HF_PLUGIN["pip_extras"])
    assert "torch" in extras
    assert "diffusers" in extras
    assert "transformers" in extras
    assert "soundfile" in extras


def test_plugin_system_packages_includes_ffmpeg():
    """ffmpeg required for mp3/opus codec."""
    assert "ffmpeg" in HF_PLUGIN["system_packages"]


def test_sniff_true_for_stable_audio_open_repo():
    info = _fake_info(
        "stabilityai/stable-audio-open-1.0",
        tags=["text-to-audio", "diffusers"],
        siblings=["model_index.json", "scheduler/scheduler_config.json"],
    )
    assert HF_PLUGIN["sniff"](info) is True


def test_sniff_false_when_text_to_audio_tag_missing():
    """Missing the modality tag means this plugin should NOT match,
    even if the name says stable-audio."""
    info = _fake_info(
        "stabilityai/stable-audio-open-1.0",
        tags=["text-generation"],
        siblings=["model_index.json"],
    )
    assert HF_PLUGIN["sniff"](info) is False


def test_sniff_false_when_model_index_missing():
    """Missing model_index.json means a non-diffusers shape."""
    info = _fake_info(
        "stabilityai/stable-audio-open-1.0",
        tags=["text-to-audio"],
        siblings=["pytorch_model.bin", "config.json"],
    )
    assert HF_PLUGIN["sniff"](info) is False


def test_sniff_false_for_musicgen_repo():
    """MusicGen has the text-to-audio tag and may have model_index.json
    but is NOT stable-audio shaped (different runtime needed)."""
    info = _fake_info(
        "facebook/musicgen-small",
        tags=["text-to-audio"],
        siblings=["model_index.json"],
    )
    assert HF_PLUGIN["sniff"](info) is False


def test_sniff_false_for_audioldm_repo():
    """AudioLDM is a different audio-gen architecture; not stable-audio."""
    info = _fake_info(
        "cvssp/audioldm",
        tags=["text-to-audio"],
        siblings=["model_index.json"],
    )
    assert HF_PLUGIN["sniff"](info) is False


def test_sniff_false_for_unrelated_repo():
    info = _fake_info(
        "Qwen/Qwen3-8B-GGUF",
        tags=["chat", "text-generation"],
        siblings=["model.gguf"],
    )
    assert HF_PLUGIN["sniff"](info) is False


def test_sniff_case_insensitive_on_repo_name():
    """Stable-Audio with different case should still match."""
    info = _fake_info(
        "Org/Stable-Audio-Open-Custom",
        tags=["text-to-audio"],
        siblings=["model_index.json"],
    )
    assert HF_PLUGIN["sniff"](info) is True


def test_resolve_synthesizes_manifest():
    info = _fake_info(
        "stabilityai/stable-audio-open-1.0",
        tags=["text-to-audio"],
        siblings=["model_index.json"],
        card=SimpleNamespace(license="apache-2.0"),
    )
    resolved = HF_PLUGIN["resolve"](
        "stabilityai/stable-audio-open-1.0", None, info,
    )
    m = resolved.manifest
    assert m["model_id"] == "stable-audio-open-1.0"
    assert m["modality"] == "audio/generation"
    assert m["hf_repo"] == "stabilityai/stable-audio-open-1.0"
    assert m["license"] == "apache-2.0"


def test_resolve_capabilities_have_music_and_sfx_flags():
    info = _fake_info(
        "stabilityai/stable-audio-open-1.0",
        tags=["text-to-audio"],
        siblings=["model_index.json"],
    )
    resolved = HF_PLUGIN["resolve"](
        "stabilityai/stable-audio-open-1.0", None, info,
    )
    caps = resolved.manifest["capabilities"]
    assert caps["supports_music"] is True
    assert caps["supports_sfx"] is True
    assert caps["default_duration"] == 10.0
    assert caps["max_duration"] == 47.0
    assert caps["default_sample_rate"] == 44100
    assert caps["default_steps"] == 50
    assert caps["default_guidance"] == 7.0


def test_resolve_backend_path_points_to_stable_audio_runtime():
    info = _fake_info(
        "stabilityai/stable-audio-open-1.0",
        tags=["text-to-audio"],
        siblings=["model_index.json"],
    )
    resolved = HF_PLUGIN["resolve"](
        "stabilityai/stable-audio-open-1.0", None, info,
    )
    assert resolved.backend_path == HF_PLUGIN["runtime_path"]


def test_resolve_pip_extras_declared():
    info = _fake_info(
        "stabilityai/stable-audio-open-1.0",
        tags=["text-to-audio"],
        siblings=["model_index.json"],
    )
    resolved = HF_PLUGIN["resolve"](
        "stabilityai/stable-audio-open-1.0", None, info,
    )
    extras = " ".join(resolved.manifest["pip_extras"])
    assert "torch" in extras
    assert "diffusers" in extras


def test_resolve_system_packages_includes_ffmpeg():
    info = _fake_info(
        "stabilityai/stable-audio-open-1.0",
        tags=["text-to-audio"],
        siblings=["model_index.json"],
    )
    resolved = HF_PLUGIN["resolve"](
        "stabilityai/stable-audio-open-1.0", None, info,
    )
    assert "ffmpeg" in resolved.manifest["system_packages"]


def test_search_yields_stable_audio_results_only():
    api = MagicMock()
    repo1 = SimpleNamespace(
        id="stabilityai/stable-audio-open-1.0", downloads=10000,
    )
    repo2 = SimpleNamespace(
        id="facebook/musicgen-small", downloads=5000,
    )
    api.list_models.return_value = iter([repo1, repo2])
    out = list(HF_PLUGIN["search"](api, "audio", sort="downloads", limit=10))
    # MusicGen filtered out by the post-filter (not stable-audio in name).
    assert len(out) == 1
    assert out[0].uri == "hf://stabilityai/stable-audio-open-1.0"
    assert out[0].model_id == "stable-audio-open-1.0"
    assert out[0].modality == "audio/generation"


def test_search_filters_by_text_to_audio_tag():
    api = MagicMock()
    api.list_models.return_value = iter([])
    list(HF_PLUGIN["search"](api, "audio", sort="downloads", limit=10))
    _, kwargs = api.list_models.call_args
    assert kwargs["filter"] == "text-to-audio"


def test_resolve_manifest_includes_description_with_repo():
    info = _fake_info(
        "stabilityai/stable-audio-open-1.0",
        tags=["text-to-audio"],
        siblings=["model_index.json"],
    )
    resolved = HF_PLUGIN["resolve"](
        "stabilityai/stable-audio-open-1.0", None, info,
    )
    assert "stabilityai/stable-audio-open-1.0" in resolved.manifest["description"]


def test_resolve_no_card_data_yields_none_license():
    info = _fake_info(
        "stabilityai/stable-audio-open-1.0",
        tags=["text-to-audio"],
        siblings=["model_index.json"],
        card=None,
    )
    resolved = HF_PLUGIN["resolve"](
        "stabilityai/stable-audio-open-1.0", None, info,
    )
    assert resolved.manifest["license"] is None
