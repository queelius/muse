"""Tests for the audio_transcription HF plugin (faster-whisper)."""
from unittest.mock import MagicMock

from muse.modalities.audio_transcription.hf import HF_PLUGIN
from muse.core.discovery import REQUIRED_HF_PLUGIN_KEYS
from muse.core.resolvers import ResolvedModel


def _fake_info(siblings=None, tags=None):
    info = MagicMock()
    info.siblings = [MagicMock(rfilename=f) for f in (siblings or [])]
    info.tags = tags or []
    info.card_data = MagicMock(license=None)
    return info


def test_plugin_has_all_required_keys():
    for key in REQUIRED_HF_PLUGIN_KEYS:
        assert key in HF_PLUGIN


def test_plugin_metadata_correct():
    assert HF_PLUGIN["modality"] == "audio/transcription"
    assert HF_PLUGIN["runtime_path"].endswith(":FasterWhisperModel")
    assert HF_PLUGIN["priority"] == 100
    assert "ffmpeg" in HF_PLUGIN["system_packages"]


def test_sniff_true_on_ct2_shape_with_asr_tag():
    info = _fake_info(
        siblings=["model.bin", "config.json", "tokenizer.json"],
        tags=["automatic-speech-recognition"],
    )
    assert HF_PLUGIN["sniff"](info) is True


def test_sniff_false_without_asr_tag():
    info = _fake_info(
        siblings=["model.bin", "config.json", "tokenizer.json"],
        tags=["text-generation"],
    )
    assert HF_PLUGIN["sniff"](info) is False


def test_sniff_false_with_asr_tag_but_wrong_shape():
    info = _fake_info(
        siblings=["model.safetensors", "config.json"],
        tags=["automatic-speech-recognition"],
    )
    assert HF_PLUGIN["sniff"](info) is False


def test_resolve_returns_resolved_model():
    info = _fake_info(
        siblings=["model.bin", "config.json", "tokenizer.json"],
        tags=["automatic-speech-recognition"],
    )
    result = HF_PLUGIN["resolve"]("Systran/faster-whisper-tiny", None, info)
    assert isinstance(result, ResolvedModel)
    assert result.manifest["modality"] == "audio/transcription"
    assert result.manifest["model_id"] == "faster-whisper-tiny"


def test_search_yields_results():
    fake_api = MagicMock()
    fake_repo = MagicMock(id="Systran/faster-whisper-base", downloads=200)
    fake_api.list_models.return_value = [fake_repo]
    rows = list(HF_PLUGIN["search"](fake_api, "whisper", sort="downloads", limit=20))
    assert len(rows) == 1
    assert rows[0].modality == "audio/transcription"
