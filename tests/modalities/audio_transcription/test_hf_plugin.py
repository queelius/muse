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


QWEN_FILES = ['config.json', 'model.safetensors', 'processor_config.json',
              'tokenizer.json', 'tokenizer_config.json', 'generation_config.json',
              'chat_template.jinja']
QWEN_REVISION = 'bcd2b5b7f32b480ab5790554cfa8347f246a14f3'


def _qwen_info(repo='Qwen/Qwen3-ASR-1.7B-hf', files=None):
    from types import SimpleNamespace
    return SimpleNamespace(id=repo, sha=QWEN_REVISION,
                           siblings=[SimpleNamespace(rfilename=f) for f in (QWEN_FILES if files is None else files)],
                           tags=['automatic-speech-recognition'], card_data={'license': 'apache-2.0'})


def test_qwen_exact_repo_and_required_files():
    for repo in ['Qwen/Qwen3-ASR-1.7B-hf', 'qwen/QWEN3-asr-1.7b-HF']:
        assert HF_PLUGIN['sniff'](_qwen_info(repo))
    for repo in ['Qwen/Qwen3-ASR-0.6B-hf', 'other/Qwen3-ASR-1.7B-hf', 'Qwen/Qwen3-ASR-1.7B']:
        assert not HF_PLUGIN['sniff'](_qwen_info(repo))
    for missing in QWEN_FILES:
        assert not HF_PLUGIN['sniff'](_qwen_info(files=[f for f in QWEN_FILES if f != missing]))


def test_qwen_manifest_and_pinned_download(monkeypatch, tmp_path):
    from muse.modalities.audio_transcription import hf
    resolved = HF_PLUGIN['resolve']('Qwen/Qwen3-ASR-1.7B-hf', None, _qwen_info())
    assert resolved.backend_path == 'muse.modalities.audio_transcription.runtimes.qwen3_asr:Qwen3ASRRuntime'
    assert resolved.manifest['revision'] == QWEN_REVISION
    assert resolved.manifest['capabilities'] == {
        'device': 'auto', 'memory_gb': 5.0, 'supports_word_timestamps': False,
        'supports_segment_timestamps': False, 'supports_translation': False,
    }
    assert resolved.manifest['pip_extras'] == ['torch>=2.11', 'torchcodec>=0.12', 'transformers>=5.13.0']
    calls = []
    def download(**kwargs):
        calls.append(kwargs)
        return str(tmp_path)
    monkeypatch.setattr(hf, 'snapshot_download', download)
    assert resolved.download(tmp_path) == tmp_path
    assert calls[0]['revision'] == QWEN_REVISION
    assert set(calls[0]['allow_patterns']) == set(QWEN_FILES + ['README.md'])
