"""Tests for muse.models.kokoro_82m: Kokoro TTS adapter."""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from muse.modalities.audio_speech.protocol import AudioChunk, AudioResult, TTSModel


class TestKokoroModel:
    def _make_adapter(self):
        from muse.models.kokoro_82m import Model as KokoroModel

        mock_pipeline = MagicMock()
        result = MagicMock()
        result.audio = torch.randn(24000)
        mock_pipeline.return_value = [result]

        adapter = object.__new__(KokoroModel)
        adapter._pipeline = mock_pipeline
        adapter._device = "cpu"
        return adapter

    def test_protocol_conformance(self):
        assert isinstance(self._make_adapter(), TTSModel)

    def test_model_id(self):
        assert self._make_adapter().model_id == "kokoro-82m"

    def test_sample_rate(self):
        assert self._make_adapter().sample_rate == 24000

    def test_synthesize_returns_audio_result(self):
        result = self._make_adapter().synthesize("Hello")
        assert isinstance(result, AudioResult)
        assert result.sample_rate == 24000
        assert len(result.audio) == 24000

    def test_synthesize_passes_voice(self):
        adapter = self._make_adapter()
        adapter.synthesize("Hello", voice="am_adam", speed=1.2)
        adapter._pipeline.assert_called_once_with("Hello", voice="am_adam", speed=1.2)

    def test_stream_yields_chunks(self):
        adapter = self._make_adapter()
        chunks = list(adapter.synthesize_stream("Hello"))
        assert len(chunks) == 1
        assert isinstance(chunks[0], AudioChunk)

    def test_voices_list(self):
        from muse.models.kokoro_82m import KOKORO_VOICES
        assert "af_heart" in KOKORO_VOICES
        assert "am_adam" in KOKORO_VOICES
        assert len(KOKORO_VOICES) > 50


def test_kokoro_has_lowercase_voices_property():
    """routes.py + registry look for `voices` (lowercase); KokoroModel must satisfy."""
    from muse.models.kokoro_82m import Model as KokoroModel

    assert "voices" in dir(KokoroModel), "KokoroModel must expose a `voices` attribute/property"

    # Verify via an instance (bypassing __init__) that it returns the VOICES list
    adapter = object.__new__(KokoroModel)
    assert hasattr(adapter, "voices")
    assert isinstance(adapter.voices, list)
    assert len(adapter.voices) > 0
    assert adapter.voices is KokoroModel.VOICES


def test_kokoro_init_does_not_pass_local_dir_as_repo_id(monkeypatch):
    """Regression: KPipeline.repo_id must be an HF-style 'namespace/name',
    never a filesystem path. Catalog.load_backend passes both hf_repo and
    local_dir; KokoroModel must forward hf_repo only.

    Symptom of the bug: 'Repo id must be in the form ... or namespace/name'
    when the HF cache snapshot directory was passed as repo_id.
    """
    from unittest import mock

    # Patch the symbols KokoroModel imports inside __init__
    import sys
    fake_kokoro = mock.MagicMock()
    fake_kpipeline = mock.MagicMock()
    fake_kokoro.KPipeline = fake_kpipeline
    monkeypatch.setitem(sys.modules, "kokoro", fake_kokoro)

    # torch is already importable in the test env but we only need a device branch
    from muse.models.kokoro_82m import Model as KokoroModel
    _ = KokoroModel(
        hf_repo="hexgrad/Kokoro-82M",
        local_dir="/home/user/.cache/huggingface/hub/models--hexgrad--Kokoro-82M/snapshots/abc",
        device="cpu",
    )
    # KPipeline was called with repo_id=hf_repo, NOT with local_dir
    call_kwargs = fake_kpipeline.call_args.kwargs
    assert call_kwargs["repo_id"] == "hexgrad/Kokoro-82M"
    assert "/home/user" not in call_kwargs.get("repo_id", "")


def test_manifest_has_required_fields():
    from muse.models.kokoro_82m import MANIFEST
    assert MANIFEST["model_id"] == "kokoro-82m"
    assert MANIFEST["modality"] == "audio/speech"
    assert "hf_repo" in MANIFEST
    assert "pip_extras" in MANIFEST
