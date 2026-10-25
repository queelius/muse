"""Qwen ASR contract tests; model inference stays mocked."""
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from muse.modalities.audio_alignment.decoding import DecodedAudio
from muse.modalities.audio_transcription.protocol import Segment, TranscriptionModel


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    q = import_module('muse.modalities.audio_transcription.runtimes.qwen3_asr')
    torch = MagicMock()
    processor = MagicMock()
    inputs = {'input_ids': SimpleNamespace(shape=(1, 3))}
    batch = MagicMock()
    batch.to.return_value = inputs
    processor.apply_chat_template.return_value = batch
    processor.decode.return_value = [{'language': 'English', 'transcription': 'hello world'}]
    model = MagicMock()
    model.to.return_value = model
    monkeypatch.setattr(q, 'torch', torch)
    monkeypatch.setattr(q, 'AutoProcessor', SimpleNamespace(from_pretrained=lambda *a, **kw: processor))
    monkeypatch.setattr(q, 'AutoModelForMultimodalLM', SimpleNamespace(from_pretrained=lambda *a, **kw: model))
    waveform = MagicMock()
    monkeypatch.setattr(q, 'decode_audio', lambda *a, **kw: DecodedAudio(waveform, 16000, 2.5))
    path = tmp_path / 'audio.wav'
    path.write_bytes(b'audio')
    backend = q.Qwen3ASRRuntime(model_id='qwen', hf_repo='unused', device='cpu')
    return backend, processor, model, str(path), waveform


@pytest.mark.parametrize('prompt', [None, '', 'Vocabulary: Muse'])
def test_prompt_only_when_sent(runtime, prompt):
    backend, processor, _, path, _ = runtime
    backend.transcribe(path, prompt=prompt)
    conversation = processor.apply_chat_template.call_args.args[0][0]
    contexts = [m['content'][0]['text'] for m in conversation if m['role'] == 'system']
    assert contexts == ([prompt] if prompt else [])


def test_whole_file_result_and_language_prefill(runtime):
    backend, processor, model, path, waveform = runtime
    result = backend.transcribe(path, language='en')
    assert isinstance(backend, TranscriptionModel)
    assert result.text == 'hello world'
    assert result.duration == 2.5
    assert result.language == 'en'
    assert result.task == 'transcribe'
    assert result.segments == [Segment(0, 0.0, 2.5, 'hello world', None)]
    conversation = processor.apply_chat_template.call_args.args[0][0]
    assert conversation[-1]['content'][0]['text'] == 'language English<asr_text>'
    assert conversation[-2]['content'][0]['audio'] is waveform.squeeze.return_value.numpy.return_value
    # Only generated tokens, excluding the three-token prompt, reach decode.
    model.generate.return_value.__getitem__.assert_called_once_with((slice(None), slice(3, None)))


def test_runtime_refuses_oversize_before_decode(runtime, monkeypatch):
    backend, processor, _, path, _ = runtime
    monkeypatch.setenv('MUSE_ASR_MAX_MB', '0')
    with pytest.raises(ValueError, match='MUSE_ASR_MAX_MB'):
        backend.transcribe(path)
    processor.apply_chat_template.assert_not_called()
