"""FasterWhisperModel runtime: mocked-dep tests."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _fake_info(language="en", duration=1.0):
    return SimpleNamespace(language=language, duration=duration)


def _fake_segment(id, start, end, text, words=None):
    return SimpleNamespace(id=id, start=start, end=end, text=text, words=words or [])


@pytest.fixture(autouse=True)
def _reset_module_sentinels():
    """Reset the module-top torch/WhisperModel sentinels between tests."""
    import muse.modalities.audio_transcription.runtimes.faster_whisper as mod
    orig = (mod.torch, mod.WhisperModel)
    yield
    mod.torch, mod.WhisperModel = orig


def test_transcribe_assembles_result_from_segments():
    import muse.modalities.audio_transcription.runtimes.faster_whisper as mod
    mod.torch = MagicMock(cuda=MagicMock(is_available=MagicMock(return_value=False)))
    mod.torch.backends = MagicMock(mps=None)
    fake_whisper = MagicMock()
    fake_whisper.transcribe.return_value = (
        iter([
            _fake_segment(0, 0.0, 1.0, "hello"),
            _fake_segment(1, 1.0, 2.0, "world"),
        ]),
        _fake_info("en", 2.0),
    )
    mod.WhisperModel = MagicMock(return_value=fake_whisper)

    m = mod.FasterWhisperModel(
        model_id="whisper-tiny", hf_repo="Systran/faster-whisper-tiny",
        local_dir="/fake/weights", device="cpu",
    )
    r = m.transcribe("/fake/audio.wav", task="transcribe")
    assert r.text == "hello world"
    assert r.language == "en"
    assert r.duration == 2.0
    assert r.task == "transcribe"
    assert len(r.segments) == 2
    assert r.segments[0].words is None


def test_word_timestamps_populates_segment_words():
    import muse.modalities.audio_transcription.runtimes.faster_whisper as mod
    mod.torch = MagicMock(cuda=MagicMock(is_available=MagicMock(return_value=False)))
    mod.torch.backends = MagicMock(mps=None)
    fake_whisper = MagicMock()
    fake_word = SimpleNamespace(word="hello", start=0.0, end=0.5)
    fake_whisper.transcribe.return_value = (
        iter([_fake_segment(0, 0.0, 1.0, "hello", words=[fake_word])]),
        _fake_info(),
    )
    mod.WhisperModel = MagicMock(return_value=fake_whisper)

    m = mod.FasterWhisperModel(
        model_id="whisper-tiny", hf_repo="x", local_dir="/fake", device="cpu",
    )
    r = m.transcribe("/fake/a.wav", word_timestamps=True)
    assert r.segments[0].words is not None
    assert r.segments[0].words[0].word == "hello"
    _, kw = fake_whisper.transcribe.call_args
    assert kw["word_timestamps"] is True


def test_task_translate_is_forwarded():
    import muse.modalities.audio_transcription.runtimes.faster_whisper as mod
    mod.torch = MagicMock(cuda=MagicMock(is_available=MagicMock(return_value=False)))
    mod.torch.backends = MagicMock(mps=None)
    fake_whisper = MagicMock()
    fake_whisper.transcribe.return_value = (iter([]), _fake_info("en", 0.0))
    mod.WhisperModel = MagicMock(return_value=fake_whisper)

    m = mod.FasterWhisperModel(
        model_id="whisper-tiny", hf_repo="x", local_dir="/fake", device="cpu",
    )
    m.transcribe("/fake/a.wav", task="translate")

    _, kw = fake_whisper.transcribe.call_args
    assert kw["task"] == "translate"


def test_device_auto_selects_cuda_when_available():
    import muse.modalities.audio_transcription.runtimes.faster_whisper as mod
    mod.torch = MagicMock(cuda=MagicMock(is_available=MagicMock(return_value=True)))
    mod.torch.backends = MagicMock(mps=None)
    captured_kwargs = {}

    def constructor(path, **kw):
        captured_kwargs.update(kw)
        return MagicMock()

    mod.WhisperModel = MagicMock(side_effect=constructor)

    mod.FasterWhisperModel(
        model_id="whisper-tiny", hf_repo="x", local_dir="/fake", device="auto",
    )
    assert captured_kwargs["device"] == "cuda"
    assert captured_kwargs["compute_type"] == "float16"


def test_device_cpu_uses_int8_compute_type():
    import muse.modalities.audio_transcription.runtimes.faster_whisper as mod
    mod.torch = MagicMock(cuda=MagicMock(is_available=MagicMock(return_value=False)))
    mod.torch.backends = MagicMock(mps=None)
    captured_kwargs = {}

    def constructor(path, **kw):
        captured_kwargs.update(kw)
        return MagicMock()

    mod.WhisperModel = MagicMock(side_effect=constructor)

    mod.FasterWhisperModel(
        model_id="whisper-tiny", hf_repo="x", local_dir="/fake", device="cpu",
    )
    assert captured_kwargs["device"] == "cpu"
    assert captured_kwargs["compute_type"] == "int8"


def test_raises_when_faster_whisper_not_installed(monkeypatch):
    """If faster_whisper import fails, constructor raises a clear error."""
    import muse.modalities.audio_transcription.runtimes.faster_whisper as mod
    mod.torch = MagicMock(cuda=MagicMock(is_available=MagicMock(return_value=False)))
    mod.torch.backends = MagicMock(mps=None)
    mod.WhisperModel = None
    monkeypatch.setattr(mod, "_ensure_deps", lambda: None)
    with pytest.raises(RuntimeError, match="faster-whisper is not installed"):
        mod.FasterWhisperModel(
            model_id="whisper-tiny", hf_repo="x", local_dir="/fake", device="cpu",
        )
