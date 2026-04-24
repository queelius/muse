"""Protocol + dataclass shape tests for audio/transcription."""
from muse.modalities.audio_transcription import (
    MODALITY,
    Word,
    Segment,
    TranscriptionResult,
    TranscriptionModel,
)


def test_modality_tag_is_mime_shaped():
    assert MODALITY == "audio/transcription"


def test_word_dataclass_roundtrip():
    w = Word(word="hello", start=0.0, end=0.5)
    assert w.word == "hello"
    assert w.end == 0.5


def test_segment_with_and_without_words():
    s1 = Segment(id=0, start=0.0, end=1.0, text="hi", words=None)
    assert s1.words is None
    s2 = Segment(
        id=1, start=1.0, end=2.0, text="world",
        words=[Word("world", 1.0, 2.0)],
    )
    assert len(s2.words) == 1
    assert s2.words[0].word == "world"


def test_transcription_result_minimal():
    r = TranscriptionResult(
        text="hi world", language="en", duration=2.0,
        segments=[
            Segment(id=0, start=0.0, end=1.0, text="hi", words=None),
            Segment(id=1, start=1.0, end=2.0, text="world", words=None),
        ],
        task="transcribe",
    )
    assert r.text == "hi world"
    assert r.language == "en"
    assert r.task == "transcribe"
    assert len(r.segments) == 2


def test_transcription_model_protocol_accepts_structural_impl():
    """A class that implements `transcribe(...)` satisfies the protocol
    without inheriting from TranscriptionModel."""
    class Fake:
        def transcribe(self, audio_path, **kwargs):
            return TranscriptionResult(
                text="", language="en", duration=0.0,
                segments=[], task="transcribe",
            )
    assert isinstance(Fake(), TranscriptionModel)


def test_transcription_model_protocol_rejects_missing_method():
    """A class without `transcribe(...)` fails the isinstance check."""
    class Missing:
        pass
    assert not isinstance(Missing(), TranscriptionModel)
