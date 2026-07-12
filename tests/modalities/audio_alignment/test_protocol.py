from muse.modalities.audio_alignment.protocol import (
    AlignmentWord,
    AudioAlignmentModel,
    AudioAlignmentResult,
)


def test_result_construction():
    result = AudioAlignmentResult(
        text="Hello",
        language="English",
        duration_seconds=1.0,
        words=[AlignmentWord("Hello", 0.1, 0.6, 0.9)],
    )
    assert result.words[0].word == "Hello"
    assert result.words[0].confidence == 0.9


def test_protocol_accepts_duck_type():
    class _Fake:
        model_id = "fake"

        def align(
            self, audio_path, transcript, *, language=None,
            max_duration_seconds=None,
        ):
            return AudioAlignmentResult(transcript, language, 1.0, [])

    assert isinstance(_Fake(), AudioAlignmentModel)


def test_protocol_rejects_missing_align():
    class _NoAlign:
        model_id = "fake"

    assert not isinstance(_NoAlign(), AudioAlignmentModel)
