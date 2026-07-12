from muse.modalities.audio_quality.protocol import (
    AudioQualityModel,
    AudioQualityResult,
    AudioQualityScore,
)


def test_result_construction():
    result = AudioQualityResult(
        scores={
            "naturalness": AudioQualityScore(
                value=4.2, minimum=1.0, maximum=5.0,
            ),
        },
        primary_score="naturalness",
    )
    assert result.scores["naturalness"].value == 4.2
    assert result.primary_score == "naturalness"


def test_protocol_accepts_duck_type():
    class _Fake:
        model_id = "fake"

        def assess(self, audio_path):
            return AudioQualityResult(
                scores={"quality": AudioQualityScore(1.0)},
                primary_score="quality",
            )

    assert isinstance(_Fake(), AudioQualityModel)


def test_protocol_rejects_missing_assess():
    class _NoAssess:
        model_id = "fake"

    assert not isinstance(_NoAssess(), AudioQualityModel)
