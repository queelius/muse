"""Tests for muse.models.kokoro_82m: Kokoro TTS adapter."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from muse.modalities.audio_speech.protocol import AudioChunk, AudioResult, TTSModel
from muse.modalities.audio_speech.word_timing import TimedSpan


def _write_local_bundle(root):
    from muse.models.kokoro_82m import KOKORO_VOICES

    root.mkdir(parents=True)
    (root / "config.json").write_text("{}", encoding="utf-8")
    (root / "kokoro-v1_0.pth").write_bytes(b"weights")
    voices_dir = root / "voices"
    voices_dir.mkdir()
    for voice in KOKORO_VOICES:
        (voices_dir / f"{voice}.pt").write_bytes(b"voice")
    return root


def _install_fake_kokoro(monkeypatch):
    fake_kokoro = MagicMock()
    fake_kokoro.KModel = MagicMock(name="KModel")
    fake_kokoro.KPipeline = MagicMock(name="KPipeline")
    monkeypatch.setitem(sys.modules, "kokoro", fake_kokoro)
    return fake_kokoro


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
        adapter._local_voice_paths = None
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

    def test_synthesize_defaults_voice_when_none(self):
        # The /v1/audio/speech route declares `voice: str | None = None` and
        # always forwards `voice=req.voice`, so an omitted voice arrives as an
        # explicit None. The default must still apply (kwargs.get default only
        # fires when the key is absent, not when it is present-but-None).
        adapter = self._make_adapter()
        adapter.synthesize("Hello", voice=None)
        adapter._pipeline.assert_called_once_with("Hello", voice="af_heart", speed=1.0)

    def test_stream_defaults_voice_when_none(self):
        adapter = self._make_adapter()
        list(adapter.synthesize_stream("Hello", voice=None))
        adapter._pipeline.assert_called_once_with("Hello", voice="af_heart", speed=1.0)

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


def test_kokoro_local_snapshot_never_uses_repo_fallback(monkeypatch, tmp_path):
    """A pulled model must construct both Kokoro objects from absolute files."""
    local_dir = _write_local_bundle(tmp_path / "snapshot")
    voice_blob = tmp_path / "voice-blob-without-extension"
    voice_blob.write_bytes(b"voice")
    local_voice_path = local_dir / "voices" / "am_adam.pt"
    local_voice_path.unlink()
    local_voice_path.symlink_to(voice_blob)
    fake_kokoro = _install_fake_kokoro(monkeypatch)
    local_model = fake_kokoro.KModel.return_value

    from muse.models.kokoro_82m import Model as KokoroModel

    adapter = KokoroModel(
        hf_repo="hexgrad/Kokoro-82M",
        local_dir=str(local_dir),
        device="cpu",
    )

    fake_kokoro.KModel.assert_called_once_with(
        config=str((local_dir / "config.json").absolute()),
        model=str((local_dir / "kokoro-v1_0.pth").absolute()),
    )
    model_kwargs = fake_kokoro.KModel.call_args.kwargs
    pipeline_kwargs = fake_kokoro.KPipeline.call_args.kwargs
    assert "repo_id" not in model_kwargs
    assert "repo_id" not in pipeline_kwargs
    assert pipeline_kwargs == {
        "lang_code": "a",
        "model": local_model,
        "device": "cpu",
    }

    result = MagicMock()
    result.audio = torch.ones(8)
    adapter._pipeline.return_value = [result]
    audio = adapter.synthesize("Hello", voice="am_adam")
    adapter._pipeline.assert_called_once_with(
        "Hello",
        voice=str(local_voice_path.absolute()),
        speed=1.0,
    )
    assert adapter._pipeline.call_args.kwargs["voice"].endswith(".pt")
    assert audio.metadata["voice"] == "am_adam"


def test_kokoro_local_stream_uses_absolute_voice_path(monkeypatch, tmp_path):
    local_dir = _write_local_bundle(tmp_path / "snapshot")
    fake_kokoro = _install_fake_kokoro(monkeypatch)

    from muse.models.kokoro_82m import Model as KokoroModel

    adapter = KokoroModel(local_dir=str(local_dir), device="cpu")
    result = MagicMock()
    result.audio = torch.ones(8)
    adapter._pipeline.return_value = [result]

    list(adapter.synthesize_stream("Hello", voice="bf_emma"))

    adapter._pipeline.assert_called_once_with(
        "Hello",
        voice=str((local_dir / "voices" / "bf_emma.pt").absolute()),
        speed=1.0,
    )
    assert fake_kokoro.KModel.called


def test_kokoro_missing_local_artifact_fails_before_construction(monkeypatch, tmp_path):
    local_dir = _write_local_bundle(tmp_path / "snapshot")
    missing_voice = local_dir / "voices" / "af_heart.pt"
    missing_voice.unlink()
    fake_kokoro = _install_fake_kokoro(monkeypatch)

    from muse.models.kokoro_82m import Model as KokoroModel

    with pytest.raises(FileNotFoundError) as exc_info:
        KokoroModel(local_dir=str(local_dir), device="cpu")

    message = str(exc_info.value)
    assert str(missing_voice) in message
    assert "muse pull kokoro-82m" in message
    fake_kokoro.KModel.assert_not_called()
    fake_kokoro.KPipeline.assert_not_called()


def test_kokoro_unknown_local_voice_fails_without_pipeline_call(monkeypatch, tmp_path):
    local_dir = _write_local_bundle(tmp_path / "snapshot")
    _install_fake_kokoro(monkeypatch)

    from muse.models.kokoro_82m import Model as KokoroModel

    adapter = KokoroModel(local_dir=str(local_dir), device="cpu")
    with pytest.raises(ValueError, match="Unknown bundled Kokoro voice"):
        adapter.synthesize("Hello", voice="download_me")
    adapter._pipeline.assert_not_called()


def test_kokoro_without_local_snapshot_can_use_repo_id(monkeypatch):
    fake_kokoro = _install_fake_kokoro(monkeypatch)

    from muse.models.kokoro_82m import Model as KokoroModel

    KokoroModel(hf_repo="org/repo", device="cpu")

    fake_kokoro.KModel.assert_not_called()
    fake_kokoro.KPipeline.assert_called_once_with(
        lang_code="a",
        repo_id="org/repo",
        device="cpu",
    )


def test_manifest_has_required_fields():
    from muse.models.kokoro_82m import MANIFEST
    assert MANIFEST["model_id"] == "kokoro-82m"
    assert MANIFEST["modality"] == "audio/speech"
    assert "hf_repo" in MANIFEST
    assert "pip_extras" in MANIFEST
    assert MANIFEST["revision"] == "f3ff3571791e39611d31c381e3a41a3af07b4987"
    assert "kokoro==0.9.4" in MANIFEST["pip_extras"]
    assert all(not requirement.startswith("misaki") for requirement in MANIFEST["pip_extras"])


def _token(text, start=None, end=None, whitespace=""):
    return SimpleNamespace(text=text, whitespace=whitespace, start_ts=start, end_ts=end)


def _chunk(graphemes, tokens, samples=24000):
    """Match KPipeline.Result fields without loading Kokoro or weights."""
    return SimpleNamespace(
        graphemes=graphemes, tokens=tokens,
        audio=SimpleNamespace(numpy=lambda: np.ones(samples, dtype=np.float32)),
    )


def _timing_adapter(monkeypatch, chunks, lang_code="a"):
    from muse.models.kokoro_82m import Model

    fake_kokoro = _install_fake_kokoro(monkeypatch)
    pipeline = fake_kokoro.KPipeline.return_value
    pipeline.lang_code = {"en-us": "a", "en-gb": "b"}.get(lang_code.lower(), lang_code.lower())
    pipeline.return_value = chunks
    return Model(device="cpu", lang_code=lang_code)


@pytest.mark.parametrize("kwargs", [{}, {"timestamps": False}, {"timestamps": None}])
def test_timestamps_off_returns_no_spans(monkeypatch, kwargs):
    # Legacy results need only audio, not graphemes or tokens.
    adapter = _timing_adapter(monkeypatch, [SimpleNamespace(audio=_chunk("", []).audio)])
    result = adapter.synthesize("Hello", **kwargs)
    assert result.metadata == {"voice": "af_heart"}
    np.testing.assert_array_equal(result.audio, np.ones(24000, dtype=np.float32))


def test_token_spans_locate_tokens_in_text(monkeypatch):
    adapter = _timing_adapter(monkeypatch, [_chunk("Hello, world.", [
        _token("Hello", .1, .4), _token(",", whitespace=" "),
        _token("world", .5, .9), _token(".", .9, .9),
    ])])
    result = adapter.synthesize("Hello, world.", timestamps=True)
    off = 0.0
    assert result.metadata == {
        "voice": "af_heart", "timing_calibration": None,
        "token_spans": [TimedSpan(0, 5, .1 + off, .4 + off),
                        TimedSpan(7, 12, .5 + off, .9 + off)],
    }


@pytest.mark.parametrize("text", ["One.\n\nTwo.", "One. Two."])
def test_multi_chunk_offsets_accumulate(monkeypatch, text):
    adapter = _timing_adapter(monkeypatch, [
        _chunk("One.", [_token("One", .1, .4)]),
        _chunk("Two.", [_token("Two", .1, .4)], samples=12000),
    ])
    result = adapter.synthesize(text, timestamps=True)
    second = text.index("Two")
    assert result.metadata["token_spans"] == [
        TimedSpan(0, 3, .1, .4), TimedSpan(second, second + 3, 1.1, 1.4),
    ]
    assert len(result.audio) == 36000
    assert all(0 <= span.start <= span.end <= len(result.audio) / 24000
               for span in result.metadata["token_spans"])


def test_link_token_maps_inside_link(monkeypatch):
    text = "[Coulomb](/kuːˈlɒm/) law"
    adapter = _timing_adapter(monkeypatch, [_chunk("Coulomb law", [
        _token("Coulomb", .1, .4), _token("law", .5, .9),
    ])])
    result = adapter.synthesize(text, timestamps=True)
    assert result.metadata["token_spans"] == [
        TimedSpan(1, 8, .1, .4), TimedSpan(21, 24, .5, .9),
    ]


def test_unmappable_token_is_dropped(monkeypatch):
    adapter = _timing_adapter(monkeypatch, [_chunk("One One", [
        _token("missing", .0, .1), _token("One", .1, .4),
        _token("missing", .4, .5), _token("One", .5, .9),
    ])])
    result = adapter.synthesize("One One", timestamps=True)
    assert result.metadata["token_spans"] == [
        TimedSpan(0, 3, .1, .4), TimedSpan(4, 7, .5, .9),
    ]


def test_untimed_token_advances_character_cursor(monkeypatch):
    adapter = _timing_adapter(monkeypatch, [_chunk("One One", [
        _token("One"), _token("One", .5, .9),
    ])])
    assert adapter.synthesize("One One", timestamps=True).metadata["token_spans"] == [
        TimedSpan(4, 7, .5, .9),
    ]


@pytest.mark.parametrize("lang_code", ["a", "b", "en-us", "en-gb", "EN-US", "EN-GB", "A", "B"])
def test_english_timestamps_preserve_speed(monkeypatch, lang_code):
    adapter = _timing_adapter(monkeypatch, [_chunk("One", [_token("One", .1, .3)])], lang_code)
    result = adapter.synthesize("One", timestamps=True, speed=1.5)
    assert result.metadata["token_spans"] == [TimedSpan(0, 3, .1, .3)]
    adapter._pipeline.assert_called_once_with("One", voice="af_heart", speed=1.5)


@pytest.mark.parametrize("lang_code", ["e", "f", "h", "i", "j", "p", "z"])
def test_non_english_timestamps_rejected(monkeypatch, lang_code):
    adapter = _timing_adapter(monkeypatch, [_chunk("Hello", [])], lang_code)
    from muse.modalities.audio_speech import TimestampsUnavailableError

    with pytest.raises(TimestampsUnavailableError, match="timestamps"):
        adapter.synthesize("Hello", timestamps=True)
    adapter._pipeline.assert_not_called()
    assert adapter.synthesize("Hello").metadata == {"voice": "af_heart"}


def test_render_ipa_and_caps(monkeypatch):
    from muse.models.kokoro_82m import MANIFEST, KOKORO_TIMING_OFFSET_SECONDS

    adapter = _timing_adapter(monkeypatch, [])
    assert adapter.render_ipa("Coulomb", "kuːˈlɒm") == "[Coulomb](/kuːˈlɒm/)"
    assert adapter.default_voice == "af_heart"
    assert KOKORO_TIMING_OFFSET_SECONDS == 0.0
    assert MANIFEST["capabilities"]["supports_word_timestamps"] is True
    assert MANIFEST["capabilities"]["pronunciation_kinds"] == ["ipa", "respell"]
    assert "Word timings come from the model's duration predictor (checked against the forced aligner, see docs/benchmarks/2026-10-24-kokoro-timing-agreement.md)" in MANIFEST["limits"]
    assert not any("No word timings" in limit for limit in MANIFEST["limits"])


def test_normalized_punctuation_does_not_jump_over_repeated_words(monkeypatch):
    text = '“Yes” Yes "Yes"'
    adapter = _timing_adapter(monkeypatch, [_chunk(text, [
        _token('"'), _token("Yes", .1, .3), _token('"'),
        _token("Yes", .4, .6), _token('"'), _token("Yes", .7, .9), _token('"'),
    ])])
    assert adapter.synthesize(text, timestamps=True).metadata["token_spans"] == [
        TimedSpan(1, 4, .1, .3), TimedSpan(6, 9, .4, .6), TimedSpan(11, 14, .7, .9),
    ]


@pytest.mark.parametrize("text, token", [("another", "a"), ("beta", "a"), ("with", "it")])
def test_short_token_does_not_match_inside_other_word(text, token):
    from muse.models.kokoro_82m import _map_chunk_tokens

    assert _map_chunk_tokens(text, [_token(token, .1, .4)], 0, 0.0) == []


def test_token_does_not_skip_unmapped_lexical_word():
    from muse.models.kokoro_82m import _map_chunk_tokens

    spans = _map_chunk_tokens("One two One", [
        _token("One", .1, .3), _token("One", .6, .9),
    ], 0, 0.0)
    assert spans == [
        TimedSpan(0, 3, .1, .3),
    ]


def test_missed_chunk_disables_spans_for_later_repeated_text(monkeypatch):
    adapter = _timing_adapter(monkeypatch, [
        _chunk("First.", [_token("First", .1, .4)]),
        _chunk("YES.", [_token("YES", .1, .4)]),
        _chunk("Yes.", [_token("Yes", .1, .4)]),
    ])
    result = adapter.synthesize("First.\n\nYes.\n\nYes.", timestamps=True)
    assert result.metadata["token_spans"] == [TimedSpan(0, 5, .1, .4)]
    assert len(result.audio) == 72000


def test_chunk_does_not_skip_non_whitespace_gap(monkeypatch):
    adapter = _timing_adapter(monkeypatch, [_chunk("Yes.", [_token("Yes", .1, .4)])])
    result = adapter.synthesize("Skipped.\n\nYes.", timestamps=True)
    assert result.metadata["token_spans"] == []
    assert len(result.audio) == 24000


@pytest.mark.parametrize("tokens, expected", [
    ([_token("Albert Camus", .1, .6)], [TimedSpan(1, 13, .1, .6)]),
    ([_token("Albert", .1, .3), _token("Camus", .3, .6)],
     [TimedSpan(1, 7, .1, .3), TimedSpan(8, 13, .3, .6)]),
])
def test_multi_word_link_maps_inside_brackets(monkeypatch, tokens, expected):
    text = "[Albert Camus](/ˈælbɚt kæmˈuː/) wrote."
    adapter = _timing_adapter(monkeypatch, [_chunk("Albert Camus wrote.", tokens + [
        _token("wrote", .6, .9),
    ])])
    spans = adapter.synthesize(text, timestamps=True).metadata["token_spans"]
    assert spans == expected + [TimedSpan(32, 37, .6, .9)]


@pytest.mark.parametrize("split_chunks", [False, True])
def test_two_links_and_repeated_word_keep_original_coordinates(monkeypatch, split_chunks):
    text = "[One](/wʌn/) One [Two](/tuː/) One"
    first_tokens = [_token("One", .1, .3), _token("One", .4, .6)]
    second_tokens = [_token("Two", .1, .3), _token("One", .4, .6)]
    if split_chunks:
        chunks = [_chunk("One One", first_tokens), _chunk("Two One", second_tokens)]
        expected_second = [TimedSpan(18, 21, 1.1, 1.3), TimedSpan(30, 33, 1.4, 1.6)]
    else:
        second_tokens = [_token("Two", .7, .8), _token("One", .8, .9)]
        chunks = [_chunk("One One Two One", first_tokens + second_tokens)]
        expected_second = [TimedSpan(18, 21, .7, .8), TimedSpan(30, 33, .8, .9)]
    adapter = _timing_adapter(monkeypatch, chunks)
    spans = adapter.synthesize(text, timestamps=True).metadata["token_spans"]
    assert spans == [
        TimedSpan(1, 4, .1, .3), TimedSpan(13, 16, .4, .6),
    ] + expected_second


def test_manifest_pins_preshed_below_broken_release():
    assert "preshed<3.0.14" in MANIFEST["pip_extras"]
