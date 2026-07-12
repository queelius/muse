from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from muse.modalities.audio_alignment.decoding import DecodedAudio
from muse.modalities.audio_alignment.protocol import (
    UnsupportedAlignmentLanguageError,
)
from muse.modalities.audio_alignment.runtimes import qwen3_forced_aligner as runtime


@pytest.mark.parametrize("value,expected", [
    (None, None),
    ("auto", None),
    ("en", "English"),
    ("ZH_cn", "Chinese"),
    ("yue", "Cantonese"),
    ("pt-BR", "Portuguese"),
    ("Japanese", "Japanese"),
])
def test_normalize_language_aliases(value, expected):
    assert runtime.normalize_language(value) == expected


def test_normalize_language_rejects_unknown():
    with pytest.raises(UnsupportedAlignmentLanguageError, match="Klingon"):
        runtime.normalize_language("Klingon")


def test_timestamp_marker_confidence_selects_only_marker_positions(
    monkeypatch,
):
    monkeypatch.setattr(runtime, "torch", torch)
    original_softmax = torch.softmax
    softmax_shapes = []

    def recording_softmax(value, *args, **kwargs):
        softmax_shapes.append(tuple(value.shape))
        return original_softmax(value, *args, **kwargs)

    monkeypatch.setattr(torch, "softmax", recording_softmax)
    logits = torch.tensor([[
        [5.0, 0.0],
        [0.0, 0.0],
        [0.0, 5.0],
    ]])
    input_ids = torch.tensor([[99, 1, 99]])
    values = runtime._timestamp_marker_confidences(logits, input_ids, 99)
    assert values == pytest.approx([0.993307, 0.993307], rel=1e-5)
    assert softmax_shapes == [(2, 2)]


class _Inputs(dict):
    def __init__(self, input_ids=None):
        super().__init__(input_ids=(
            torch.tensor([[99, 99, 99, 99]])
            if input_ids is None
            else input_ids
        ))
        self.move_args = None

    def to(self, *args):
        self.move_args = args
        return self


class _Processor:
    def __init__(self, *, words=None, input_ids=None):
        self.words = words or ["Hello", "world"]
        self.inputs = _Inputs(input_ids)
        self.prepared = None

    def split_words_for_alignment(self, text, language):
        self.split = (text, language)
        return list(self.words)

    def prepare_forced_aligner_inputs(self, **kwargs):
        self.prepared = kwargs
        return self.inputs, [list(self.words)]

    def decode_forced_alignment(self, **kwargs):
        self.decoded = kwargs
        return [[
            {"text": "Hello", "start_time": -0.1, "end_time": 0.4},
            {"text": "world", "start_time": 0.5, "end_time": 1.2},
        ]]


class _Model:
    def __init__(self):
        self.config = SimpleNamespace(timestamp_token_id=99)

    def __call__(self, **kwargs):
        logits = torch.tensor([[
            [5.0, 0.0], [5.0, 0.0],
            [5.0, 0.0], [5.0, 0.0],
        ]])
        return SimpleNamespace(logits=logits)


def test_align_runs_processor_and_returns_clamped_words(monkeypatch):
    monkeypatch.setattr(runtime, "torch", torch)
    captured = {}

    def fake_decode(path, *, sample_rate, max_duration_seconds):
        captured.update(
            path=path, sample_rate=sample_rate,
            max_duration_seconds=max_duration_seconds,
        )
        return DecodedAudio(torch.ones(1, 16000), 16000, 1.0)

    monkeypatch.setattr(runtime, "decode_audio", fake_decode)
    aligner = runtime.Qwen3ForcedAlignerRuntime.__new__(
        runtime.Qwen3ForcedAlignerRuntime
    )
    aligner.model_id = "qwen-aligner"
    aligner._device = "cpu"
    aligner._dtype = torch.float32
    aligner._sample_rate = 16000
    aligner._max_duration_seconds = 300.0
    aligner._max_input_tokens = 8192
    aligner._max_reference_words = 2048
    aligner._processor = _Processor()
    aligner._model = _Model()

    result = aligner.align(
        "clip.wav", "  Hello world  ", language="en",
        max_duration_seconds=600,
    )
    assert captured["max_duration_seconds"] == 300
    assert aligner._processor.prepared["language"] == "English"
    assert aligner._processor.inputs.move_args == ("cpu", torch.float32)
    assert result.text == "Hello world"
    assert result.language == "English"
    assert [(word.start, word.end) for word in result.words] == [
        (0.0, 0.4), (0.5, 1.0),
    ]
    assert all(word.confidence is not None for word in result.words)
    assert result.metadata["clamped_timestamp_count"] == 2


def test_rejects_reference_word_limit_before_audio_decode(monkeypatch):
    monkeypatch.setattr(runtime, "torch", torch)
    decode = MagicMock()
    monkeypatch.setattr(runtime, "decode_audio", decode)
    aligner = runtime.Qwen3ForcedAlignerRuntime.__new__(
        runtime.Qwen3ForcedAlignerRuntime
    )
    aligner._processor = _Processor(words=["a", "b", "c"])
    aligner._max_reference_words = 2

    with pytest.raises(ValueError, match="3 alignable words; maximum is 2"):
        aligner.align("clip.wav", "a b c", language="en")
    decode.assert_not_called()


def test_rejects_prepared_input_over_model_context_before_inference(
    monkeypatch,
):
    monkeypatch.setattr(runtime, "torch", torch)
    monkeypatch.setattr(
        runtime,
        "decode_audio",
        lambda *args, **kwargs: DecodedAudio(
            torch.ones(1, 16000), 16000, 1.0,
        ),
    )
    inputs = torch.tensor([[99, 1, 99, 99, 1, 99]])
    processor = _Processor(input_ids=inputs)
    aligner = runtime.Qwen3ForcedAlignerRuntime.__new__(
        runtime.Qwen3ForcedAlignerRuntime
    )
    aligner._device = "cpu"
    aligner._dtype = torch.float32
    aligner._sample_rate = 16000
    aligner._max_duration_seconds = 300.0
    aligner._max_input_tokens = 5
    aligner._max_reference_words = 2048
    aligner._processor = processor
    aligner._model = MagicMock()

    with pytest.raises(ValueError, match="6 input tokens; maximum is 5"):
        aligner.align("clip.wav", "Hello world", language="en")
    assert processor.inputs.move_args is None
    aligner._model.assert_not_called()


def test_constructor_loads_local_checkpoint_on_cpu(monkeypatch, tmp_path):
    for name in runtime._REQUIRED_FILES:
        (tmp_path / name).write_bytes(b"x")
    processor = MagicMock()
    processor_cls = MagicMock()
    processor_cls.from_pretrained.return_value = processor
    model = MagicMock()
    model.to.return_value = model
    model.config = SimpleNamespace(
        text_config=SimpleNamespace(max_position_embeddings=8192),
    )
    model_cls = MagicMock()
    model_cls.from_pretrained.return_value = model
    monkeypatch.setattr(runtime, "torch", torch)
    monkeypatch.setattr(runtime, "AutoProcessor", processor_cls)
    monkeypatch.setattr(runtime, "AutoModelForTokenClassification", model_cls)

    aligner = runtime.Qwen3ForcedAlignerRuntime(
        model_id="qwen-aligner",
        hf_repo="Qwen/Qwen3-ForcedAligner-0.6B-hf",
        local_dir=str(tmp_path),
        device="cpu",
    )
    assert aligner._device == "cpu"
    processor_cls.from_pretrained.assert_called_once_with(str(tmp_path))
    assert model_cls.from_pretrained.call_args.kwargs["dtype"] is torch.float32
    model.to.assert_called_once_with("cpu")
    assert aligner._max_input_tokens == 8192
    assert aligner._max_reference_words == 2048
