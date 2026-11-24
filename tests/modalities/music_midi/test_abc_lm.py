"""Mocked unit tests for ``AbcMidiLMRuntime`` (neural ABC -> MIDI runtime)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from muse.modalities.music_midi.protocol import MidiGenerationModel
from muse.modalities.music_midi.runtimes import abc_lm
from muse.modalities.music_midi.runtimes.abc_lm import AbcMidiLMRuntime
from muse.modalities.music_midi.smf import parse_smf


_SAMPLE_ABC = (
    "X:1\n"
    "T:Generated Reel\n"
    "M:4/4\n"
    "L:1/8\n"
    "Q:1/4=120\n"
    "K:D dor\n"
    "| D2 F2 A2 G2 | F2 E2 D4 |\n"
)


def _mock_torch_and_transformers(generated_text: str = _SAMPLE_ABC):
    fake_torch = MagicMock()
    fake_torch.float16 = "fp16_dtype"
    fake_torch.float32 = "fp32_dtype"
    fake_torch.bfloat16 = "bf16_dtype"
    fake_torch.cuda.is_available.return_value = False
    fake_torch.backends.mps.is_available.return_value = False
    fake_torch.no_grad.return_value.__enter__ = MagicMock(return_value=None)
    fake_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)

    fake_tok = MagicMock()
    fake_tok.return_value = {"input_ids": [[1, 2, 3]]}
    fake_tok.decode.return_value = generated_text

    fake_model = MagicMock()
    fake_model.generate.return_value = [[1, 2, 3, 4, 5]]
    fake_model.to.return_value = fake_model

    fake_am = MagicMock()
    fake_am.from_pretrained.return_value = fake_model
    fake_at = MagicMock()
    fake_at.from_pretrained.return_value = fake_tok

    return fake_torch, fake_am, fake_at, fake_model, fake_tok


def test_abc_lm_runtime_loads_and_generates_smf():
    fake_torch, fake_am, fake_at, fake_model, fake_tok = _mock_torch_and_transformers()

    with patch.object(abc_lm, "torch", fake_torch), \
         patch.object(abc_lm, "AutoModelForCausalLM", fake_am), \
         patch.object(abc_lm, "AutoTokenizer", fake_at):
        rt = AbcMidiLMRuntime(
            model_id="sander-wood-text-to-music",
            hf_repo="sander-wood/text-to-music",
            local_dir="/tmp/fake-abc-model",
            device="auto",
            dtype="float16",
        )
        assert isinstance(rt, MidiGenerationModel)
        assert rt._device == "cpu"
        # CPU demotes float16 -> float32 via dtype_for_name
        assert fake_am.from_pretrained.call_args.kwargs["torch_dtype"] == "fp32_dtype"
        assert fake_am.from_pretrained.call_args.args[0] == "/tmp/fake-abc-model"

        res = rt.generate(
            "medieval flute jig",
            key="D",
            mode="dorian",
            bpm=128,
            bars=4,
            instruments=["flute"],
            layers=["base"],
            seed=7,
            temperature=0.8,
        )

    assert res.model_id == "sander-wood-text-to-music"
    assert res.seed == 7
    assert res.midi_bytes[:4] == b"MThd"
    assert res.score.bpm == 128
    assert res.score.key == "D"
    assert res.score.mode == "dorian"
    assert res.score.tracks[0].instrument == "flute"
    assert res.score.tracks[0].gm_program == 73
    assert "abc" in res.metadata
    fake_torch.manual_seed.assert_called_once_with(7)

    parsed = parse_smf(res.midi_bytes)
    assert parsed.bpm == 128
    assert len(parsed.tracks[0].notes) == 7


def test_abc_lm_runtime_raises_when_transformers_missing():
    with patch.object(abc_lm, "torch", None), \
         patch.object(abc_lm, "AutoModelForCausalLM", None), \
         patch.object(abc_lm, "AutoTokenizer", None), \
         patch.object(abc_lm, "_ensure_deps", lambda: None):
        with pytest.raises(RuntimeError, match="transformers"):
            AbcMidiLMRuntime(model_id="abc-test", hf_repo="org/abc-test")


def test_abc_lm_runtime_cuda_path_moves_model_to_cuda():
    fake_torch, fake_am, fake_at, fake_model, _ = _mock_torch_and_transformers()
    fake_torch.cuda.is_available.return_value = True

    with patch.object(abc_lm, "torch", fake_torch), \
         patch.object(abc_lm, "AutoModelForCausalLM", fake_am), \
         patch.object(abc_lm, "AutoTokenizer", fake_at), \
         patch("muse.core.runtime_helpers.cuda_runtime_available", return_value=True):
        rt = AbcMidiLMRuntime(
            model_id="abc-cuda",
            hf_repo="sander-wood/text-to-music",
            local_dir="/tmp/abc-cuda",
            device="auto",
            dtype="float16",
        )
        assert rt._device == "cuda"
        assert fake_am.from_pretrained.call_args.kwargs["torch_dtype"] == "fp16_dtype"
        fake_model.to.assert_called_once_with("cuda")
