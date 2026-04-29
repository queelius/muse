"""Tests for BartSeq2SeqRuntime (transformers AutoModelForSeq2SeqLM wrapper)."""
from unittest.mock import MagicMock, patch

import pytest

import muse.modalities.text_summarization.runtimes.bart_seq2seq as bart_mod
from muse.modalities.text_summarization import SummarizationResult
from muse.modalities.text_summarization.runtimes.bart_seq2seq import (
    LENGTH_TO_MAX_TOKENS,
    VALID_FORMATS,
    VALID_LENGTHS,
    BartSeq2SeqRuntime,
)


def _fake_tensor(seq_len: int):
    """Return a MagicMock that mimics a torch tensor's shape access."""
    t = MagicMock()
    # tensor.shape[-1] -> seq_len
    t.shape = (1, seq_len)
    # tensor.to(device) -> tensor (chainable)
    t.to.return_value = t
    # tensor[0] -> tensor (decode walks the first batch element)
    t.__getitem__ = lambda self, idx: t
    return t


def _fake_tokenizer(input_len_full=200, input_len_truncated=200):
    """Simulate transformers AutoTokenizer behavior.

    The tokenizer returns a BatchEncoding-like dict. Two calls are made:
    one without truncation (to measure full length) and one with
    truncation (to feed the model). We let the test parameterize both.
    """
    tok = MagicMock()
    full_encoded = {"input_ids": _fake_tensor(input_len_full)}
    truncated_encoded = {"input_ids": _fake_tensor(input_len_truncated)}

    call_state = {"calls": 0}

    def _tok_call(*args, **kwargs):
        call_state["calls"] += 1
        return full_encoded if call_state["calls"] == 1 else truncated_encoded

    tok.side_effect = _tok_call
    tok.decode.return_value = "summary text"
    return tok


def _patched_runtime(
    *,
    input_len_full=200,
    input_len_truncated=200,
    output_len=42,
    device="cpu",
    max_input_tokens=1024,
    default_length="medium",
    default_format="paragraph",
):
    """Build a BartSeq2SeqRuntime with all heavy deps mocked."""
    fake_tok = _fake_tokenizer(input_len_full, input_len_truncated)
    fake_tok_class = MagicMock()
    fake_tok_class.from_pretrained.return_value = fake_tok

    fake_model = MagicMock()
    fake_model.to.return_value = fake_model
    fake_model.eval.return_value = None
    fake_model.generate.return_value = _fake_tensor(output_len)
    fake_model_class = MagicMock()
    fake_model_class.from_pretrained.return_value = fake_model

    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False
    fake_torch.backends.mps.is_available.return_value = False
    fake_torch.float16 = "float16"
    fake_torch.float32 = "float32"
    fake_torch.bfloat16 = "bfloat16"

    with patch.object(bart_mod, "AutoModelForSeq2SeqLM", fake_model_class), \
            patch.object(bart_mod, "AutoTokenizer", fake_tok_class), \
            patch.object(bart_mod, "torch", fake_torch):
        rt = BartSeq2SeqRuntime(
            model_id="test",
            hf_repo="org/repo",
            local_dir=None,
            device=device,
            dtype="float32",
            default_length=default_length,
            default_format=default_format,
            max_input_tokens=max_input_tokens,
        )
    return rt, fake_tok, fake_model, fake_tok_class, fake_model_class, fake_torch


def test_length_to_max_tokens_mapping_is_frozen():
    assert LENGTH_TO_MAX_TOKENS == {"short": 80, "medium": 180, "long": 400}


def test_valid_lengths_match_mapping_keys():
    assert VALID_LENGTHS == frozenset({"short", "medium", "long"})


def test_valid_formats_includes_paragraph_and_bullets():
    assert VALID_FORMATS == frozenset({"paragraph", "bullets"})


def test_runtime_constructs_with_local_dir_preference():
    fake_tok = MagicMock()
    fake_tok_class = MagicMock()
    fake_tok_class.from_pretrained.return_value = fake_tok
    fake_model = MagicMock()
    fake_model.to.return_value = fake_model
    fake_model_class = MagicMock()
    fake_model_class.from_pretrained.return_value = fake_model
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False
    fake_torch.backends.mps.is_available.return_value = False

    with patch.object(bart_mod, "AutoModelForSeq2SeqLM", fake_model_class), \
            patch.object(bart_mod, "AutoTokenizer", fake_tok_class), \
            patch.object(bart_mod, "torch", fake_torch):
        BartSeq2SeqRuntime(
            model_id="m",
            hf_repo="org/repo",
            local_dir="/tmp/cache/abc",
            device="cpu",
        )
    # AutoTokenizer.from_pretrained called with local_dir
    args, _ = fake_tok_class.from_pretrained.call_args
    assert args[0] == "/tmp/cache/abc"
    # AutoModelForSeq2SeqLM.from_pretrained called with local_dir
    args2, _ = fake_model_class.from_pretrained.call_args
    assert args2[0] == "/tmp/cache/abc"


def test_runtime_falls_back_to_hf_repo_when_no_local_dir():
    fake_tok_class = MagicMock()
    fake_tok_class.from_pretrained.return_value = MagicMock()
    fake_model = MagicMock()
    fake_model.to.return_value = fake_model
    fake_model_class = MagicMock()
    fake_model_class.from_pretrained.return_value = fake_model
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False
    fake_torch.backends.mps.is_available.return_value = False

    with patch.object(bart_mod, "AutoModelForSeq2SeqLM", fake_model_class), \
            patch.object(bart_mod, "AutoTokenizer", fake_tok_class), \
            patch.object(bart_mod, "torch", fake_torch):
        BartSeq2SeqRuntime(
            model_id="m",
            hf_repo="facebook/bart-large-cnn",
            local_dir=None,
            device="cpu",
        )
    args, _ = fake_tok_class.from_pretrained.call_args
    assert args[0] == "facebook/bart-large-cnn"


def test_runtime_calls_to_with_device():
    rt, _, fake_model, _, _, _ = _patched_runtime(device="cpu")
    fake_model.to.assert_called_with("cpu")


def test_summarize_short_uses_max_new_tokens_80():
    rt, fake_tok, fake_model, _, _, _ = _patched_runtime()
    rt.summarize("hello world", length="short")
    _, kwargs = fake_model.generate.call_args
    assert kwargs["max_new_tokens"] == 80


def test_summarize_medium_uses_max_new_tokens_180():
    rt, fake_tok, fake_model, _, _, _ = _patched_runtime()
    rt.summarize("hello world", length="medium")
    _, kwargs = fake_model.generate.call_args
    assert kwargs["max_new_tokens"] == 180


def test_summarize_long_uses_max_new_tokens_400():
    rt, fake_tok, fake_model, _, _, _ = _patched_runtime()
    rt.summarize("hello world", length="long")
    _, kwargs = fake_model.generate.call_args
    assert kwargs["max_new_tokens"] == 400


def test_summarize_returns_summarization_result():
    rt, _, _, _, _, _ = _patched_runtime()
    out = rt.summarize("hello", length="short", format="paragraph")
    assert isinstance(out, SummarizationResult)
    assert out.summary == "summary text"
    assert out.length == "short"
    assert out.format == "paragraph"
    assert out.model_id == "test"


def test_summarize_default_length_when_omitted():
    rt, _, fake_model, _, _, _ = _patched_runtime(default_length="long")
    out = rt.summarize("hello")
    assert out.length == "long"
    _, kwargs = fake_model.generate.call_args
    assert kwargs["max_new_tokens"] == 400


def test_summarize_default_format_when_omitted():
    rt, _, _, _, _, _ = _patched_runtime(default_format="bullets")
    out = rt.summarize("hello")
    assert out.format == "bullets"


def test_summarize_invalid_length_raises():
    rt, _, _, _, _, _ = _patched_runtime()
    with pytest.raises(ValueError, match="length"):
        rt.summarize("hello", length="WRONG")


def test_summarize_invalid_format_raises():
    rt, _, _, _, _, _ = _patched_runtime()
    with pytest.raises(ValueError, match="format"):
        rt.summarize("hello", format="WRONG")


def test_summarize_records_truncation_warning_when_input_too_long():
    """Full input is 2000 tokens but max_input_tokens=1024."""
    rt, _, _, _, _, _ = _patched_runtime(
        input_len_full=2000, input_len_truncated=1024, max_input_tokens=1024,
    )
    out = rt.summarize("very long text")
    assert out.metadata.get("truncation_warning") is True
    assert out.metadata["truncated_from_tokens"] == 2000
    assert out.metadata["truncated_to_tokens"] == 1024


def test_summarize_no_truncation_warning_when_input_fits():
    rt, _, _, _, _, _ = _patched_runtime(
        input_len_full=200, input_len_truncated=200, max_input_tokens=1024,
    )
    out = rt.summarize("short input")
    assert "truncation_warning" not in out.metadata


def test_summarize_prompt_tokens_reflects_truncated_input():
    rt, _, _, _, _, _ = _patched_runtime(
        input_len_full=2000, input_len_truncated=1024, max_input_tokens=1024,
    )
    out = rt.summarize("very long text")
    assert out.prompt_tokens == 1024


def test_summarize_completion_tokens_reflects_output_length():
    rt, _, _, _, _, _ = _patched_runtime(output_len=42)
    out = rt.summarize("hello")
    assert out.completion_tokens == 42


def test_runtime_raises_when_transformers_missing():
    with patch.object(bart_mod, "AutoModelForSeq2SeqLM", None), \
            patch.object(bart_mod, "AutoTokenizer", None), \
            patch.object(bart_mod, "_ensure_deps", lambda: None), \
            patch.object(bart_mod, "torch", MagicMock()):
        with pytest.raises(RuntimeError, match="transformers"):
            BartSeq2SeqRuntime(
                model_id="m", hf_repo="org/repo", local_dir=None,
            )


def test_select_device_auto_falls_back_to_cpu_with_no_torch():
    with patch.object(bart_mod, "torch", None):
        from muse.modalities.text_summarization.runtimes.bart_seq2seq import (
            _select_device,
        )
        assert _select_device("auto") == "cpu"


def test_select_device_auto_picks_cuda():
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = True
    with patch.object(bart_mod, "torch", fake_torch):
        from muse.modalities.text_summarization.runtimes.bart_seq2seq import (
            _select_device,
        )
        assert _select_device("auto") == "cuda"


def test_select_device_auto_picks_mps_when_cuda_absent():
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False
    fake_torch.backends.mps.is_available.return_value = True
    with patch.object(bart_mod, "torch", fake_torch):
        from muse.modalities.text_summarization.runtimes.bart_seq2seq import (
            _select_device,
        )
        assert _select_device("auto") == "mps"


def test_select_device_explicit_passes_through():
    with patch.object(bart_mod, "torch", MagicMock()):
        from muse.modalities.text_summarization.runtimes.bart_seq2seq import (
            _select_device,
        )
        assert _select_device("cuda:1") == "cuda:1"


def test_resolve_dtype_returns_none_without_torch():
    with patch.object(bart_mod, "torch", None):
        from muse.modalities.text_summarization.runtimes.bart_seq2seq import (
            _resolve_dtype,
        )
        assert _resolve_dtype("float16") is None


def test_resolve_dtype_maps_known_strings():
    fake_torch = MagicMock()
    fake_torch.float16 = "FP16_TENSOR"
    fake_torch.bfloat16 = "BF16_TENSOR"
    fake_torch.float32 = "FP32_TENSOR"
    with patch.object(bart_mod, "torch", fake_torch):
        from muse.modalities.text_summarization.runtimes.bart_seq2seq import (
            _resolve_dtype,
        )
        assert _resolve_dtype("float16") == "FP16_TENSOR"
        assert _resolve_dtype("fp16") == "FP16_TENSOR"
        assert _resolve_dtype("bfloat16") == "BF16_TENSOR"
        assert _resolve_dtype("bf16") == "BF16_TENSOR"
        assert _resolve_dtype("float32") == "FP32_TENSOR"
        assert _resolve_dtype("fp32") == "FP32_TENSOR"
        # Unknown string falls back to float32 (safe default).
        assert _resolve_dtype("unknown_dtype") == "FP32_TENSOR"


def test_set_inference_mode_calls_eval_when_method_present():
    """Calling _set_inference_mode on a model with .eval should invoke it."""
    from muse.modalities.text_summarization.runtimes.bart_seq2seq import (
        _set_inference_mode,
    )
    m = MagicMock()
    _set_inference_mode(m)
    m.eval.assert_called_once()


def test_set_inference_mode_no_op_when_method_missing():
    from muse.modalities.text_summarization.runtimes.bart_seq2seq import (
        _set_inference_mode,
    )

    class NoEval:
        pass

    # Must not raise; nothing to call.
    _set_inference_mode(NoEval())


def test_summarize_passes_input_ids_to_generate():
    rt, fake_tok, fake_model, _, _, _ = _patched_runtime()
    rt.summarize("hello")
    args, _ = fake_model.generate.call_args
    # First positional arg is the input_ids tensor (which is a MagicMock).
    assert args[0] is not None


def test_summarize_decodes_first_batch_element():
    rt, fake_tok, fake_model, _, _, _ = _patched_runtime()
    rt.summarize("hello")
    # decode was called with skip_special_tokens=True
    _, dec_kwargs = fake_tok.decode.call_args
    assert dec_kwargs.get("skip_special_tokens") is True


def test_summarize_metadata_does_not_contain_length_or_format():
    """Runtime metadata should NOT carry length/format; the dataclass
    fields are the canonical home, codec echoes them via setdefault."""
    rt, _, _, _, _, _ = _patched_runtime()
    out = rt.summarize("hello")
    assert "length" not in out.metadata
    assert "format" not in out.metadata
