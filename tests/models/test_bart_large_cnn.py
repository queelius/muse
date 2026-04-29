"""Tests for the bundled bart_large_cnn script (fully mocked).

Module-level imports of `muse.models.bart_large_cnn` are DELIBERATELY
avoided: another test in the suite (test_discovery_robust_to_broken_deps)
pops `muse.models.*` from sys.modules and re-imports them, which means
a top-level `import muse.models.bart_large_cnn as bart_script` captures
a stale module reference once that test has run. We re-resolve the
live module inside helpers/each test instead.
"""
from unittest.mock import MagicMock, patch

import pytest

from muse.modalities.text_summarization import SummarizationResult


def _bart_script():
    """Resolve the live module each call so test_discovery's sys.modules
    eviction doesn't leave us holding a stale reference."""
    import importlib
    return importlib.import_module("muse.models.bart_large_cnn")


def _manifest():
    return _bart_script().MANIFEST


def _fake_tensor(seq_len: int):
    """Return a MagicMock that mimics a torch tensor's shape access."""
    t = MagicMock()
    t.shape = (1, seq_len)
    t.to.return_value = t
    t.__getitem__ = lambda self, idx: t
    return t


def _fake_tokenizer(input_len_full=200, input_len_truncated=200):
    tok = MagicMock()
    full_encoded = {"input_ids": _fake_tensor(input_len_full)}
    truncated_encoded = {"input_ids": _fake_tensor(input_len_truncated)}
    state = {"calls": 0}

    def _tok_call(*args, **kwargs):
        state["calls"] += 1
        return full_encoded if state["calls"] == 1 else truncated_encoded

    tok.side_effect = _tok_call
    tok.decode.return_value = "a summary of the input text."
    return tok


def _patched_setup(
    *,
    input_len_full=200,
    input_len_truncated=200,
    output_len=42,
    with_cuda=False,
    max_input_tokens=1024,
):
    """Install fake transformers + torch on the live bart_large_cnn module.
    Returns (bart_module, fake_tok_class, fake_model_class, fake_torch).
    """
    bart = _bart_script()

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
    fake_torch.cuda.is_available.return_value = with_cuda
    fake_torch.backends.mps.is_available.return_value = False

    bart.AutoModelForSeq2SeqLM = fake_model_class
    bart.AutoTokenizer = fake_tok_class
    bart.torch = fake_torch
    return bart, fake_tok, fake_model, fake_tok_class, fake_model_class, fake_torch


@pytest.fixture(autouse=True)
def _reset_module_sentinels():
    yield
    bart = _bart_script()
    bart.torch = None
    bart.AutoModelForSeq2SeqLM = None
    bart.AutoTokenizer = None


def test_manifest_required_fields():
    m = _manifest()
    assert m["model_id"] == "bart-large-cnn"
    assert m["modality"] == "text/summarization"
    assert m["hf_repo"] == "facebook/bart-large-cnn"
    assert "pip_extras" in m
    assert "torch>=2.1.0" in m["pip_extras"]
    assert any("transformers" in x for x in m["pip_extras"])


def test_manifest_capabilities_shape():
    caps = _manifest()["capabilities"]
    assert caps["device"] == "cpu"
    assert caps["default_length"] == "medium"
    assert caps["default_format"] == "paragraph"
    assert caps["supports_dialog_summarization"] is False
    assert caps["max_input_tokens"] == 1024
    assert "memory_gb" in caps


def test_manifest_allow_patterns_includes_bpe_files():
    """BART tokenizer requires both vocab.json and merges.txt; both must
    survive the allow_patterns filter or tokenizer load fails."""
    patterns = _manifest()["allow_patterns"]
    assert any("merges.txt" in p for p in patterns)
    assert any("vocab.json" in p for p in patterns)


def test_manifest_license_is_apache_2_0():
    assert _manifest()["license"] == "Apache 2.0"


def test_model_class_exists():
    bart = _bart_script()
    assert hasattr(bart, "Model")
    assert bart.Model.model_id == "bart-large-cnn"


def test_model_construction_lazy_imports():
    """Model.__init__ should call _ensure_deps and read transformers
    via the module's sentinels so tests can patch them."""
    bart, _, _, fake_tok_class, fake_model_class, _ = _patched_setup()
    m = bart.Model(hf_repo="facebook/bart-large-cnn", local_dir=None)
    assert m._device == "cpu"
    fake_tok_class.from_pretrained.assert_called_once()
    fake_model_class.from_pretrained.assert_called_once()


def test_model_prefers_local_dir():
    bart, _, _, fake_tok_class, fake_model_class, _ = _patched_setup()
    bart.Model(
        hf_repo="facebook/bart-large-cnn", local_dir="/tmp/bart",
    )
    args, _ = fake_tok_class.from_pretrained.call_args
    assert args[0] == "/tmp/bart"
    args2, _ = fake_model_class.from_pretrained.call_args
    assert args2[0] == "/tmp/bart"


def test_model_falls_back_to_hf_repo_when_no_local_dir():
    bart, _, _, fake_tok_class, _, _ = _patched_setup()
    bart.Model(hf_repo="facebook/bart-large-cnn", local_dir=None)
    args, _ = fake_tok_class.from_pretrained.call_args
    assert args[0] == "facebook/bart-large-cnn"


def test_summarize_returns_summarization_result():
    bart, _, _, _, _, _ = _patched_setup()
    m = bart.Model(hf_repo="facebook/bart-large-cnn", local_dir=None)
    out = m.summarize("hello world", length="short", format="paragraph")
    assert isinstance(out, SummarizationResult)
    assert out.summary == "a summary of the input text."
    assert out.length == "short"
    assert out.format == "paragraph"
    assert out.model_id == "bart-large-cnn"


def test_summarize_short_uses_max_new_tokens_80():
    bart, _, fake_model, _, _, _ = _patched_setup()
    m = bart.Model(hf_repo="facebook/bart-large-cnn", local_dir=None)
    m.summarize("hello", length="short")
    _, kwargs = fake_model.generate.call_args
    assert kwargs["max_new_tokens"] == 80


def test_summarize_medium_uses_max_new_tokens_180():
    bart, _, fake_model, _, _, _ = _patched_setup()
    m = bart.Model(hf_repo="facebook/bart-large-cnn", local_dir=None)
    m.summarize("hello", length="medium")
    _, kwargs = fake_model.generate.call_args
    assert kwargs["max_new_tokens"] == 180


def test_summarize_long_uses_max_new_tokens_400():
    bart, _, fake_model, _, _, _ = _patched_setup()
    m = bart.Model(hf_repo="facebook/bart-large-cnn", local_dir=None)
    m.summarize("hello", length="long")
    _, kwargs = fake_model.generate.call_args
    assert kwargs["max_new_tokens"] == 400


def test_summarize_default_length_when_omitted():
    bart, _, fake_model, _, _, _ = _patched_setup()
    m = bart.Model(hf_repo="facebook/bart-large-cnn", local_dir=None)
    out = m.summarize("hello")
    # Default is medium per MANIFEST capabilities.
    assert out.length == "medium"
    _, kwargs = fake_model.generate.call_args
    assert kwargs["max_new_tokens"] == 180


def test_summarize_default_format_when_omitted():
    bart, _, _, _, _, _ = _patched_setup()
    m = bart.Model(hf_repo="facebook/bart-large-cnn", local_dir=None)
    out = m.summarize("hello")
    assert out.format == "paragraph"


def test_summarize_invalid_length_raises():
    bart, _, _, _, _, _ = _patched_setup()
    m = bart.Model(hf_repo="facebook/bart-large-cnn", local_dir=None)
    with pytest.raises(ValueError, match="length"):
        m.summarize("hello", length="WRONG")


def test_summarize_invalid_format_raises():
    bart, _, _, _, _, _ = _patched_setup()
    m = bart.Model(hf_repo="facebook/bart-large-cnn", local_dir=None)
    with pytest.raises(ValueError, match="format"):
        m.summarize("hello", format="WRONG")


def test_summarize_records_truncation_warning_when_input_too_long():
    bart, _, _, _, _, _ = _patched_setup(
        input_len_full=2000, input_len_truncated=1024,
    )
    m = bart.Model(
        hf_repo="facebook/bart-large-cnn",
        local_dir=None,
        max_input_tokens=1024,
    )
    out = m.summarize("very long text")
    assert out.metadata.get("truncation_warning") is True
    assert out.metadata["truncated_from_tokens"] == 2000
    assert out.metadata["truncated_to_tokens"] == 1024


def test_summarize_no_truncation_warning_when_input_fits():
    bart, _, _, _, _, _ = _patched_setup(
        input_len_full=200, input_len_truncated=200,
    )
    m = bart.Model(
        hf_repo="facebook/bart-large-cnn",
        local_dir=None,
        max_input_tokens=1024,
    )
    out = m.summarize("short input")
    assert "truncation_warning" not in out.metadata


def test_summarize_prompt_tokens_reflects_truncated_input():
    bart, _, _, _, _, _ = _patched_setup(
        input_len_full=2000, input_len_truncated=1024,
    )
    m = bart.Model(
        hf_repo="facebook/bart-large-cnn",
        local_dir=None,
        max_input_tokens=1024,
    )
    out = m.summarize("long")
    assert out.prompt_tokens == 1024


def test_summarize_completion_tokens_reflects_output_length():
    bart, _, _, _, _, _ = _patched_setup(output_len=42)
    m = bart.Model(hf_repo="facebook/bart-large-cnn", local_dir=None)
    out = m.summarize("hello")
    assert out.completion_tokens == 42


def test_model_raises_when_transformers_missing():
    """Stub _ensure_deps so it leaves AutoModelForSeq2SeqLM as None
    (simulates transformers not being installed in the venv)."""
    bart = _bart_script()
    fake_torch = MagicMock()
    with patch.object(bart, "AutoModelForSeq2SeqLM", None), \
            patch.object(bart, "AutoTokenizer", None), \
            patch.object(bart, "_ensure_deps", lambda: None), \
            patch.object(bart, "torch", fake_torch):
        with pytest.raises(RuntimeError, match="transformers"):
            bart.Model(
                hf_repo="facebook/bart-large-cnn", local_dir=None,
            )


def test_select_device_auto_falls_back_to_cpu_with_no_torch():
    bart = _bart_script()
    with patch.object(bart, "torch", None):
        assert bart._select_device("auto") == "cpu"


def test_select_device_auto_picks_cuda():
    bart = _bart_script()
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = True
    with patch.object(bart, "torch", fake_torch):
        assert bart._select_device("auto") == "cuda"


def test_select_device_explicit_passes_through():
    bart = _bart_script()
    with patch.object(bart, "torch", MagicMock()):
        assert bart._select_device("cuda:1") == "cuda:1"


def test_set_inference_mode_safe_when_method_missing():
    """The helper must not raise on objects without .eval."""
    bart = _bart_script()

    class NoEval:
        pass

    bart._set_inference_mode(NoEval())  # no exception


def test_summarize_decodes_with_skip_special_tokens():
    bart, fake_tok, _, _, _, _ = _patched_setup()
    m = bart.Model(hf_repo="facebook/bart-large-cnn", local_dir=None)
    m.summarize("hello")
    _, dec_kwargs = fake_tok.decode.call_args
    assert dec_kwargs.get("skip_special_tokens") is True
