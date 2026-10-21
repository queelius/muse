"""Tests for the bundled bart_large_cnn script (fully mocked).

The script aliases the generic `BartSeq2SeqRuntime` as its `Model`;
runtime behavior is covered by tests/modalities/text_summarization/
runtimes/test_bart_seq2seq.py. These tests pin the MANIFEST and check
that the runtime honors it when constructed the way `load_backend` does
(`model_id` + capabilities splatted as kwargs).

Module-level imports of `muse.models.bart_large_cnn` are DELIBERATELY
avoided: test_discovery_robust_to_broken_deps pops `muse.models.*` from
sys.modules and re-imports them, so a top-level import would go stale.
"""
import importlib
from unittest.mock import MagicMock, patch

_RT = "muse.modalities.text_summarization.runtimes.bart_seq2seq"


def _manifest():
    return importlib.import_module("muse.models.bart_large_cnn").MANIFEST


def test_manifest_required_fields():
    m = _manifest()
    assert m["model_id"] == "bart-large-cnn"
    assert m["modality"] == "text/summarization"
    assert m["hf_repo"] == "facebook/bart-large-cnn"
    assert "torch>=2.1.0" in m["pip_extras"]
    assert any("transformers" in x for x in m["pip_extras"])


def test_manifest_capabilities_shape():
    caps = _manifest()["capabilities"]
    assert caps["device"] == "auto"
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


def test_model_is_bart_seq2seq_runtime():
    from muse.modalities.text_summarization.runtimes.bart_seq2seq import (
        BartSeq2SeqRuntime,
    )
    assert importlib.import_module("muse.models.bart_large_cnn").Model is (
        BartSeq2SeqRuntime
    )


def _fake_tensor(seq_len: int):
    t = MagicMock()
    t.shape = (1, seq_len)
    t.to.return_value = t
    t.__getitem__ = lambda self, idx: t
    t.tolist.return_value = list(range(seq_len))
    return t


def test_manifest_capabilities_drive_runtime_construction():
    mod = importlib.import_module("muse.models.bart_large_cnn")
    manifest = mod.MANIFEST

    fake_tok = MagicMock(return_value={"input_ids": _fake_tensor(10)})
    fake_tok.decode.return_value = "a summary"
    fake_tok.all_special_ids = [0, 1]
    fake_model = MagicMock()
    fake_model.to.return_value = fake_model
    fake_model.generate.return_value = _fake_tensor(12)
    tok_cls = MagicMock()
    tok_cls.from_pretrained.return_value = fake_tok
    model_cls = MagicMock()
    model_cls.from_pretrained.return_value = fake_model
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False
    fake_torch.backends.mps.is_available.return_value = False

    kwargs = {"model_id": manifest["model_id"], **manifest["capabilities"]}
    with patch(f"{_RT}.torch", fake_torch), \
            patch(f"{_RT}.AutoTokenizer", tok_cls), \
            patch(f"{_RT}.AutoModelForSeq2SeqLM", model_cls):
        m = mod.Model(
            hf_repo=manifest["hf_repo"], local_dir="/fake/bart", **kwargs,
        )
        r = m.summarize("some long article text")

    tok_cls.from_pretrained.assert_called_once_with("/fake/bart")
    fake_model.to.assert_called_once_with("cpu")  # auto, no accelerator
    assert fake_model.generate.call_args.kwargs["max_new_tokens"] == 180
    assert r.length == "medium"
    assert r.format == "paragraph"
    assert r.model_id == "bart-large-cnn"
    assert r.completion_tokens == 10  # 12 ids minus the 2 specials
