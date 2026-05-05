"""HFVision2SeqRuntime: mocked-dep tests.

Module-level sentinels (torch, AutoModelForVision2Seq, AutoProcessor)
get patched per-test; the autouse fixture restores them on teardown.

The runtime is wrapped around a vision-encoder + text-decoder pair.
We mock the model.generate() to return a tensor and processor methods
to round-trip the decode.
"""
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _reset_module_sentinels():
    """Reset module-top sentinels between tests (deferred-imports pattern)."""
    import muse.modalities.image_ocr.runtimes.hf_vision2seq as mod
    orig = (mod.torch, mod.AutoModelForVision2Seq, mod.AutoProcessor)
    yield
    mod.torch, mod.AutoModelForVision2Seq, mod.AutoProcessor = orig


def _make_outputs(token_count=10):
    """Build a fake generate() return tensor with a usable .shape[-1]."""
    out = MagicMock()
    out.shape = MagicMock()
    out.shape.__getitem__ = MagicMock(return_value=token_count)
    return out


def _wire_basic_runtime(mod, *, decoded_text="hello world", token_count=10):
    """Install fake torch + AutoProcessor + AutoModelForVision2Seq.

    The processor returns a stand-in for inputs that has a .to(device)
    method so the runtime's `inputs.to(self._device)` chain works.
    """
    mod.torch = MagicMock()
    mod.torch.cuda.is_available.return_value = False
    mod.torch.backends = MagicMock(mps=None)

    # Processor: callable() returns a "BatchEncoding"-like object that
    # has a .to(device) method (chainable). batch_decode returns a list
    # of strings.
    processor = MagicMock()
    encoded = MagicMock()
    encoded.to = MagicMock(return_value=encoded)
    # When the processor is called with positional image (and possibly
    # return_tensors=...), it returns `encoded`.
    processor.return_value = encoded
    processor.batch_decode = MagicMock(return_value=[decoded_text])
    # tokenizer().input_ids.to(device) chain for prompt encoding.
    # The .to() return value needs a .shape so the runtime can
    # compute prompt_len for completion_tokens accounting.
    tok_moved = MagicMock()
    tok_moved.shape = (1, 5)  # batch=1, prompt_seq_len=5
    tok_out = MagicMock()
    tok_out.input_ids = MagicMock()
    tok_out.input_ids.to = MagicMock(return_value=tok_moved)
    processor.tokenizer = MagicMock(return_value=tok_out)
    # No post_process_generation by default.
    if hasattr(processor, "post_process_generation"):
        del processor.post_process_generation

    proc_factory = MagicMock()
    proc_factory.from_pretrained = MagicMock(return_value=processor)
    mod.AutoProcessor = proc_factory

    # Model: from_pretrained returns a model with .to() chain and
    # .generate(...) returning a fake outputs tensor.
    model_obj = MagicMock()
    model_obj.to = MagicMock(return_value=model_obj)
    model_obj.generate = MagicMock(return_value=_make_outputs(token_count))

    model_factory = MagicMock()
    model_factory.from_pretrained = MagicMock(return_value=model_obj)
    mod.AutoModelForVision2Seq = model_factory

    return processor, model_obj


def test_ocr_returns_extracted_text():
    import muse.modalities.image_ocr.runtimes.hf_vision2seq as mod
    processor, _ = _wire_basic_runtime(mod, decoded_text="hello world")
    runtime = mod.HFVision2SeqRuntime(
        model_id="trocr-test",
        hf_repo="microsoft/trocr-base-printed",
        device="cpu",
    )
    result = runtime.ocr(MagicMock(name="pil_image"))
    assert result.text == "hello world"
    assert result.model_id == "trocr-test"
    # completion_tokens is the new tokens only, not the full sequence.
    # _make_outputs returned token_count=10; decoder bos prepends 1;
    # so completion_tokens == 10 - 1 == 9.
    assert result.completion_tokens == 9


def test_ocr_max_new_tokens_override():
    import muse.modalities.image_ocr.runtimes.hf_vision2seq as mod
    _, model_obj = _wire_basic_runtime(mod)
    runtime = mod.HFVision2SeqRuntime(
        model_id="m", hf_repo="x", device="cpu", max_new_tokens=256,
    )
    runtime.ocr(MagicMock(), max_new_tokens=64)
    gen_kwargs = model_obj.generate.call_args.kwargs
    assert gen_kwargs["max_new_tokens"] == 64


def test_ocr_default_max_new_tokens_from_constructor():
    """When the per-call max_new_tokens is None, constructor default applies."""
    import muse.modalities.image_ocr.runtimes.hf_vision2seq as mod
    _, model_obj = _wire_basic_runtime(mod)
    runtime = mod.HFVision2SeqRuntime(
        model_id="m", hf_repo="x", device="cpu", max_new_tokens=300,
    )
    runtime.ocr(MagicMock())
    gen_kwargs = model_obj.generate.call_args.kwargs
    assert gen_kwargs["max_new_tokens"] == 300


def test_ocr_prompt_forwards_decoder_input_ids():
    """Setting prompt='<s_doc>' tokenizes it and passes decoder_input_ids."""
    import muse.modalities.image_ocr.runtimes.hf_vision2seq as mod
    processor, model_obj = _wire_basic_runtime(mod)
    runtime = mod.HFVision2SeqRuntime(
        model_id="nougat-test",
        hf_repo="facebook/nougat-base",
        device="cpu",
    )
    runtime.ocr(MagicMock(), prompt="<s_doc>")
    # Tokenizer was called with the prompt.
    processor.tokenizer.assert_called_once_with(
        "<s_doc>", return_tensors="pt",
    )
    gen_kwargs = model_obj.generate.call_args.kwargs
    assert "decoder_input_ids" in gen_kwargs
    # Forwarded value is the .to() result of the tokenizer's input_ids.
    expected = processor.tokenizer.return_value.input_ids.to.return_value
    assert gen_kwargs["decoder_input_ids"] is expected


def test_ocr_completion_tokens_excludes_prompt():
    """Regression: completion_tokens must report newly-generated
    tokens only, not the full sequence including the decoder prompt.
    Wrong accounting was inflating Nougat usage by 5x+."""
    import muse.modalities.image_ocr.runtimes.hf_vision2seq as mod
    # token_count=20: 5 prompt tokens + 15 newly generated.
    _wire_basic_runtime(mod, token_count=20)
    runtime = mod.HFVision2SeqRuntime(
        model_id="nougat", hf_repo="x", device="cpu",
    )
    result = runtime.ocr(MagicMock(), prompt="<s_doc>")
    # 20 total - 5 prompt = 15 newly-generated tokens.
    assert result.completion_tokens == 15


def test_ocr_completion_tokens_no_prompt_subtracts_bos():
    """Without prompt, decoder_start adds 1 token; subtract it."""
    import muse.modalities.image_ocr.runtimes.hf_vision2seq as mod
    _wire_basic_runtime(mod, token_count=10)
    runtime = mod.HFVision2SeqRuntime(
        model_id="trocr", hf_repo="x", device="cpu",
    )
    result = runtime.ocr(MagicMock())
    assert result.completion_tokens == 9


def test_ocr_prompt_none_skips_decoder_input_ids():
    import muse.modalities.image_ocr.runtimes.hf_vision2seq as mod
    _, model_obj = _wire_basic_runtime(mod)
    runtime = mod.HFVision2SeqRuntime(
        model_id="m", hf_repo="x", device="cpu",
    )
    runtime.ocr(MagicMock(), prompt=None)
    gen_kwargs = model_obj.generate.call_args.kwargs
    assert "decoder_input_ids" not in gen_kwargs


def test_ocr_num_beams_forwards():
    import muse.modalities.image_ocr.runtimes.hf_vision2seq as mod
    _, model_obj = _wire_basic_runtime(mod)
    runtime = mod.HFVision2SeqRuntime(
        model_id="m", hf_repo="x", device="cpu",
    )
    runtime.ocr(MagicMock(), num_beams=4)
    gen_kwargs = model_obj.generate.call_args.kwargs
    assert gen_kwargs["num_beams"] == 4


def test_ocr_uses_post_process_generation_when_available():
    """Nougat-style processors expose post_process_generation; runtime
    routes through it for LaTeX cleanup. Test the path."""
    import muse.modalities.image_ocr.runtimes.hf_vision2seq as mod
    processor, _ = _wire_basic_runtime(mod, decoded_text="raw\nwith\ttabs")
    processor.post_process_generation = MagicMock(return_value="cleaned latex")
    runtime = mod.HFVision2SeqRuntime(
        model_id="nougat", hf_repo="x", device="cpu",
    )
    result = runtime.ocr(MagicMock())
    assert result.text == "cleaned latex"
    # Verify the post-process function received the raw decoded string.
    args, _kwargs = processor.post_process_generation.call_args
    assert args[0] == "raw\nwith\ttabs"


def test_ocr_post_process_generation_typeerror_falls_back_to_no_kwargs():
    """If the processor's post_process_generation doesn't take
    fix_markdown=, the runtime catches TypeError and retries
    positionally."""
    import muse.modalities.image_ocr.runtimes.hf_vision2seq as mod
    processor, _ = _wire_basic_runtime(mod, decoded_text="raw")
    call_count = {"n": 0}

    def _post(text, **kwargs):
        call_count["n"] += 1
        if "fix_markdown" in kwargs:
            raise TypeError("unexpected keyword argument 'fix_markdown'")
        return f"cleaned: {text}"

    processor.post_process_generation = _post
    runtime = mod.HFVision2SeqRuntime(
        model_id="m", hf_repo="x", device="cpu",
    )
    result = runtime.ocr(MagicMock())
    assert result.text == "cleaned: raw"
    assert call_count["n"] == 2  # first with fix_markdown, then without


def test_local_dir_preferred_over_hf_repo():
    """When local_dir is set, the runtime loads from the snapshot path,
    not the HF repo id."""
    import muse.modalities.image_ocr.runtimes.hf_vision2seq as mod
    processor, model_obj = _wire_basic_runtime(mod)
    runtime = mod.HFVision2SeqRuntime(
        model_id="m",
        hf_repo="microsoft/trocr-base-printed",
        local_dir="/tmp/local-snapshot",
        device="cpu",
    )
    # AutoProcessor.from_pretrained received the local dir, not the hf id.
    proc_call_arg = mod.AutoProcessor.from_pretrained.call_args.args[0]
    model_call_arg = mod.AutoModelForVision2Seq.from_pretrained.call_args.args[0]
    assert proc_call_arg == "/tmp/local-snapshot"
    assert model_call_arg == "/tmp/local-snapshot"


def test_raises_when_torch_not_installed(monkeypatch):
    import muse.modalities.image_ocr.runtimes.hf_vision2seq as mod
    monkeypatch.setattr(mod, "_ensure_deps", lambda: None)
    mod.torch = None
    mod.AutoModelForVision2Seq = MagicMock()
    mod.AutoProcessor = MagicMock()
    with pytest.raises(RuntimeError, match="torch is not installed"):
        mod.HFVision2SeqRuntime(model_id="m", hf_repo="x", device="cpu")


def test_raises_when_transformers_not_installed(monkeypatch):
    import muse.modalities.image_ocr.runtimes.hf_vision2seq as mod
    monkeypatch.setattr(mod, "_ensure_deps", lambda: None)
    mod.torch = MagicMock()
    mod.AutoModelForVision2Seq = None
    mod.AutoProcessor = None
    with pytest.raises(RuntimeError, match="transformers is not installed"):
        mod.HFVision2SeqRuntime(model_id="m", hf_repo="x", device="cpu")


def test_partial_transformers_install_also_raises(monkeypatch):
    """If only AutoProcessor (or only AutoModelForVision2Seq) is missing
    we still raise the unified error."""
    import muse.modalities.image_ocr.runtimes.hf_vision2seq as mod
    monkeypatch.setattr(mod, "_ensure_deps", lambda: None)
    mod.torch = MagicMock()
    mod.AutoModelForVision2Seq = MagicMock()
    mod.AutoProcessor = None
    with pytest.raises(RuntimeError, match="transformers is not installed"):
        mod.HFVision2SeqRuntime(model_id="m", hf_repo="x", device="cpu")
