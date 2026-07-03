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
    orig = (
        mod.torch, mod.AutoModelForVision2Seq, mod.AutoProcessor,
        mod.AutoImageProcessor, mod.AutoTokenizer,
        mod._LAST_IMPORT_ERROR,
    )
    yield
    (
        mod.torch, mod.AutoModelForVision2Seq, mod.AutoProcessor,
        mod.AutoImageProcessor, mod.AutoTokenizer,
        mod._LAST_IMPORT_ERROR,
    ) = orig


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
    # The runtime detects "unified vs split" by checking for
    # `image_processor` (or legacy `feature_extractor`) on the
    # AutoProcessor return. The default fixture is the unified path
    # (TrOCR/Nougat shape); attach a placeholder image_processor so
    # the constructor takes the existing branch without needing
    # AutoImageProcessor mocked.
    processor.image_processor = MagicMock()
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
    """The error must surface the version + class context so the user
    isn't told 'transformers is not installed' when transformers IS
    installed but the specific Auto* class isn't (e.g. transformers 5.x
    dropped AutoModelForVision2Seq).
    """
    import muse.modalities.image_ocr.runtimes.hf_vision2seq as mod
    monkeypatch.setattr(mod, "_ensure_deps", lambda: None)
    mod.torch = MagicMock()
    mod.AutoModelForVision2Seq = None
    mod.AutoProcessor = None
    with pytest.raises(RuntimeError, match="AutoModelForImageTextToText"):
        mod.HFVision2SeqRuntime(model_id="m", hf_repo="x", device="cpu")


def test_partial_transformers_install_also_raises(monkeypatch):
    """If only AutoProcessor (or only AutoModelForVision2Seq) is missing
    we still raise the unified error."""
    import muse.modalities.image_ocr.runtimes.hf_vision2seq as mod
    monkeypatch.setattr(mod, "_ensure_deps", lambda: None)
    mod.torch = MagicMock()
    mod.AutoModelForVision2Seq = MagicMock()
    mod.AutoProcessor = None
    with pytest.raises(RuntimeError, match="AutoModelForImageTextToText"):
        mod.HFVision2SeqRuntime(model_id="m", hf_repo="x", device="cpu")


def test_ensure_deps_prefers_transformers5_class_when_available():
    """Regression watchdog: transformers 5.x renamed
    AutoModelForVision2Seq -> AutoModelForImageTextToText. The runtime
    must try the new name first and fall back to the old name on
    older transformers versions.

    Simulates the transformers 5.x case: the new class imports cleanly,
    the old name raises ImportError. Expect the sentinel populated
    from the new class.
    """
    import muse.modalities.image_ocr.runtimes.hf_vision2seq as mod
    fake_transformers = MagicMock()
    fake_new_class = MagicMock()
    fake_new_class.__name__ = "AutoModelForImageTextToText"
    fake_transformers.AutoModelForImageTextToText = fake_new_class
    # Don't expose AutoModelForVision2Seq on the fake module (it's gone
    # in transformers 5.x).
    delattr_target = "AutoModelForVision2Seq"
    if hasattr(fake_transformers, delattr_target):
        delattr(fake_transformers, delattr_target)
    fake_transformers.AutoProcessor = MagicMock()

    import sys
    real_transformers = sys.modules.get("transformers")
    sys.modules["transformers"] = fake_transformers
    try:
        mod.torch = None
        mod.AutoModelForVision2Seq = None
        mod.AutoProcessor = None
        mod._LAST_IMPORT_ERROR = None
        mod._ensure_deps()
        assert mod.AutoModelForVision2Seq is fake_new_class, (
            "Expected AutoModelForVision2Seq sentinel to hold the "
            "new transformers 5.x class"
        )
    finally:
        if real_transformers is not None:
            sys.modules["transformers"] = real_transformers
        else:
            sys.modules.pop("transformers", None)


def test_split_component_path_when_processor_lacks_image_processor(monkeypatch):
    """Regression for v0.40.5: when AutoProcessor returns a tokenizer
    (because the repo lacks `preprocessor_config.json`), the runtime
    must fall back to AutoImageProcessor + AutoTokenizer.

    Repos that hit this path: TexTeller and other older
    vision-encoder-decoder OCR repos.
    """
    import muse.modalities.image_ocr.runtimes.hf_vision2seq as mod
    import muse.core.image_preprocessing as ip_mod

    # Set up: AutoProcessor returns a tokenizer-only (no image_processor,
    # no feature_extractor). The runtime should detect this and load
    # AutoImageProcessor separately via build_image_processor tier 2.
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False
    fake_torch.backends = MagicMock(mps=None)
    mod.torch = fake_torch

    # Tokenizer-shaped object: callable for text, batch_decode method,
    # input_ids chain. NO image_processor / feature_extractor attribute.
    tokenizer = MagicMock(spec=["__call__", "batch_decode"])
    tok_moved = MagicMock()
    tok_moved.shape = (1, 5)
    tok_out = MagicMock()
    tok_out.input_ids = MagicMock()
    tok_out.input_ids.to = MagicMock(return_value=tok_moved)
    tokenizer.return_value = tok_out
    tokenizer.batch_decode = MagicMock(return_value=["formula"])

    proc_factory = MagicMock()
    proc_factory.from_pretrained = MagicMock(return_value=tokenizer)
    mod.AutoProcessor = proc_factory

    # Patch the core module's _load_auto_image_processor to return our
    # sentinel image_processor (build_image_processor tier 2: auto succeeds).
    image_processor = MagicMock()
    encoded = MagicMock()
    encoded.to = MagicMock(return_value=encoded)
    image_processor.return_value = encoded
    monkeypatch.setattr(ip_mod, "_load_auto_image_processor",
                        lambda src: image_processor)

    # Model: standard.
    model_obj = MagicMock()
    model_obj.to = MagicMock(return_value=model_obj)
    model_obj.generate = MagicMock(return_value=_make_outputs(token_count=12))
    model_factory = MagicMock()
    model_factory.from_pretrained = MagicMock(return_value=model_obj)
    mod.AutoModelForVision2Seq = model_factory

    runtime = mod.HFVision2SeqRuntime(
        model_id="texteller", hf_repo="OleehyO/TexTeller", device="cpu",
    )
    # The constructor should have detected the split case.
    assert runtime._processor is None
    assert runtime._tokenizer is tokenizer
    assert runtime._image_processor is image_processor

    # Inference should use the image_processor for the image side and
    # the tokenizer for batch_decode.
    result = runtime.ocr(MagicMock(name="pil_image"))
    assert result.text == "formula"
    image_processor.assert_called_once()
    tokenizer.batch_decode.assert_called_once()


def test_constructor_split_path_uses_derived_when_auto_image_processor_fails(tmp_path, monkeypatch):
    """End-to-end: when AutoProcessor returns a tokenizer-only AND
    AutoImageProcessor.from_pretrained raises (the TexTeller case),
    the constructor must wire DerivedImageProcessor into
    self._image_processor via build_image_processor's fallback ladder.

    Regression watchdog: the fallback path from the core module
    (build_image_processor tier 3) must still fire through the runtime.

    Encoder hints must include explicit image_mean/image_std: tier 3
    only trusts hints that carry ground-truth normalization stats
    (v0.5x fix), so num_channels/image_size alone are not enough to
    drive DerivedImageProcessor."""
    import muse.modalities.image_ocr.runtimes.hf_vision2seq as mod
    import json
    import torch as real_torch
    import muse.core.image_preprocessing as ip_mod

    # Drop a config.json with grayscale encoder hints (including explicit
    # normalization stats) into the snapshot dir.
    (tmp_path / "config.json").write_text(json.dumps({
        "encoder": {
            "num_channels": 1, "image_size": 448,
            "image_mean": [0.5], "image_std": [0.5],
        },
    }))

    mod.torch = real_torch
    # AutoProcessor returns a tokenizer-only object (TexTeller shape).
    tokenizer = MagicMock(spec=["__call__", "batch_decode"])
    tokenizer.batch_decode = MagicMock(return_value=["x"])
    proc_factory = MagicMock()
    proc_factory.from_pretrained = MagicMock(return_value=tokenizer)
    mod.AutoProcessor = proc_factory
    # Patch core module's _load_auto_image_processor to raise so tier 3 fires.
    monkeypatch.setattr(ip_mod, "_load_auto_image_processor",
                        lambda src: (_ for _ in ()).throw(RuntimeError("no pp")))

    model_obj = MagicMock()
    model_obj.to = MagicMock(return_value=model_obj)
    model_factory = MagicMock()
    model_factory.from_pretrained = MagicMock(return_value=model_obj)
    mod.AutoModelForVision2Seq = model_factory

    runtime = mod.HFVision2SeqRuntime(
        model_id="texteller",
        hf_repo="OleehyO/TexTeller",
        local_dir=str(tmp_path),
        device="cpu",
    )
    assert runtime._processor is None
    assert runtime._tokenizer is tokenizer
    from muse.core.image_preprocessing import DerivedImageProcessor
    assert isinstance(runtime._image_processor, DerivedImageProcessor)
    assert runtime._image_processor.num_channels == 1
    assert runtime._image_processor.height == 448


def test_ensure_deps_falls_back_to_legacy_class_on_transformers4():
    """Symmetric regression: when only the legacy class is available
    (transformers 4.x), the sentinel should hold it."""
    import muse.modalities.image_ocr.runtimes.hf_vision2seq as mod

    # Build a fake module where the new class import RAISES but the
    # old class succeeds. We can't easily delattr the new class to
    # cause an ImportError, so use a dynamic __getattr__ on a custom
    # module-shaped class.
    class _FakeT4:
        AutoProcessor = MagicMock()
        # Old class exists.
        AutoModelForVision2Seq = MagicMock(name="LegacyVision2Seq")

        def __getattr__(self, name):
            # New class doesn't exist in transformers 4.x.
            if name == "AutoModelForImageTextToText":
                raise ImportError(
                    "cannot import AutoModelForImageTextToText (transformers 4.x)"
                )
            raise AttributeError(name)

    fake = _FakeT4()
    import sys
    real_transformers = sys.modules.get("transformers")
    sys.modules["transformers"] = fake
    try:
        mod.torch = None
        mod.AutoModelForVision2Seq = None
        mod.AutoProcessor = None
        mod._LAST_IMPORT_ERROR = None
        mod._ensure_deps()
        assert mod.AutoModelForVision2Seq is fake.AutoModelForVision2Seq, (
            "Expected fallback to legacy AutoModelForVision2Seq in "
            "transformers 4.x"
        )
    finally:
        if real_transformers is not None:
            sys.modules["transformers"] = real_transformers
        else:
            sys.modules.pop("transformers", None)


def test_constructor_accepts_image_processor_overrides_kwarg(tmp_path, monkeypatch):
    """When a manifest declares capabilities.image_processor_overrides,
    the runtime forwards them to build_image_processor.

    Regression for v0.42.1: the override hatch must reach the core
    helper through the runtime's constructor, bypassing AutoImageProcessor."""
    import muse.modalities.image_ocr.runtimes.hf_vision2seq as mod
    import torch as real_torch
    import muse.core.image_preprocessing as ip_mod
    mod.torch = real_torch

    # AutoProcessor returns a tokenizer-only object (TexTeller shape).
    tokenizer = MagicMock(spec=["__call__", "batch_decode"])
    tokenizer.batch_decode = MagicMock(return_value=["x"])
    proc_factory = MagicMock()
    proc_factory.from_pretrained = MagicMock(return_value=tokenizer)
    mod.AutoProcessor = proc_factory

    # _load_auto_image_processor: should NOT be called because overrides is set.
    auto_ip_called = []
    def _must_not_call(src):
        auto_ip_called.append(src)
        raise AssertionError(
            "AutoImageProcessor must not be called when overrides is set"
        )
    monkeypatch.setattr(ip_mod, "_load_auto_image_processor", _must_not_call)

    model_obj = MagicMock()
    model_obj.to = MagicMock(return_value=model_obj)
    model_factory = MagicMock()
    model_factory.from_pretrained = MagicMock(return_value=model_obj)
    mod.AutoModelForVision2Seq = model_factory

    runtime = mod.HFVision2SeqRuntime(
        model_id="texteller",
        hf_repo="OleehyO/TexTeller",
        local_dir=str(tmp_path),
        device="cpu",
        image_processor_overrides={"num_channels": 1, "image_size": 448},
    )
    # The core DerivedImageProcessor was used via build_image_processor.
    from muse.core.image_preprocessing import DerivedImageProcessor
    assert isinstance(runtime._image_processor, DerivedImageProcessor)
    assert runtime._image_processor.num_channels == 1
    assert runtime._image_processor.height == 448
    assert not auto_ip_called, "AutoImageProcessor must not be called"
