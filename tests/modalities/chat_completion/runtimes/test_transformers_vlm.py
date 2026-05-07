"""HFVisionLanguageModel: mocked-dep tests.

Module-level sentinels (torch, AutoModelForImageTextToText, AutoProcessor,
TextIteratorStreamer) get patched per-test; the autouse fixture restores
them on teardown.
"""
from unittest.mock import MagicMock
import pytest


@pytest.fixture(autouse=True)
def _reset_module_sentinels():
    import muse.modalities.chat_completion.runtimes.transformers_vlm as mod
    orig = (
        mod.torch, mod.AutoModelForImageTextToText,
        mod.AutoModelForVision2Seq, mod.AutoProcessor,
        mod.TextIteratorStreamer,
    )
    yield
    (
        mod.torch, mod.AutoModelForImageTextToText,
        mod.AutoModelForVision2Seq, mod.AutoProcessor,
        mod.TextIteratorStreamer,
    ) = orig


def test_raises_when_torch_not_installed(monkeypatch):
    import muse.modalities.chat_completion.runtimes.transformers_vlm as mod
    monkeypatch.setattr(mod, "_ensure_deps", lambda: None)
    mod.torch = None
    mod.AutoModelForImageTextToText = MagicMock()
    mod.AutoProcessor = MagicMock()
    with pytest.raises(RuntimeError, match="torch is not installed"):
        mod.HFVisionLanguageModel(
            model_id="m", hf_repo="x", device="cpu",
        )


def test_raises_when_transformers_too_old(monkeypatch):
    """When neither AutoModelForImageTextToText nor AutoModelForVision2Seq
    is importable, the error must mention transformers >= 4.46."""
    import muse.modalities.chat_completion.runtimes.transformers_vlm as mod
    monkeypatch.setattr(mod, "_ensure_deps", lambda: None)
    mod.torch = MagicMock()
    mod.AutoModelForImageTextToText = None
    mod.AutoModelForVision2Seq = None
    mod.AutoProcessor = None
    with pytest.raises(RuntimeError, match=r"transformers .* 4\.46"):
        mod.HFVisionLanguageModel(
            model_id="m", hf_repo="x", device="cpu",
        )


def test_prepare_inputs_extracts_text_and_images():
    """Walk a multimodal messages list; return (chat_template_text, images_list).

    Image parts arrive as {type: image, image: <PIL>} (route layer rewrites
    image_url to image). Text parts stay as strings.
    """
    import muse.modalities.chat_completion.runtimes.transformers_vlm as mod
    fake_processor = MagicMock()
    fake_processor.apply_chat_template = MagicMock(return_value="<rendered>")
    img1, img2 = MagicMock(name="pil1"), MagicMock(name="pil2")

    text, images = mod._prepare_inputs(
        [{
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in these?"},
                {"type": "image", "image": img1},
                {"type": "image", "image": img2},
            ],
        }],
        fake_processor,
    )
    assert text == "<rendered>"
    assert images == [img1, img2]
    # The processor's apply_chat_template should have been called with
    # template-shape content (no PIL objects, just type tags).
    template_msgs = fake_processor.apply_chat_template.call_args.args[0]
    assert template_msgs[0]["content"] == [
        {"type": "text", "text": "What's in these?"},
        {"type": "image"},
        {"type": "image"},
    ]


def test_prepare_inputs_string_content_passthrough():
    """Legacy `content: "string"` shape preserved unchanged."""
    import muse.modalities.chat_completion.runtimes.transformers_vlm as mod
    fake_processor = MagicMock()
    fake_processor.apply_chat_template = MagicMock(return_value="<text-only>")

    text, images = mod._prepare_inputs(
        [{"role": "user", "content": "hello"}],
        fake_processor,
    )
    assert text == "<text-only>"
    assert images == []


def test_prepare_inputs_undecoded_image_url_raises():
    """Defensive: route layer must rewrite image_url to image. If it
    doesn't, the runtime raises so the bug surfaces clearly."""
    import muse.modalities.chat_completion.runtimes.transformers_vlm as mod
    fake_processor = MagicMock()
    with pytest.raises(ValueError, match="image_url part reached runtime"):
        mod._prepare_inputs(
            [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:..."}},
                ],
            }],
            fake_processor,
        )


def _wire_runtime(mod, decoded_text="hello world", token_count=10, prompt_len=5):
    """Install fake torch + AutoProcessor + AutoModelForImageTextToText.

    The processor returns inputs that have a .to(device) chain and an
    input_ids tensor with .shape[-1] == prompt_len. The model returns a
    generate() result with .shape[-1] == prompt_len + (token_count - prompt_len).
    """
    mod.torch = MagicMock()
    mod.torch.cuda.is_available.return_value = False
    mod.torch.backends = MagicMock(mps=None)

    processor = MagicMock()
    processor.apply_chat_template = MagicMock(return_value="<rendered>")
    encoded = MagicMock()
    encoded.to = MagicMock(return_value=encoded)
    encoded.__getitem__ = lambda self, k: _ids_tensor(prompt_len)
    processor.return_value = encoded
    processor.batch_decode = MagicMock(return_value=[decoded_text])
    tokenizer = MagicMock()
    processor.tokenizer = tokenizer

    proc_factory = MagicMock()
    proc_factory.from_pretrained = MagicMock(return_value=processor)
    mod.AutoProcessor = proc_factory

    model_obj = MagicMock()
    model_obj.to = MagicMock(return_value=model_obj)
    out_tensor = MagicMock()
    out_tensor.shape = MagicMock()
    out_tensor.shape.__getitem__ = MagicMock(return_value=token_count)
    out_tensor.__getitem__ = MagicMock(return_value=_ids_slice(token_count - prompt_len))
    model_obj.generate = MagicMock(return_value=out_tensor)

    model_factory = MagicMock()
    model_factory.from_pretrained = MagicMock(return_value=model_obj)
    mod.AutoModelForImageTextToText = model_factory
    mod.AutoModelForVision2Seq = None
    return processor, model_obj


def _ids_tensor(length):
    t = MagicMock()
    t.shape = MagicMock()
    t.shape.__getitem__ = MagicMock(return_value=length)
    return t


def _ids_slice(length):
    s = MagicMock()
    s.__len__ = MagicMock(return_value=length)
    return s


def test_chat_text_only_returns_chat_result():
    import muse.modalities.chat_completion.runtimes.transformers_vlm as mod
    processor, _ = _wire_runtime(mod, decoded_text="text reply")
    runtime = mod.HFVisionLanguageModel(
        model_id="vlm-test", hf_repo="any/vlm", device="cpu",
    )
    result = runtime.chat([{"role": "user", "content": "hi"}])
    assert result.model_id == "vlm-test"
    assert result.choices[0].message["content"] == "text reply"
    assert result.choices[0].finish_reason == "stop"
    # Processor was called WITHOUT images=
    proc_kwargs = processor.call_args.kwargs
    assert "images" not in proc_kwargs


def test_chat_with_images_passes_them_to_processor():
    import muse.modalities.chat_completion.runtimes.transformers_vlm as mod
    processor, _ = _wire_runtime(mod)
    runtime = mod.HFVisionLanguageModel(
        model_id="vlm", hf_repo="x", device="cpu",
    )
    img = MagicMock(name="pil")
    runtime.chat([{
        "role": "user",
        "content": [
            {"type": "text", "text": "describe"},
            {"type": "image", "image": img},
        ],
    }])
    proc_kwargs = processor.call_args.kwargs
    assert proc_kwargs.get("images") == [img]


def test_chat_completion_tokens_excludes_prompt():
    """Regression: completion_tokens reports newly-generated tokens only."""
    import muse.modalities.chat_completion.runtimes.transformers_vlm as mod
    _wire_runtime(mod, prompt_len=7, token_count=20)
    runtime = mod.HFVisionLanguageModel(
        model_id="vlm", hf_repo="x", device="cpu",
    )
    result = runtime.chat([{"role": "user", "content": "x"}])
    assert result.usage["prompt_tokens"] == 7
    assert result.usage["completion_tokens"] == 13
    assert result.usage["total_tokens"] == 20


def test_chat_max_tokens_kwarg_overrides_constructor_default():
    import muse.modalities.chat_completion.runtimes.transformers_vlm as mod
    _, model_obj = _wire_runtime(mod)
    runtime = mod.HFVisionLanguageModel(
        model_id="m", hf_repo="x", device="cpu", max_new_tokens=300,
    )
    runtime.chat([{"role": "user", "content": "x"}], max_tokens=64)
    gen_kwargs = model_obj.generate.call_args.kwargs
    assert gen_kwargs["max_new_tokens"] == 64


def test_supports_multi_image_flag_propagates():
    import muse.modalities.chat_completion.runtimes.transformers_vlm as mod
    _wire_runtime(mod)
    runtime = mod.HFVisionLanguageModel(
        model_id="m", hf_repo="x", device="cpu", supports_multi_image=True,
    )
    assert runtime.supports_multi_image is True


def test_chat_stream_yields_chunks_then_final_finish_reason():
    """The streamer yields chat chunks per-token, with a final empty-delta
    chunk carrying finish_reason='stop'."""
    import muse.modalities.chat_completion.runtimes.transformers_vlm as mod
    _wire_runtime(mod)

    fake_streamer = MagicMock()
    fake_streamer.__iter__ = MagicMock(return_value=iter(["he", "llo", " world"]))
    streamer_factory = MagicMock(return_value=fake_streamer)
    mod.TextIteratorStreamer = streamer_factory

    runtime = mod.HFVisionLanguageModel(
        model_id="m", hf_repo="x", device="cpu",
    )
    chunks = list(runtime.chat_stream([{"role": "user", "content": "x"}]))
    assert len(chunks) == 4
    assert chunks[0].delta == {"content": "he"}
    assert chunks[1].delta == {"content": "llo"}
    assert chunks[2].delta == {"content": " world"}
    assert chunks[3].finish_reason == "stop"
    assert chunks[3].delta == {}


def test_chat_stream_calls_streamer_end_when_generate_raises():
    """Regression: if generate() raises, streamer.end() must still be called
    (in the finally block) so the consumer iterator doesn't deadlock."""
    import muse.modalities.chat_completion.runtimes.transformers_vlm as mod
    _wire_runtime(mod)

    fake_streamer = MagicMock()
    fake_streamer.__iter__ = MagicMock(return_value=iter([]))
    fake_streamer.end = MagicMock()
    streamer_factory = MagicMock(return_value=fake_streamer)
    mod.TextIteratorStreamer = streamer_factory

    runtime = mod.HFVisionLanguageModel(
        model_id="m", hf_repo="x", device="cpu",
    )
    # Make generate raise inside the daemon thread.
    runtime._model.generate = MagicMock(side_effect=RuntimeError("CUDA OOM"))

    chunks = list(runtime.chat_stream([{"role": "user", "content": "x"}]))
    # The exception fires in the daemon thread; the consumer drains the
    # empty streamer (end() was called in finally) and finishes cleanly.
    fake_streamer.end.assert_called()
    # The final finish_reason chunk must still be emitted.
    assert chunks[-1].finish_reason == "stop"


def test_chat_stream_thread_is_daemon():
    """Regression: the generate thread must be daemon=True so it doesn't
    prevent interpreter shutdown on consumer abandonment."""
    import muse.modalities.chat_completion.runtimes.transformers_vlm as mod
    _wire_runtime(mod)

    captured_threads: list = []
    original_thread_class = mod.threading.Thread

    class CapturingThread(original_thread_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            captured_threads.append(self)

    fake_streamer = MagicMock()
    fake_streamer.__iter__ = MagicMock(return_value=iter(["hi"]))
    mod.TextIteratorStreamer = MagicMock(return_value=fake_streamer)

    mod.threading.Thread = CapturingThread
    try:
        runtime = mod.HFVisionLanguageModel(
            model_id="m", hf_repo="x", device="cpu",
        )
        list(runtime.chat_stream([{"role": "user", "content": "x"}]))
    finally:
        mod.threading.Thread = original_thread_class

    assert captured_threads, "expected at least one Thread to be created"
    assert all(t.daemon for t in captured_threads), (
        "chat_stream's generate thread must be daemon=True"
    )
