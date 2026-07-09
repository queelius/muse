"""Tests for TranslationRuntime (transformers AutoModelForSeq2SeqLM wrapper
with per-family dispatch: m2m100, nllb, opus_mt, madlad)."""
from unittest.mock import MagicMock, patch

import pytest

import muse.modalities.text_translation.runtimes.hf_translation as hf_mod
from muse.modalities.text_translation.protocol import (
    TranslationResult,
    UnsupportedLanguageError,
)
from muse.modalities.text_translation.runtimes.hf_translation import (
    TranslationRuntime,
    _family_for,
)
from muse.modalities.text_translation.runtimes.nllb_codes import ISO_TO_FLORES


# ---------------------------------------------------------------------------
# _family_for: pure dispatch function
# ---------------------------------------------------------------------------


def test_family_for_m2m100():
    assert _family_for("facebook/m2m100_418M") == "m2m100"


def test_family_for_nllb():
    assert _family_for("facebook/nllb-200-distilled-600M") == "nllb"


def test_family_for_opus_mt():
    assert _family_for("Helsinki-NLP/opus-mt-en-es") == "opus_mt"


def test_family_for_madlad():
    assert _family_for("google/madlad400-3b-mt") == "madlad"


def test_family_for_is_case_insensitive():
    assert _family_for("FACEBOOK/M2M100_418M") == "m2m100"
    assert _family_for("SOME/NLLB-repo") == "nllb"
    assert _family_for("Some/OPUS-MT-en-fr") == "opus_mt"
    assert _family_for("Some/MADLAD400") == "madlad"


def test_family_for_unknown_repo_raises_value_error():
    with pytest.raises(ValueError, match="unknown translation family"):
        _family_for("some-org/totally-unknown-repo")


# ---------------------------------------------------------------------------
# Fakes / harness
# ---------------------------------------------------------------------------


def _fake_tensor(seq_len: int, batch: int = 1):
    """MagicMock mimicking a torch tensor's shape + .to() chaining."""
    t = MagicMock()
    t.shape = (batch, seq_len)
    t.to.return_value = t
    return t


def _fake_torch():
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False
    fake_torch.backends.mps.is_available.return_value = False
    fake_torch.float16 = "float16"
    fake_torch.float32 = "float32"
    fake_torch.bfloat16 = "bfloat16"
    return fake_torch


def _build_runtime(
    hf_repo: str,
    *,
    num_beams: int = 4,
    source_language=None,
    target_language=None,
    generate_return=None,
    decode_side_effect=None,
    tokenizer_extra=None,
):
    """Construct a TranslationRuntime with transformers/torch fully mocked.

    Returns (runtime, fake_tokenizer, fake_model) so tests can inspect
    calls made against them.
    """
    fake_tok = MagicMock()
    # tokenizer(texts, return_tensors="pt", padding=True) -> BatchEncoding-like
    fake_tok.return_value = {
        "input_ids": _fake_tensor(8),
        "attention_mask": _fake_tensor(8),
    }
    fake_tok.decode.side_effect = decode_side_effect or (
        lambda ids, **kw: f"decoded-{ids}"
    )
    if tokenizer_extra:
        tokenizer_extra(fake_tok)
    fake_tok_class = MagicMock()
    fake_tok_class.from_pretrained.return_value = fake_tok

    fake_model = MagicMock()
    fake_model.to.return_value = fake_model
    fake_model.eval.return_value = None
    fake_model.generate.return_value = generate_return or ["out-0"]
    fake_model_class = MagicMock()
    fake_model_class.from_pretrained.return_value = fake_model

    fake_torch = _fake_torch()

    with patch.object(hf_mod, "AutoModelForSeq2SeqLM", fake_model_class), \
            patch.object(hf_mod, "AutoTokenizer", fake_tok_class), \
            patch.object(hf_mod, "torch", fake_torch):
        rt = TranslationRuntime(
            model_id="test",
            hf_repo=hf_repo,
            local_dir=None,
            device="cpu",
            num_beams=num_beams,
            source_language=source_language,
            target_language=target_language,
        )
    return rt, fake_tok, fake_model


# ---------------------------------------------------------------------------
# Construction: unknown-repo fallback / ValueError
# ---------------------------------------------------------------------------


def test_construct_unknown_repo_without_pair_raises_value_error():
    with pytest.raises(ValueError, match="unknown translation family"):
        _build_runtime("some-org/mystery-model")


def test_construct_unknown_repo_with_pair_falls_back_to_opus_mt():
    rt, fake_tok, fake_model = _build_runtime(
        "some-org/mystery-model", source_language="en", target_language="es",
    )
    assert rt._family == "opus_mt"
    out = rt.translate(["hello"], source="en", target="es")
    assert isinstance(out, TranslationResult)


# ---------------------------------------------------------------------------
# m2m100
# ---------------------------------------------------------------------------


def test_m2m100_sets_src_lang_and_forced_bos_via_get_lang_id():
    def extra(tok):
        tok.get_lang_id.return_value = 42

    rt, fake_tok, fake_model = _build_runtime(
        "facebook/m2m100_418M", tokenizer_extra=extra,
    )
    rt.translate(["hello"], source="en", target="es")
    assert fake_tok.src_lang == "en"
    fake_tok.get_lang_id.assert_called_once_with("es")
    _, kwargs = fake_model.generate.call_args
    assert kwargs["forced_bos_token_id"] == 42


def test_m2m100_supported_languages_derives_from_lang_code_to_id():
    def extra(tok):
        tok.lang_code_to_id = {"en": 1, "es": 2, "fr": 3}

    rt, fake_tok, fake_model = _build_runtime(
        "facebook/m2m100_418M", tokenizer_extra=extra,
    )
    supported = rt.supported_languages()
    assert supported["en"] == ["es", "fr"]
    assert supported["es"] == ["en", "fr"]
    assert supported["fr"] == ["en", "es"]


# ---------------------------------------------------------------------------
# nllb
# ---------------------------------------------------------------------------


def test_nllb_maps_en_to_eng_latn_and_sets_forced_bos():
    def extra(tok):
        tok.convert_tokens_to_ids.return_value = 999

    rt, fake_tok, fake_model = _build_runtime(
        "facebook/nllb-200-distilled-600M", tokenizer_extra=extra,
    )
    rt.translate(["hello"], source="en", target="es")
    assert fake_tok.src_lang == ISO_TO_FLORES["en"]
    fake_tok.convert_tokens_to_ids.assert_called_once_with(ISO_TO_FLORES["es"])
    _, kwargs = fake_model.generate.call_args
    assert kwargs["forced_bos_token_id"] == 999


def test_nllb_raises_unsupported_language_error_on_unmapped_source():
    rt, fake_tok, fake_model = _build_runtime("facebook/nllb-200-distilled-600M")
    with pytest.raises(UnsupportedLanguageError) as excinfo:
        rt.translate(["hello"], source="xx", target="en")
    assert excinfo.value.code == "xx"


def test_nllb_raises_unsupported_language_error_on_unmapped_target():
    rt, fake_tok, fake_model = _build_runtime("facebook/nllb-200-distilled-600M")
    with pytest.raises(UnsupportedLanguageError) as excinfo:
        rt.translate(["hello"], source="en", target="xx")
    assert excinfo.value.code == "xx"


def test_nllb_fy_has_no_flores_mapping():
    # Real-tokenizer verification against facebook/nllb-200-distilled-600M
    # on 2026-07-09 confirmed it has no "fry_Latn" token: NLLB-200 does not
    # cover Western Frisian. A prior "fy" -> "fry_Latn" entry would have
    # forced an <unk> BOS token and produced garbage; "fy" must stay
    # absent from ISO_TO_FLORES until NLLB actually ships the language.
    assert "fy" not in ISO_TO_FLORES


def test_nllb_raises_unsupported_language_error_for_fy():
    rt, fake_tok, fake_model = _build_runtime("facebook/nllb-200-distilled-600M")
    with pytest.raises(UnsupportedLanguageError) as excinfo:
        rt.translate(["hello"], source="en", target="fy")
    assert excinfo.value.code == "fy"


# ---------------------------------------------------------------------------
# opus_mt
# ---------------------------------------------------------------------------


def test_opus_mt_translates_declared_pair():
    rt, fake_tok, fake_model = _build_runtime(
        "Helsinki-NLP/opus-mt-en-es",
        source_language="en",
        target_language="es",
    )
    out = rt.translate(["hello"], source="en", target="es")
    assert isinstance(out, TranslationResult)


def test_opus_mt_refuses_non_declared_pair():
    rt, fake_tok, fake_model = _build_runtime(
        "Helsinki-NLP/opus-mt-en-es",
        source_language="en",
        target_language="es",
    )
    with pytest.raises(UnsupportedLanguageError):
        rt.translate(["bonjour"], source="fr", target="es")


def test_opus_mt_supported_languages_is_pair_only():
    rt, fake_tok, fake_model = _build_runtime(
        "Helsinki-NLP/opus-mt-en-es",
        source_language="en",
        target_language="es",
    )
    assert rt.supported_languages() == {"en": ["es"]}


# ---------------------------------------------------------------------------
# madlad
# ---------------------------------------------------------------------------


def test_madlad_prepends_target_prefix():
    rt, fake_tok, fake_model = _build_runtime("google/madlad400-3b-mt")
    rt.translate(["hello"], source="en", target="es")
    args, kwargs = fake_tok.call_args
    prefixed_texts = args[0] if args else kwargs.get("text")
    assert prefixed_texts == ["<2es> hello"]


def test_madlad_prepends_prefix_per_input_in_batch():
    rt, fake_tok, fake_model = _build_runtime(
        "google/madlad400-3b-mt",
        generate_return=["out-0", "out-1"],
    )
    rt.translate(["hello", "world"], source="en", target="fr")
    args, kwargs = fake_tok.call_args
    prefixed_texts = args[0] if args else kwargs.get("text")
    assert prefixed_texts == ["<2fr> hello", "<2fr> world"]


# ---------------------------------------------------------------------------
# Batch / shared generation behavior
# ---------------------------------------------------------------------------


def test_batch_returns_one_string_per_input():
    rt, fake_tok, fake_model = _build_runtime(
        "facebook/m2m100_418M",
        generate_return=["out-0", "out-1", "out-2"],
        decode_side_effect=lambda ids, **kw: f"decoded-{ids}",
    )
    out = rt.translate(["a", "b", "c"], source="en", target="es")
    assert out.texts == ["decoded-out-0", "decoded-out-1", "decoded-out-2"]


def test_num_beams_forwarded_to_generate():
    rt, fake_tok, fake_model = _build_runtime("facebook/m2m100_418M", num_beams=2)
    rt.translate(["hello"], source="en", target="es")
    _, kwargs = fake_model.generate.call_args
    assert kwargs["num_beams"] == 2


def test_num_beams_defaults_to_4():
    rt, fake_tok, fake_model = _build_runtime("facebook/m2m100_418M")
    rt.translate(["hello"], source="en", target="es")
    _, kwargs = fake_model.generate.call_args
    assert kwargs["num_beams"] == 4


def test_max_new_tokens_derived_from_longest_input():
    # input_ids.shape[-1] == 8 (from _fake_tensor default) ->
    # max_new_tokens = min(1024, 2 * 8 + 16) == 32
    rt, fake_tok, fake_model = _build_runtime("facebook/m2m100_418M")
    rt.translate(["hello"], source="en", target="es")
    _, kwargs = fake_model.generate.call_args
    assert kwargs["max_new_tokens"] == 32


def test_max_new_tokens_capped_at_1024():
    fake_tok = MagicMock()
    fake_tok.return_value = {
        "input_ids": _fake_tensor(600),
        "attention_mask": _fake_tensor(600),
    }
    fake_tok.decode.side_effect = lambda ids, **kw: f"decoded-{ids}"
    fake_tok.get_lang_id.return_value = 1
    fake_tok_class = MagicMock()
    fake_tok_class.from_pretrained.return_value = fake_tok

    fake_model = MagicMock()
    fake_model.to.return_value = fake_model
    fake_model.generate.return_value = ["out-0"]
    fake_model_class = MagicMock()
    fake_model_class.from_pretrained.return_value = fake_model

    fake_torch = _fake_torch()

    with patch.object(hf_mod, "AutoModelForSeq2SeqLM", fake_model_class), \
            patch.object(hf_mod, "AutoTokenizer", fake_tok_class), \
            patch.object(hf_mod, "torch", fake_torch):
        rt = TranslationRuntime(
            model_id="test", hf_repo="facebook/m2m100_418M",
            local_dir=None, device="cpu",
        )
    rt.translate(["hello"], source="en", target="es")
    _, kwargs = fake_model.generate.call_args
    assert kwargs["max_new_tokens"] == 1024


def test_translate_returns_translation_result():
    rt, fake_tok, fake_model = _build_runtime("facebook/m2m100_418M")
    out = rt.translate(["hello"], source="en", target="es")
    assert isinstance(out, TranslationResult)


def test_decode_called_with_skip_special_tokens():
    rt, fake_tok, fake_model = _build_runtime("facebook/m2m100_418M")
    rt.translate(["hello"], source="en", target="es")
    _, dec_kwargs = fake_tok.decode.call_args
    assert dec_kwargs.get("skip_special_tokens") is True


# ---------------------------------------------------------------------------
# Deferred imports
# ---------------------------------------------------------------------------


def test_runtime_raises_when_transformers_missing():
    with patch.object(hf_mod, "AutoModelForSeq2SeqLM", None), \
            patch.object(hf_mod, "AutoTokenizer", None), \
            patch.object(hf_mod, "_ensure_deps", lambda: None), \
            patch.object(hf_mod, "torch", MagicMock()):
        with pytest.raises(RuntimeError, match="transformers"):
            TranslationRuntime(
                model_id="m", hf_repo="facebook/m2m100_418M", local_dir=None,
            )


def test_select_device_auto_falls_back_to_cpu_with_no_torch():
    with patch.object(hf_mod, "torch", None):
        from muse.modalities.text_translation.runtimes.hf_translation import (
            _select_device,
        )
        assert _select_device("auto") == "cpu"


def test_select_device_auto_picks_cuda():
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = True
    with patch.object(hf_mod, "torch", fake_torch):
        from muse.modalities.text_translation.runtimes.hf_translation import (
            _select_device,
        )
        assert _select_device("auto") == "cuda"


def test_set_inference_mode_calls_eval_when_method_present():
    from muse.modalities.text_translation.runtimes.hf_translation import (
        _set_inference_mode,
    )
    m = MagicMock()
    _set_inference_mode(m)
    m.eval.assert_called_once()


# ---------------------------------------------------------------------------
# Q1: padded batches must carry the attention mask into generate()
# ---------------------------------------------------------------------------


def test_generate_passes_device_moved_attention_mask_to_model_generate():
    """A padded batch's attention_mask must reach model.generate(...),
    device-moved the same way input_ids is. Uses the shared _build_runtime
    harness (via tokenizer_extra) with batch-sized (batch=2) input_ids +
    attention_mask, rather than a bespoke one-off fake tokenizer, so this
    coverage stays wired to the shared fake (Q4)."""
    fake_input_ids = _fake_tensor(8, batch=2)
    fake_attention_mask = _fake_tensor(8, batch=2)

    def extra(tok):
        tok.return_value = {
            "input_ids": fake_input_ids,
            "attention_mask": fake_attention_mask,
        }
        tok.get_lang_id.return_value = 1

    rt, fake_tok, fake_model = _build_runtime(
        "facebook/m2m100_418M",
        tokenizer_extra=extra,
        generate_return=["out-0", "out-1"],
    )
    rt.translate(["hello", "world"], source="en", target="es")

    _, kwargs = fake_model.generate.call_args
    assert kwargs["attention_mask"] is fake_attention_mask
    fake_attention_mask.to.assert_called_once_with("cpu")
    fake_input_ids.to.assert_called_once_with("cpu")


# ---------------------------------------------------------------------------
# Q2: empty input handling
# ---------------------------------------------------------------------------


def test_translate_empty_list_returns_empty_result_without_touching_tokenizer():
    rt, fake_tok, fake_model = _build_runtime("facebook/m2m100_418M")
    fake_tok.reset_mock()
    fake_model.reset_mock()

    out = rt.translate([], source="en", target="es")

    assert out == TranslationResult(texts=[])
    fake_tok.assert_not_called()
    fake_model.generate.assert_not_called()


def test_translate_single_empty_string_flows_through_normally():
    rt, fake_tok, fake_model = _build_runtime(
        "facebook/m2m100_418M",
        generate_return=["out-0"],
        decode_side_effect=lambda ids, **kw: f"decoded-{ids}",
    )
    out = rt.translate([""], source="en", target="es")
    assert out.texts == ["decoded-out-0"]
    fake_tok.assert_called_once()
