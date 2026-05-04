"""Tests for encode_ocr (the /v1/images/ocr response builder)."""
from muse.modalities.image_ocr.codec import encode_ocr
from muse.modalities.image_ocr.protocol import OcrResult


def test_encode_ocr_envelope_shape():
    result = OcrResult(
        text="hello world",
        model_id="trocr-base-printed",
        completion_tokens=4,
    )
    body = encode_ocr(result)
    assert body["model"] == "trocr-base-printed"
    assert body["text"] == "hello world"
    assert body["usage"]["completion_tokens"] == 4
    assert body["id"].startswith("ocr-")


def test_encode_ocr_id_unique_per_call():
    """Per-call uuid keeps logs and traces correlatable. Two calls with
    identical results must NOT collide on id."""
    a = encode_ocr(OcrResult(text="x", model_id="m"))
    b = encode_ocr(OcrResult(text="x", model_id="m"))
    assert a["id"] != b["id"]


def test_encode_ocr_zero_tokens_default():
    body = encode_ocr(OcrResult(text="x", model_id="m"))
    assert body["usage"]["completion_tokens"] == 0


def test_encode_ocr_preserves_text_with_special_chars():
    """LaTeX output (Nougat) contains backslashes, braces, etc.
    The codec must surface them as-is."""
    latex = r"\\begin{equation} \\frac{1}{2} \\end{equation}"
    body = encode_ocr(OcrResult(text=latex, model_id="nougat-base"))
    assert body["text"] == latex


def test_encode_ocr_empty_text():
    body = encode_ocr(OcrResult(text="", model_id="m"))
    assert body["text"] == ""
