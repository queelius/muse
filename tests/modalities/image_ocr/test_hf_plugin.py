"""Tests for the image_ocr HF plugin."""
from unittest.mock import MagicMock

from muse.modalities.image_ocr.hf import HF_PLUGIN
from muse.core.discovery import REQUIRED_HF_PLUGIN_KEYS
from muse.core.resolvers import ResolvedModel


def _fake_info(siblings=None, tags=None, repo_id="org/repo"):
    info = MagicMock()
    info.siblings = [MagicMock(rfilename=f) for f in (siblings or [])]
    info.tags = tags or []
    info.card_data = MagicMock(license=None)
    info.id = repo_id
    return info


def test_plugin_has_all_required_keys():
    for key in REQUIRED_HF_PLUGIN_KEYS:
        assert key in HF_PLUGIN


def test_plugin_metadata_correct():
    assert HF_PLUGIN["modality"] == "image/ocr"
    assert HF_PLUGIN["runtime_path"].endswith(":HFVision2SeqRuntime")
    assert HF_PLUGIN["priority"] == 110


def test_sniff_true_on_image_to_text_tag():
    info = _fake_info(tags=["image-to-text"])
    assert HF_PLUGIN["sniff"](info) is True


def test_sniff_false_on_image_text_to_text_only():
    """VLMs that ONLY carry image-text-to-text (no image-to-text)
    belong to the future #97 modality. Defer them."""
    info = _fake_info(tags=["image-text-to-text"])
    assert HF_PLUGIN["sniff"](info) is False


def test_sniff_true_when_both_image_tags_present():
    """Vision-encoder-decoder OCR repos (TexTeller, some Nougat
    variants) carry BOTH image-to-text AND image-text-to-text tags.
    The image-to-text presence proves an OCR-shape mode; claim it
    here. Pure VLMs (Llava, Qwen-VL) only have image-text-to-text.
    """
    info = _fake_info(tags=["image-to-text", "image-text-to-text"])
    assert HF_PLUGIN["sniff"](info) is True


def test_sniff_true_on_texteller_name_with_vlm_tag():
    """Regression watchdog: TexTeller's HF repo carries the
    image-text-to-text tag for SDK-routing reasons, but it's an
    OCR model. Repo-name allowlist beats the tag check.
    """
    info = _fake_info(
        tags=["image-text-to-text", "vision-encoder-decoder"],
        repo_id="OleehyO/TexTeller",
    )
    assert HF_PLUGIN["sniff"](info) is True


def test_sniff_false_on_random_tag():
    info = _fake_info(tags=["text-generation"])
    assert HF_PLUGIN["sniff"](info) is False


def test_sniff_repo_name_fallback_trocr():
    """Some checkpoints ship without the canonical tag; repo-name
    fallback catches them."""
    info = _fake_info(tags=[], repo_id="custom-org/my-trocr-finetune")
    assert HF_PLUGIN["sniff"](info) is True


def test_sniff_repo_name_fallback_nougat():
    info = _fake_info(tags=[], repo_id="some-org/nougat-base-finetune")
    assert HF_PLUGIN["sniff"](info) is True


def test_sniff_repo_name_fallback_texteller():
    info = _fake_info(tags=[], repo_id="OleehyO/TexTeller")
    assert HF_PLUGIN["sniff"](info) is True


def test_resolve_sets_supports_handwritten_for_handwritten_repo():
    info = _fake_info(
        tags=["image-to-text"],
        repo_id="microsoft/trocr-large-handwritten",
    )
    result = HF_PLUGIN["resolve"](
        "microsoft/trocr-large-handwritten", None, info,
    )
    assert isinstance(result, ResolvedModel)
    caps = result.manifest["capabilities"]
    assert caps.get("supports_handwritten") is True


def test_resolve_sets_supports_math_for_nougat():
    info = _fake_info(
        tags=["image-to-text"],
        repo_id="facebook/nougat-base",
    )
    result = HF_PLUGIN["resolve"]("facebook/nougat-base", None, info)
    assert result.manifest["capabilities"].get("supports_math") is True


def test_resolve_sets_supports_math_for_texteller():
    info = _fake_info(
        tags=["image-to-text"],
        repo_id="OleehyO/TexTeller",
    )
    result = HF_PLUGIN["resolve"]("OleehyO/TexTeller", None, info)
    assert result.manifest["capabilities"].get("supports_math") is True


def test_resolve_no_capability_flags_for_plain_trocr():
    """Plain TrOCR (printed) gets no advisory flags. The route doesn't
    enforce them; clients filter via /v1/models."""
    info = _fake_info(
        tags=["image-to-text"],
        repo_id="microsoft/trocr-base-printed",
    )
    result = HF_PLUGIN["resolve"](
        "microsoft/trocr-base-printed", None, info,
    )
    caps = result.manifest["capabilities"]
    assert caps.get("supports_handwritten") is None
    assert caps.get("supports_math") is None
    assert caps["device"] == "auto"


def test_resolve_routes_to_vision2seq_runtime():
    info = _fake_info(tags=["image-to-text"], repo_id="microsoft/trocr")
    result = HF_PLUGIN["resolve"]("microsoft/trocr", None, info)
    assert "HFVision2SeqRuntime" in result.backend_path
    assert result.manifest["modality"] == "image/ocr"


def test_resolve_model_id_kebab_case():
    info = _fake_info(tags=["image-to-text"], repo_id="microsoft/TrOCR-Base-Printed")
    result = HF_PLUGIN["resolve"](
        "microsoft/TrOCR-Base-Printed", None, info,
    )
    assert result.manifest["model_id"] == "trocr-base-printed"


def test_search_yields_results():
    fake_api = MagicMock()
    fake_repo = MagicMock(id="org/some-trocr", downloads=42)
    fake_api.list_models.return_value = [fake_repo]
    rows = list(HF_PLUGIN["search"](
        fake_api, "trocr", sort="downloads", limit=20,
    ))
    assert len(rows) == 1
    assert rows[0].modality == "image/ocr"
    assert rows[0].uri == "hf://org/some-trocr"


def test_pip_extras_includes_pillow():
    """Pillow must be in pip_extras so a fresh-venv pull-and-load
    works without ImportError."""
    extras = HF_PLUGIN["pip_extras"]
    assert any("Pillow" in e for e in extras)
