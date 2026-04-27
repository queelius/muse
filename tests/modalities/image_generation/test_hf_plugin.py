"""Tests for the image_generation HF plugin (diffusers text-to-image)."""
from unittest.mock import MagicMock

from muse.modalities.image_generation.hf import HF_PLUGIN
from muse.core.discovery import REQUIRED_HF_PLUGIN_KEYS
from muse.core.resolvers import ResolvedModel


def _fake_info(siblings=None, tags=None):
    info = MagicMock()
    info.siblings = [MagicMock(rfilename=f) for f in (siblings or [])]
    info.tags = tags or []
    info.card_data = MagicMock(license=None)
    return info


def test_plugin_has_all_required_keys():
    for key in REQUIRED_HF_PLUGIN_KEYS:
        assert key in HF_PLUGIN


def test_plugin_metadata_correct():
    assert HF_PLUGIN["modality"] == "image/generation"
    assert HF_PLUGIN["runtime_path"].endswith(":DiffusersText2ImageModel")
    assert HF_PLUGIN["priority"] == 100


def test_sniff_true_on_diffusers_text_to_image_repo():
    info = _fake_info(
        siblings=["model_index.json", "unet/diffusion_pytorch_model.safetensors"],
        tags=["text-to-image", "diffusers"],
    )
    assert HF_PLUGIN["sniff"](info) is True


def test_sniff_false_without_text_to_image_tag():
    info = _fake_info(
        siblings=["model_index.json"],
        tags=["image-to-image"],
    )
    assert HF_PLUGIN["sniff"](info) is False


def test_sniff_false_without_model_index_json():
    info = _fake_info(
        siblings=["model.safetensors"],
        tags=["text-to-image"],
    )
    assert HF_PLUGIN["sniff"](info) is False


def test_resolve_returns_resolved_model_with_turbo_defaults():
    """Repo name containing 'turbo' should default steps=1, guidance=0."""
    info = _fake_info(
        siblings=["model_index.json"],
        tags=["text-to-image"],
    )
    result = HF_PLUGIN["resolve"]("stabilityai/sdxl-turbo", None, info)
    assert isinstance(result, ResolvedModel)
    caps = result.manifest["capabilities"]
    assert caps["default_steps"] == 1
    assert caps["default_guidance"] == 0.0


def test_resolve_flux_schnell_defaults():
    info = _fake_info(siblings=["model_index.json"], tags=["text-to-image"])
    result = HF_PLUGIN["resolve"]("black-forest-labs/FLUX.1-schnell", None, info)
    caps = result.manifest["capabilities"]
    assert caps["default_steps"] == 4
    assert caps["default_size"] == [1024, 1024]


def test_resolve_default_fallback():
    """Non-turbo, non-flux gets the conservative fallback."""
    info = _fake_info(siblings=["model_index.json"], tags=["text-to-image"])
    result = HF_PLUGIN["resolve"]("random-org/random-sd", None, info)
    caps = result.manifest["capabilities"]
    assert caps["default_steps"] == 25
    assert caps["default_guidance"] == 7.5


def test_search_yields_results_with_modality_tag():
    fake_api = MagicMock()
    fake_repo = MagicMock(id="org/repo", downloads=42)
    fake_api.list_models.return_value = [fake_repo]
    rows = list(HF_PLUGIN["search"](fake_api, "sdxl", sort="downloads", limit=20))
    assert len(rows) == 1
    assert rows[0].modality == "image/generation"
