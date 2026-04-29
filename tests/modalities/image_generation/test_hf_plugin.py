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


def test_resolve_advertises_supports_img2img():
    """Resolver-pulled diffusers models advertise img2img support by default."""
    info = _fake_info(siblings=["model_index.json"], tags=["text-to-image"])
    result = HF_PLUGIN["resolve"]("org/anything", None, info)
    assert result.manifest["capabilities"]["supports_img2img"] is True


def test_resolve_advertises_supports_inpainting_and_variations():
    """Resolver-pulled diffusers models advertise inpainting + variations by default.

    Both flags default True since AutoPipelineForInpainting and
    AutoPipelineForImage2Image work on essentially any diffusers t2i
    checkpoint via from_pipe.
    """
    info = _fake_info(siblings=["model_index.json"], tags=["text-to-image"])
    result = HF_PLUGIN["resolve"]("org/anything", None, info)
    caps = result.manifest["capabilities"]
    assert caps["supports_inpainting"] is True
    assert caps["supports_variations"] is True


def test_search_yields_results_with_modality_tag():
    fake_api = MagicMock()
    fake_repo = MagicMock(id="org/repo", downloads=42)
    fake_api.list_models.return_value = [fake_repo]
    rows = list(HF_PLUGIN["search"](fake_api, "sdxl", sort="downloads", limit=20))
    assert len(rows) == 1
    assert rows[0].modality == "image/generation"


def test_resolve_download_filters_to_fp16_when_variants_exist():
    """When info shows fp16 variants, _download requests fp16-only patterns."""
    from pathlib import Path
    from unittest.mock import patch

    info = _fake_info(
        siblings=[
            "model_index.json",
            "unet/diffusion_pytorch_model.safetensors",
            "unet/diffusion_pytorch_model.fp16.safetensors",
            "sd_xl_turbo_1.0.safetensors",  # top-level standalone, should be excluded
        ],
        tags=["text-to-image"],
    )
    with patch(
        "muse.modalities.image_generation.hf.snapshot_download",
        return_value="/tmp/fake",
    ) as fake_snapshot:
        result = HF_PLUGIN["resolve"]("stabilityai/sdxl-turbo", None, info)
        result.download(Path("/tmp/cache"))

    patterns = fake_snapshot.call_args.kwargs["allow_patterns"]
    assert "*/*.fp16.safetensors" in patterns
    assert "*/*.fp16.bin" in patterns
    # Bare safetensors patterns must NOT be present (they would also match fp32)
    assert "*/*.safetensors" not in patterns
    assert "*/*.bin" not in patterns
    # Configs and tokenizer assets always allowed
    assert "model_index.json" in patterns
    assert "*/*.json" in patterns


def test_resolve_download_uses_bare_pattern_when_no_fp16_variant():
    """FLUX/SD3 repos ship single-precision weights; download must include them."""
    from pathlib import Path
    from unittest.mock import patch

    info = _fake_info(
        siblings=[
            "model_index.json",
            "transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
            "vae/diffusion_pytorch_model.safetensors",
        ],
        tags=["text-to-image"],
    )
    with patch(
        "muse.modalities.image_generation.hf.snapshot_download",
        return_value="/tmp/fake",
    ) as fake_snapshot:
        result = HF_PLUGIN["resolve"](
            "black-forest-labs/FLUX.1-schnell", None, info,
        )
        result.download(Path("/tmp/cache"))

    patterns = fake_snapshot.call_args.kwargs["allow_patterns"]
    assert "*/*.safetensors" in patterns
    assert "*/*.bin" in patterns
    # No fp16-only patterns when the repo doesn't have variants
    assert "*/*.fp16.safetensors" not in patterns
    assert "model_index.json" in patterns
