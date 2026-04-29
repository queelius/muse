"""Tests for the image_segmentation HF resolver plugin."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from muse.modalities.image_segmentation.hf import HF_PLUGIN


def _fake_info(*, repo_id="x", siblings=(), tags=(), card_data=None):
    return SimpleNamespace(
        id=repo_id,
        siblings=[SimpleNamespace(rfilename=f) for f in siblings],
        tags=list(tags),
        card_data=card_data,
    )


# ---------------- plugin metadata ----------------


def test_priority_is_110():
    assert HF_PLUGIN["priority"] == 110


def test_modality_is_image_segmentation():
    assert HF_PLUGIN["modality"] == "image/segmentation"


def test_runtime_path_points_at_sam2_runtime():
    assert HF_PLUGIN["runtime_path"] == (
        "muse.modalities.image_segmentation.runtimes.sam2_runtime"
        ":SAM2Runtime"
    )


def test_pip_extras_includes_torch_transformers_pillow_numpy():
    extras = " ".join(HF_PLUGIN["pip_extras"])
    assert "torch" in extras
    assert "transformers" in extras
    assert "Pillow" in extras
    assert "numpy" in extras


# ---------------- _sniff ----------------


def test_sniff_mask_generation_tag():
    info = _fake_info(
        repo_id="facebook/sam2-hiera-tiny",
        tags=["mask-generation"],
    )
    assert HF_PLUGIN["sniff"](info) is True


def test_sniff_image_segmentation_tag():
    info = _fake_info(
        repo_id="CIDAS/clipseg-rd64-refined",
        tags=["image-segmentation"],
    )
    assert HF_PLUGIN["sniff"](info) is True


def test_sniff_rejects_unrelated_repo():
    info = _fake_info(
        repo_id="runwayml/stable-diffusion-v1-5",
        tags=["text-to-image", "diffusers"],
    )
    assert HF_PLUGIN["sniff"](info) is False


def test_sniff_rejects_repo_with_no_tags():
    info = _fake_info(repo_id="user/empty", tags=[])
    assert HF_PLUGIN["sniff"](info) is False


def test_sniff_handles_missing_tags_attr():
    info = SimpleNamespace(id="user/x", siblings=[])
    # No `tags` attr; sniff must return False, not crash.
    assert HF_PLUGIN["sniff"](info) is False


# ---------------- _resolve: SAM-2 family ----------------


def test_resolve_sam2_hiera_tiny_capabilities():
    info = _fake_info(
        repo_id="facebook/sam2-hiera-tiny",
        tags=["mask-generation"],
    )
    resolved = HF_PLUGIN["resolve"](
        "facebook/sam2-hiera-tiny", None, info,
    )
    caps = resolved.manifest["capabilities"]
    assert caps["max_masks"] == 64
    assert caps["memory_gb"] == 0.8
    assert caps["supports_text_prompts"] is False
    assert caps["supports_point_prompts"] is True
    assert caps["supports_box_prompts"] is True
    assert caps["supports_automatic"] is True


def test_resolve_sam2_hiera_base_plus_capabilities():
    info = _fake_info(
        repo_id="facebook/sam2-hiera-base-plus",
        tags=["mask-generation"],
    )
    resolved = HF_PLUGIN["resolve"](
        "facebook/sam2-hiera-base-plus", None, info,
    )
    caps = resolved.manifest["capabilities"]
    assert caps["max_masks"] == 64
    assert caps["memory_gb"] == 1.5


def test_resolve_sam2_hiera_large_capabilities():
    info = _fake_info(
        repo_id="facebook/sam2-hiera-large",
        tags=["mask-generation"],
    )
    resolved = HF_PLUGIN["resolve"](
        "facebook/sam2-hiera-large", None, info,
    )
    caps = resolved.manifest["capabilities"]
    assert caps["max_masks"] == 64
    assert caps["memory_gb"] == 2.5


def test_resolve_original_sam_family():
    info = _fake_info(
        repo_id="facebook/sam-vit-base",
        tags=["mask-generation"],
    )
    resolved = HF_PLUGIN["resolve"](
        "facebook/sam-vit-base", None, info,
    )
    caps = resolved.manifest["capabilities"]
    assert caps["supports_point_prompts"] is True
    assert caps["supports_text_prompts"] is False


def test_resolve_clipseg_flips_capability_flags():
    info = _fake_info(
        repo_id="CIDAS/clipseg-rd64-refined",
        tags=["image-segmentation"],
    )
    resolved = HF_PLUGIN["resolve"](
        "CIDAS/clipseg-rd64-refined", None, info,
    )
    caps = resolved.manifest["capabilities"]
    assert caps["supports_text_prompts"] is True
    assert caps["supports_point_prompts"] is False
    assert caps["supports_box_prompts"] is False
    assert caps["supports_automatic"] is False


def test_resolve_fallback_capabilities():
    info = _fake_info(
        repo_id="user/some-other-segmenter",
        tags=["mask-generation"],
    )
    resolved = HF_PLUGIN["resolve"](
        "user/some-other-segmenter", None, info,
    )
    caps = resolved.manifest["capabilities"]
    assert caps["max_masks"] == 16
    assert caps["supports_point_prompts"] is True
    assert caps["supports_box_prompts"] is True
    assert caps["supports_automatic"] is True
    assert caps["supports_text_prompts"] is False


# ---------------- _resolve: manifest shape ----------------


def test_resolve_modality_is_image_segmentation():
    info = _fake_info(
        repo_id="facebook/sam2-hiera-tiny",
        tags=["mask-generation"],
    )
    resolved = HF_PLUGIN["resolve"](
        "facebook/sam2-hiera-tiny", None, info,
    )
    assert resolved.manifest["modality"] == "image/segmentation"


def test_resolve_backend_path_points_at_sam2_runtime():
    info = _fake_info(
        repo_id="facebook/sam2-hiera-tiny",
        tags=["mask-generation"],
    )
    resolved = HF_PLUGIN["resolve"](
        "facebook/sam2-hiera-tiny", None, info,
    )
    assert resolved.backend_path == (
        "muse.modalities.image_segmentation.runtimes.sam2_runtime"
        ":SAM2Runtime"
    )


def test_resolve_pip_extras_includes_transformers_torch():
    info = _fake_info(
        repo_id="facebook/sam2-hiera-tiny",
        tags=["mask-generation"],
    )
    resolved = HF_PLUGIN["resolve"](
        "facebook/sam2-hiera-tiny", None, info,
    )
    extras = " ".join(resolved.manifest["pip_extras"])
    assert "transformers" in extras
    assert "torch" in extras
    assert "Pillow" in extras
    assert "numpy" in extras


def test_resolve_synthesizes_lowercase_model_id():
    info = _fake_info(
        repo_id="Facebook/SAM2-Hiera-Tiny",
        tags=["mask-generation"],
    )
    resolved = HF_PLUGIN["resolve"](
        "Facebook/SAM2-Hiera-Tiny", None, info,
    )
    assert resolved.manifest["model_id"] == "sam2-hiera-tiny"


def test_resolve_capabilities_default_device_auto():
    info = _fake_info(
        repo_id="facebook/sam2-hiera-tiny",
        tags=["mask-generation"],
    )
    resolved = HF_PLUGIN["resolve"](
        "facebook/sam2-hiera-tiny", None, info,
    )
    caps = resolved.manifest["capabilities"]
    assert caps["device"] == "auto"


def test_resolve_carries_card_license():
    info = _fake_info(
        repo_id="facebook/sam2-hiera-tiny",
        tags=["mask-generation"],
        card_data=SimpleNamespace(license="apache-2.0"),
    )
    resolved = HF_PLUGIN["resolve"](
        "facebook/sam2-hiera-tiny", None, info,
    )
    assert resolved.manifest["license"] == "apache-2.0"


# ---------------- _search ----------------


def test_search_yields_uri_and_modality():
    api = MagicMock()
    api.list_models.return_value = [
        SimpleNamespace(id="facebook/sam2-hiera-tiny", downloads=1000),
        SimpleNamespace(id="facebook/sam2-hiera-large", downloads=500),
    ]
    results = list(HF_PLUGIN["search"](api, "sam2", sort="downloads", limit=10))
    uris = [r.uri for r in results]
    assert "hf://facebook/sam2-hiera-tiny" in uris
    assert "hf://facebook/sam2-hiera-large" in uris


def test_search_calls_list_models_with_mask_generation_filter():
    api = MagicMock()
    api.list_models.return_value = []
    list(HF_PLUGIN["search"](api, "sam", sort="downloads", limit=5))
    api.list_models.assert_called_once()
    _, kw = api.list_models.call_args
    assert kw.get("filter") == "mask-generation"


def test_search_results_carry_modality():
    api = MagicMock()
    api.list_models.return_value = [
        SimpleNamespace(id="facebook/sam2-hiera-tiny", downloads=1000),
    ]
    results = list(HF_PLUGIN["search"](api, "sam2", sort="downloads", limit=10))
    assert all(r.modality == "image/segmentation" for r in results)
