"""Tests for the image_upscale HF resolver plugin."""
from types import SimpleNamespace
from unittest.mock import MagicMock

from muse.modalities.image_upscale.hf import HF_PLUGIN


def _fake_info(*, repo_id="x", siblings=(), tags=()):
    return SimpleNamespace(
        id=repo_id,
        siblings=[SimpleNamespace(rfilename=f) for f in siblings],
        tags=list(tags),
        card_data=None,
    )


def test_priority_is_105():
    """Priority 105: between image/animation (110) and image/generation (100)."""
    assert HF_PLUGIN["priority"] == 105


def test_modality_is_image_upscale():
    assert HF_PLUGIN["modality"] == "image/upscale"


def test_runtime_path_points_at_diffusers_upscale_runtime():
    assert HF_PLUGIN["runtime_path"] == (
        "muse.modalities.image_upscale.runtimes.diffusers_upscaler"
        ":DiffusersUpscaleRuntime"
    )


def test_pip_extras_includes_diffusers_torch_transformers():
    extras = " ".join(HF_PLUGIN["pip_extras"])
    assert "diffusers" in extras
    assert "torch" in extras
    assert "transformers" in extras


def test_sniff_x4_upscaler():
    info = _fake_info(
        repo_id="stabilityai/stable-diffusion-x4-upscaler",
        siblings=[
            "model_index.json",
            "unet/diffusion_pytorch_model.safetensors",
        ],
        tags=["image-to-image", "diffusers"],
    )
    assert HF_PLUGIN["sniff"](info) is True


def test_sniff_rejects_non_upscaler_diffusers_repo():
    """A regular SD checkpoint with model_index.json + i2i tag is NOT an upscaler."""
    info = _fake_info(
        repo_id="runwayml/stable-diffusion-v1-5",
        siblings=["model_index.json"],
        tags=["image-to-image", "diffusers"],
    )
    assert HF_PLUGIN["sniff"](info) is False


def test_sniff_rejects_repo_without_model_index():
    info = _fake_info(
        repo_id="user/some-upscaler",
        siblings=["other.txt"],
        tags=["image-to-image"],
    )
    assert HF_PLUGIN["sniff"](info) is False


def test_sniff_rejects_repo_without_i2i_tag():
    info = _fake_info(
        repo_id="stabilityai/stable-diffusion-x4-upscaler",
        siblings=["model_index.json"],
        tags=["text-to-image"],
    )
    assert HF_PLUGIN["sniff"](info) is False


def test_sniff_aura_currently_excluded():
    """AuraSR repos don't ship model_index.json (custom torch module).
    Even if they did, our v0.25.0 plugin doesn't carry an Aura runtime."""
    info = _fake_info(
        repo_id="fal/AuraSR-v2",
        siblings=["aura_sr.py", "model.safetensors"],
        tags=["image-to-image"],
    )
    # No model_index.json -> not claimed.
    assert HF_PLUGIN["sniff"](info) is False


def test_sniff_super_resolution_name():
    info = _fake_info(
        repo_id="acme/super-resolution-pipeline",
        siblings=["model_index.json"],
        tags=["image-to-image"],
    )
    assert HF_PLUGIN["sniff"](info) is True


def test_sniff_ldm_super_name():
    info = _fake_info(
        repo_id="acme/ldm-super-resolution",
        siblings=["model_index.json"],
        tags=["image-to-image"],
    )
    assert HF_PLUGIN["sniff"](info) is True


def test_resolve_x4_upscaler_capabilities():
    info = _fake_info(
        repo_id="stabilityai/stable-diffusion-x4-upscaler",
        siblings=["model_index.json"],
        tags=["image-to-image"],
    )
    resolved = HF_PLUGIN["resolve"](
        "stabilityai/stable-diffusion-x4-upscaler", None, info,
    )
    caps = resolved.manifest["capabilities"]
    assert caps["default_scale"] == 4
    assert caps["supported_scales"] == [4]
    assert caps["default_steps"] == 20
    assert caps["default_guidance"] == 9.0
    assert caps["device"] == "cuda"


def test_resolve_fallback_capabilities():
    info = _fake_info(
        repo_id="user/some-other-upscaler",
        siblings=["model_index.json"],
        tags=["image-to-image"],
    )
    resolved = HF_PLUGIN["resolve"](
        "user/some-other-upscaler", None, info,
    )
    caps = resolved.manifest["capabilities"]
    assert caps["default_scale"] == 4
    assert caps["supported_scales"] == [4]


def test_resolve_modality_is_image_upscale():
    info = _fake_info(
        repo_id="stabilityai/stable-diffusion-x4-upscaler",
        siblings=["model_index.json"],
        tags=["image-to-image"],
    )
    resolved = HF_PLUGIN["resolve"](
        "stabilityai/stable-diffusion-x4-upscaler", None, info,
    )
    assert resolved.manifest["modality"] == "image/upscale"


def test_resolve_backend_path_points_at_diffusers_upscaler():
    info = _fake_info(
        repo_id="stabilityai/stable-diffusion-x4-upscaler",
        siblings=["model_index.json"],
        tags=["image-to-image"],
    )
    resolved = HF_PLUGIN["resolve"](
        "stabilityai/stable-diffusion-x4-upscaler", None, info,
    )
    assert resolved.backend_path == (
        "muse.modalities.image_upscale.runtimes.diffusers_upscaler"
        ":DiffusersUpscaleRuntime"
    )


def test_resolve_pip_extras_includes_diffusers_and_transformers():
    info = _fake_info(
        repo_id="stabilityai/stable-diffusion-x4-upscaler",
        siblings=["model_index.json"],
        tags=["image-to-image"],
    )
    resolved = HF_PLUGIN["resolve"](
        "stabilityai/stable-diffusion-x4-upscaler", None, info,
    )
    extras = " ".join(resolved.manifest["pip_extras"])
    assert "diffusers" in extras
    assert "transformers" in extras
    assert "torch" in extras


def test_resolve_synthesizes_model_id():
    info = _fake_info(
        repo_id="stabilityai/stable-diffusion-x4-upscaler",
        siblings=["model_index.json"],
        tags=["image-to-image"],
    )
    resolved = HF_PLUGIN["resolve"](
        "stabilityai/stable-diffusion-x4-upscaler", None, info,
    )
    assert resolved.manifest["model_id"] == "stable-diffusion-x4-upscaler"


def test_search_yields_only_upscaler_results():
    api = MagicMock()
    api.list_models.return_value = [
        SimpleNamespace(
            id="stabilityai/stable-diffusion-x4-upscaler", downloads=1000,
        ),
        # NOT an upscaler; gets filtered out.
        SimpleNamespace(
            id="runwayml/stable-diffusion-v1-5", downloads=2000,
        ),
        SimpleNamespace(
            id="acme/ldm-super-resolution", downloads=50,
        ),
    ]
    results = list(HF_PLUGIN["search"](api, "upscaler", sort="downloads", limit=10))
    uris = [r.uri for r in results]
    assert any("stable-diffusion-x4-upscaler" in u for u in uris)
    assert any("ldm-super-resolution" in u for u in uris)
    assert not any("stable-diffusion-v1-5" in u for u in uris)


def test_search_results_carry_modality():
    api = MagicMock()
    api.list_models.return_value = [
        SimpleNamespace(
            id="stabilityai/stable-diffusion-x4-upscaler", downloads=1000,
        ),
    ]
    results = list(HF_PLUGIN["search"](api, "x4", sort="downloads", limit=10))
    assert all(r.modality == "image/upscale" for r in results)
