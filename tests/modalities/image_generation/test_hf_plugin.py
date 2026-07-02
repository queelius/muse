"""Tests for the image_generation HF plugin (diffusers text-to-image)."""
from unittest.mock import MagicMock

from muse.modalities.image_generation.hf import HF_PLUGIN
from muse.core.discovery import REQUIRED_HF_PLUGIN_KEYS
from muse.core.resolvers import ResolvedModel


def _fake_info(siblings=None, tags=None, pipeline_tag=None):
    info = MagicMock()
    info.siblings = [MagicMock(rfilename=f) for f in (siblings or [])]
    info.tags = tags or []
    # Set explicitly so getattr doesn't auto-vivify a truthy MagicMock;
    # HF's canonical text-to-image signal lives here, not always in `tags`.
    info.pipeline_tag = pipeline_tag
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


def test_sniff_true_via_pipeline_tag_when_absent_from_tags_list():
    """Canonical `pipeline_tag` is the authoritative text-to-image signal.

    Many community SD checkpoints (e.g. PublicPrompts/All-In-One-Pixel-Model)
    set `pipeline_tag="text-to-image"` but do NOT mirror it into the loose
    `tags` bag (which only carries `diffusers:StableDiffusionPipeline`).
    The sniff must read the structured field, not just the redundant mirror.
    """
    info = _fake_info(
        siblings=[
            "model_index.json",
            "unet/diffusion_pytorch_model.bin",
            "vae/diffusion_pytorch_model.bin",
        ],
        tags=["diffusers", "diffusers:StableDiffusionPipeline", "region:us"],
        pipeline_tag="text-to-image",
    )
    assert HF_PLUGIN["sniff"](info) is True


def test_sniff_false_when_pipeline_tag_is_unrelated():
    """A repo whose pipeline_tag is something else (and no t2i tag) is rejected."""
    info = _fake_info(
        siblings=["model_index.json"],
        tags=["diffusers", "diffusers:StableDiffusionInpaintPipeline"],
        pipeline_tag="image-to-image",
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


def test_sniff_true_on_lora_adapter_repo():
    """Real shape of nerijs/pixel-art-xl (verified 2026-07-02): no
    model_index.json, one top-level safetensors, lora + base_model tags."""
    info = _fake_info(
        siblings=[".gitattributes", "README.md", "pixel-art-xl.safetensors"],
        tags=[
            "diffusers", "text-to-image", "stable-diffusion", "lora",
            "base_model:stabilityai/stable-diffusion-xl-base-1.0",
            "base_model:adapter:stabilityai/stable-diffusion-xl-base-1.0",
        ],
        pipeline_tag="text-to-image",
    )
    assert HF_PLUGIN["sniff"](info) is True


def test_sniff_false_on_safetensors_without_lora_signal():
    """A bare safetensors artifact repo (no lora tag, no adapter tag) is
    NOT claimed, even when tagged text-to-image."""
    info = _fake_info(
        siblings=["model.safetensors"],
        tags=["text-to-image", "diffusers"],
        pipeline_tag="text-to-image",
    )
    assert HF_PLUGIN["sniff"](info) is False


def test_sniff_false_on_lora_repo_without_text_to_image_signal():
    """A LoRA for some non-t2i pipeline is not ours."""
    info = _fake_info(
        siblings=["adapter.safetensors"],
        tags=["lora", "base_model:adapter:some/llm"],
        pipeline_tag="text-generation",
    )
    assert HF_PLUGIN["sniff"](info) is False


def _lora_info(siblings=None, tags=None):
    return _fake_info(
        siblings=siblings or ["README.md", "pixel-art-xl.safetensors"],
        tags=tags or [
            "diffusers", "text-to-image", "lora",
            "base_model:stabilityai/stable-diffusion-xl-base-1.0",
            "base_model:adapter:stabilityai/stable-diffusion-xl-base-1.0",
        ],
        pipeline_tag="text-to-image",
    )


def test_resolve_lora_extracts_base_from_adapter_tag():
    from unittest.mock import patch
    with patch(
        "muse.modalities.image_generation.hf._estimate_repo_weights_gb",
        return_value=None,
    ):
        result = HF_PLUGIN["resolve"]("nerijs/pixel-art-xl", None, _lora_info())
    caps = result.manifest["capabilities"]
    assert caps["lora_adapter"] is True
    assert caps["base_model"] == "stabilityai/stable-diffusion-xl-base-1.0"
    assert caps["lora_scale"] == 1.0
    assert result.manifest["model_id"] == "pixel-art-xl"
    assert result.manifest["modality"] == "image/generation"
    assert "peft" in result.manifest["pip_extras"]


def test_resolve_lora_falls_back_to_plain_base_model_tag():
    from unittest.mock import patch
    info = _lora_info(tags=[
        "text-to-image", "lora",
        "base_model:runwayml/stable-diffusion-v1-5",
    ])
    with patch(
        "muse.modalities.image_generation.hf._estimate_repo_weights_gb",
        return_value=None,
    ):
        result = HF_PLUGIN["resolve"]("org/some-lora", None, info)
    assert result.manifest["capabilities"]["base_model"] == (
        "runwayml/stable-diffusion-v1-5"
    )


def test_resolve_lora_without_base_tag_omits_base_model():
    """No base tag: resolve succeeds WITHOUT base_model; the post-overlay
    validation in catalog.pull rejects it unless --base supplied it."""
    from unittest.mock import patch
    info = _lora_info(tags=["text-to-image", "lora"])
    with patch(
        "muse.modalities.image_generation.hf._estimate_repo_weights_gb",
        return_value=None,
    ):
        result = HF_PLUGIN["resolve"]("org/tagless-lora", None, info)
    caps = result.manifest["capabilities"]
    assert caps["lora_adapter"] is True
    assert "base_model" not in caps


def test_resolve_lora_defaults_derive_from_base_id():
    """An SDXL base yields the sdxl defaults (25 steps, 1024), proving
    _infer_defaults ran against the BASE, not the adapter repo name."""
    from unittest.mock import patch
    with patch(
        "muse.modalities.image_generation.hf._estimate_repo_weights_gb",
        return_value=None,
    ):
        result = HF_PLUGIN["resolve"]("nerijs/pixel-art-xl", None, _lora_info())
    caps = result.manifest["capabilities"]
    assert caps["default_steps"] == 25
    assert caps["default_size"] == [1024, 1024]


def test_resolve_lora_multiple_safetensors_fails_actionably():
    import pytest
    from muse.core.resolvers import ResolverError
    info = _lora_info(
        siblings=["a.safetensors", "b.safetensors", "README.md"],
    )
    with pytest.raises(ResolverError, match="a.safetensors"):
        HF_PLUGIN["resolve"]("org/multi-lora", None, info)


def test_resolve_lora_memory_estimate_from_base_weights():
    from unittest.mock import patch
    with patch(
        "muse.modalities.image_generation.hf._estimate_repo_weights_gb",
        return_value=6.9,
    ) as est:
        result = HF_PLUGIN["resolve"]("nerijs/pixel-art-xl", None, _lora_info())
    est.assert_called_once_with("stabilityai/stable-diffusion-xl-base-1.0")
    assert result.manifest["capabilities"]["memory_gb"] == 7.2  # 6.9 + 0.3


def test_resolve_lora_base_override_wins_over_tag_declared_base():
    """I2: --base wins over the tag-declared base AND re-derives
    generation defaults from the OVERRIDE base, not the tag base. The
    fixture's tags declare an SDXL base (25 steps / 1024); overriding
    to sdxl-turbo must flip capabilities.base_model AND the derived
    defaults to turbo values (1 step / 0.0 guidance)."""
    from unittest.mock import patch
    with patch(
        "muse.modalities.image_generation.hf._estimate_repo_weights_gb",
        return_value=None,
    ):
        result = HF_PLUGIN["resolve"](
            "nerijs/pixel-art-xl", None, _lora_info(),
            base_override="sdxl-turbo",
        )
    caps = result.manifest["capabilities"]
    assert caps["base_model"] == "sdxl-turbo"
    assert caps["default_steps"] == 1
    assert caps["default_guidance"] == 0.0


def test_resolve_lora_base_override_satisfies_tagless_repo():
    """A tagless adapter repo (no base_model tag) resolves successfully
    when --base is supplied, instead of omitting base_model."""
    from unittest.mock import patch
    info = _lora_info(tags=["text-to-image", "lora"])
    with patch(
        "muse.modalities.image_generation.hf._estimate_repo_weights_gb",
        return_value=None,
    ):
        result = HF_PLUGIN["resolve"](
            "org/tagless-lora", None, info, base_override="flux-schnell",
        )
    caps = result.manifest["capabilities"]
    assert caps["base_model"] == "flux-schnell"


def test_resolve_non_lora_ignores_base_override():
    """A non-LoRA t2i repo has no base to override; base_override is a
    silent no-op for it."""
    info = _fake_info(
        siblings=["model_index.json", "unet/diffusion_pytorch_model.safetensors"],
        tags=["text-to-image", "diffusers"],
    )
    result = HF_PLUGIN["resolve"](
        "some/regular-t2i", None, info, base_override="sdxl-turbo",
    )
    assert "base_model" not in result.manifest["capabilities"]


def test_resolve_lora_download_patterns():
    from pathlib import Path
    from unittest.mock import patch
    with patch(
        "muse.modalities.image_generation.hf._estimate_repo_weights_gb",
        return_value=None,
    ), patch(
        "muse.modalities.image_generation.hf.snapshot_download",
        return_value="/tmp/fake",
    ) as snap:
        result = HF_PLUGIN["resolve"]("nerijs/pixel-art-xl", None, _lora_info())
        result.download(Path("/tmp/cache"))
    patterns = snap.call_args.kwargs["allow_patterns"]
    assert "*.safetensors" in patterns
    assert "*.json" in patterns
    # No subfolder pipeline patterns for an adapter-only repo.
    assert "*/*.safetensors" not in patterns
