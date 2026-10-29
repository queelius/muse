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


_KLEIN_REPO = "black-forest-labs/FLUX.2-klein-4B"
_KLEIN_SIBLINGS = [
    "model_index.json",
    "flux-2-klein-4b.safetensors",  # top-level single file, never fetched
    "transformer/diffusion_pytorch_model.safetensors",
    "vae/diffusion_pytorch_model.safetensors",
    "text_encoder/model-00001-of-00002.safetensors",
]


def _klein_info():
    info = _fake_info(siblings=_KLEIN_SIBLINGS, tags=["text-to-image"])
    info.card_data = MagicMock(license="apache-2.0")
    return info


def test_klein_repo_resolves_to_reference_runtime():
    result = HF_PLUGIN["resolve"](_KLEIN_REPO, None, _klein_info())
    assert result.backend_path == (
        "muse.modalities.image_generation.runtimes.reference"
        ":DiffusersReferenceModel"
    )
    caps = result.manifest["capabilities"]
    assert caps == {
        "pipeline_class": "Flux2KleinPipeline", "default_size": [1024, 1024],
        "default_steps": 4, "default_guidance": 1.0, "cpu_offload": "model",
        "device": "cuda", "supports_reference_images": True,
        "max_reference_images": 4, "size_multiple": 16,
        "max_pixels": 4194304,
        "quality_presets": {
            "low": {"steps": 4, "max_pixels": 1048576},
            "medium": {"steps": 4},
            "high": {"steps": 6},
        },
        "supports_negative_prompt": False, "supports_guidance": False,
        "supports_seeded_generation": True, "supports_img2img": False,
        "supports_inpainting": False, "supports_variations": False,
    }
    assert result.manifest["pip_extras"] == [
        "torch>=2.1.0", "diffusers>=0.37.0", "transformers>=4.51.0",
        "accelerate", "Pillow", "safetensors",
    ]
    assert result.manifest["license"] == "apache-2.0"
    assert result.manifest["description"] == f"Diffusers reference-image model: {_KLEIN_REPO}"


def test_klein_base_repo_not_reference_family():
    """Without a curated reference_runtime opt-in, base repos stay excluded."""
    from unittest.mock import patch
    from muse.modalities.image_generation.hf import _reference_family

    with patch("muse.core.curated.load_curated", return_value=[]):
        assert _reference_family("black-forest-labs/FLUX.2-klein-base-4B") is None
    assert _reference_family(_KLEIN_REPO) is not None


def test_klein_download_keeps_every_subfolder_file():
    from pathlib import Path
    from unittest.mock import patch
    from huggingface_hub.utils import filter_repo_objects

    files = [
        "model_index.json", "flux-2-klein-4b.safetensors",
        "transformer/config.json", "transformer/diffusion_pytorch_model.safetensors",
        "text_encoder/config.json", "text_encoder/model-00001-of-00002.safetensors",
        "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json",
        "tokenizer/chat_template.jinja", "tokenizer/vocab.json", "tokenizer/merges.txt",
        "vae/config.json", "vae/diffusion_pytorch_model.safetensors",
        "scheduler/scheduler_config.json", "README.md",
    ]
    with patch(
        "muse.modalities.image_generation.hf.snapshot_download", return_value="/tmp/fake",
    ) as fake_snapshot:
        result = HF_PLUGIN["resolve"](_KLEIN_REPO, None, _klein_info())
        result.download(Path("/tmp/cache"))
    patterns = fake_snapshot.call_args.kwargs["allow_patterns"]
    kept = set(filter_repo_objects(files, allow_patterns=patterns))
    assert kept == {f for f in files if "/" in f} | {"model_index.json"}
    assert "tokenizer/chat_template.jinja" in kept


def test_klein_download_excludes_top_level_single_file():
    from pathlib import Path
    from unittest.mock import patch

    with patch(
        "muse.modalities.image_generation.hf.snapshot_download",
        return_value="/tmp/fake",
    ) as fake_snapshot:
        result = HF_PLUGIN["resolve"](_KLEIN_REPO, None, _klein_info())
        result.download(Path("/tmp/cache"))
    patterns = fake_snapshot.call_args.kwargs["allow_patterns"]
    assert "*/*.safetensors" in patterns
    assert all(p == "model_index.json" or p.startswith("*/") for p in patterns)


def test_klein_capabilities_are_fresh_copies():
    a = HF_PLUGIN["resolve"](_KLEIN_REPO, None, _klein_info())
    a.manifest["capabilities"]["quality_presets"]["low"]["steps"] = 99
    b = HF_PLUGIN["resolve"](_KLEIN_REPO, None, _klein_info())
    assert b.manifest["capabilities"]["quality_presets"]["low"]["steps"] == 4


def test_other_repos_unchanged():
    info = _fake_info(
        siblings=["model_index.json", "unet/diffusion_pytorch_model.safetensors"],
        tags=["text-to-image"],
    )
    result = HF_PLUGIN["resolve"]("stabilityai/sdxl-turbo", None, info)
    assert result.backend_path.endswith(":DiffusersText2ImageModel")


# ---------------- klein LoRA adapters (Limbicnation/pixel-art-lora) ----------------

_LIMBIC_REPO = "Limbicnation/pixel-art-lora"
_LIMBIC_SIBLINGS = [
    ".gitattributes", "LICENSE", "README.md", "comfyui_workflow.json",
    "config.json", "config.yaml",
    "pytorch_lora_weights.comfyui.safetensors",
    "pytorch_lora_weights.safetensors",
    "samples/01_a_brave_knight_in_shining_armor_holding_.png",
]


def _limbic_info():
    info = _fake_info(
        siblings=_LIMBIC_SIBLINGS,
        tags=[
            "diffusers", "lora", "flux", "pixel-art", "text-to-image",
            "FLUX.2-klein-4B",
            "base_model:black-forest-labs/FLUX.2-klein-4B",
            "base_model:adapter:black-forest-labs/FLUX.2-klein-4B",
            "license:apache-2.0",
        ],
        pipeline_tag="text-to-image",
    )
    info.card_data = MagicMock(license="apache-2.0")
    return info


def test_sniff_true_on_limbicnation_lora():
    assert HF_PLUGIN["sniff"](_limbic_info()) is True


def test_resolve_lora_klein_base_routes_to_reference():
    from unittest.mock import patch
    from muse.modalities.image_generation.hf import _resolve_lora

    with patch(
        "muse.modalities.image_generation.hf._estimate_repo_weights_gb",
    ) as est:
        r = _resolve_lora(_LIMBIC_REPO, _limbic_info())
    assert r.backend_path.endswith("runtimes.reference:DiffusersReferenceModel")
    caps = r.manifest["capabilities"]
    assert caps["lora_adapter"] is True
    assert caps["lora_scale"] == 1.0
    assert caps["base_model"] == "black-forest-labs/FLUX.2-klein-4B"
    assert caps["pipeline_class"] == "Flux2KleinPipeline"
    assert caps["cpu_offload"] == "model"
    assert caps["device"] == "cuda"
    assert caps["size_multiple"] == 16
    assert caps["max_pixels"] == 4194304
    assert caps["max_reference_images"] == 4
    assert caps["supports_reference_images"] is True
    # The SDXL diffusers paths do not exist on the reference runtime.
    assert caps["supports_img2img"] is False
    assert caps["supports_inpainting"] is False
    assert caps["supports_variations"] is False
    # A klein base's repo weight sum double-counts its single-file
    # checkpoint and ignores offload; sizing comes from the base entry.
    assert "memory_gb" not in caps
    est.assert_not_called()
    assert "peft" in r.manifest["pip_extras"]
    assert "diffusers>=0.37.0" in r.manifest["pip_extras"]
    assert r.manifest["license"] == "apache-2.0"
    assert r.manifest["model_id"] == "pixel-art-lora"


def test_resolve_lora_klein_via_plugin_resolve():
    r = HF_PLUGIN["resolve"](_LIMBIC_REPO, None, _limbic_info())
    assert r.backend_path.endswith(":DiffusersReferenceModel")


def test_resolve_lora_klein_base_override_catalog_id():
    """`--base flux2-klein-4b` (a muse id) routes to the reference runtime,
    even when the adapter's own tags declare no base."""
    info = _limbic_info()
    info.tags = ["text-to-image", "lora"]
    r = HF_PLUGIN["resolve"](
        _LIMBIC_REPO, None, info, base_override="flux2-klein-4b",
    )
    assert r.backend_path.endswith(":DiffusersReferenceModel")
    assert r.manifest["capabilities"]["base_model"] == "flux2-klein-4b"


def test_resolve_lora_klein_base_named_base_still_reference():
    """The klein name test ignores the reference family's `base` exclusion."""
    from unittest.mock import patch
    info = _limbic_info()
    info.tags = [
        "text-to-image", "lora",
        "base_model:adapter:black-forest-labs/FLUX.2-klein-base-4B",
    ]
    with patch(
        "muse.modalities.image_generation.hf._estimate_repo_weights_gb",
        return_value=None,
    ):
        r = HF_PLUGIN["resolve"](_LIMBIC_REPO, None, info)
    assert r.backend_path.endswith(":DiffusersReferenceModel")


def test_resolve_lora_sdxl_override_on_klein_tagged_repo_uses_diffusers():
    """The EFFECTIVE base decides: an SDXL --base keeps the SDXL runtime."""
    from unittest.mock import patch
    with patch(
        "muse.modalities.image_generation.hf._estimate_repo_weights_gb",
        return_value=None,
    ):
        r = HF_PLUGIN["resolve"](
            _LIMBIC_REPO, None, _limbic_info(), base_override="sdxl-turbo",
        )
    assert r.backend_path.endswith(":DiffusersText2ImageModel")


def test_resolve_lora_picks_diffusers_weight_name():
    from pathlib import Path
    from unittest.mock import patch
    from huggingface_hub.utils import filter_repo_objects

    with patch(
        "muse.modalities.image_generation.hf.snapshot_download",
        return_value="/tmp/fake",
    ) as snap:
        r = HF_PLUGIN["resolve"](_LIMBIC_REPO, None, _limbic_info())
        r.download(Path("/tmp/cache"))
    patterns = snap.call_args.kwargs["allow_patterns"]
    assert patterns == ["pytorch_lora_weights.safetensors", "*.json"]
    assert r.allow_patterns == ("pytorch_lora_weights.safetensors", "*.json")
    kept = set(filter_repo_objects(_LIMBIC_SIBLINGS, allow_patterns=patterns))
    assert kept == {
        "pytorch_lora_weights.safetensors", "config.json",
        "comfyui_workflow.json",
    }


def test_resolve_lora_several_safetensors_without_diffusers_name_fails():
    import pytest
    from muse.core.resolvers import ResolverError
    info = _limbic_info()
    info.siblings = [
        MagicMock(rfilename=f)
        for f in ("a.safetensors", "b.comfyui.safetensors", "README.md")
    ]
    with pytest.raises(ResolverError, match="exactly one"):
        HF_PLUGIN["resolve"](_LIMBIC_REPO, None, info)


def test_resolve_lora_sdxl_unchanged():
    from pathlib import Path
    from unittest.mock import patch
    with patch(
        "muse.modalities.image_generation.hf._estimate_repo_weights_gb",
        return_value=6.9,
    ), patch(
        "muse.modalities.image_generation.hf.snapshot_download",
        return_value="/tmp/fake",
    ) as snap:
        r = HF_PLUGIN["resolve"]("nerijs/pixel-art-xl", None, _lora_info())
        r.download(Path("/tmp/cache"))
    assert r.backend_path.endswith(":DiffusersText2ImageModel")
    caps = r.manifest["capabilities"]
    assert caps["supports_img2img"] is True
    assert caps["memory_gb"] == 7.2
    assert "pipeline_class" not in caps and "cpu_offload" not in caps
    assert "diffusers>=0.27.0" in r.manifest["pip_extras"]
    assert snap.call_args.kwargs["allow_patterns"] == [
        "*.safetensors", "*.json", "*.txt",
    ]
    assert r.allow_patterns == ("*.safetensors", "*.json", "*.txt")


# ---------------- klein base opt-in (curated reference_runtime: true) ----------------

_KLEIN_BASE_REPO = "black-forest-labs/FLUX.2-klein-base-4B"
_WALK_REPO = "svntax-dev/pixel_spritesheet_4walk_small_lora_v1"
_WALK_FINAL = "pixel_4walk_small_flux2_klein_base_4b_v1.safetensors"
_WALK_SIBLINGS = [
    ".gitattributes", "README.md",
    "flux2_klein_image_edit_4b_base_pixel_spritesheet_workflow.json",
    "images/knight.png",
    _WALK_FINAL,
    "pixel_4walk_small_flux2_klein_base_4b_v1_000002750.safetensors",
]


def _curated_stub(*, opted_in=True):
    from muse.core.curated import CuratedEntry
    caps = {"reference_runtime": True, "memory_gb": 9.5} if opted_in else {}
    return [CuratedEntry(
        id="flux2-klein-base-4b", bundled=False,
        uri=f"hf://{_KLEIN_BASE_REPO}", modality="image/generation",
        size_gb=16.0, description=None, tags=(), capabilities=caps,
    )]


def _klein_base_info():
    info = _fake_info(
        siblings=[s.replace("flux-2-klein-4b", "flux-2-klein-base-4b")
                  for s in _KLEIN_SIBLINGS],
        tags=["text-to-image", "image-to-image"], pipeline_tag="image-to-image",
    )
    info.card_data = MagicMock(license="apache-2.0")
    return info


def _walk_info():
    info = _fake_info(
        siblings=_WALK_SIBLINGS,
        tags=[
            "diffusers", "text-to-image", "lora", "template:diffusion-lora",
            f"base_model:{_KLEIN_BASE_REPO}",
            f"base_model:adapter:{_KLEIN_BASE_REPO}", "license:apache-2.0",
        ],
        pipeline_tag="text-to-image",
    )
    info.card_data = MagicMock(license="apache-2.0")
    return info


def _assert_base_presets(caps):
    assert caps["pipeline_class"] == "Flux2KleinPipeline"
    assert caps["default_steps"] == 28
    assert caps["default_guidance"] == 4.0
    # Undistilled: guidance is real CFG.
    assert caps["supports_guidance"] is True
    assert caps["quality_presets"]["medium"] == {"steps": 28}
    assert all(p["steps"] >= 20 for p in caps["quality_presets"].values())
    assert caps["cpu_offload"] == "model"
    assert caps["size_multiple"] == 16


def test_klein_base_opt_in_routes_to_reference_runtime():
    from unittest.mock import patch
    with patch("muse.core.curated.load_curated", return_value=_curated_stub()):
        r = HF_PLUGIN["resolve"](_KLEIN_BASE_REPO, None, _klein_base_info())
    assert r.backend_path.endswith("runtimes.reference:DiffusersReferenceModel")
    _assert_base_presets(r.manifest["capabilities"])
    assert r.manifest["pip_extras"][1] == "diffusers>=0.37.0"


def test_klein_base_without_opt_in_keeps_exclusion():
    from unittest.mock import patch
    for entries in ([], _curated_stub(opted_in=False)):
        with patch("muse.core.curated.load_curated", return_value=entries):
            r = HF_PLUGIN["resolve"](_KLEIN_BASE_REPO, None, _klein_base_info())
        assert r.backend_path.endswith(":DiffusersText2ImageModel")


def test_klein_distilled_repo_ignores_opt_in_lookup():
    """The distilled repo keeps its 4-step presets whatever curated says."""
    from unittest.mock import patch
    with patch("muse.core.curated.load_curated", return_value=_curated_stub()):
        r = HF_PLUGIN["resolve"](_KLEIN_REPO, None, _klein_info())
    assert r.manifest["capabilities"]["default_steps"] == 4


def test_real_curated_opts_klein_base_in():
    r = HF_PLUGIN["resolve"](_KLEIN_BASE_REPO, None, _klein_base_info())
    assert r.backend_path.endswith(":DiffusersReferenceModel")
    _assert_base_presets(r.manifest["capabilities"])


def test_lora_on_opted_in_base_gets_base_presets():
    from unittest.mock import patch
    with patch("muse.core.curated.load_curated", return_value=_curated_stub()):
        tagged = HF_PLUGIN["resolve"](_WALK_REPO, None, _walk_info())
        pinned = HF_PLUGIN["resolve"](
            _WALK_REPO, None, _walk_info(), base_override="flux2-klein-base-4b",
        )
    for r in (tagged, pinned):
        assert r.backend_path.endswith(":DiffusersReferenceModel")
        caps = r.manifest["capabilities"]
        _assert_base_presets(caps)
        assert caps["lora_adapter"] is True
        assert "memory_gb" not in caps
    assert pinned.manifest["capabilities"]["base_model"] == "flux2-klein-base-4b"


def test_lora_on_klein_base_without_opt_in_keeps_distilled_presets():
    from unittest.mock import patch
    with patch("muse.core.curated.load_curated", return_value=[]):
        r = HF_PLUGIN["resolve"](_WALK_REPO, None, _walk_info())
    assert r.backend_path.endswith(":DiffusersReferenceModel")
    assert r.manifest["capabilities"]["default_steps"] == 4


def test_lora_prefers_final_weight_over_intermediate_checkpoints():
    from pathlib import Path
    from unittest.mock import patch
    from huggingface_hub.utils import filter_repo_objects

    with patch(
        "muse.modalities.image_generation.hf.snapshot_download",
        return_value="/tmp/fake",
    ) as snap:
        r = HF_PLUGIN["resolve"](_WALK_REPO, None, _walk_info())
        r.download(Path("/tmp/cache"))
    patterns = snap.call_args.kwargs["allow_patterns"]
    assert r.allow_patterns == tuple(patterns)
    kept = set(filter_repo_objects(_WALK_SIBLINGS, allow_patterns=patterns))
    assert {f for f in kept if f.endswith(".safetensors")} == {_WALK_FINAL}


def test_lora_unrelated_weight_names_still_fail():
    import pytest
    from muse.core.resolvers import ResolverError
    info = _walk_info()
    info.siblings = [
        MagicMock(rfilename=f)
        for f in ("walk_v1.safetensors", "run_v1_000002750.safetensors")
    ]
    with pytest.raises(ResolverError, match="exactly one"):
        HF_PLUGIN["resolve"](_WALK_REPO, None, info)
