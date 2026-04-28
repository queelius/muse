"""Tests for the image_animation HF plugin (fused-checkpoint AnimateDiff variants)."""
from unittest.mock import MagicMock

from muse.modalities.image_animation.hf import HF_PLUGIN
from muse.core.discovery import REQUIRED_HF_PLUGIN_KEYS
from muse.core.resolvers import ResolvedModel


def _fake_info(repo_id="org/repo", siblings=None, tags=None):
    info = MagicMock()
    info.id = repo_id
    info.siblings = [MagicMock(rfilename=f) for f in (siblings or [])]
    info.tags = tags or []
    info.card_data = MagicMock(license=None)
    return info


def test_plugin_has_all_required_keys():
    for key in REQUIRED_HF_PLUGIN_KEYS:
        assert key in HF_PLUGIN


def test_plugin_metadata_correct():
    assert HF_PLUGIN["modality"] == "image/animation"
    assert HF_PLUGIN["runtime_path"].endswith(":AnimateDiffRuntime")
    # priority 110: tag + repo-name pattern, more specific than the generic
    # text-classification catch-all (200) but less specific than file-pattern
    # plugins (100).
    assert HF_PLUGIN["priority"] == 110


def test_sniff_true_on_animatediff_repo():
    """Repo with model_index.json + text-to-video tag + 'animate' in name."""
    info = _fake_info(
        repo_id="guoyww/animatediff-motion-adapter-v1-5-3",
        siblings=["model_index.json", "unet/diffusion_pytorch_model.safetensors"],
        tags=["text-to-video", "diffusers"],
    )
    assert HF_PLUGIN["sniff"](info) is True


def test_sniff_true_on_motion_in_name():
    """Repo with model_index.json + text-to-video tag + 'motion' in name."""
    info = _fake_info(
        repo_id="someorg/motion-adapter-v3",
        siblings=["model_index.json"],
        tags=["text-to-video"],
    )
    assert HF_PLUGIN["sniff"](info) is True


def test_sniff_false_without_text_to_video_tag():
    """Has model_index.json + animate in name but no text-to-video tag."""
    info = _fake_info(
        repo_id="someorg/animatediff-something",
        siblings=["model_index.json"],
        tags=["text-to-image"],
    )
    assert HF_PLUGIN["sniff"](info) is False


def test_sniff_false_without_animate_or_motion_in_name():
    """Has model_index.json + text-to-video tag but generic repo name."""
    info = _fake_info(
        repo_id="someorg/generic-video",
        siblings=["model_index.json"],
        tags=["text-to-video"],
    )
    assert HF_PLUGIN["sniff"](info) is False


def test_resolve_animatelcm_uses_lcm_defaults():
    """AnimateLCM repos get steps=4 guidance=1.0 + base_model + supports_text_to_animation."""
    info = _fake_info(
        repo_id="wangfuyun/AnimateLCM",
        siblings=["model_index.json"],
        tags=["text-to-video"],
    )
    result = HF_PLUGIN["resolve"]("wangfuyun/AnimateLCM", None, info)
    assert isinstance(result, ResolvedModel)
    caps = result.manifest["capabilities"]
    assert caps["default_steps"] == 4
    assert caps["default_guidance"] == 1.0
    assert "base_model" in caps
    assert caps["supports_text_to_animation"] is True


def test_search_yields_results_with_modality_tag():
    """Search filters by text-to-video and yields image/animation rows."""
    fake_api = MagicMock()
    fake_repo = MagicMock(id="org/animatediff-thing", downloads=42)
    fake_api.list_models.return_value = [fake_repo]
    rows = list(HF_PLUGIN["search"](fake_api, "animate", sort="downloads", limit=20))
    assert len(rows) == 1
    assert rows[0].modality == "image/animation"
    # Confirm the search filter targets text-to-video
    fake_api.list_models.assert_called_once()
    call_kwargs = fake_api.list_models.call_args.kwargs
    assert call_kwargs["filter"] == "text-to-video"
