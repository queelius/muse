"""Per-family backend_path dispatch in the 3d/generation HF resolver.

The dispatcher picks a runtime path based on repo name. Shap-E gets
ShapERuntime; TRELLIS gets TRELLISRuntime; Hunyuan3D-2 gets
Hunyuan3DRuntime. Unknown repos (TripoSR, Wonder3D) fall through to
TripoSRRuntime (regression watchdog).
"""
import pytest


def test_runtime_path_for_shap_e():
    from muse.modalities.model_3d_generation.hf import _family_for
    assert _family_for("openai/shap-e").runtime_path.endswith(":ShapERuntime")


def test_runtime_path_for_trellis_now_dispatches_to_TRELLISRuntime():
    """v0.44.0 promotes TRELLIS from TripoSR fallback to dedicated runtime."""
    from muse.modalities.model_3d_generation.hf import _family_for
    assert _family_for("JeffreyXiang/TRELLIS-image-large").runtime_path.endswith(
        ":TRELLISRuntime"
    )


def test_runtime_path_for_wonder3d():
    """Wonder3D is deferred; falls back to TripoSR."""
    from muse.modalities.model_3d_generation.hf import _family_for
    assert _family_for("flamehaze1115/wonder3d-v1.0").runtime_path.endswith(":TripoSRRuntime")


def test_runtime_path_for_triposr():
    """Regression watchdog: TripoSR repos always route to TripoSRRuntime."""
    from muse.modalities.model_3d_generation.hf import _family_for
    assert _family_for("stabilityai/TripoSR").runtime_path.endswith(":TripoSRRuntime")


@pytest.mark.parametrize(
    "repo_id,required_substrings",
    [
        ("openai/shap-e", ("torch", "diffusers", "trimesh")),
        ("JeffreyXiang/TRELLIS-image-large", ("torch", "transformers", "trimesh", "trellis")),
        ("tencent/Hunyuan3D-2", ("torch", "trimesh", "hy3dgen")),
    ],
)
def test_pip_extras_for_family_includes_required_substrings(repo_id, required_substrings):
    from muse.modalities.model_3d_generation.hf import _family_for
    extras = _family_for(repo_id).pip_extras
    for sub in required_substrings:
        assert any(sub in e for e in extras), (
            f"pip_extras for {repo_id!r} must include {sub!r}; got {extras!r}"
        )


def test_pip_extras_for_triposr_default():
    """Non-family repos get the original TripoSR pip_extras (fallback path)."""
    from muse.modalities.model_3d_generation.hf import _family_for
    extras = _family_for("some/unknown-3d-repo").pip_extras
    assert any("tsr" in e for e in extras)


def test_family_for_shap_e_returns_shape_e_family():
    from muse.modalities.model_3d_generation.hf import _family_for, _SHAPE_E_RUNTIME_PATH
    family = _family_for("openai/shap-e")
    assert family.runtime_path == _SHAPE_E_RUNTIME_PATH
    assert family.capability_overrides.get("supports_text_to_3d") is True
    assert family.capability_overrides.get("supports_image_to_3d") is False
    assert family.trust_remote_code is False


def test_family_for_unknown_returns_default_triposr_family():
    from muse.modalities.model_3d_generation.hf import _family_for, _TRIPOSR_RUNTIME_PATH
    family = _family_for("some/unknown-repo")
    assert family.runtime_path == _TRIPOSR_RUNTIME_PATH
    assert family.capability_overrides == {}
    assert family.trust_remote_code is False


def test_family_for_rejects_false_positive_substring():
    """Regression: _family_for must use word-boundary matching so that
    e.g. `my-reshape-enhancer` does NOT match `shape-e`."""
    from muse.modalities.model_3d_generation.hf import (
        _family_for, _TRIPOSR_RUNTIME_PATH,
    )
    family = _family_for("user/my-reshape-enhancer")
    assert family.runtime_path == _TRIPOSR_RUNTIME_PATH


def test_family_for_accepts_word_boundary_match():
    """Legitimate fork-style names (e.g. `my-shap-e-fork`) should still match."""
    from muse.modalities.model_3d_generation.hf import (
        _family_for, _SHAPE_E_RUNTIME_PATH,
    )
    family = _family_for("someuser/my-shap-e-fork")
    assert family.runtime_path == _SHAPE_E_RUNTIME_PATH


def test_family_default_has_empty_system_packages():
    from muse.modalities.model_3d_generation.hf import _DEFAULT_FAMILY
    assert _DEFAULT_FAMILY.system_packages == ()


def test_shape_e_family_has_empty_system_packages():
    from muse.modalities.model_3d_generation.hf import _family_for
    family = _family_for("openai/shap-e")
    assert family.system_packages == ()


def test_family_for_trellis_has_trust_remote_code_true():
    from muse.modalities.model_3d_generation.hf import _family_for
    family = _family_for("JeffreyXiang/TRELLIS-image-large")
    assert family.trust_remote_code is True


def test_family_for_trellis_has_correct_capability_overrides():
    """TRELLIS-image-large is image-only. The override sets the flags
    explicitly so the hint-list path is bypassed."""
    from muse.modalities.model_3d_generation.hf import _family_for
    family = _family_for("JeffreyXiang/TRELLIS-image-large")
    assert family.capability_overrides.get("supports_image_to_3d") is True
    assert family.capability_overrides.get("supports_text_to_3d") is False


def test_resolve_trellis_manifest_includes_trust_remote_code_capability():
    """Integration: the synthesized manifest carries trust_remote_code=True
    in capabilities so the runtime constructor receives it via the kwargs splat."""
    from unittest.mock import MagicMock
    from muse.modalities.model_3d_generation.hf import _resolve
    info = MagicMock()
    info.card_data = None
    resolved = _resolve("JeffreyXiang/TRELLIS-image-large", None, info)
    caps = resolved.manifest["capabilities"]
    assert caps.get("trust_remote_code") is True
    assert caps.get("supports_image_to_3d") is True
    assert caps.get("supports_text_to_3d") is False


def test_runtime_path_for_hunyuan3d_now_dispatches_to_Hunyuan3DRuntime():
    """v0.45.0 promotes Hunyuan3D-2 from TripoSR fallback to dedicated runtime."""
    from muse.modalities.model_3d_generation.hf import _family_for
    assert _family_for("tencent/Hunyuan3D-2").runtime_path.endswith(":Hunyuan3DRuntime")


def test_family_for_hunyuan3d_has_trust_remote_code_true():
    from muse.modalities.model_3d_generation.hf import _family_for
    family = _family_for("tencent/Hunyuan3D-2")
    assert family.trust_remote_code is True


def test_family_for_hunyuan3d_has_dual_direction_capabilities():
    """Hunyuan3D-2 is muse's first 3D runtime supporting BOTH directions."""
    from muse.modalities.model_3d_generation.hf import _family_for
    family = _family_for("tencent/Hunyuan3D-2")
    assert family.capability_overrides.get("supports_image_to_3d") is True
    assert family.capability_overrides.get("supports_text_to_3d") is True


def test_resolve_hunyuan3d_manifest_includes_dual_direction_capabilities():
    """Integration: synthesized manifest carries both flags + trust_remote_code."""
    from unittest.mock import MagicMock
    from muse.modalities.model_3d_generation.hf import _resolve
    info = MagicMock()
    info.card_data = None
    resolved = _resolve("tencent/Hunyuan3D-2", None, info)
    caps = resolved.manifest["capabilities"]
    assert caps.get("trust_remote_code") is True
    assert caps.get("supports_image_to_3d") is True
    assert caps.get("supports_text_to_3d") is True
