"""Per-family backend_path dispatch in the 3d/generation HF resolver.

The dispatcher picks a runtime path based on repo name. Shap-E gets
ShapERuntime; TRELLIS gets TRELLISRuntime; Hunyuan3D-2 gets
Hunyuan3DRuntime; TripoSR gets TripoSRRuntime. Any other repo (Wonder3D,
InstantMesh, ...) has no family: _family_for returns None and _resolve
raises ResolverError (curated pulls skip the sniff, so this is the gate).
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


def test_wonder3d_has_no_family():
    """Wonder3D is deferred; it must not fall back to TripoSR."""
    from muse.modalities.model_3d_generation.hf import _family_for
    assert _family_for("flamehaze1115/wonder3d-v1.0") is None


def test_resolve_refuses_repos_outside_the_known_families():
    """Curated pulls pass modality= and skip the sniff, so _resolve itself
    must refuse a family with no runtime."""
    from types import SimpleNamespace

    from muse.core.resolvers import ResolverError
    from muse.modalities.model_3d_generation.hf import _resolve

    info = SimpleNamespace(
        id="flamehaze1115/wonder3d-v1.0", sha="a" * 40, tags=["image-to-3d"],
        siblings=[], card_data=None,
    )
    with pytest.raises(ResolverError, match="no 3d/generation runtime"):
        _resolve("flamehaze1115/wonder3d-v1.0", None, info)


def test_runtime_path_for_triposr():
    """Regression watchdog: TripoSR repos always route to TripoSRRuntime."""
    from muse.modalities.model_3d_generation.hf import _family_for
    assert _family_for("stabilityai/TripoSR").runtime_path.endswith(":TripoSRRuntime")


@pytest.mark.parametrize(
    "repo_id,required_substrings",
    [
        ("openai/shap-e", ("torch", "diffusers", "trimesh")),
        (
            "JeffreyXiang/TRELLIS-image-large",
            ("torch", "transformers", "trimesh", "rembg", "easydict", "utils3d"),
        ),
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


def test_pip_extras_for_triposr():
    from muse.modalities.model_3d_generation.hf import _family_for
    extras = _family_for("stabilityai/TripoSR").pip_extras
    assert any("tsr" in e for e in extras)


def test_family_for_shap_e_returns_shape_e_family():
    from muse.modalities.model_3d_generation.hf import _family_for, _SHAPE_E_RUNTIME_PATH
    family = _family_for("openai/shap-e")
    assert family.runtime_path == _SHAPE_E_RUNTIME_PATH
    assert family.capability_overrides.get("supports_text_to_3d") is True
    assert family.capability_overrides.get("supports_image_to_3d") is False
    assert not hasattr(family, "trust_remote_code")


def test_family_for_unknown_returns_none():
    from muse.modalities.model_3d_generation.hf import _family_for
    assert _family_for("some/unknown-repo") is None


def test_family_for_triposr_is_the_plain_triposr_family():
    from muse.modalities.model_3d_generation.hf import _family_for, _TRIPOSR_RUNTIME_PATH
    family = _family_for("stabilityai/TripoSR")
    assert family.runtime_path == _TRIPOSR_RUNTIME_PATH
    assert family.capability_overrides == {}
    assert not hasattr(family, "trust_remote_code")


def test_family_for_rejects_false_positive_substring():
    """Regression: _family_for must use word-boundary matching so that
    e.g. `my-reshape-enhancer` does NOT match `shape-e`."""
    from muse.modalities.model_3d_generation.hf import _family_for
    assert _family_for("user/my-reshape-enhancer") is None


def test_family_for_accepts_word_boundary_match():
    """Legitimate fork-style names (e.g. `my-shap-e-fork`) should still match."""
    from muse.modalities.model_3d_generation.hf import (
        _family_for, _SHAPE_E_RUNTIME_PATH,
    )
    family = _family_for("someuser/my-shap-e-fork")
    assert family.runtime_path == _SHAPE_E_RUNTIME_PATH


def test_family_default_has_empty_system_packages():
    from muse.modalities.model_3d_generation.hf import _family_for
    assert _family_for("stabilityai/TripoSR").system_packages == ()


def test_shape_e_family_has_empty_system_packages():
    from muse.modalities.model_3d_generation.hf import _family_for
    family = _family_for("openai/shap-e")
    assert family.system_packages == ()


def test_family_for_trellis_uses_installed_sdk_not_transformers_remote_code():
    from muse.modalities.model_3d_generation.hf import _family_for
    family = _family_for("JeffreyXiang/TRELLIS-image-large")
    assert not hasattr(family, "trust_remote_code")


def test_family_for_trellis_declares_exact_reviewed_source_and_submodule():
    from muse.modalities.model_3d_generation.hf import _family_for

    family = _family_for("JeffreyXiang/TRELLIS-image-large")
    assert family.system_packages == ("git", "nvcc")
    assert family.python_sources == ({
        "type": "git",
        "name": "trellis",
        "url": "https://github.com/microsoft/TRELLIS.git",
        "revision": "442aa1e1afb9014e80681d3bf604e8d728a86ee7",
        "sparse_paths": ("trellis",),
        "required_paths": (
            "trellis/__init__.py",
            "trellis/pipelines/trellis_image_to_3d.py",
            "trellis/representations/mesh/flexicubes/flexicubes.py",
        ),
        "pth_path": ".",
        "submodules": ({
            "path": "trellis/representations/mesh/flexicubes",
            "url": "https://github.com/MaxtirError/FlexiCubes.git",
            "revision": "815e075a2a400d06c48d94c347674344ed6ae5c5",
        },),
    },)
    assert not any("microsoft/TRELLIS.git" in extra for extra in family.pip_extras)


def test_family_for_trellis_has_correct_capability_overrides():
    """TRELLIS-image-large is image-only. The override sets the flags
    explicitly so the hint-list path is bypassed."""
    from muse.modalities.model_3d_generation.hf import _family_for
    family = _family_for("JeffreyXiang/TRELLIS-image-large")
    assert family.capability_overrides.get("supports_image_to_3d") is True
    assert family.capability_overrides.get("supports_text_to_3d") is False


def test_resolve_trellis_manifest_omits_transformers_remote_code_capability():
    """The installed TRELLIS SDK is not Transformers dynamic remote code."""
    from unittest.mock import MagicMock
    from muse.modalities.model_3d_generation.hf import _family_for, _resolve
    info = MagicMock()
    info.card_data = None
    resolved = _resolve("JeffreyXiang/TRELLIS-image-large", None, info)
    caps = resolved.manifest["capabilities"]
    assert "trust_remote_code" not in caps
    assert caps.get("supports_image_to_3d") is True
    assert caps.get("supports_text_to_3d") is False
    family_sources = _family_for("JeffreyXiang/TRELLIS-image-large").python_sources
    assert resolved.manifest["python_sources"] == list(family_sources)
    assert resolved.manifest["python_sources"][0] is not family_sources[0]


def test_runtime_path_for_hunyuan3d_now_dispatches_to_Hunyuan3DRuntime():
    """v0.45.0 promotes Hunyuan3D-2 from TripoSR fallback to dedicated runtime."""
    from muse.modalities.model_3d_generation.hf import _family_for
    assert _family_for("tencent/Hunyuan3D-2").runtime_path.endswith(":Hunyuan3DRuntime")


def test_family_for_hunyuan3d_uses_installed_sdk_not_transformers_remote_code():
    from muse.modalities.model_3d_generation.hf import _family_for
    family = _family_for("tencent/Hunyuan3D-2")
    assert not hasattr(family, "trust_remote_code")


def test_family_for_hunyuan3d_has_dual_direction_capabilities():
    """Hunyuan3D-2 is muse's first 3D runtime supporting BOTH directions."""
    from muse.modalities.model_3d_generation.hf import _family_for
    family = _family_for("tencent/Hunyuan3D-2")
    assert family.capability_overrides.get("supports_image_to_3d") is True
    assert family.capability_overrides.get("supports_text_to_3d") is True


def test_resolve_hunyuan3d_manifest_includes_dual_direction_capabilities():
    """The installed Hunyuan SDK supplies both supported directions."""
    from unittest.mock import MagicMock
    from muse.modalities.model_3d_generation.hf import _resolve
    info = MagicMock()
    info.card_data = None
    resolved = _resolve("tencent/Hunyuan3D-2", None, info)
    caps = resolved.manifest["capabilities"]
    assert "trust_remote_code" not in caps
    assert caps.get("supports_image_to_3d") is True
    assert caps.get("supports_text_to_3d") is True
