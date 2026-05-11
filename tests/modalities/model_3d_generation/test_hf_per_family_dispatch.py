"""Per-family backend_path dispatch in the 3d/generation HF resolver.

The dispatcher picks a runtime path based on repo name. Shap-E gets
the new ShapERuntime path; TRELLIS / Wonder3D / Hunyuan3D-2 fall
through to TripoSR until their dedicated runtimes ship in v0.44.0+.
TripoSR repos remain on TripoSR (regression watchdog).
"""


def test_runtime_path_for_shap_e():
    from muse.modalities.model_3d_generation.hf import _runtime_path_for
    assert _runtime_path_for("openai/shap-e").endswith(":ShapERuntime")


def test_runtime_path_for_trellis_now_dispatches_to_TRELLISRuntime():
    """v0.44.0 promotes TRELLIS from TripoSR fallback to dedicated runtime."""
    from muse.modalities.model_3d_generation.hf import _runtime_path_for
    assert _runtime_path_for("JeffreyXiang/TRELLIS-image-large").endswith(
        ":TRELLISRuntime"
    )


def test_runtime_path_for_wonder3d():
    """Until v0.44.0, Wonder3D still falls back to TripoSR."""
    from muse.modalities.model_3d_generation.hf import _runtime_path_for
    assert _runtime_path_for("flamehaze1115/wonder3d-v1.0").endswith(":TripoSRRuntime")


def test_runtime_path_for_triposr():
    """Regression watchdog: TripoSR repos always route to TripoSRRuntime."""
    from muse.modalities.model_3d_generation.hf import _runtime_path_for
    assert _runtime_path_for("stabilityai/TripoSR").endswith(":TripoSRRuntime")


def test_pip_extras_for_shap_e():
    from muse.modalities.model_3d_generation.hf import (
        _SHAPE_E_RUNTIME_PATH, _pip_extras_for,
    )
    extras = _pip_extras_for(_SHAPE_E_RUNTIME_PATH)
    assert any("diffusers" in e for e in extras)
    assert any("trimesh" in e for e in extras)
    assert any("torch" in e for e in extras)


def test_pip_extras_for_triposr_default():
    """Non-Shap-E paths get the original TripoSR pip_extras."""
    from muse.modalities.model_3d_generation.hf import _pip_extras_for
    extras = _pip_extras_for(
        "muse.modalities.model_3d_generation.runtimes.triposr:TripoSRRuntime"
    )
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


def test_text_capable_hints_does_not_overlap_with_family_overrides():
    """Regression: hints list and family capability_overrides should not
    both claim the same family. The override path owns Shap-E's text-to-3d
    capability."""
    from muse.modalities.model_3d_generation.hf import (
        _FAMILIES, _TEXT_CAPABLE_NAME_HINTS,
    )
    for family in _FAMILIES:
        if "supports_text_to_3d" in family.capability_overrides:
            for hint in family.name_hints:
                assert hint not in _TEXT_CAPABLE_NAME_HINTS, (
                    f"Family hint {hint!r} also in _TEXT_CAPABLE_NAME_HINTS; "
                    f"this is double-dispatch (override AND hint list). "
                    f"Remove from hint list since the override owns it."
                )


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


def test_pip_extras_for_trellis_includes_transformers_and_trimesh():
    from muse.modalities.model_3d_generation.hf import (
        _TRELLIS_RUNTIME_PATH, _pip_extras_for,
    )
    extras = _pip_extras_for(_TRELLIS_RUNTIME_PATH)
    assert any("transformers" in e for e in extras)
    assert any("trimesh" in e for e in extras)
    assert any("torch" in e for e in extras)
