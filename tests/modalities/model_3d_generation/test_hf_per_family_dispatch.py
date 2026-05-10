"""Per-family backend_path dispatch in the 3d/generation HF resolver.

The dispatcher picks a runtime path based on repo name. Shap-E gets
the new ShapERuntime path; TRELLIS / Wonder3D / Hunyuan3D-2 fall
through to TripoSR until their dedicated runtimes ship in v0.44.0+.
TripoSR repos remain on TripoSR (regression watchdog).
"""


def test_runtime_path_for_shap_e():
    from muse.modalities.model_3d_generation.hf import _runtime_path_for
    assert _runtime_path_for("openai/shap-e").endswith(":ShapERuntime")


def test_runtime_path_for_trellis():
    """Until v0.45.0, TRELLIS still falls back to TripoSR."""
    from muse.modalities.model_3d_generation.hf import _runtime_path_for
    assert _runtime_path_for("JeffreyXiang/TRELLIS-image-large").endswith(":TripoSRRuntime")


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
