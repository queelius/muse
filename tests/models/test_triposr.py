"""Bundled triposr smoke tests: MANIFEST shape + Model class wired
to TripoSRRuntime, no real weights required.

The bundled triposr script is a thin re-export of TripoSRRuntime, so
the tests stay structural: manifest values, Model identity, presence
of the protocol method, and pip_extras coverage of the runtime's
deferred imports. Real-weights inference is exercised by the runtime
test (`tests/modalities/model_3d_generation/test_triposr_runtime.py`)
and the fresh-venv smoke matrix.
"""
from __future__ import annotations


def test_manifest_required_keys():
    from muse.models.triposr import MANIFEST

    assert MANIFEST["model_id"] == "triposr"
    assert MANIFEST["modality"] == "3d/generation"
    assert MANIFEST["hf_repo"] == "stabilityai/TripoSR"
    assert MANIFEST["license"] == "MIT"


def test_manifest_capabilities():
    """Image-to-3d only; text-to-3d disabled. CUDA-by-default; GLB output."""
    from muse.models.triposr import MANIFEST

    caps = MANIFEST["capabilities"]
    assert caps["supports_image_to_3d"] is True
    assert caps["supports_text_to_3d"] is False
    assert caps["device"] == "cuda"
    assert caps["output_format"] == "glb"
    assert "memory_gb" in caps
    assert caps["memory_gb"] == 1.5


def test_model_is_triposr_runtime_alias():
    """Model is a thin re-export of TripoSRRuntime, so a Model instance
    satisfies the ImageTo3DBackend protocol structurally.

    Don't instantiate (no real weights); just check the alias and the
    presence of the protocol method at the class level.
    """
    from muse.models.triposr import Model
    from muse.modalities.model_3d_generation.runtimes.triposr import (
        TripoSRRuntime,
    )

    assert Model is TripoSRRuntime
    assert hasattr(Model, "image_to_3d")
    assert callable(getattr(Model, "image_to_3d"))


def test_pip_extras_include_runtime_imports():
    """The runtime's deferred imports must each appear in pip_extras
    (otherwise `muse pull triposr` won't install them and the worker
    crashes with ImportError on first use). This mirrors the v0.30.0
    pip_extras audit pattern for sd-turbo/kokoro: bundled scripts are
    the source of truth for fresh-venv installs.
    """
    from muse.models.triposr import MANIFEST

    extras = " ".join(MANIFEST["pip_extras"])
    # The runtime's `_ensure_deps` lazy-imports torch, tsr, trimesh,
    # PIL (Pillow). einops is pulled in transitively via tsr's own
    # internal imports; declaring it explicitly keeps the contract
    # self-describing.
    for required in ("torch", "tsr", "trimesh", "Pillow"):
        assert required in extras, (
            f"runtime imports {required} but it's missing from pip_extras"
        )


def test_pip_extras_pin_minimum_torch():
    """Fresh-venv installs need a torch new enough for TripoSR; pin >=2.1."""
    from muse.models.triposr import MANIFEST

    torch_specs = [e for e in MANIFEST["pip_extras"] if e.startswith("torch")]
    assert torch_specs, "torch missing from pip_extras"
    # First entry should be the bare `torch>=2.1.0` (not torchvision).
    assert any(
        e.startswith("torch>=") and "torchvision" not in e
        for e in torch_specs
    ), f"torch>=2.1.0 minimum not declared in {torch_specs!r}"


def test_license_is_mit():
    """TripoSR ships under MIT; verifies the license field hasn't drifted."""
    from muse.models.triposr import MANIFEST

    assert MANIFEST["license"] == "MIT"
