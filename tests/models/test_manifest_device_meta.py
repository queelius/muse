"""Meta-test: every bundled script declares a valid capabilities.device.

The v0.48.0 device policy makes the control plane (LoadDirector admission /
eviction, supervisor servability, idle sweeper) size a model against a memory
pool derived from ``manifest.capabilities.device`` via ``.get("device", "cpu")``.
A bundled script that OMITS the key is therefore sized against host RAM even
when it loads on CUDA (its constructor default resolves ``"auto"`` -> cuda),
so its VRAM is never counted or evicted and the GPU can OOM under pressure.

This guard asserts the invariant documented in CLAUDE.md: every CUDA-safe
bundled model declares ``device: "auto"`` (and the one CPU-only ONNX model
declares ``"cpu"``). Concretely: every bundled script's MANIFEST capabilities
carries a ``device`` key whose value is one of the four accepted pools.
"""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest

VALID_DEVICES = {"auto", "cpu", "cuda", "mps"}


def _bundled_scripts() -> list[Path]:
    here = Path(__file__).resolve()
    repo_root = here.parents[2]
    models_dir = repo_root / "src" / "muse" / "models"
    return sorted(p for p in models_dir.glob("*.py") if p.name != "__init__.py")


def _read_manifest(script: Path) -> dict:
    mod = importlib.import_module("muse.models." + script.stem)
    return getattr(mod, "MANIFEST", {})


@pytest.mark.parametrize("script", _bundled_scripts(), ids=lambda p: p.stem)
def test_bundled_script_declares_valid_device(script: Path):
    manifest = _read_manifest(script)
    caps = manifest.get("capabilities", {})
    assert "device" in caps, (
        f"{script.stem}: MANIFEST['capabilities'] omits 'device'. The control "
        f"plane defaults a missing key to 'cpu' and sizes VRAM against host "
        f"RAM (GPU OOM risk). Declare 'device': 'auto' (CUDA-safe) or 'cpu' "
        f"(CPU-only)."
    )
    assert caps["device"] in VALID_DEVICES, (
        f"{script.stem}: capabilities.device={caps['device']!r} is not one of "
        f"{sorted(VALID_DEVICES)}."
    )
