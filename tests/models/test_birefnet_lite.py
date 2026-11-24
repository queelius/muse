"""birefnet-lite: ZhengPeng7/BiRefNet_lite bundled model script tests."""
from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import muse.modalities.image_matting.runtimes.birefnet as birefnet_mod
from muse.modalities.image_matting.runtimes.birefnet import BiRefNetRuntime
from muse.models.birefnet_lite import MANIFEST, Model


def test_birefnet_lite_manifest():
    assert MANIFEST["model_id"] == "birefnet-lite"
    assert MANIFEST["modality"] == "image/matting"
    assert MANIFEST["hf_repo"] == "ZhengPeng7/BiRefNet_lite"
    assert MANIFEST["license"] == "MIT"
    caps = MANIFEST["capabilities"]
    assert caps["device"] == "auto"
    assert caps["memory_gb"] == 1.5
    assert caps["input_size"] == 1024
    dists = {e.split(">")[0].split("=")[0] for e in MANIFEST["pip_extras"]}
    assert {"torch", "transformers", "Pillow", "numpy", "einops"} <= dists
    for key in ("summary", "strengths", "limits", "quality_tier", "license", "description"):
        assert MANIFEST.get(key), key


def test_model_is_birefnet_runtime():
    assert Model is BiRefNetRuntime


_BLOCK_HEAVY = """
import importlib.abc, sys

class _NoHeavy(importlib.abc.MetaPathFinder):
    _BLOCKED = ("torch", "transformers", "PIL")
    def find_spec(self, name, path=None, target=None):
        top = name.split(".", 1)[0]
        if top in self._BLOCKED:
            raise ImportError(f"{top} blocked for this test")
        return None

sys.meta_path.insert(0, _NoHeavy())
"""


def test_discovered_and_importable_without_heavy_deps(tmp_path):
    code = _BLOCK_HEAVY + (
        "import muse.models.birefnet_lite as m\n"
        "assert m.MANIFEST['model_id'] == 'birefnet-lite'\n"
        "from muse.core.catalog import known_models\n"
        "assert 'birefnet-lite' in known_models(), sorted(known_models())\n"
        "for mod in ('torch', 'transformers', 'PIL'):\n"
        "    assert mod not in sys.modules, f'{mod} was imported'\n"
        "print('ok')\n"
    )
    env = {**os.environ, "MUSE_CATALOG_DIR": str(tmp_path / "catalog")}
    env.pop("MUSE_MODELS_DIR", None)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.strip().endswith("ok")


def test_model_instantiates_with_mocked_deps(monkeypatch):
    fake_model = MagicMock()
    fake_model.to.return_value = fake_model
    fake_auto = MagicMock()
    fake_auto.from_pretrained.return_value = fake_model
    fake_torch = SimpleNamespace(
        float16="fp16",
        bfloat16="bf16",
        float32="fp32",
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
    )
    monkeypatch.setattr(birefnet_mod, "torch", fake_torch)
    monkeypatch.setattr(birefnet_mod, "AutoModelForImageSegmentation", fake_auto)

    m = Model(
        hf_repo=MANIFEST["hf_repo"],
        local_dir="/tmp/birefnet-lite-local",
        model_id="birefnet-lite",
        **MANIFEST["capabilities"],
    )
    assert m.model_id == "birefnet-lite"
    assert m._device == "cpu"
    fake_auto.from_pretrained.assert_called_once_with(
        "/tmp/birefnet-lite-local",
        trust_remote_code=True,
        torch_dtype="fp32",
    )
