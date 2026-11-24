"""Unit tests for ``src/muse/modalities/image_matting/hf.py`` (single-file
plugin load, sniffing, resolution, and search)."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

HF_PLUGIN_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "muse"
    / "modalities"
    / "image_matting"
    / "hf.py"
)


def _load_plugin_standalone():
    """Load hf.py the exact way Muse's resolver does (single-file import,
    proving there are no relative imports)."""
    spec = importlib.util.spec_from_file_location("image_matting_hf_standalone", HF_PLUGIN_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _info(
    repo_id: str,
    *,
    tags: list[str] | None = None,
    pipeline_tag: str | None = "image-segmentation",
    sha: str = "0123456789abcdef0123456789abcdef01234567",
    license_name: str = "mit",
    siblings: list[str] | None = None,
):
    return SimpleNamespace(
        id=repo_id,
        tags=tags if tags is not None else ["image-segmentation", "transformers"],
        pipeline_tag=pipeline_tag,
        sha=sha,
        card_data=SimpleNamespace(license=license_name),
        siblings=[
            SimpleNamespace(rfilename=f)
            for f in (siblings or ["model.safetensors", "config.json", "birefnet.py"])
        ],
    )


def test_plugin_metadata_and_priority():
    mod = _load_plugin_standalone()
    plugin = mod.HF_PLUGIN
    assert plugin["modality"] == "image/matting"
    assert plugin["priority"] == 95
    assert plugin["runtime_path"] == (
        "muse.modalities.image_matting.runtimes.birefnet:BiRefNetRuntime"
    )
    assert "torch>=2.1.0" in plugin["pip_extras"]
    assert "transformers>=4.40.0" in plugin["pip_extras"]
    assert "Pillow>=10.0" in plugin["pip_extras"]
    assert "numpy" in plugin["pip_extras"]
    assert "einops" in plugin["pip_extras"]


@pytest.mark.parametrize(
    "repo_id",
    [
        "ZhengPeng7/BiRefNet_lite",
        "ZhengPeng7/BiRefNet",
        "ZhengPeng7/BiRefNet_HR",
        "briaai/RMBG-2.0",
        "briaai/RMBG-1.4",
        "PramaLLC/BEN2",
        "org/ben2-base",
    ],
)
def test_sniff_matches_matting_repos(repo_id):
    mod = _load_plugin_standalone()
    assert mod.HF_PLUGIN["sniff"](_info(repo_id)) is True


@pytest.mark.parametrize(
    "info",
    [
        _info("facebook/sam2-hiera-tiny", tags=["mask-generation"]),
        _info("nvidia/segformer-b0-finetuned-ade-512-512"),
        _info("ZhengPeng7/BiRefNet-GGUF", tags=["gguf"]),
        _info("org/birefnet-llm", pipeline_tag="text-generation"),
        _info("stabilityai/sdxl-turbo", pipeline_tag="text-to-image"),
    ],
)
def test_sniff_rejects_non_matting_repos(info):
    mod = _load_plugin_standalone()
    assert mod.HF_PLUGIN["sniff"](info) is False


def test_resolve_builds_manifest_and_download(monkeypatch, tmp_path):
    mod = _load_plugin_standalone()
    commit_sha = "deadbeef" * 5
    info = _info("ZhengPeng7/BiRefNet_lite", sha=commit_sha, license_name="mit")

    resolved = mod.HF_PLUGIN["resolve"]("ZhengPeng7/BiRefNet_lite", None, info)
    manifest = resolved.manifest
    assert manifest["model_id"] == "birefnet_lite"
    assert manifest["modality"] == "image/matting"
    assert manifest["hf_repo"] == "ZhengPeng7/BiRefNet_lite"
    assert manifest["revision"] == commit_sha
    assert manifest["license"] == "mit"
    assert manifest["capabilities"]["device"] == "auto"
    assert manifest["capabilities"]["memory_gb"] == 1.5
    assert manifest["capabilities"]["input_size"] == 1024
    assert resolved.backend_path == (
        "muse.modalities.image_matting.runtimes.birefnet:BiRefNetRuntime"
    )

    dl_mock = MagicMock(return_value=str(tmp_path / "snap"))
    monkeypatch.setattr(mod, "snapshot_download", dl_mock)
    out_path = resolved.download(tmp_path)
    assert out_path == tmp_path / "snap"
    call_kwargs = dl_mock.call_args.kwargs
    assert call_kwargs["repo_id"] == "ZhengPeng7/BiRefNet_lite"
    assert call_kwargs["revision"] == commit_sha
    assert "*.py" in call_kwargs["allow_patterns"]
    assert "*.safetensors" in call_kwargs["allow_patterns"]


def test_resolve_hr_and_rmbg_capabilities():
    mod = _load_plugin_standalone()
    hr = mod.HF_PLUGIN["resolve"](
        "ZhengPeng7/BiRefNet_HR", None, _info("ZhengPeng7/BiRefNet_HR")
    )
    assert hr.manifest["capabilities"]["memory_gb"] == 4.0
    assert hr.manifest["capabilities"]["input_size"] == 2048

    rmbg = mod.HF_PLUGIN["resolve"](
        "briaai/RMBG-2.0", None, _info("briaai/RMBG-2.0")
    )
    assert rmbg.manifest["capabilities"]["memory_gb"] == 2.0
    assert rmbg.manifest["capabilities"]["input_size"] == 1024


def test_search_filters_to_matting_repos():
    mod = _load_plugin_standalone()
    api = MagicMock()
    api.list_models.return_value = [
        SimpleNamespace(id="ZhengPeng7/BiRefNet_lite", downloads=5000),
        SimpleNamespace(id="nvidia/segformer-b0", downloads=9000),
        SimpleNamespace(id="briaai/RMBG-2.0", downloads=12000),
    ]
    results = list(mod.HF_PLUGIN["search"](api, "background", sort="downloads", limit=10))
    assert [r.uri for r in results] == [
        "hf://ZhengPeng7/BiRefNet_lite",
        "hf://briaai/RMBG-2.0",
    ]
    assert all(r.modality == "image/matting" for r in results)
