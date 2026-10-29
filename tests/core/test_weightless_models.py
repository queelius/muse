"""Weightless model scripts (`MANIFEST["weights"] = "none"`).

A script that declares no weights pulls a venv only (zero downloads), gets an
empty owned `local_dir`, and carries a `code` provenance receipt over its own
source instead of an HF revision. The seam is opt-in: a script without
`hf_repo` that does not declare `weights: none` is still skipped exactly as
before.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from muse.core import catalog_models
from muse.core.catalog import (
    _read_catalog,
    known_models,
    load_backend,
    model_load_blocker,
    pull,
    remove,
)
from muse.core.catalog_models import (
    _reset_known_models_cache,
    get_manifest,
)
from muse.core.catalog_store import _catalog_dir, _reset_read_catalog_cache
from muse.core.provenance import ArtifactComponent, model_revision, revision_headers


TOY_SCRIPT = '''
MANIFEST = {
    "model_id": "toy-tools",
    "modality": "image/generation",
    "weights": "none",
    "description": "deterministic toy tools",
    "pip_extras": ["pillow"],
    "capabilities": {"device": "cpu", "memory_gb": 0.5},
}


class Model:
    def __init__(self, *, hf_repo, local_dir, device="cpu", **kwargs):
        self.hf_repo = hf_repo
        self.local_dir = local_dir
        self.device = device
        self.kwargs = kwargs
'''

NO_REPO_SCRIPT = '''
MANIFEST = {
    "model_id": "toy-norepo",
    "modality": "image/generation",
    "capabilities": {"memory_gb": 0.5},
}


class Model:
    def __init__(self, **_):
        pass
'''


@pytest.fixture
def models_dir(tmp_path, monkeypatch):
    bundled = tmp_path / "bundled_models"
    bundled.mkdir()
    (bundled / "toy_tools.py").write_text(TOY_SCRIPT)
    user = tmp_path / "user_models"
    user.mkdir()
    monkeypatch.setattr(catalog_models, "_bundled_models_dir", lambda: bundled)
    monkeypatch.setattr(catalog_models, "_user_models_dir", lambda: user)
    monkeypatch.delenv("MUSE_MODELS_DIR", raising=False)
    _reset_known_models_cache()
    _reset_read_catalog_cache()
    yield bundled
    _reset_known_models_cache()
    _reset_read_catalog_cache()


def _fake_create_venv(target: Path) -> None:
    python = Path(target) / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\n")
    python.chmod(0o755)


def _forbidden(*_args, **_kwargs):
    raise AssertionError("a weightless pull must not download anything")


@pytest.fixture
def no_install():
    with patch("muse.core.catalog_pull.create_venv", side_effect=_fake_create_venv), \
         patch("muse.core.catalog_pull.install_into_venv") as install, \
         patch("muse.core.catalog_pull.install_python_sources"), \
         patch("muse.core.catalog_pull.check_system_packages", return_value=[]), \
         patch("muse.core.catalog_pull.dedupe_after_install"), \
         patch("muse.core.catalog_pull.snapshot_download", side_effect=_forbidden), \
         patch("muse.core.catalog_pull.seed_repo", side_effect=_forbidden), \
         patch("muse.core.artifacts.download_hf_artifact_bundle", side_effect=_forbidden):
        yield install


@pytest.fixture
def pulled(models_dir, no_install):
    model_id = pull("toy-tools")
    _reset_read_catalog_cache()
    return model_id


def test_catalog_entry_for_weightless_script(models_dir):
    entry = known_models()["toy-tools"]
    assert entry.hf_repo == ""
    assert entry.extra["weights"] == "none"
    assert entry.bundled is True


def test_script_source_sha256_hashes_script_bytes(models_dir):
    from muse.core.catalog_models import script_source_sha256

    expected = hashlib.sha256(TOY_SCRIPT.encode()).hexdigest()
    assert script_source_sha256("toy-tools") == expected


def test_weightless_pull_downloads_nothing(pulled, models_dir, no_install):
    assert pulled == "toy-tools"
    row = _read_catalog()["toy-tools"]
    assert row["weights"] == "none"
    assert "hf_repo" not in row and "revision" not in row
    local_dir = Path(row["local_dir"])
    assert local_dir == _catalog_dir() / "local" / "toy-tools"
    assert local_dir.is_dir() and not any(local_dir.iterdir())
    assert row["enabled"] is True
    assert row["python_path"].endswith("/bin/python")
    receipt = row["artifact_provenance"]
    assert len(receipt) == 1
    assert receipt[0]["kind"] == "code"
    assert receipt[0]["model_id"] == "toy-tools"
    assert receipt[0]["muse_version"]
    assert receipt[0]["source_sha256"] == hashlib.sha256(
        (models_dir / "toy_tools.py").read_bytes()
    ).hexdigest()
    # The venv still gets muse[server] plus the declared pip_extras.
    installed = [call.args[1] for call in no_install.call_args_list]
    assert ["pillow"] in installed


def test_weightless_local_dir_is_private(pulled):
    local_dir = Path(_read_catalog()["toy-tools"]["local_dir"])
    assert (local_dir.parent.stat().st_mode & 0o077) == 0


def test_weightless_repull_is_idempotent(pulled, no_install):
    first = _read_catalog()["toy-tools"]
    pull("toy-tools")
    _reset_read_catalog_cache()
    second = _read_catalog()["toy-tools"]
    assert second["local_dir"] == first["local_dir"]
    assert second["artifact_provenance"] == first["artifact_provenance"]


def test_weightless_load_blocker_is_none_after_pull(pulled):
    assert model_load_blocker("toy-tools", _read_catalog()["toy-tools"]) is None


def test_weightless_receipt_mismatch_blocks_load(pulled, models_dir):
    (models_dir / "toy_tools.py").write_text(TOY_SCRIPT + "\n# edited\n")
    blocker = model_load_blocker("toy-tools", _read_catalog()["toy-tools"])
    assert blocker is not None
    assert "provenance" in blocker
    assert "re-run `muse pull toy-tools`" in blocker


def test_weightless_missing_receipt_blocks_load(pulled):
    row = dict(_read_catalog()["toy-tools"])
    row.pop("artifact_provenance")
    blocker = model_load_blocker("toy-tools", row)
    assert blocker is not None and "re-run `muse pull toy-tools`" in blocker


def test_weightless_revision_headers(pulled, monkeypatch):
    import muse

    monkeypatch.setattr(muse, "__version__", "1.2.3")
    row = _read_catalog()["toy-tools"]
    rev = model_revision("toy-tools", row, get_manifest("toy-tools"))
    assert rev.revision is None
    assert rev.components == (
        ArtifactComponent("code", "muse/toy-tools", "muse-1.2.3"),)
    sha = row["artifact_provenance"][0]["source_sha256"]
    expected = hashlib.sha256(f"{sha}:1.2.3".encode()).hexdigest()[:16]
    assert rev.artifacts_digest == expected
    headers = dict(revision_headers(rev))
    assert b"x-muse-model-revision" not in headers
    assert headers[b"x-muse-model-artifacts"] == expected.encode()


def test_weightless_digest_follows_running_muse_version(pulled, monkeypatch):
    # A muse upgrade can change the library code a weightless alias script
    # imports without changing the script, so the running version is folded
    # into the digest; the recorded receipt version is not enforced.
    import muse

    row = _read_catalog()["toy-tools"]
    manifest = get_manifest("toy-tools")
    monkeypatch.setattr(muse, "__version__", "1.2.3")
    old = model_revision("toy-tools", row, manifest)
    monkeypatch.setattr(muse, "__version__", "1.2.4")
    new = model_revision("toy-tools", row, manifest)
    assert old.artifacts_digest != new.artifacts_digest
    assert new.components[0].revision == "muse-1.2.4"
    assert new.revision is None
    assert b"x-muse-model-revision" not in dict(revision_headers(new))


def test_load_backend_constructs_weightless_model(pulled):
    row = _read_catalog()["toy-tools"]
    fake_class = MagicMock()
    fake_module = MagicMock()
    fake_module.Model = fake_class
    with patch("muse.core.catalog._import_backend_module", return_value=fake_module):
        load_backend("toy-tools", device="cpu")
    kwargs = fake_class.call_args.kwargs
    assert kwargs["hf_repo"] == ""
    assert kwargs["local_dir"] == row["local_dir"]
    assert kwargs["model_id"] == "toy-tools"
    assert kwargs["device"] == "cpu"


def test_load_backend_real_class(pulled):
    backend = load_backend("toy-tools")
    assert backend.hf_repo == ""
    assert Path(backend.local_dir).is_dir()


def test_remove_purge_deletes_weightless_dirs(pulled):
    row = _read_catalog()["toy-tools"]
    local_dir = Path(row["local_dir"])
    venv = Path(row["venv_path"])
    remove("toy-tools", purge=True)
    _reset_read_catalog_cache()
    assert "toy-tools" not in _read_catalog()
    assert not local_dir.exists()
    assert not venv.exists()


def test_remove_purge_refuses_foreign_weightless_local_dir(pulled, tmp_path):
    from muse.core.catalog import _write_catalog
    from muse.core.catalog_store import CatalogError

    foreign = tmp_path / "elsewhere"
    foreign.mkdir()
    catalog = _read_catalog()
    catalog["toy-tools"]["local_dir"] = str(foreign)
    _write_catalog(catalog)
    with pytest.raises(CatalogError):
        remove("toy-tools", purge=True)
    assert foreign.is_dir()


def test_refresh_tolerates_weightless_entry(pulled):
    from muse.cli_impl import refresh

    with patch.object(refresh, "_run_pip", return_value=None) as run_pip, \
         patch.object(refresh, "install_python_sources"), \
         patch.object(refresh, "dedupe_after_install"):
        result = refresh.refresh_one("toy-tools")
    assert result.state == "ok", result
    assert run_pip.called
    row = _read_catalog()["toy-tools"]
    assert Path(row["local_dir"]).is_dir()
    assert model_load_blocker("toy-tools", row) is None


def test_storage_inspection_has_no_issues_for_weightless(pulled):
    from muse.core.storage import inspect_storage

    report = inspect_storage(include_shared=False)
    assert [i for i in report.issues if i.model_id == "toy-tools"] == []
    assert "toy-tools" not in report.shared_hf_models


def test_supervisor_sizes_weightless_from_memory_gb(pulled):
    from muse.cli_impl.supervisor import _has_memory_data

    entry = dict(_read_catalog()["toy-tools"])
    entry["manifest"] = get_manifest("toy-tools")
    has_data, memory_gb, device = _has_memory_data(entry)
    assert has_data is True
    assert memory_gb == pytest.approx(0.5)
    assert device == "cpu"


def test_bundled_script_without_hf_repo_and_without_weights_none_still_skipped(
    models_dir, no_install,
):
    (models_dir / "toy_norepo.py").write_text(NO_REPO_SCRIPT)
    _reset_known_models_cache()
    assert "toy-norepo" not in known_models()
    with pytest.raises(KeyError):
        pull("toy-norepo")


@pytest.mark.parametrize(
    "extra_manifest",
    [
        '"hf_repo": "org/repo",',
        '"revision": "' + "a" * 40 + '",',
        '"hf_artifacts": [],',
        '"allow_patterns": ["*.bin"],',
    ],
)
def test_weightless_script_cannot_also_declare_downloads(
    models_dir, no_install, extra_manifest,
):
    from muse.core.catalog_store import CatalogError

    script = TOY_SCRIPT.replace(
        '"weights": "none",', '"weights": "none",\n    ' + extra_manifest,
    )
    (models_dir / "toy_tools.py").write_text(script)
    _reset_known_models_cache()
    with pytest.raises(CatalogError, match="weights"):
        pull("toy-tools")
    assert "toy-tools" not in _read_catalog()


def test_weightless_hf_repo_none_is_accepted(models_dir, no_install):
    script = TOY_SCRIPT.replace('"weights": "none",', '"weights": "none",\n    "hf_repo": None,')
    (models_dir / "toy_tools.py").write_text(script)
    _reset_known_models_cache()
    assert known_models()["toy-tools"].hf_repo == ""
    assert pull("toy-tools") == "toy-tools"


def test_models_info_reports_weightless(pulled):
    from muse.cli_impl.models_info_display import format_info

    text = format_info(
        "toy-tools",
        catalog_known=known_models(),
        catalog_data=_read_catalog()["toy-tools"],
    )
    assert "hf_repo:         (none: weightless)" in text
    assert "weights:         none" in text


def test_weightless_script_must_declare_memory_gb(models_dir, no_install):
    from muse.core.catalog_store import CatalogError

    script = TOY_SCRIPT.replace('"memory_gb": 0.5', '"memory_gb": 0')
    (models_dir / "toy_tools.py").write_text(script)
    _reset_known_models_cache()
    with pytest.raises(CatalogError, match="memory_gb"):
        pull("toy-tools")


def test_catalog_schema_accepts_code_receipt_and_rejects_bad_ones(pulled):
    from muse.core.catalog_store import _CatalogSchemaError, _validate_catalog_shape

    row = _read_catalog()["toy-tools"]
    _validate_catalog_shape({"toy-tools": row})
    bad_digest = dict(row, artifact_provenance=[
        dict(row["artifact_provenance"][0], source_sha256="abc"),
    ])
    with pytest.raises(_CatalogSchemaError, match="source_sha256"):
        _validate_catalog_shape({"toy-tools": bad_digest})
    with pytest.raises(_CatalogSchemaError, match="weights"):
        _validate_catalog_shape({"toy-tools": dict(row, weights="hf")})
