"""Resolver pulls must not replace a catalog entry owned by another source.

Regression for the 2026-10-18 install audit: resolver plugins derive the
catalog id from the repo basename, so `hf://alice/Foo-GGUF@q4_k_m` and
`hf://bob/Foo-GGUF@q4_k_m` both synthesize `foo-gguf-q4-k-m` and the second
pull silently overwrote the first.
"""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from muse.core.catalog import (
    ModelIdConflictError,
    _read_catalog,
    _reset_known_models_cache,
    _reset_read_catalog_cache,
    pull,
)
from muse.core.curated import CuratedEntry
from muse.core.resolvers import (
    ResolvedModel,
    Resolver,
    _reset_registry_for_tests,
    register_resolver,
)


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
    _reset_known_models_cache()
    _reset_read_catalog_cache()
    _reset_registry_for_tests()
    yield tmp_path
    _reset_registry_for_tests()
    _reset_known_models_cache()
    _reset_read_catalog_cache()


class _BasenameResolver(Resolver):
    """Mimics the HF plugins: catalog id is the lowercased repo basename."""

    scheme = "hf"

    def __init__(self):
        self.downloads: list[str] = []

    def resolve(self, uri):
        ref = uri.split("://", 1)[1]
        repo, _, variant = ref.partition("@")
        model_id = repo.split("/", 1)[-1].lower()

        def _download(cache_root, _repo=repo):
            self.downloads.append(_repo)
            return cache_root / "weights" / _repo.replace("/", "--")

        return ResolvedModel(
            manifest={
                "model_id": model_id,
                "modality": "chat/completion",
                "hf_repo": repo,
                "pip_extras": [],
                "capabilities": {"variant": variant or None},
            },
            backend_path="fake.mod:Cls",
            download=_download,
        )

    def search(self, q, **k):
        return []


@pytest.fixture
def resolver():
    r = _BasenameResolver()
    register_resolver(r)
    return r


@pytest.fixture
def no_install():
    with patch("muse.core.catalog_pull.create_venv") as create, \
         patch("muse.core.catalog_pull.install_into_venv") as install, \
         patch("muse.core.catalog_pull.check_system_packages", return_value=[]):
        yield create, install


def _catalog_bytes(root) -> bytes:
    return (root / "catalog.json").read_bytes()


def test_different_publisher_same_id_is_refused_before_any_work(
    _isolate, resolver, no_install,
):
    create, install = no_install
    pull("hf://alice/Foo-GGUF@q4_k_m")
    before = _catalog_bytes(_isolate)
    create.reset_mock()
    install.reset_mock()

    with pytest.raises(ModelIdConflictError) as exc:
        pull("hf://bob/Foo-GGUF@q4_k_m")

    message = str(exc.value)
    assert "hf://alice/Foo-GGUF@q4_k_m" in message
    assert "hf://bob/Foo-GGUF@q4_k_m" in message
    assert "--id" in message
    assert _catalog_bytes(_isolate) == before
    assert resolver.downloads == ["alice/Foo-GGUF"]
    create.assert_not_called()
    install.assert_not_called()


def test_repull_of_same_source_is_allowed_despite_case(_isolate, resolver, no_install):
    pull("hf://alice/Foo-GGUF@q4_k_m")
    pull("hf://Alice/foo-gguf@Q4_K_M")

    entry = _read_catalog()["foo-gguf"]
    assert entry["hf_repo"] == "Alice/foo-gguf"
    assert resolver.downloads == ["alice/Foo-GGUF", "Alice/foo-gguf"]


def test_explicit_id_installs_second_publisher_alongside(_isolate, resolver, no_install):
    pull("hf://alice/Foo-GGUF@q4_k_m")
    pull("hf://bob/Foo-GGUF@q4_k_m", model_id="bob-foo")

    catalog = _read_catalog()
    assert catalog["foo-gguf"]["source"] == "hf://alice/Foo-GGUF@q4_k_m"
    assert catalog["bob-foo"]["source"] == "hf://bob/Foo-GGUF@q4_k_m"
    assert catalog["bob-foo"]["manifest"]["model_id"] == "bob-foo"


def test_bare_id_repull_of_explicit_id_keeps_its_source(_isolate, resolver, no_install):
    pull("hf://bob/Foo-GGUF@q4_k_m", model_id="bob-foo")
    pull("bob-foo")

    assert _read_catalog()["bob-foo"]["source"] == "hf://bob/Foo-GGUF@q4_k_m"


def test_same_repo_new_variant_under_same_id_is_an_update(
    _isolate, resolver, no_install,
):
    pull("hf://alice/Foo-GGUF@q4_k_m", model_id="foo")
    pull("hf://alice/Foo-GGUF@q8_0", model_id="foo")

    assert _read_catalog()["foo"]["source"] == "hf://alice/Foo-GGUF@q8_0"


def test_resolver_pull_cannot_take_a_bundled_script_id(_isolate, resolver, no_install):
    create, _ = no_install
    with pytest.raises(ModelIdConflictError) as exc:
        pull("hf://mallory/kokoro-82m")
    assert "bundled" in str(exc.value)
    assert resolver.downloads == []
    create.assert_not_called()
    assert "kokoro-82m" not in _read_catalog()


def test_explicit_id_cannot_take_a_bundled_script_id(_isolate, resolver, no_install):
    with pytest.raises(ModelIdConflictError):
        pull("hf://alice/Foo-GGUF@q4_k_m", model_id="kokoro-82m")
    assert resolver.downloads == []


def test_explicit_id_cannot_take_another_curated_id(_isolate, resolver, no_install):
    curated = CuratedEntry(
        id="friendly", bundled=False, uri="hf://carol/Bar", modality=None,
        size_gb=None, description=None, tags=(), capabilities={},
    )
    with patch(
        "muse.core.catalog_pull.find_curated",
        side_effect=lambda i: curated if i == "friendly" else None,
    ), patch(
        "muse.core.catalog_models.find_curated",
        side_effect=lambda i: curated if i == "friendly" else None,
    ):
        with pytest.raises(ModelIdConflictError):
            pull("hf://alice/Foo-GGUF@q4_k_m", model_id="friendly")
    assert resolver.downloads == []


def test_explicit_id_requires_a_resolver_uri(_isolate, resolver, no_install):
    with pytest.raises(ValueError, match="resolver URI"):
        pull("kokoro-82m", model_id="voice")


def test_explicit_id_is_validated_as_a_path_component(_isolate, resolver, no_install):
    with pytest.raises(ValueError):
        pull("hf://alice/Foo-GGUF@q4_k_m", model_id="../escape")
    assert resolver.downloads == []


def test_curated_pull_may_replace_its_own_entry_from_a_new_uri(
    _isolate, resolver, no_install,
):
    """A curated id is owned by its curated.yaml row; when Muse updates
    that row's URI, re-pulling the curated id must follow it."""
    catalog_path = _isolate / "catalog.json"
    catalog_path.write_text(json.dumps({
        "friendly": {
            "hf_repo": "olduser/Bar",
            "source": "hf://olduser/Bar",
            "manifest": {
                "model_id": "friendly", "modality": "chat/completion",
                "hf_repo": "olduser/Bar", "backend_path": "fake.mod:Cls",
            },
            "enabled": True,
        },
    }))
    _reset_read_catalog_cache()
    curated = CuratedEntry(
        id="friendly", bundled=False, uri="hf://carol/Bar", modality=None,
        size_gb=None, description=None, tags=(), capabilities={},
    )
    with patch(
        "muse.core.catalog_pull.find_curated",
        side_effect=lambda i: curated if i == "friendly" else None,
    ), patch(
        "muse.core.catalog_models.find_curated",
        side_effect=lambda i: curated if i == "friendly" else None,
    ):
        pull("friendly")

    assert _read_catalog()["friendly"]["source"] == "hf://carol/Bar"


class TestPullCli:
    """`muse pull` surfaces conflicts as a one-line error and threads --id."""

    def test_conflict_exits_2_without_traceback(self, monkeypatch):
        from typer.testing import CliRunner

        import muse.core.catalog as catalog
        from muse.cli import app

        def _conflict(identifier, *, base_override=None, model_id=None):
            raise ModelIdConflictError("model id 'foo' is already installed")

        monkeypatch.setattr(catalog, "pull", _conflict)
        result = CliRunner().invoke(app, ["pull", "hf://bob/foo", "--no-probe"])

        assert result.exit_code == 2
        assert "already installed" in result.output
        assert result.exception is None or isinstance(result.exception, SystemExit)

    def test_id_is_threaded_to_catalog_pull(self, monkeypatch):
        from typer.testing import CliRunner

        import muse.core.catalog as catalog
        from muse.cli import app

        calls = []
        monkeypatch.setattr(
            catalog, "pull",
            lambda identifier, **kw: calls.append((identifier, kw)),
        )
        result = CliRunner().invoke(
            app, ["pull", "hf://bob/foo", "--id", "bob-foo", "--no-probe"],
        )

        assert result.exit_code == 0, result.output
        assert calls == [("hf://bob/foo", {"base_override": None, "model_id": "bob-foo"})]

    def test_id_with_non_uri_exits_2(self, monkeypatch):
        from typer.testing import CliRunner

        import muse.core.catalog as catalog
        from muse.cli import app

        monkeypatch.setattr(catalog, "pull", lambda *a, **k: pytest.fail("pulled"))
        result = CliRunner().invoke(app, ["pull", "kokoro-82m", "--id", "voice"])

        assert result.exit_code == 2
        assert "resolver URIs" in result.output

    def test_invalid_id_exits_2(self, monkeypatch):
        from typer.testing import CliRunner

        import muse.core.catalog as catalog
        from muse.cli import app

        monkeypatch.setattr(catalog, "pull", lambda *a, **k: pytest.fail("pulled"))
        result = CliRunner().invoke(app, ["pull", "hf://bob/foo", "--id", "a/b"])

        assert result.exit_code == 2
        assert "invalid model id" in result.output
