"""Unit tests for cli_impl/models_list row-builder + filters.

Distinct from tests/test_cli.py which subprocesses `python -m muse.cli`
end-to-end. These tests poke the row-building logic directly: faster,
more precise, and don't need a full venv to run.
"""
from __future__ import annotations

import json
from io import StringIO
from unittest.mock import patch

import pytest

from muse.cli_impl.models_list import (
    _ListRow,
    _model_memory_display,
    build_rows,
    filter_rows,
    run_models_list,
)


@pytest.fixture
def sample_rows():
    return [
        _ListRow(
            id="kokoro-82m", modality="audio/speech", status="enabled_loaded",
            description="TTS", mem_str="~0.5 GB CPU", mem_gb=0.5,
            mem_device="CPU",
        ),
        _ListRow(
            id="sd-turbo", modality="image/generation", status="disabled",
            description="img2img", mem_str="~4.0 GB GPU", mem_gb=4.0,
            mem_device="GPU",
        ),
        _ListRow(
            id="bart-large-cnn", modality="text/summarization",
            status="recommended", description="summarization",
            mem_str="~1.5 GB CPU", mem_gb=1.5, mem_device="CPU",
        ),
        _ListRow(
            id="bark-small", modality="audio/speech", status="available",
            description="TTS", mem_str="~3.0 GB GPU", mem_gb=3.0,
            mem_device="GPU",
        ),
    ]


# Filters -----------------------------------------------------------------


def test_filter_rows_no_filters_returns_all(sample_rows):
    out = filter_rows(
        sample_rows, modality=None, installed=False, available=False,
    )
    assert len(out) == 4


def test_filter_rows_modality(sample_rows):
    out = filter_rows(
        sample_rows, modality="audio/speech", installed=False, available=False,
    )
    assert {r.id for r in out} == {"kokoro-82m", "bark-small"}


def test_filter_rows_installed_only(sample_rows):
    """--installed includes every catalog-enabled state plus disabled."""
    out = filter_rows(
        sample_rows, modality=None, installed=True, available=False,
    )
    assert {r.id for r in out} == {"kokoro-82m", "sd-turbo"}


def test_filter_rows_installed_includes_enabled_unloaded():
    """--installed must catch enabled_unloaded too: it's still a catalog row."""
    rows = [
        _ListRow(
            id="loaded", modality="x", status="enabled_loaded",
            description="d", mem_str="-", mem_gb=None, mem_device="GPU",
        ),
        _ListRow(
            id="unloaded", modality="x", status="enabled_unloaded",
            description="d", mem_str="-", mem_gb=None, mem_device="GPU",
        ),
        _ListRow(
            id="off", modality="x", status="disabled",
            description="d", mem_str="-", mem_gb=None, mem_device="GPU",
        ),
        _ListRow(
            id="rec", modality="x", status="recommended",
            description="d", mem_str="-", mem_gb=None, mem_device="GPU",
        ),
    ]
    out = filter_rows(rows, modality=None, installed=True, available=False)
    assert {r.id for r in out} == {"loaded", "unloaded", "off"}


def test_filter_rows_available_only(sample_rows):
    """--available includes both recommended and unbundled-but-known."""
    out = filter_rows(
        sample_rows, modality=None, installed=False, available=True,
    )
    assert {r.id for r in out} == {"bart-large-cnn", "bark-small"}


def test_filter_rows_compose(sample_rows):
    """Multiple filters AND together."""
    out = filter_rows(
        sample_rows, modality="audio/speech", installed=True, available=False,
    )
    assert {r.id for r in out} == {"kokoro-82m"}


# Memory display ----------------------------------------------------------


def test_memory_display_prefers_measured_over_annotation():
    extra = {"device": "cuda", "memory_gb": 5.0}
    catalog = {"measurements": {"cuda": {"peak_bytes": 1024**3 * 3}}}
    s, gb, device = _model_memory_display(extra, catalog)
    assert gb == 3.0  # measured wins
    assert device == "GPU"
    assert "GB GPU" in s
    assert s == "3.0 GB GPU"
    assert "~" not in s, "measured values should not have ~ prefix"


def test_memory_display_falls_back_to_annotation():
    extra = {"device": "cpu", "memory_gb": 0.7}
    s, gb, device = _model_memory_display(extra, None)
    assert gb == 0.7
    assert device == "CPU"
    assert s.startswith("~"), "annotation should have ~ prefix"


def test_memory_display_no_data():
    s, gb, device = _model_memory_display({}, None)
    assert gb is None
    assert s == "-"


def test_memory_display_reads_mps_bucket():
    # On Apple Silicon the probe persists its peak under the "mps" bucket.
    # A GPU-side model (device auto/mps) must surface that measurement, not
    # fall through to "-".
    extra = {"device": "auto"}
    catalog = {"measurements": {"mps": {"peak_bytes": 1024**3 * 2}}}
    s, gb, device = _model_memory_display(extra, catalog)
    assert gb == 2.0
    assert device == "GPU"
    assert s == "2.0 GB GPU"


def test_memory_display_mps_device_pin_reads_mps_bucket():
    extra = {"device": "mps"}
    catalog = {"measurements": {"mps": {"peak_bytes": 1024**3 * 4}}}
    s, gb, device = _model_memory_display(extra, catalog)
    assert gb == 4.0
    assert device == "GPU"


# JSON output -------------------------------------------------------------


def test_json_output_shape(sample_rows):
    """run_models_list --json emits a list of dicts with stable schema."""
    captured = StringIO()
    with patch("muse.cli_impl.models_list.build_rows", return_value=sample_rows):
        with patch("sys.stdout", captured):
            rc = run_models_list(
                modality=None, installed=False, available=False,
                as_json=True, no_color=False,
            )
    assert rc == 0
    parsed = json.loads(captured.getvalue())
    assert len(parsed) == 4
    keys = set(parsed[0])
    assert {"id", "modality", "status", "description", "memory"} <= keys


def test_json_output_filter_excludes_correct_rows(sample_rows):
    """Negative assertions are precise via JSON: --available excludes
    enabled and disabled rows, no exceptions."""
    captured = StringIO()
    with patch("muse.cli_impl.models_list.build_rows", return_value=sample_rows):
        with patch("sys.stdout", captured):
            run_models_list(
                modality=None, installed=False, available=True,
                as_json=True, no_color=False,
            )
    parsed = json.loads(captured.getvalue())
    statuses = {r["status"] for r in parsed}
    assert statuses == {"recommended", "available"}
    assert "enabled_loaded" not in statuses
    assert "enabled_unloaded" not in statuses
    assert "disabled" not in statuses


# v0.40.0 lazy-load: enabled_unloaded state ---------------------------------


def test_build_rows_classifies_loaded_vs_unloaded(monkeypatch, tmp_path):
    """When the director reports a model as currently loaded, it
    classifies as enabled_loaded. Catalog-enabled but not in director
    state -> enabled_unloaded."""
    from muse.cli_impl import models_list as ml

    # Two pulled+enabled models: one is in director.loaded, the other isn't.
    # Stub catalog/curated/known so build_rows returns deterministic shape.
    fake_known = {
        "loaded-id": _FakeKnown("loaded-id", "audio/speech", "loaded model"),
        "unloaded-id": _FakeKnown("unloaded-id", "audio/speech", "unloaded model"),
    }
    fake_catalog = {
        "loaded-id": {"enabled": True},
        "unloaded-id": {"enabled": True},
    }
    fake_loaded = {"loaded-id"}  # what the director thinks is loaded

    monkeypatch.setattr(ml, "_load_known_models", lambda: fake_known)
    monkeypatch.setattr(ml, "_load_curated_entries", lambda: {})
    monkeypatch.setattr(ml, "_read_catalog_data", lambda: fake_catalog)
    monkeypatch.setattr(ml, "_is_pulled_for_row", lambda mid: True)
    monkeypatch.setattr(ml, "_is_enabled_for_row", lambda mid: True)
    monkeypatch.setattr(ml, "_director_loaded_ids", lambda: fake_loaded)

    rows = ml.build_rows()
    by_id = {r.id: r for r in rows}
    assert by_id["loaded-id"].status == "enabled_loaded"
    assert by_id["unloaded-id"].status == "enabled_unloaded"


def test_build_rows_director_unreachable_treats_all_as_unloaded(monkeypatch):
    """When the director isn't reachable (CLI used outside a running
    supervisor), every catalog-enabled row falls back to enabled_unloaded."""
    from muse.cli_impl import models_list as ml

    fake_known = {
        "a": _FakeKnown("a", "audio/speech", "a"),
        "b": _FakeKnown("b", "audio/speech", "b"),
    }
    fake_catalog = {
        "a": {"enabled": True},
        "b": {"enabled": True},
    }

    monkeypatch.setattr(ml, "_load_known_models", lambda: fake_known)
    monkeypatch.setattr(ml, "_load_curated_entries", lambda: {})
    monkeypatch.setattr(ml, "_read_catalog_data", lambda: fake_catalog)
    monkeypatch.setattr(ml, "_is_pulled_for_row", lambda mid: True)
    monkeypatch.setattr(ml, "_is_enabled_for_row", lambda mid: True)
    # Empty set -> nothing is loaded; CLI doesn't know runtime state
    monkeypatch.setattr(ml, "_director_loaded_ids", lambda: set())

    rows = ml.build_rows()
    statuses = {r.id: r.status for r in rows}
    assert statuses == {"a": "enabled_unloaded", "b": "enabled_unloaded"}


class _FakeKnown:
    """Minimal stand-in for muse.core.catalog.CatalogEntry used in tests."""
    def __init__(self, model_id, modality, description):
        self.model_id = model_id
        self.modality = modality
        self.description = description
        self.extra = {}


# v0.47.4: read loaded ids from the PUBLIC /v1/models endpoint ------------


class _FakeResp:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


def _fake_httpx_client(*, captured_urls, resp=None, raise_exc=None):
    """Build a fake httpx.Client class recording GET urls."""

    class _FakeClient:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url):
            captured_urls.append(url)
            if raise_exc is not None:
                raise raise_exc
            return resp

    return _FakeClient


def test_director_loaded_ids_uses_public_v1_models(monkeypatch):
    """`_director_loaded_ids` reads the PUBLIC GET /v1/models endpoint
    (no admin token), returning the ids of entries with loaded == True.
    /v1/models carries `loaded: bool` per entry since v0.47.3."""
    from muse.cli_impl import models_list as ml

    monkeypatch.delenv("MUSE_ADMIN_TOKEN", raising=False)
    monkeypatch.setenv("MUSE_SERVER", "http://server:8000")

    urls: list[str] = []
    payload = {"object": "list", "data": [
        {"id": "hot-a", "loaded": True},
        {"id": "hot-b", "loaded": True},
        {"id": "cold-c", "loaded": False},
    ]}
    monkeypatch.setattr(
        "httpx.Client",
        _fake_httpx_client(captured_urls=urls, resp=_FakeResp(200, payload)),
    )

    loaded = ml._director_loaded_ids()
    assert loaded == {"hot-a", "hot-b"}
    assert urls == ["http://server:8000/v1/models"]


def test_director_loaded_ids_works_without_token(monkeypatch):
    """The endpoint is public: a loaded set is returned even with no
    MUSE_ADMIN_TOKEN set (the old admin-only path returned empty here)."""
    from muse.cli_impl import models_list as ml

    monkeypatch.delenv("MUSE_ADMIN_TOKEN", raising=False)
    monkeypatch.delenv("MUSE_SERVER", raising=False)

    urls: list[str] = []
    payload = {"data": [{"id": "loaded-one", "loaded": True}]}
    monkeypatch.setattr(
        "httpx.Client",
        _fake_httpx_client(captured_urls=urls, resp=_FakeResp(200, payload)),
    )

    assert ml._director_loaded_ids() == {"loaded-one"}
    # Default base URL when MUSE_SERVER is unset.
    assert urls == ["http://localhost:8000/v1/models"]


def test_director_loaded_ids_server_unreachable_returns_empty(monkeypatch):
    """When /v1/models is unreachable, return an empty set so every
    catalog-enabled row falls back to enabled_unloaded."""
    import httpx

    from muse.cli_impl import models_list as ml

    monkeypatch.setattr(
        "httpx.Client",
        _fake_httpx_client(
            captured_urls=[],
            raise_exc=httpx.ConnectError("down", request=None),
        ),
    )
    assert ml._director_loaded_ids() == set()


def test_director_loaded_ids_non_200_returns_empty(monkeypatch):
    """A non-200 /v1/models response yields an empty loaded set."""
    from muse.cli_impl import models_list as ml

    monkeypatch.setattr(
        "httpx.Client",
        _fake_httpx_client(
            captured_urls=[], resp=_FakeResp(503, {"data": []}),
        ),
    )
    assert ml._director_loaded_ids() == set()


def test_director_loaded_ids_ignores_unloaded_and_malformed(monkeypatch):
    """Entries that are unloaded, malformed, or missing an id do not
    contribute to the loaded set."""
    from muse.cli_impl import models_list as ml

    payload = {"data": [
        {"id": "good", "loaded": True},
        {"id": "unloaded", "loaded": False},
        {"loaded": True},          # no id
        {"id": "", "loaded": True},  # empty id
        "not-a-dict",
        {"id": "no-loaded-key"},   # absent loaded -> treated as not loaded
    ]}
    monkeypatch.setattr(
        "httpx.Client",
        _fake_httpx_client(captured_urls=[], resp=_FakeResp(200, payload)),
    )
    assert ml._director_loaded_ids() == {"good"}


def test_json_output_no_memory_for_unprobed_unannotated():
    """Rows with no memory data emit memory: null, not a fake key."""
    rows = [_ListRow(
        id="x", modality="m", status="available", description="d",
        mem_str="-", mem_gb=None, mem_device="GPU",
    )]
    captured = StringIO()
    with patch("muse.cli_impl.models_list.build_rows", return_value=rows):
        with patch("sys.stdout", captured):
            run_models_list(
                modality=None, installed=False, available=False,
                as_json=True, no_color=False,
            )
    parsed = json.loads(captured.getvalue())
    assert parsed[0]["memory"] is None
