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
            id="kokoro-82m", modality="audio/speech", status="enabled",
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
    """--installed includes both enabled and disabled."""
    out = filter_rows(
        sample_rows, modality=None, installed=True, available=False,
    )
    assert {r.id for r in out} == {"kokoro-82m", "sd-turbo"}


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
    assert "enabled" not in statuses
    assert "disabled" not in statuses


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
