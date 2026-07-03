"""Tests for the shared CPU-offload dispatch helper.

Covers `resolve_offload_mode` (global config override > per-model
capability > off) and `place_pipeline` (mutually-exclusive dispatch
between .to(device), enable_model_cpu_offload, and
enable_sequential_cpu_offload, plus best-effort vae tiling/slicing).
"""
from __future__ import annotations

import pytest

from muse.core import config
from muse.modalities.video_generation.runtimes._offload import (
    place_pipeline,
    resolve_offload_mode,
)


@pytest.fixture(autouse=True)
def _reset_config(monkeypatch):
    monkeypatch.delenv("MUSE_VIDEO_CPU_OFFLOAD", raising=False)
    config.reset_config()
    yield
    config.reset_config()


class FakePipe:
    """Records which placement/tiling methods were called."""

    def __init__(self, *, has_tiling=True):
        self.to_calls = []
        self.model_offload_calls = []
        self.sequential_offload_calls = []
        self.vae_tiling_called = False
        self.vae_slicing_called = False
        if has_tiling:
            self.enable_vae_tiling = self._enable_vae_tiling
            self.enable_vae_slicing = self._enable_vae_slicing

    def to(self, device):
        self.to_calls.append(device)
        return self

    def enable_model_cpu_offload(self, device=None):
        self.model_offload_calls.append(device)

    def enable_sequential_cpu_offload(self, device=None):
        self.sequential_offload_calls.append(device)

    def _enable_vae_tiling(self):
        self.vae_tiling_called = True

    def _enable_vae_slicing(self):
        self.vae_slicing_called = True


# --- resolve_offload_mode -------------------------------------------------


def test_resolve_offload_mode_capability_model():
    assert resolve_offload_mode("model") == "model"


def test_resolve_offload_mode_capability_sequential():
    assert resolve_offload_mode("sequential") == "sequential"


@pytest.mark.parametrize("raw", [None, "off", "false", "none", "", "no", "0"])
def test_resolve_offload_mode_off_values(raw):
    assert resolve_offload_mode(raw) is None


def test_resolve_offload_mode_unknown_value_warns_and_disables(caplog):
    assert resolve_offload_mode("bogus") is None


def test_resolve_offload_mode_global_override_beats_capability_off(monkeypatch):
    monkeypatch.setenv("MUSE_VIDEO_CPU_OFFLOAD", "off")
    config.reset_config()
    assert resolve_offload_mode("sequential") is None


def test_resolve_offload_mode_global_override_beats_capability_model(monkeypatch):
    monkeypatch.setenv("MUSE_VIDEO_CPU_OFFLOAD", "model")
    config.reset_config()
    assert resolve_offload_mode(None) == "model"


def test_resolve_offload_mode_unset_override_falls_through_to_capability():
    assert resolve_offload_mode("sequential") == "sequential"


# --- place_pipeline ---------------------------------------------------


def test_place_pipeline_model_mode_calls_enable_model_cpu_offload():
    pipe = FakePipe()
    out = place_pipeline(pipe, "cuda", cpu_offload="model")
    assert pipe.model_offload_calls == ["cuda"]
    assert pipe.to_calls == []
    assert pipe.sequential_offload_calls == []
    assert out is pipe


def test_place_pipeline_sequential_mode_calls_enable_sequential_cpu_offload():
    pipe = FakePipe()
    out = place_pipeline(pipe, "cuda", cpu_offload="sequential")
    assert pipe.sequential_offload_calls == ["cuda"]
    assert pipe.to_calls == []
    assert pipe.model_offload_calls == []
    assert out is pipe


@pytest.mark.parametrize("mode", [None, "off", "false"])
def test_place_pipeline_off_calls_to(mode):
    pipe = FakePipe()
    place_pipeline(pipe, "cuda", cpu_offload=mode)
    assert pipe.to_calls == ["cuda"]
    assert pipe.model_offload_calls == []
    assert pipe.sequential_offload_calls == []


def test_place_pipeline_cpu_device_calls_neither():
    pipe = FakePipe()
    out = place_pipeline(pipe, "cpu", cpu_offload="sequential")
    assert pipe.to_calls == []
    assert pipe.model_offload_calls == []
    assert pipe.sequential_offload_calls == []
    assert out is pipe


def test_place_pipeline_vae_tiling_calls_helpers_when_present():
    pipe = FakePipe(has_tiling=True)
    place_pipeline(pipe, "cuda", cpu_offload=None, vae_tiling=True)
    assert pipe.vae_tiling_called
    assert pipe.vae_slicing_called


def test_place_pipeline_vae_tiling_no_crash_when_absent():
    pipe = FakePipe(has_tiling=False)
    # Must not raise even though the mock lacks enable_vae_tiling/slicing.
    place_pipeline(pipe, "cuda", cpu_offload=None, vae_tiling=True)


def test_place_pipeline_vae_tiling_false_skips_helpers():
    pipe = FakePipe(has_tiling=True)
    place_pipeline(pipe, "cuda", cpu_offload=None, vae_tiling=False)
    assert not pipe.vae_tiling_called
    assert not pipe.vae_slicing_called


def test_place_pipeline_global_override_beats_capability(monkeypatch):
    monkeypatch.setenv("MUSE_VIDEO_CPU_OFFLOAD", "off")
    config.reset_config()
    pipe = FakePipe()
    place_pipeline(pipe, "cuda", cpu_offload="sequential")
    assert pipe.to_calls == ["cuda"]
    assert pipe.sequential_offload_calls == []
