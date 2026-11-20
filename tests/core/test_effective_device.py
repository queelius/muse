"""One device-precedence rule: operator pin > manifest pin > --device > auto."""
from __future__ import annotations

import pytest

from muse.core.placement import effective_device


@pytest.mark.parametrize("override, cap, requested, expected", [
    ("cuda", "cpu", "mps", "cuda"),       # operator pin beats everything
    ("auto", "cpu", "cuda", "auto"),      # set-device auto un-pins, beats --device
    (None, "cpu", "cuda", "cpu"),         # manifest pin beats --device
    (None, "auto", "cuda", "cuda"),       # an auto manifest defers to --device
    (None, None, "mps", "mps"),
    (None, None, None, "auto"),
    ("", "", "", "auto"),
    ("CUDA", None, None, "cuda"),         # case-insensitive throughout
    (None, "CPU", "cuda", "cpu"),
    (None, "Auto", "CUDA", "cuda"),
])
def test_effective_device(override, cap, requested, expected):
    assert effective_device(override, cap, requested) == expected


CASES = [
    # (device_override, manifest device, requested)
    (None, None, "auto"),
    (None, "auto", "cuda"),
    (None, "cpu", "cuda"),
    (None, "cuda", "auto"),
    ("cpu", "cuda", "mps"),
    ("cuda", "cpu", "auto"),
    ("auto", "cpu", "cuda"),
    (None, None, "mps"),
]


def _manifest(cap):
    caps = {"memory_gb": 1.0}
    if cap is not None:
        caps["device"] = cap
    return {"model_id": "m", "modality": "audio/speech",
            "backend_path": "muse.fake:Model", "capabilities": caps}


def _entry(override):
    return {"device_override": override} if override else {}


@pytest.mark.parametrize("override, cap, requested", CASES)
def test_every_precedence_site_agrees(override, cap, requested, monkeypatch):
    """load_backend's rule is effective_device; the supervisor's sizing
    device, the manifest backfill, the probe worker and `muse models list`
    must place the model the same way."""
    from types import SimpleNamespace

    from muse.cli_impl import models_list, probe_worker, supervisor

    expected = effective_device(override, cap, requested)

    # supervisor sizing device
    entry = dict(_entry(override), manifest=_manifest(cap))
    assert supervisor._effective_model_device(
        entry, measured_device="cpu", supervisor_device=requested,
    ) == expected

    # backfill: the device handed to the director
    monkeypatch.setattr(supervisor, "_read_catalog", lambda: {"m": entry})
    out = supervisor.backfill_manifest_memory(
        _manifest(cap), "m", supervisor_device=requested,
    )
    got = str(out["capabilities"].get("device") or "auto").lower()
    assert got == expected

    # probe worker: identical unless the result is auto (then torch decides)
    if expected != "auto":
        fake = SimpleNamespace(
            extra=_manifest(cap)["capabilities"], backend_path="muse.fake:Model",
        )
        assert probe_worker._resolve_device(requested, fake, override) == expected

    # models list labels the pool the model is pinned to (no --device there)
    monkeypatch.setattr(supervisor, "_host_auto_uses_cuda", lambda: False)
    _s, _gb, label = models_list._model_memory_display(
        _manifest(cap)["capabilities"], _entry(override),
    )
    listed = effective_device(override, cap, None)
    assert label == ("CPU" if listed in ("cpu", "mps", "auto") else "GPU")
