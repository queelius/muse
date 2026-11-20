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


def test_load_backend_and_supervisor_share_the_rule(monkeypatch):
    """load_backend, placement and the supervisor all call effective_device."""
    import inspect

    from muse.cli_impl import supervisor
    from muse.core import catalog, placement

    for fn in (catalog.load_backend, placement.resolve_model_placement,
               supervisor._effective_model_device):
        assert "effective_device(" in inspect.getsource(fn), fn.__qualname__
