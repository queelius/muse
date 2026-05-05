"""MemoryProbe: pynvml + psutil wrappers, mocked-dep tests.

The module exposes three pure functions:

- ``gpu_free_gb(device_id) -> float | None``
- ``cpu_free_gb() -> float``
- ``init_pynvml() -> bool`` (idempotent)

``pynvml`` is a soft dep, kept behind a deferred-import sentinel so muse
loads on hosts without an NVIDIA driver (AMD GPU, Apple Silicon, CPU-only
CI). Tests patch the sentinels directly, mirroring the pattern used by
``tests/modalities/audio_classification/runtimes/test_hf_audio_classifier.py``.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _reset_module_sentinels():
    """Reset module-top sentinels between tests (deferred-imports pattern).

    The sentinel ``pynvml`` and the ``_init_attempted`` / ``_init_ok`` flags
    are module-globals; without resetting them, a test that flipped them
    would leak state into the next test.
    """
    import muse.core.memory_probe as mod
    orig = (mod.pynvml, mod._init_attempted, mod._init_ok)
    yield
    (mod.pynvml, mod._init_attempted, mod._init_ok) = orig


# ---------- cpu_free_gb ----------------------------------------------------

def test_cpu_free_gb_returns_positive_float():
    """psutil-backed; should always succeed on any host that runs the suite."""
    import muse.core.memory_probe as mod
    free = mod.cpu_free_gb()
    assert isinstance(free, float)
    assert free > 0.0


def test_cpu_free_gb_uses_psutil_virtual_memory():
    """Wires through ``psutil.virtual_memory().available`` divided by 1 GiB."""
    import muse.core.memory_probe as mod
    fake_vm = MagicMock()
    fake_vm.available = 8 * (1024 ** 3)  # 8 GiB exact
    # Patch via the deferred-import path: cpu_free_gb does its own
    # `import psutil` inside the function, so we patch the package
    # module directly.
    with patch("psutil.virtual_memory", return_value=fake_vm):
        assert mod.cpu_free_gb() == pytest.approx(8.0)


# ---------- init_pynvml ----------------------------------------------------

def test_init_pynvml_returns_true_when_import_and_init_succeed():
    """Happy path: pynvml installed, driver healthy."""
    import muse.core.memory_probe as mod
    fake_pynvml = MagicMock()
    with patch.dict("sys.modules", {"pynvml": fake_pynvml}):
        assert mod.init_pynvml() is True
    fake_pynvml.nvmlInit.assert_called_once()
    assert mod.pynvml is fake_pynvml


def test_init_pynvml_returns_false_when_module_missing():
    """No nvidia-ml-py installed at all -> graceful False."""
    import muse.core.memory_probe as mod
    # Force ImportError by hiding the module from sys.modules.
    with patch.dict("sys.modules", {"pynvml": None}):
        assert mod.init_pynvml() is False
    assert mod.pynvml is None


def test_init_pynvml_returns_false_when_init_raises():
    """Module imports but ``nvmlInit`` fails (no driver, AMD GPU, etc.)."""
    import muse.core.memory_probe as mod
    fake_pynvml = MagicMock()
    fake_pynvml.nvmlInit.side_effect = RuntimeError("NVML driver not loaded")
    with patch.dict("sys.modules", {"pynvml": fake_pynvml}):
        assert mod.init_pynvml() is False
    # Sentinel must remain unset so subsequent calls do not believe init
    # succeeded.
    assert mod.pynvml is None


def test_init_pynvml_is_idempotent_on_success():
    """Second call must not re-import or re-init."""
    import muse.core.memory_probe as mod
    fake_pynvml = MagicMock()
    with patch.dict("sys.modules", {"pynvml": fake_pynvml}):
        assert mod.init_pynvml() is True
        assert mod.init_pynvml() is True
        assert mod.init_pynvml() is True
    # Init runs at most once per process.
    assert fake_pynvml.nvmlInit.call_count == 1


def test_init_pynvml_is_idempotent_on_failure():
    """A failed init must not be retried; sticky False."""
    import muse.core.memory_probe as mod
    fake_pynvml = MagicMock()
    fake_pynvml.nvmlInit.side_effect = RuntimeError("driver missing")
    with patch.dict("sys.modules", {"pynvml": fake_pynvml}):
        assert mod.init_pynvml() is False
        assert mod.init_pynvml() is False
    # Failure path also runs at most once.
    assert fake_pynvml.nvmlInit.call_count == 1


# ---------- gpu_free_gb ----------------------------------------------------

def test_gpu_free_gb_returns_float_when_pynvml_available():
    """Mock the full pynvml call chain: handle -> mem_info -> .free."""
    import muse.core.memory_probe as mod
    fake_pynvml = MagicMock()
    handle = MagicMock()
    fake_pynvml.nvmlDeviceGetHandleByIndex.return_value = handle
    mem_info = MagicMock()
    mem_info.free = 4 * (1024 ** 3)  # 4 GiB free
    fake_pynvml.nvmlDeviceGetMemoryInfo.return_value = mem_info
    # Pre-mark init as already done so gpu_free_gb skips the import.
    mod.pynvml = fake_pynvml
    mod._init_attempted = True
    mod._init_ok = True

    free = mod.gpu_free_gb(0)
    assert free == pytest.approx(4.0)
    fake_pynvml.nvmlDeviceGetHandleByIndex.assert_called_once_with(0)
    fake_pynvml.nvmlDeviceGetMemoryInfo.assert_called_once_with(handle)


def test_gpu_free_gb_returns_none_when_pynvml_missing():
    """No pynvml installed -> None, no exception."""
    import muse.core.memory_probe as mod
    with patch.dict("sys.modules", {"pynvml": None}):
        assert mod.gpu_free_gb(0) is None


def test_gpu_free_gb_returns_none_when_init_fails():
    """pynvml present but ``nvmlInit`` raises (no driver / AMD)."""
    import muse.core.memory_probe as mod
    fake_pynvml = MagicMock()
    fake_pynvml.nvmlInit.side_effect = RuntimeError("no driver")
    with patch.dict("sys.modules", {"pynvml": fake_pynvml}):
        assert mod.gpu_free_gb(0) is None


def test_gpu_free_gb_returns_none_on_query_error():
    """init succeeds, but the per-device query later fails (rare; e.g.
    device unplugged)."""
    import muse.core.memory_probe as mod
    fake_pynvml = MagicMock()
    fake_pynvml.nvmlDeviceGetHandleByIndex.side_effect = RuntimeError("gone")
    mod.pynvml = fake_pynvml
    mod._init_attempted = True
    mod._init_ok = True
    assert mod.gpu_free_gb(0) is None


def test_gpu_free_gb_triggers_lazy_init_on_first_call():
    """First call to ``gpu_free_gb`` must implicitly run ``init_pynvml``."""
    import muse.core.memory_probe as mod
    fake_pynvml = MagicMock()
    handle = MagicMock()
    fake_pynvml.nvmlDeviceGetHandleByIndex.return_value = handle
    mem_info = MagicMock()
    mem_info.free = 2 * (1024 ** 3)
    fake_pynvml.nvmlDeviceGetMemoryInfo.return_value = mem_info
    with patch.dict("sys.modules", {"pynvml": fake_pynvml}):
        # No explicit init_pynvml() call here.
        free = mod.gpu_free_gb(0)
    assert free == pytest.approx(2.0)
    fake_pynvml.nvmlInit.assert_called_once()


def test_gpu_free_gb_default_device_id_is_zero():
    """Sanity: signature default is 0; matches the spec."""
    import inspect
    import muse.core.memory_probe as mod
    sig = inspect.signature(mod.gpu_free_gb)
    assert sig.parameters["device_id"].default == 0
