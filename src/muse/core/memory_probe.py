"""Live memory probing for the lazy-load director (v0.40.0).

Wraps ``pynvml`` (per-device NVIDIA VRAM) and ``psutil`` (system RAM)
behind three pure functions:

- ``gpu_free_gb(device_id)``: VRAM free in gibibytes, or ``None`` when
  pynvml is unavailable, the host has no NVIDIA driver, or the per-device
  query fails (rare; e.g. a TCC device toggled off mid-process).
- ``cpu_free_gb()``: RAM available in gibibytes, via
  ``psutil.virtual_memory().available``.
- ``init_pynvml()``: idempotent. Imports pynvml once, calls ``nvmlInit``
  once, and caches the success / failure verdict for the rest of the
  process.

``pynvml`` is a soft dep: muse runs on AMD GPU hosts, Apple Silicon, and
CPU-only CI without it. The director tolerates ``None`` from
``gpu_free_gb`` by either falling back to a static budget (when one is
declared) or refusing the GPU load with a 503.

Deferred-imports pattern: a module-top sentinel ``pynvml: Any = None``
gets populated by ``init_pynvml`` on first use. Tests patch the sentinel
directly (see ``tests/core/test_memory_probe.py``).
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# Deferred-import sentinels. The first call to ``init_pynvml`` populates
# ``pynvml`` (or leaves it None on failure); ``_init_attempted`` and
# ``_init_ok`` make the call idempotent (sticky verdict, no retry).
pynvml: Any = None
_init_attempted: bool = False
_init_ok: bool = False

# 1 GiB in bytes. Both pynvml's ``nvmlDeviceGetMemoryInfo().free`` and
# psutil's ``virtual_memory().available`` are in bytes; the wrapper
# normalizes to gibibytes.
_BYTES_PER_GB: int = 1024 ** 3


def init_pynvml() -> bool:
    """Idempotent pynvml init.

    Returns ``True`` once the module has been imported and ``nvmlInit``
    has succeeded; returns ``False`` on any failure (module missing,
    driver missing, AMD card, CPU-only host). The verdict is sticky for
    the remainder of the process: a failure is not retried.

    The pre-commit hook policy bans em-dashes; we use parens above for
    aside clauses.
    """
    global pynvml, _init_attempted, _init_ok
    if _init_attempted:
        return _init_ok
    _init_attempted = True
    try:
        import pynvml as _p
        _p.nvmlInit()
    except Exception as e:  # noqa: BLE001
        # ImportError, RuntimeError (NVMLError), OSError on missing libs
        # all collapse to a single "no GPU info" verdict. We log at
        # DEBUG so a CPU-only or AMD host does not get noisy warnings.
        logger.debug("pynvml init failed: %s", e)
        _init_ok = False
        return False
    pynvml = _p
    _init_ok = True
    return True


def gpu_free_gb(device_id: int = 0) -> float | None:
    """Live VRAM free on the given NVIDIA GPU, in gibibytes.

    Returns ``None`` when:
    - pynvml is not installed,
    - ``nvmlInit`` failed (no driver, AMD GPU, CPU-only host), or
    - the per-device query raises (rare).

    Triggers a lazy ``init_pynvml`` on first call so callers do not need
    to wire init explicitly.
    """
    if not init_pynvml():
        return None
    # init_pynvml() returning True implies the sentinel is populated. Guard
    # with an explicit None-return (not assert: asserts are stripped under
    # `python -O`, and None is this function's established failure idiom).
    if pynvml is None:  # pragma: no cover - invariant backstop
        return None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    except Exception as e:  # noqa: BLE001
        logger.debug("pynvml query for device %s failed: %s", device_id, e)
        return None
    return float(mem.free) / _BYTES_PER_GB


def cpu_free_gb() -> float:
    """Live RAM available on the host, in gibibytes.

    Backed by ``psutil.virtual_memory().available``: the kernel's view of
    how much memory can be allocated without swapping (Linux: MemAvailable
    from /proc/meminfo; macOS / Windows: the equivalent platform-specific
    counter). psutil is a hard ``museq[server]`` dep so this never fails.
    """
    import psutil
    return float(psutil.virtual_memory().available) / _BYTES_PER_GB


def declared_device(capabilities: dict | None) -> str:
    """The device a manifest's capabilities declare, defaulting to "auto".

    Shared by every control-plane site that sizes or pools a model
    (LoadDirector admission and commit, IdleSweeper eviction, supervisor
    servability). Absent or empty means the model follows the worker's
    own default (`muse _worker --device auto`), which binds to the GPU
    when one exists, so the control plane must size it against the pool
    "auto" resolves to on this host, NOT against host RAM.

    Regression note (v0.50.1): these sites each defaulted the absent key
    to "cpu". Resolver-pulled manifests never carry a `device` key, so
    the director sized their GPU loads against host RAM, admission always
    "fit", eviction never ran, and workers OOM'd at spawn on a full GPU.
    """
    return str((capabilities or {}).get("device") or "auto").lower()
