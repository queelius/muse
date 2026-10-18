"""Resolve native GGUF placement from the build that will load the model."""
from __future__ import annotations

import json
import re
import subprocess
import sys
import threading
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path

from muse.core.venv import run_owned_command


@dataclass(frozen=True)
class Placement:
    device: str
    pool: str
    n_gpu_layers: int


def native_placement(
    device: str, n_gpu_layers: int, *, gpu_offload: bool, gpu_backend: str | None = None,
) -> Placement:
    """CPU requests never offload, even when the native build supports it."""
    device = str(device or "auto").lower()
    if device == "cpu" or n_gpu_layers == 0:
        return Placement("cpu", "cpu", 0)
    if not gpu_offload:
        if device not in ("", "auto"):
            raise RuntimeError(f"native GGUF GPU offload is unavailable for {device}")
        return Placement("cpu", "cpu", 0)
    backend = gpu_backend or ("metal" if sys.platform == "darwin" else "cuda")
    target = {"cuda": "cuda", "metal": "mps"}.get(backend)
    if target is None:
        raise RuntimeError(f"native GGUF GPU offload backend {backend!r} is unsupported")
    if device in ("", "auto", "gpu"):
        device = target
    elif device != target and not (target == "cuda" and device.startswith("cuda:")):
        raise RuntimeError(f"native GGUF GPU offload backend {backend} cannot use {device}")
    return Placement(device, "cpu" if target == "mps" else "cuda", n_gpu_layers)


def is_native_gguf(manifest: dict) -> bool:
    return manifest.get("backend_path", "") == (
        "muse.modalities.chat_completion.runtimes.llama_cpp:LlamaCppModel"
    )


def native_gpu_offload() -> bool:
    """Inspect the loaded native library in the current model interpreter."""
    try:
        import llama_cpp
    except ImportError:
        return False
    llama_cpp.llama_backend_init()
    return bool(llama_cpp.llama_supports_gpu_offload())


def _backend_from_system_info(info: str, *, platform: str) -> str:
    tokens = set(re.findall(r"[A-Z0-9_]+", info.upper()))
    supported = tokens & {"CUDA", "METAL"}
    unsupported = tokens & {
        "VULKAN", "SYCL", "HIP", "HIPBLAS", "ROCM", "KOMPUTE", "CANN", "OPENCL", "UNKNOWN",
    }
    if unsupported or len(supported) > 1:
        return "unknown"
    if supported:
        return supported.pop().lower()
    # Older bindings do not identify a backend. Keep the validated platform
    # convention: Metal on Darwin, CUDA elsewhere (not generic GPU support).
    return "metal" if platform == "darwin" else "cuda"


def native_gpu_backend() -> str:
    """Identify supported native families without consulting Torch CUDA."""
    import llama_cpp
    info = getattr(llama_cpp, "llama_print_system_info", lambda: b"")()
    if isinstance(info, bytes):
        info = info.decode(errors="replace")
    return _backend_from_system_info(str(info), platform=sys.platform)


_QUERY = """
import importlib.metadata
import json
import sys
from pathlib import Path
import llama_cpp
llama_cpp.llama_backend_init()
package = Path(llama_cpp.__file__).resolve()
paths = {str(package), str(package.parent)}
for path in package.parent.rglob('*'):
    if path.is_dir() or path.suffix in ('.py', '.so', '.dll', '.dylib', '.pyd') or '.so.' in path.name:
        paths.add(str(path))
bindings = getattr(llama_cpp, 'llama_cpp', None)
library = getattr(getattr(bindings, '_lib', None), '_name', None)
if library:
    paths.add(str(Path(library).resolve()))
try:
    dist = importlib.metadata.distribution('llama-cpp-python')
    for file in dist.files or ():
        if file.name in ('METADATA', 'RECORD'):
            paths.add(str(dist.locate_file(file)))
except importlib.metadata.PackageNotFoundError:
    pass
info = getattr(llama_cpp, 'llama_print_system_info', lambda: b'')()
if isinstance(info, bytes):
    info = info.decode(errors='replace')
print(json.dumps({'gpu_offload': bool(llama_cpp.llama_supports_gpu_offload()),
                  'paths': sorted(paths), 'platform': sys.platform, 'system_info': str(info)}))
"""


def _identity(path: str) -> tuple:
    try:
        stat = Path(path).stat()
        return (
            path, stat.st_dev, stat.st_ino, stat.st_size,
            stat.st_mtime_ns, stat.st_ctime_ns,
        )
    except OSError:
        return (path, None)


_cache: dict[str, tuple[tuple, tuple[bool, str], tuple[str, ...]]] = {}
_cache_lock = threading.Lock()
_queries: dict[str, Future[tuple[bool, str]]] = {}
_query_slots = threading.BoundedSemaphore(2)


def _interpreter_capability(python: str) -> tuple[bool, str]:
    # Keep the lexical interpreter path: two venvs may symlink the same binary.
    config = str(Path(python).parent.parent / "pyvenv.cfg")
    with _cache_lock:
        cached = _cache.get(python)
        if cached is not None:
            identity, capability, paths = cached
            if identity == tuple(_identity(p) for p in (python, config, *paths)):
                return capability
        pending = _queries.get(python)
        owner = pending is None
        if pending is None:
            pending = Future()
            _queries[python] = pending
    if not owner:
        # Share successes AND failures without retrying once per waiter.
        try:
            return pending.result(timeout=35)
        except TimeoutError as exc:
            raise RuntimeError("could not query native GGUF GPU offload: capability check is still running; retry") from exc
    try:
        if not _query_slots.acquire(timeout=15):
            raise RuntimeError(
                "could not query native GGUF GPU offload: capability queries are busy; retry"
            )
        try:
            try:
                result = run_owned_command([python, "-c", _QUERY], timeout=15, check=True)
            finally:
                _query_slots.release()
            record = json.loads(result.stdout.strip().splitlines()[-1])
            if not isinstance(record["gpu_offload"], bool):
                raise ValueError("invalid native capability")
            paths = tuple(str(p) for p in record["paths"])
            capability = (
                record["gpu_offload"],
                _backend_from_system_info(
                    record.get("system_info", ""), platform=record.get("platform", sys.platform),
                ),
            )
        except (OSError, subprocess.SubprocessError, ValueError, KeyError, IndexError, TypeError) as exc:
            # Unknown is not CPU-only: the separately loading worker might
            # still offload. Reject admission, without caching the failure.
            raise RuntimeError(
                f"could not query native GGUF GPU offload capability in {python}; "
                "check the model environment and retry"
            ) from exc
        identity = tuple(_identity(p) for p in (python, config, *paths))
        with _cache_lock:
            _cache[python] = (identity, capability, paths)
        pending.set_result(capability)
        return capability
    except BaseException as exc:
        pending.set_exception(exc)
        raise
    finally:
        with _cache_lock:
            _queries.pop(python, None)


def resolve_model_placement(
    model_id: str, manifest: dict, *, requested: str = "auto", entry: dict | None = None,
) -> Placement:
    """Apply operator/manifest/request precedence using the recorded interpreter."""
    if entry is None:
        from muse.core.catalog import _read_catalog
        entry = _read_catalog().get(model_id, {})
    caps = manifest.get("capabilities") or {}
    override = entry.get("device_override")
    cap = str(caps.get("device") or "auto").lower()
    device = str(override or (cap if cap != "auto" else requested) or "auto").lower()
    layers_override = entry.get("gpu_layers_override")
    layers = int(layers_override if layers_override is not None else caps.get("n_gpu_layers", -1))
    if is_native_gguf(manifest):
        supported = False
        backend = None
        if device != "cpu" and layers != 0:
            python = entry.get("python_path")
            if not python:
                raise RuntimeError(
                    "could not query native GGUF GPU offload: no recorded model interpreter"
                )
            supported, backend = _interpreter_capability(python)
        return native_placement(device, layers, gpu_offload=supported, gpu_backend=backend)
    from muse.core import memory_probe
    auto = device in ("", "auto")
    pool = memory_probe.resolve_memory_pool(
        device,
        gpu_free_gb=memory_probe.gpu_free_gb() if auto else None,
        cuda_available=memory_probe.cuda_runtime_available() if auto else False,
    )
    return Placement(pool if device == "auto" else device, pool, layers)
