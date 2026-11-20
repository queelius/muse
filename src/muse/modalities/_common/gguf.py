"""GGUF (llama.cpp) loading shared by the chat and decision runtimes.

Both runtimes keep their own module-level `Llama` sentinel (tests patch
it there) and call these helpers for the parts that must not drift: the
CUDA-wheel preparation before `import llama_cpp`, the on-disk checks, and
native GPU placement (an explicit GPU device the build cannot serve is an
error, never a silent CPU fallback).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from muse.core.llama_cuda import prepare_llama_cpp
from muse.core.placement import (
    Placement,
    native_gpu_backend,
    native_gpu_offload,
    native_placement,
)


logger = logging.getLogger(__name__)


def is_cpu_only(device: str, n_gpu_layers: int) -> bool:
    """A cpu placement or zero GPU layers: hide CUDA from a CUDA build."""
    return device == "cpu" or n_gpu_layers == 0


def import_llama(*, cpu_only: bool) -> Any:
    """Return `llama_cpp.Llama`, or None when llama-cpp-python is missing.

    A prebuilt CUDA wheel needs NVIDIA's pip runtime loaded first.
    """
    try:
        prepare_llama_cpp(cpu_only=cpu_only)
        from llama_cpp import Llama
    except Exception as e:  # noqa: BLE001
        logger.debug("llama-cpp-python unavailable: %s", e)
        return None
    return Llama


def require_llama(llama: Any) -> None:
    if llama is None:
        raise RuntimeError(
            "llama-cpp-python is not installed; run `muse pull` or "
            "install `llama-cpp-python` into this venv"
        )


def gguf_path(local_dir: str | None, gguf_file: str) -> Path:
    """The GGUF file inside `local_dir`; it must already be on disk."""
    if not local_dir:
        raise RuntimeError("local_dir is required; the GGUF file must be on disk")
    path = Path(local_dir) / gguf_file
    if not path.exists():
        raise FileNotFoundError(f"GGUF file not found: {gguf_file}")
    return path


def resolve_placement(device: str, n_gpu_layers: int) -> Placement:
    """Native device + layer count for this build and host."""
    offload = (
        native_gpu_offload() if not is_cpu_only(device, n_gpu_layers) else False
    )
    return native_placement(
        device, n_gpu_layers, gpu_offload=offload,
        gpu_backend=native_gpu_backend() if offload else None,
    )
