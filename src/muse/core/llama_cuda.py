"""Prebuilt CUDA llama-cpp-python wheels: install plan and runtime preload.

The abetlen CUDA wheels (`.../whl/cu130`) link `libcudart` and `libcublas`
but do not bundle them. NVIDIA ships both as pip packages that install to
`site-packages/nvidia/*/lib`, a directory llama.cpp's `$ORIGIN` RPATH never
searches. `prepare_llama_cpp()` loads them with RTLD_GLOBAL before
`import llama_cpp`, so the native library resolves them by soname. It is a
no-op for a CPU-only build (no `libggml-cuda` next to `libllama`).

Stdlib only: the supervisor's capability query and every GGUF runtime call
this before importing llama_cpp, and `muse --help` must stay light.
"""
from __future__ import annotations

import ctypes
import importlib.util
import logging
import os
import re
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Setting value -> (wheel index, newest verified llama-cpp-python, CUDA
# runtime pip packages). Only builds verified end to end belong here; the
# cap keeps a loose model requirement (`>=0.2.90`) from picking up a newer
# untested wheel. Raise it after verifying the newer build.
WHEELS: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "cu130": (
        "https://abetlen.github.io/llama-cpp-python/whl/cu130",
        "0.3.33",
        ("nvidia-cuda-runtime==13.0.96", "nvidia-cublas==13.1.0.3"),
    ),
}

# Load order matters: cuBLAS needs cuBLASLt, both need the runtime.
_LIBRARIES = ("libcudart.so", "libcublasLt.so", "libcublas.so")
_REQUIREMENT = re.compile(r"^\s*llama[-_.]cpp[-_.]python(?![\w.-])", re.IGNORECASE)
_PARTS = re.compile(
    r"^(\s*llama[-_.]cpp[-_.]python(?:\[[^\]]*\])?)\s*([^;]*?)\s*(;.*)?$", re.IGNORECASE,
)

_prepared: list[str] | None = None


def is_llama_cpp_requirement(requirement: str) -> bool:
    return bool(_REQUIREMENT.match(requirement))


def _capped(requirement: str, verified: str) -> str:
    """Fold the cap into the requirement's own specifier.

    A separate `llama-cpp-python<=X` requirement still makes pip download
    the newest wheel first (hundreds of MB) before it intersects the two.
    """
    match = _PARTS.match(requirement)
    if match is None or "@" in match.group(2):
        return requirement  # a direct URL pins its own build
    name, spec, marker = match.groups()
    spec = f"{spec},<={verified}" if spec else f"<={verified}"
    return f"{name}{spec}{marker or ''}"


def install_plan(packages: list[str], wheel: str | None) -> list[list[str]]:
    """Split one pip install into argument lists, run in order.

    Without a wheel setting (or without a llama-cpp-python requirement) the
    packages install as one list, unchanged. Otherwise the CUDA runtimes join
    the other packages and llama-cpp-python installs on its own, binary-only
    from the wheel index (capped at the verified version), so pip never falls
    back to a source build. That step skips pip's cache: the index serves
    each wheel from a fresh signed URL, so a cached copy is never reused and
    every install would leave another ~800 MB entry behind.
    """
    llama = [p for p in packages if is_llama_cpp_requirement(p)]
    if not wheel or not llama:
        return [list(packages)] if packages else []
    if wheel not in WHEELS:
        raise ValueError(f"unknown llama-cpp-python wheel {wheel!r}")
    index, verified, runtimes = WHEELS[wheel]
    rest = [p for p in packages if not is_llama_cpp_requirement(p)]
    return [
        [*rest, *runtimes],
        [
            "--no-cache-dir", "--only-binary=llama-cpp-python", "--extra-index-url", index,
            *(_capped(p, verified) for p in llama),
        ],
    ]


def reinstall_args(plan: list[str]) -> list[str]:
    """Replace an installed (e.g. source-built) llama-cpp-python in place.

    pip treats a CPU build of the same version as already satisfied, so a
    refresh that switches builds forces the wheel without touching its deps.
    """
    return ["--force-reinstall", "--no-deps", *plan]


IS_CUDA_BUILD_SCRIPT = (
    "import importlib.util, pathlib, sys\n"
    "s = importlib.util.find_spec('llama_cpp')\n"
    "lib = pathlib.Path(s.origin).parent / 'lib' if s and s.origin else None\n"
    "sys.exit(0 if lib and any(lib.glob('libggml-cuda*')) else 1)\n"
)


def _package_dir() -> Path | None:
    try:
        spec = importlib.util.find_spec("llama_cpp")
    except ValueError:  # a stand-in module without a spec (tests)
        return None
    if spec is None or not spec.origin:
        return None
    return Path(spec.origin).parent


def is_cuda_build() -> bool:
    package = _package_dir()
    return package is not None and any((package / "lib").glob("libggml-cuda*"))


def _runtime_libraries(site_packages: Path) -> list[Path]:
    found = []
    for stem in _LIBRARIES:
        found.extend(sorted(
            p for p in (site_packages / "nvidia").glob(f"*/lib/{stem}.*")
            if re.fullmatch(re.escape(stem) + r"\.\d+", p.name)
        ))
    return found


def prepare_llama_cpp(*, cpu_only: bool = False) -> list[str]:
    """Make a CUDA llama-cpp-python build importable; return libraries loaded.

    cpu_only hides CUDA devices from this process first, so a CPU placement
    allocates no VRAM and llama.cpp cannot offload host matmuls to the GPU.
    That only works before llama_cpp initializes; later it logs a warning.
    """
    global _prepared
    if not is_cuda_build():
        return []
    if cpu_only:
        if "llama_cpp" in sys.modules:
            logger.warning("llama_cpp already loaded; CUDA devices stay visible")
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if _prepared is not None:
        return list(_prepared)
    loaded = []
    for path in _runtime_libraries(_package_dir().parent):
        try:
            ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
        except OSError as exc:
            logger.debug("could not preload %s: %s", path, exc)
        else:
            loaded.append(str(path))
    _prepared = loaded
    return list(loaded)
