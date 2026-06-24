#!/usr/bin/env python
"""Preflight guard: verify the dev venv can run the fast test lane, then run it.

The muse fast lane (`pytest -m "not slow"`) imports the heavy ML stack
(torch, transformers, diffusers, sentence-transformers) plus server deps in
many test modules; several import them at collection time (not behind a
mock), so a venv missing those deps does not merely skip - it errors at
collection or silently runs a partial suite. This script asserts the
required deps import BEFORE running pytest, so a release cannot be
"verified" in a drifted venv.

Usage:
    python scripts/preflight.py                  # check deps, then run the lane
    python scripts/preflight.py --check-only     # check deps only, no tests
    python scripts/preflight.py -- -k resolver   # forward args after -- to pytest
"""
from __future__ import annotations

import argparse
import importlib
import subprocess
import sys

# (import_name, extra, pip_package). Import name != package name for several,
# hence the explicit mapping. Keep in sync with INSTALL_CMD and pyproject
# optional-dependencies.
REQUIRED: list[tuple[str, str, str]] = [
    ("numpy", "core", "numpy"),
    ("yaml", "core", "pyyaml"),
    ("torch", "audio", "torch"),
    ("transformers", "audio", "transformers"),
    ("scipy", "audio", "scipy"),
    ("diffusers", "images", "diffusers"),
    ("PIL", "images", "Pillow"),
    ("sentence_transformers", "embeddings", "sentence-transformers"),
    ("fastapi", "server", "fastapi"),
    ("uvicorn", "server", "uvicorn"),
    ("httpx", "server", "httpx"),
    ("psutil", "server", "psutil"),
    ("multipart", "server", "python-multipart"),
    ("pynvml", "server", "nvidia-ml-py"),
    ("pytest_asyncio", "dev", "pytest-asyncio"),
]

INSTALL_CMD = (
    'pip install -e ".[dev,server,audio,images,embeddings]" '
    "--extra-index-url https://download.pytorch.org/whl/cpu"
)


def missing_deps() -> list[tuple[str, str, str]]:
    """Return the sentinels that fail to import."""
    missing: list[tuple[str, str, str]] = []
    for import_name, extra, package in REQUIRED:
        try:
            importlib.import_module(import_name)
        except Exception:  # noqa: BLE001 - any import failure means "missing"
            missing.append((import_name, extra, package))
    return missing


def report_missing(missing: list[tuple[str, str, str]]) -> None:
    """Print an actionable error naming each missing dep and the fix."""
    print("preflight: venv is not fast-lane ready; missing imports:",
          file=sys.stderr)
    for import_name, extra, package in missing:
        print(f"  - {import_name}  (extra: {extra}, package: {package})",
              file=sys.stderr)
    print(f"\nInstall the full dev stack:\n  {INSTALL_CMD}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="muse fast-lane preflight guard")
    parser.add_argument("--check-only", action="store_true",
                        help="verify deps only; do not run pytest")
    parser.add_argument("pytest_args", nargs="*",
                        help="args forwarded to pytest (use -- to separate)")
    args = parser.parse_args(argv)

    missing = missing_deps()
    if missing:
        report_missing(missing)
        return 1
    print(f"preflight: all {len(REQUIRED)} fast-lane deps present.")
    if args.check_only:
        return 0

    cmd = [sys.executable, "-m", "pytest", "-m", "not slow", *args.pytest_args]
    print(f"preflight: running: {' '.join(cmd)}")
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    raise SystemExit(main())
