"""Make `scripts.bench.*` importable regardless of pytest invocation dir.

The bench harness lives at <repo>/scripts/bench/ (not inside the
installed muse package), so importing `scripts.bench._stats` relies on
the repo root being on sys.path. That holds for some local invocation
shapes but NOT on CI (rootdir-relative imports differ there), which
broke the `tests` workflow with ModuleNotFoundError. Pin it explicitly.
"""
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
