"""Fresh-venv smoke-test for a bundled muse model (#124, v0.32.0).

Catches the production failure mode where a bundled script's
`pip_extras` declares the deps the runtime source-imports but misses
transitive deps that `from_pretrained` (or sentence-transformers, or
diffusers) pulls in at load time.

The host muse install (with broad dev extras: server, audio, images,
embeddings, dev) typically has those transitives already, so the test
suite passes. A fresh per-model venv created via `muse pull` does NOT,
because `pull` installs ONLY `muse[server]` plus the model's declared
`pip_extras`. Transitive holes show up there.

This script reproduces the `muse pull` install path against a clean
venv and then runs the in-venv probe worker (load-only, no inference).
A failure surfaces the missing dep in the worker's stderr and the
script exits non-zero with an informative label.

Usage (local):
    python scripts/smoke_fresh_venv.py --model_id kokoro-82m
    python scripts/smoke_fresh_venv.py --model_id dinov2-small --json

Usage (CI):
    See .github/workflows/fresh-venv-smoke.yml
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import re
import subprocess
import sys
import tempfile
import time
import venv
from pathlib import Path

logger = logging.getLogger("muse.smoke")


@dataclasses.dataclass
class SmokeResult:
    """Outcome of a smoke test for one bundled model.

    `label` is the one-line summary CI surfaces in the job log (e.g.,
    "kokoro-82m: OK (12.3s)" or "kokoro-82m: FAIL (missing dep: librosa)").
    """
    model_id: str
    ok: bool
    error: str | None
    duration_s: float
    label: str


def _repo_root() -> Path:
    """Resolve the muse repo root (contains pyproject.toml).

    The script lives at <repo>/scripts/smoke_fresh_venv.py, so two
    parents up is the repo. Defensive: walk up until pyproject.toml
    appears, fall back to the two-parent default if not found.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return here.parents[1]


def _venv_python(venv_dir: Path) -> Path:
    """Return the Python interpreter inside a venv (POSIX layout)."""
    return venv_dir / "bin" / "python"


def _create_venv(target: Path) -> None:
    """Create a fresh venv at `target`. Uses stdlib venv module.

    Mirrors muse.core.venv.create_venv (subprocess `python -m venv`),
    but uses the venv module directly because we already know the
    Python interpreter we're running on is what muse pull would use.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    logger.info("creating venv at %s", target)
    venv.create(str(target), with_pip=True, clear=True)


def _install_muse(venv_python: Path, repo_root: Path) -> tuple[int, str]:
    """Install muse[server] (editable) into the venv. Returns (rc, captured)."""
    cmd = [
        str(venv_python), "-m", "pip", "install",
        "-e", f"{repo_root}[server]",
    ]
    logger.info("installing muse[server]: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout + proc.stderr


def _install_pip_extras(
    venv_python: Path,
    packages: tuple[str, ...],
) -> tuple[int, str]:
    """Install the model's pip_extras into the venv. Returns (rc, captured)."""
    if not packages:
        return 0, ""
    cmd = [str(venv_python), "-m", "pip", "install", *packages]
    logger.info("installing pip_extras: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout + proc.stderr


def _run_load_only(
    venv_python: Path,
    model_id: str,
) -> tuple[int, str]:
    """Run the muse probe worker in load-only mode. Returns (rc, captured).

    The probe worker exists as a hidden CLI subcommand
    (`muse _probe_worker --model <id> --device cpu --no-inference`).
    It calls `load_backend(model_id)`, captures memory, prints a JSON
    record on stdout, and exits 0 on success. The smoke test only cares
    that it loads without ImportError, so we run it on the CPU device.
    """
    cmd = [
        str(venv_python), "-m", "muse.cli", "_probe_worker",
        "--model", model_id,
        "--device", "cpu",
        "--no-inference",
    ]
    logger.info("running load-only probe: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout + proc.stderr


_MODULE_NOT_FOUND_RE = re.compile(
    r"ModuleNotFoundError: No module named ['\"]([^'\"]+)['\"]"
)


def _extract_failure_reason(captured: str) -> str:
    """Pick the most informative failure line from worker output.

    Recognized patterns (best to least specific):
      - ModuleNotFoundError: 'foo'
      - ImportError: cannot import name 'foo' from 'bar'
      - the `load failed:` prefix the probe worker emits
      - last non-empty line as fallback
    """
    if not captured:
        return "no output"
    m = _MODULE_NOT_FOUND_RE.search(captured)
    if m:
        return f"missing dep: {m.group(1)}"
    for line in captured.splitlines():
        if line.startswith("ImportError:"):
            return line.strip()
        if line.startswith("load failed:"):
            return line.strip()
    # Fall back to the last non-empty line of stderr-style output.
    for line in reversed(captured.splitlines()):
        s = line.strip()
        if s and not s.startswith("{"):
            return s[:200]
    return "unknown failure"


def smoke_one(
    model_id: str,
    venv_root: Path,
) -> SmokeResult:
    """Run the full smoke-test pipeline for one bundled model.

    Stages (each can fail):
      1. Look up MANIFEST + pip_extras from muse.core.catalog.known_models().
      2. Create venv.
      3. pip install -e <repo>[server].
      4. pip install <pip_extras>.
      5. python -m muse.cli _probe_worker --model <id> --device cpu --no-inference

    A failure at any stage becomes SmokeResult(ok=False) with a label
    naming the stage and the probable cause.
    """
    from muse.core.catalog import known_models

    t0 = time.monotonic()

    catalog_known = known_models()
    if model_id not in catalog_known:
        duration = time.monotonic() - t0
        return SmokeResult(
            model_id=model_id,
            ok=False,
            error=f"unknown model {model_id!r}",
            duration_s=duration,
            label=f"{model_id}: FAIL (unknown model)",
        )
    entry = catalog_known[model_id]
    pip_extras = entry.pip_extras

    venv_dir = venv_root / ".venv"
    repo_root = _repo_root()

    _create_venv(venv_dir)
    py = _venv_python(venv_dir)

    rc, captured = _install_muse(py, repo_root)
    if rc != 0:
        duration = time.monotonic() - t0
        reason = _extract_failure_reason(captured)
        return SmokeResult(
            model_id=model_id,
            ok=False,
            error=f"pip install muse[server] failed: {reason}",
            duration_s=duration,
            label=f"{model_id}: FAIL (pip muse[server]: {reason})",
        )

    rc, captured = _install_pip_extras(py, pip_extras)
    if rc != 0:
        duration = time.monotonic() - t0
        reason = _extract_failure_reason(captured)
        return SmokeResult(
            model_id=model_id,
            ok=False,
            error=f"pip install pip_extras failed: {reason}",
            duration_s=duration,
            label=f"{model_id}: FAIL (pip extras: {reason})",
        )

    rc, captured = _run_load_only(py, model_id)
    if rc != 0:
        duration = time.monotonic() - t0
        reason = _extract_failure_reason(captured)
        return SmokeResult(
            model_id=model_id,
            ok=False,
            error=f"load failed: {reason}",
            duration_s=duration,
            label=f"{model_id}: FAIL ({reason})",
        )

    duration = time.monotonic() - t0
    return SmokeResult(
        model_id=model_id,
        ok=True,
        error=None,
        duration_s=duration,
        label=f"{model_id}: OK ({duration:.1f}s)",
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns process exit code."""
    parser = argparse.ArgumentParser(
        prog="smoke_fresh_venv",
        description=(
            "Smoke-test a bundled muse model in a fresh per-model venv, "
            "verifying that pip_extras covers the runtime's load-time deps."
        ),
    )
    parser.add_argument("--model_id", required=True, help="bundled model id (e.g., kokoro-82m)")
    parser.add_argument(
        "--venv_root",
        default=None,
        help="directory to create the smoke venv under (default: tempdir)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="emit a JSON record on stdout (default: human-readable label)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
    )

    if args.venv_root:
        venv_root = Path(args.venv_root)
        venv_root.mkdir(parents=True, exist_ok=True)
    else:
        venv_root = Path(tempfile.mkdtemp(prefix=f"muse-smoke-{args.model_id}-"))

    result = smoke_one(args.model_id, venv_root)

    if args.as_json:
        record = dataclasses.asdict(result)
        # Ensure stable JSON: drop the human label from the JSON body so
        # CI artifact JSON stays uncluttered.
        print(json.dumps(record, indent=2))
    else:
        print(result.label)

    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
