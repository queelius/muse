"""Fresh-venv smoke-test for a bundled muse model (#124, v0.32.0).

Catches the production failure mode where a bundled script's
`pip_extras` declares the deps the runtime source-imports but misses
transitive deps that `from_pretrained` (or sentence-transformers, or
diffusers) pulls in at load time.

The host muse install (with broad dev extras: server, audio, images,
embeddings, dev) typically has those transitives already, so the test
suite passes. A fresh per-model venv created via `muse pull` does NOT,
because `pull` installs ONLY `museq[server]` plus the model's declared
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
import os
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
    """Install museq[server] (editable) into the venv. Returns (rc, captured)."""
    cmd = [
        str(venv_python), "-m", "pip", "install",
        "-e", f"{repo_root}[server]",
    ]
    logger.info("installing museq[server]: %s", " ".join(cmd))
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
    *,
    env: dict[str, str] | None = None,
) -> tuple[int, str]:
    """Run the muse probe worker in load-only mode. Returns (rc, captured).

    The probe worker exists as a hidden CLI subcommand
    (`muse _probe_worker --model <id> --device cpu --no-inference`).
    It calls `load_backend(model_id)`, captures memory, prints a JSON
    record on stdout, and exits 0 on success. The smoke test only cares
    that it loads without ImportError, so we run it on the CPU device.

    `env` overrides the subprocess environment (e.g. `MUSE_CATALOG_DIR`
    for a curated resolver-based model whose manifest was persisted to
    an isolated catalog by `_smoke_curated_resolver`). Defaults to the
    current process environment when omitted, matching bundled-script
    smoke runs which have no catalog dependency.
    """
    cmd = [
        str(venv_python), "-m", "muse.cli", "_probe_worker",
        "--model", model_id,
        "--device", "cpu",
        "--no-inference",
    ]
    logger.info("running load-only probe: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
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


def _smoke_curated_resolver(model_id: str, uri: str, venv_root: Path) -> SmokeResult:
    """Smoke-test a curated resolver-based (non-bundled) model id.

    Bundled scripts (the common matrix case) are discoverable via
    `known_models()` with no prior pull, so `smoke_one` can create a
    plain venv, install the declared `pip_extras`, and run the load-only
    probe directly. Curated resolver entries (e.g. `opus-mt-en-es`, a
    `hf://Helsinki-NLP/opus-mt-en-es` URI with no bundled script) have no
    discovered script: `known_models()` only sees them once a real
    `muse pull` synthesizes and PERSISTS a manifest into catalog.json.

    Rather than re-implement resolve + venv-create + pip-install +
    weight-download, this shells out to `muse pull <id> --no-probe`
    scoped to an isolated `MUSE_CATALOG_DIR` under `venv_root` -- the
    exact path a user runs, exercising `catalog._pull_via_resolver`'s
    real venv/pip_extras/weight-download machinery instead of a smoke-
    script-only reimplementation. `--no-probe` skips the (heavier,
    inference-touching) memory probe; the load-only check below is the
    smoke assertion, mirroring the bundled-script path.
    """
    t0 = time.monotonic()
    catalog_dir = venv_root / "catalog"
    catalog_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["MUSE_CATALOG_DIR"] = str(catalog_dir)

    repo_root = _repo_root()
    cmd = [
        sys.executable, "-m", "muse.cli", "pull", model_id, "--no-probe",
    ]
    logger.info("pulling curated resolver model: %s", " ".join(cmd))
    proc = subprocess.run(
        cmd, capture_output=True, text=True, env=env, cwd=str(repo_root),
    )
    if proc.returncode != 0:
        duration = time.monotonic() - t0
        reason = _extract_failure_reason(proc.stdout + proc.stderr)
        return SmokeResult(
            model_id=model_id,
            ok=False,
            error=f"muse pull failed: {reason}",
            duration_s=duration,
            label=f"{model_id}: FAIL (pull: {reason})",
        )

    catalog_path = catalog_dir / "catalog.json"
    try:
        entry = json.loads(catalog_path.read_text()).get(model_id, {})
    except (OSError, ValueError) as e:
        duration = time.monotonic() - t0
        return SmokeResult(
            model_id=model_id,
            ok=False,
            error=f"catalog unreadable after pull: {e}",
            duration_s=duration,
            label=f"{model_id}: FAIL (no catalog after pull)",
        )
    python_path = entry.get("python_path")
    if not python_path:
        duration = time.monotonic() - t0
        return SmokeResult(
            model_id=model_id,
            ok=False,
            error="pulled catalog entry has no python_path",
            duration_s=duration,
            label=f"{model_id}: FAIL (no python_path after pull)",
        )

    rc, captured = _run_load_only(Path(python_path), model_id, env=env)
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
    from muse.core.curated import find_curated

    t0 = time.monotonic()

    catalog_known = known_models()
    if model_id not in catalog_known:
        # Not a discovered bundled script. A curated resolver-based id
        # (has a `uri`, no bundled script) needs an actual `muse pull`
        # to synthesize + persist its manifest before it is loadable at
        # all -- see _smoke_curated_resolver for why.
        curated = find_curated(model_id)
        if curated is not None and curated.uri:
            return _smoke_curated_resolver(model_id, curated.uri, venv_root)
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
            error=f"pip install museq[server] failed: {reason}",
            duration_s=duration,
            label=f"{model_id}: FAIL (pip museq[server]: {reason})",
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
