"""`muse models refresh`: re-install muse[server,<extras>] into per-model venvs.

Use case: muse[server] gets a new dep (e.g. python-multipart in v0.13.1)
and existing per-model venvs created by older `muse pull` calls are
stale. Without this command, hitting the new code path inside an old
venv crashes the worker; the user has fixed this by hand multiple
times. `muse models refresh --all` upgrades every venv in one pass.

Behavior:
  - Inspect catalog.json for the target model_id(s).
  - For each: invoke <venv>/bin/pip install --upgrade -e <muse>[server,<modality-extras>].
  - Then (unless --no-extras): pip install --upgrade <model's pip_extras...>.
  - Continue past failures; aggregate at the end.

The supervisor is NOT restarted. To pick up a refreshed venv, the
operator runs `Ctrl+C; muse serve` themselves. This is documented.
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from muse.core.catalog import _read_catalog, get_manifest, is_enabled

logger = logging.getLogger(__name__)


# Map a modality tag to the muse-side optional-deps extras names.
# `server` is added unconditionally on top of these. The map is
# hand-maintained: each entry corresponds to a pyproject [project.
# optional-dependencies] block (see pyproject.toml). New modalities
# either bring a new extra (then add a row here) or reuse `server`
# (then map to []).
MODALITY_EXTRAS: dict[str, list[str]] = {
    "audio/speech": ["audio"],
    "audio/transcription": [],
    "audio/embedding": [],
    "audio/generation": [],
    "image/generation": ["images"],
    "image/animation": ["images"],
    "image/upscale": ["images"],
    "image/embedding": [],
    "image/segmentation": [],
    "embedding/text": ["embeddings"],
    "text/classification": [],
    "text/rerank": [],
    "text/summarization": [],
    "video/generation": ["images"],
    "chat/completion": [],
}


@dataclass
class RefreshResult:
    """Outcome record for one model's venv refresh."""

    model_id: str
    state: str  # "ok" | "failed" | "skipped"
    message: str = ""
    pip_output: str = ""
    extras: list[str] = field(default_factory=list)


def _infer_extras(modality: str) -> list[str]:
    """Look up muse[server,<extras>] for a modality tag. Unknown -> []."""
    return list(MODALITY_EXTRAS.get(modality, []))


def _muse_repo_root() -> Path:
    """Locate the muse source tree to install in editable mode.

    Walks parents looking for pyproject.toml. Falls back to the current
    working directory when running from a wheel install (no pyproject
    in any parent of __file__). Documented in the spec: users on a
    PyPI install of muse can still refresh; pip will pull muse from
    PyPI rather than installing editable.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def _pip_target(extras: list[str]) -> str:
    """Build the install spec: <muse-repo-path>[server,extra1,extra2,...].

    `server` is always present; modality extras append. Bracket-comma
    syntax matches PEP 508 extras.
    """
    extras_set = ["server", *extras]
    spec = ",".join(extras_set)
    return f"{_muse_repo_root()}[{spec}]"


def refresh_one(
    model_id: str,
    *,
    no_extras: bool = False,
) -> RefreshResult:
    """Refresh a single model's venv.

    Two pip invocations:
      1. install --upgrade -e <muse>[server,<modality-extras>]
      2. install --upgrade <model's pip_extras...>  (skipped if --no-extras
         or pip_extras is empty)

    On any non-zero pip exit, returns RefreshResult(state='failed') with
    captured stdout+stderr and skips step 2. Catalog entries that point
    at a missing python_path return state='skipped'.
    """
    catalog = _read_catalog()
    entry = catalog.get(model_id)
    if not entry:
        return RefreshResult(model_id, "skipped", "not in catalog")
    python_path = entry.get("python_path")
    if not python_path or not Path(python_path).exists():
        return RefreshResult(
            model_id,
            "skipped",
            f"python_path missing or not found: {python_path}",
        )

    try:
        manifest = get_manifest(model_id)
    except KeyError:
        manifest = {}
    modality = manifest.get("modality") or entry.get("modality") or ""
    pip_extras_list = list(manifest.get("pip_extras") or ())
    muse_extras = _infer_extras(modality)
    target = _pip_target(muse_extras)

    cmd = [python_path, "-m", "pip", "install", "--upgrade", "-e", target]
    logger.info("refresh %s: %s", model_id, " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return RefreshResult(
            model_id,
            "failed",
            "muse[server] install failed",
            proc.stdout + proc.stderr,
            extras=muse_extras,
        )

    if not no_extras and pip_extras_list:
        cmd2 = [python_path, "-m", "pip", "install", "--upgrade", *pip_extras_list]
        logger.info("refresh %s extras: %s", model_id, " ".join(cmd2))
        proc2 = subprocess.run(cmd2, capture_output=True, text=True)
        if proc2.returncode != 0:
            return RefreshResult(
                model_id,
                "failed",
                "pip_extras install failed",
                proc2.stdout + proc2.stderr,
                extras=muse_extras,
            )

    return RefreshResult(model_id, "ok", extras=muse_extras)


def _select_targets(
    *,
    model_id: str | None,
    all_: bool,
    enabled_only: bool,
) -> list[str] | None:
    """Resolve the --all/--enabled/<id> flags to a sorted list of targets.

    Returns None on usage error (caller prints a message + exits 2).
    Alphabetical order keeps `--all` deterministic across runs (same
    output ordering, same JSON shape).
    """
    catalog = _read_catalog()
    if all_:
        return sorted(catalog.keys())
    if enabled_only:
        return sorted(mid for mid in catalog if is_enabled(mid))
    if model_id is not None:
        return [model_id]
    return None


def run_refresh(
    *,
    model_id: str | None = None,
    all_: bool = False,
    enabled_only: bool = False,
    no_extras: bool = False,
    as_json: bool = False,
) -> int:
    """Entry point for `muse models refresh`.

    Returns 0 if every selected target succeeded or was skipped, 1 if
    any failed, 2 on usage error (no targets selected).
    """
    targets = _select_targets(
        model_id=model_id, all_=all_, enabled_only=enabled_only,
    )
    if targets is None:
        print(
            "error: pass <model_id>, --all, or --enabled",
            file=sys.stderr,
        )
        return 2

    if not targets:
        print("no targets selected")
        return 0

    results = [refresh_one(mid, no_extras=no_extras) for mid in targets]

    if as_json:
        print(json.dumps([asdict(r) for r in results], indent=2))
    else:
        marks = {"ok": "OK", "failed": "FAIL", "skipped": "skip"}
        for r in results:
            mark = marks.get(r.state, r.state)
            line = f"  {r.model_id:24s} [{mark:5s}] {r.message}"
            print(line)
            if r.state == "failed" and r.pip_output:
                # Show last few lines of pip output so the user can
                # diagnose without re-running with --json.
                tail = "\n".join(r.pip_output.strip().splitlines()[-5:])
                print(f"    pip output (last 5 lines):\n    {tail}")

    n_ok = sum(1 for r in results if r.state == "ok")
    n_failed = sum(1 for r in results if r.state == "failed")
    n_skipped = sum(1 for r in results if r.state == "skipped")
    if not as_json:
        print(f"\n{n_ok} ok, {n_failed} failed, {n_skipped} skipped")
    return 0 if n_failed == 0 else 1
