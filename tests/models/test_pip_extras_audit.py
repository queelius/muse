"""Static audit: bundled-script pip_extras tuples cover their runtime imports.

Reads each src/muse/models/*.py via ast.parse, walks the import nodes
(top-level + inside-function), maps each module to its PyPI dist via
MODULE_TO_PYPI, and asserts every PyPI dist is declared in the script's
MANIFEST['pip_extras'].

This is a "best effort" check: runtimes that load modules via importlib
are out of scope (none in muse today). The audit defends against the
most common pip_extras gap, where a runtime module is added, its
top-level `from X import Y` is added, but pip_extras is not updated;
fresh-venv installs from `muse pull <id>` then crash at load time.
"""
from __future__ import annotations

import ast
import importlib
import re
from pathlib import Path

import pytest


# Maps a Python top-level module name to the PyPI distribution name
# that provides it. Build incrementally; new bundled scripts add
# entries.
MODULE_TO_PYPI: dict[str, str] = {
    "torch": "torch",
    "transformers": "transformers",
    "diffusers": "diffusers",
    "accelerate": "accelerate",
    "PIL": "Pillow",
    "safetensors": "safetensors",
    "sentence_transformers": "sentence-transformers",
    "scipy": "scipy",
    "inflect": "inflect",
    "unidecode": "unidecode",
    "numpy": "numpy",
    "librosa": "librosa",
    "kokoro": "kokoro",
    "misaki": "misaki",
    "soundfile": "soundfile",
    "einops": "einops",
    "imageio": "imageio",
}

# Top-level module names that the audit should skip:
#  - this project's own packages (muse.*)
#  - the standard library (best-effort enumeration)
#  - private-prefix sentinels in lazy-imports (start with underscore)
SKIP_NAMES: set[str] = {
    "muse",
    "abc",
    "argparse",
    "ast",
    "asyncio",
    "base64",
    "collections",
    "contextlib",
    "copy",
    "csv",
    "dataclasses",
    "datetime",
    "enum",
    "functools",
    "glob",
    "hashlib",
    "io",
    "itertools",
    "json",
    "logging",
    "math",
    "os",
    "pathlib",
    "queue",
    "random",
    "re",
    "shutil",
    "signal",
    "socket",
    "string",
    "struct",
    "subprocess",
    "sys",
    "tempfile",
    "threading",
    "time",
    "traceback",
    "types",
    "typing",
    "unittest",
    "urllib",
    "uuid",
    "warnings",
    "weakref",
    "zlib",
    "__future__",
}


def _bundled_scripts() -> list[Path]:
    """Return all src/muse/models/*.py except __init__.py."""
    here = Path(__file__).resolve()
    repo_root = here.parents[2]
    models_dir = repo_root / "src" / "muse" / "models"
    return sorted(p for p in models_dir.glob("*.py") if p.name != "__init__.py")


def _walk_imports(tree: ast.AST) -> set[str]:
    """Return the set of top-level module names imported in tree.

    Handles:
      - import X.Y         -> X
      - import X.Y as Z    -> X
      - from X.Y import Z  -> X (only when level == 0; relative
                                 imports skip)
    """
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                out.add(node.module.split(".", 1)[0])
    return out


def _read_manifest(script: Path) -> dict:
    """Parse MANIFEST = { ... } from a script via importlib.

    Bundled scripts declare MANIFEST at module top with no heavy
    imports gated by it; importing the module is safe even on a
    bare install.
    """
    rel = "muse.models." + script.stem
    mod = importlib.import_module(rel)
    return getattr(mod, "MANIFEST", {})


def _strip_pip_extras_to_dists(extras: tuple[str, ...]) -> set[str]:
    """('torch>=2.1', 'misaki[en]') -> {'torch', 'misaki'}.

    Drops version specifiers and bracketed extras so the audit can
    compare bare PyPI dist names.
    """
    out: set[str] = set()
    for entry in extras:
        bare = re.sub(r"\[.*?\]", "", entry)
        bare = re.split(r"[<>=!~]", bare, maxsplit=1)[0]
        out.add(bare.strip())
    return out


def _classify_module(name: str) -> str | None:
    """Return PyPI dist name for `name`, or None if module should be skipped.

    Calls pytest.fail when the module isn't classified anywhere; this
    forces the maintainer to update MODULE_TO_PYPI when a new bundled
    script imports a new module.
    """
    if name in MODULE_TO_PYPI:
        return MODULE_TO_PYPI[name]
    if name in SKIP_NAMES:
        return None
    if name.startswith("_"):
        return None
    pytest.fail(
        f"unknown module {name!r}: add to MODULE_TO_PYPI in "
        f"tests/models/test_pip_extras_audit.py to enable the audit, "
        f"or to SKIP_NAMES if it's stdlib/local"
    )
    return None  # unreachable; pytest.fail raises


@pytest.mark.parametrize(
    "script", _bundled_scripts(), ids=lambda p: p.stem,
)
def test_pip_extras_covers_runtime_imports(script: Path):
    """Every bundled script's pip_extras includes the PyPI dists for
    its runtime imports."""
    src = script.read_text()
    tree = ast.parse(src)
    imported = _walk_imports(tree)
    needed: set[str] = set()
    for module_name in imported:
        dist = _classify_module(module_name)
        if dist:
            needed.add(dist)

    manifest = _read_manifest(script)
    declared = _strip_pip_extras_to_dists(tuple(manifest.get("pip_extras", ())))

    missing = needed - declared
    assert not missing, (
        f"{script.name}: pip_extras is missing {sorted(missing)}; "
        f"runtime imports {sorted(needed)} but declared {sorted(declared)}. "
        f"Add the missing entries to the script's MANIFEST['pip_extras']."
    )


def test_module_to_pypi_table_lookup_works_for_known_imports():
    """Smoke: every entry in MODULE_TO_PYPI maps cleanly through
    _classify_module. Guards against typos in the table itself."""
    for module_name, expected_dist in MODULE_TO_PYPI.items():
        assert _classify_module(module_name) == expected_dist
