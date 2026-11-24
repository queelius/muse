"""The voice catalog and light routes stay importable without numpy/torch.

The gateway serves voice listings, details, reference audio and metadata
edits from these modules, so their module-level imports are limited to the
stdlib, the web stack and muse.core (plus each other).
"""
import ast
import sys
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[3] / "src" / "muse" / "modalities" / "audio_speech"
ALLOWED_PREFIXES = (
    "fastapi", "starlette", "pydantic", "muse.core",
    "muse.modalities.audio_speech.voice_",
)


def _top_level_modules(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    out = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            out += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            out.append(node.module)
    return out


@pytest.mark.parametrize("name", ["voice_errors.py", "voice_catalog.py", "light.py"])
def test_light_modules_import_only_allowed_modules(name):
    path = PKG / name
    if not path.exists():
        pytest.skip(f"{name} not written yet")
    for mod in _top_level_modules(path):
        top = mod.split(".")[0]
        assert top in sys.stdlib_module_names or top == "__future__" or mod.startswith(
            ALLOWED_PREFIXES), f"{name} imports {mod} at module level"
