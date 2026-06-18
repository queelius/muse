"""Packaging guards for non-Python runtime data files.

`curated.yaml` (recommended-models list) and `chat_formats.yaml` (GGUF
chat-template map) are loaded at runtime via importlib.resources. They
MUST be declared as setuptools package-data, or they are excluded from
the built wheel and every pip-installed user sees zero curated models and
degraded GGUF chat-format detection.

Regression guard for the v0.46.0 fix: editable installs find these files
on the source tree regardless of packaging config, so unit tests that only
call load_curated() would not have caught the wheel exclusion. These tests
assert the declaration itself plus the source files' presence; the release
ritual additionally smoke-installs the wheel and asserts the curated count.
"""
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - 3.10 fallback
    tomllib = None


def test_runtime_yaml_data_files_exist():
    assert (_ROOT / "src/muse/curated.yaml").is_file()
    assert (_ROOT / "src/muse/chat_formats.yaml").is_file()


def test_yaml_data_files_declared_as_package_data():
    """Without this declaration the yaml files don't ship in the wheel."""
    if tomllib is not None:
        with open(_ROOT / "pyproject.toml", "rb") as f:
            cfg = tomllib.load(f)
        pkg_data = (
            cfg.get("tool", {}).get("setuptools", {}).get("package-data", {})
        )
        patterns = [p for pats in pkg_data.values() for p in pats]
        assert any(p.endswith("*.yaml") or p == "*.yaml" for p in patterns), (
            "curated.yaml / chat_formats.yaml must be declared in "
            "[tool.setuptools.package-data]; otherwise they are excluded "
            "from the wheel (see test docstring)."
        )
    else:  # pragma: no cover - exercised only on Python 3.10
        text = (_ROOT / "pyproject.toml").read_text()
        assert "[tool.setuptools.package-data]" in text
        assert "*.yaml" in text
