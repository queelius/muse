"""Regression guard: no stray `MUSE_*` env reads outside `muse.core.config`.

After the config migration (Tasks 1-7), every `MUSE_*` environment
variable read is routed through `muse.core.config.get(...)`, except the
two bootstrap keys (`MUSE_CATALOG_DIR`, `MUSE_CONFIG`) that `config.py`
itself must read directly in order to locate the config file before it
can be loaded. This test AST-walks every module under `src/muse/` and
fails if it finds a raw `os.environ.get("MUSE_...")`, `os.getenv(
"MUSE_...")`, or `os.environ["MUSE_..."]` anywhere outside
`core/config.py`.

Mirrors `tests/core/test_runtime_helpers_meta.py`.
"""
import ast
import pathlib

SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "muse"
ALLOW = {"core/config.py"}  # only the bootstrap reads live here


def _muse_env_reads(tree) -> list[str]:
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # os.environ.get("MUSE_...") / os.getenv("MUSE_...")
        args = node.args
        if not args or not isinstance(args[0], ast.Constant):
            continue
        if not (isinstance(args[0].value, str) and args[0].value.startswith("MUSE_")):
            continue
        fn = node.func
        if isinstance(fn, ast.Attribute) and fn.attr in {"get", "getenv"}:
            hits.append(args[0].value)
    # os.environ["MUSE_..."] subscripts
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            v = node.slice.value
            if isinstance(v, str) and v.startswith("MUSE_"):
                hits.append(v)
    return hits


def test_no_stray_muse_env_reads_outside_config():
    offenders = {}
    for path in SRC.rglob("*.py"):
        rel = path.relative_to(SRC).as_posix()
        if rel in ALLOW:
            continue
        hits = _muse_env_reads(ast.parse(path.read_text()))
        if hits:
            offenders[rel] = hits
    assert not offenders, f"stray MUSE_* env reads (route via muse.core.config): {offenders}"
