"""Meta-test: no modality route may leak a caught exception's str(e) (or an
f-string interpolation of it) into a 500 error_response body.

Finding 1 (v0.58.1 review): 14 sites across the modality wire layer did
`error_response(500, "...", str(e))`, leaking filesystem paths, CUDA driver
text, or other backend-implementation detail to external clients. The
repo's own hardened pattern (text_translation/routes.py) is:
`logger.exception(...)` server-side + a generic client-facing message.

This test AST-walks every `src/muse/modalities/*/routes.py`, finds every
`except ... as <name>:` handler, and flags any `error_response(500, ...)`
call inside that handler whose message argument is `str(<name>)` or an
f-string that interpolates `<name>` (or an attribute/call rooted at
<name>, e.g. `str(e.args)`). A 400 (or other 4xx) leaking str(e) is NOT
flagged here -- that's client-input detail, not backend internals, and is
out of scope for this finding.
"""
from __future__ import annotations

import ast
from pathlib import Path

MODALITIES_DIR = Path(__file__).resolve().parents[2] / "src" / "muse" / "modalities"


def _root_name(node: ast.AST) -> str | None:
    """Walk down an expression to find the root Name id, if any.

    Handles `e`, `e.args`, `str(e)`, `str(e.args[0])`, etc. -- anything
    whose ultimate root is a bare Name.
    """
    while True:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            node = node.value
        elif isinstance(node, ast.Subscript):
            node = node.value
        elif isinstance(node, ast.Call):
            # str(e), repr(e), f(e) -- descend into the first argument.
            if node.args:
                node = node.args[0]
            else:
                return None
        else:
            return None


def _mentions_name(node: ast.AST, name: str) -> bool:
    """True if `node` is an expression that leaks `name`'s value: either
    directly (str(e), e.args, ...) or via an f-string FormattedValue."""
    if isinstance(node, ast.JoinedStr):
        return any(
            isinstance(part, ast.FormattedValue)
            and _root_name(part.value) == name
            for part in node.values
        )
    return _root_name(node) == name


def _is_500(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value == 500


def _offenders_in_file(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    offenders: list[str] = []

    class Visitor(ast.NodeVisitor):
        def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
            if node.name:
                for call in ast.walk(node):
                    if (
                        isinstance(call, ast.Call)
                        and isinstance(call.func, ast.Name)
                        and call.func.id == "error_response"
                        and len(call.args) >= 3
                        and _is_500(call.args[0])
                        and _mentions_name(call.args[2], node.name)
                    ):
                        offenders.append(
                            f"{path}:{call.lineno}: error_response(500, ...) "
                            f"leaks caught exception {node.name!r}"
                        )
            self.generic_visit(node)

    Visitor().visit(tree)
    return offenders


def _all_routes_files() -> list[Path]:
    return sorted(MODALITIES_DIR.glob("*/routes.py"))


def test_no_modality_route_leaks_exception_str_in_500_body():
    files = _all_routes_files()
    assert len(files) >= 15, (
        f"expected many modalities/*/routes.py files, found {len(files)}; "
        "the glob or directory layout may have changed"
    )

    all_offenders: list[str] = []
    for path in files:
        all_offenders.extend(_offenders_in_file(path))

    assert not all_offenders, (
        "The following routes leak a caught exception's str(e) into a "
        "500 error_response body (see text_translation/routes.py for the "
        "hardened pattern -- logger.exception server-side + generic "
        "client message):\n" + "\n".join(all_offenders)
    )


# ---------------------------------------------------------------------------
# Spot behavior tests: a handful of concrete backends across modalities that
# raise an exception carrying a secret-looking string, verifying the string
# never reaches the client body but does reach the server log (via
# logger.exception, captured through Python's logging -> caplog).
# ---------------------------------------------------------------------------

_SECRET = "secret /home/alex/.muse/venvs/leak/model.bin"


def test_text_rerank_backend_error_does_not_leak_exception_text(caplog):
    from unittest.mock import MagicMock

    from muse.core.registry import ModalityRegistry
    from muse.core.server import create_app
    from muse.modalities.text_rerank import MODALITY, build_router

    from fastapi.testclient import TestClient

    backend = MagicMock()
    backend.model_id = "bge-reranker-v2-m3"
    backend.rerank.side_effect = RuntimeError(_SECRET)
    reg = ModalityRegistry()
    reg.register(MODALITY, backend, manifest={"model_id": "bge-reranker-v2-m3"})
    app = create_app(registry=reg, routers={MODALITY: build_router(reg)})
    client = TestClient(app, raise_server_exceptions=False)

    with caplog.at_level("ERROR"):
        r = client.post("/v1/rerank", json={"query": "q", "documents": ["a", "b"]})

    assert r.status_code == 500
    assert _SECRET not in r.text
    assert r.json()["error"]["code"] == "internal_error"
    assert any(_SECRET in rec.getMessage() or _SECRET in str(rec.exc_info)
               for rec in caplog.records), "secret should reach the server log"


def test_embedding_text_backend_error_does_not_leak_exception_text(caplog):
    from muse.core.registry import ModalityRegistry
    from muse.core.server import create_app
    from muse.modalities.embedding_text.routes import build_router

    from fastapi.testclient import TestClient

    class _RaisingModel:
        model_id = "fake-embed"
        dimensions = 4

        def embed(self, input, *, dimensions=None, **_):
            raise RuntimeError(_SECRET)

    reg = ModalityRegistry()
    reg.register("embedding/text", _RaisingModel())
    app = create_app(registry=reg, routers={"embedding/text": build_router(reg)})
    client = TestClient(app, raise_server_exceptions=False)

    with caplog.at_level("ERROR"):
        r = client.post(
            "/v1/embeddings", json={"input": "hello", "model": "fake-embed"},
        )

    assert r.status_code == 500
    assert _SECRET not in r.text
    assert r.json()["error"]["code"] == "internal_error"
    assert any(_SECRET in rec.getMessage() or _SECRET in str(rec.exc_info)
               for rec in caplog.records), "secret should reach the server log"


def test_text_summarization_backend_error_does_not_leak_exception_text(caplog):
    from unittest.mock import MagicMock

    from muse.core.registry import ModalityRegistry
    from muse.core.server import create_app
    from muse.modalities.text_summarization import MODALITY, build_router

    from fastapi.testclient import TestClient

    backend = MagicMock()
    backend.model_id = "bart-large-cnn"
    backend.summarize.side_effect = RuntimeError(_SECRET)
    reg = ModalityRegistry()
    reg.register(MODALITY, backend, manifest={"model_id": "bart-large-cnn"})
    app = create_app(registry=reg, routers={MODALITY: build_router(reg)})
    client = TestClient(app, raise_server_exceptions=False)

    with caplog.at_level("ERROR"):
        r = client.post("/v1/summarize", json={"text": "some long text here"})

    assert r.status_code == 500
    assert _SECRET not in r.text
    assert r.json()["error"]["code"] == "internal_error"
    assert any(_SECRET in rec.getMessage() or _SECRET in str(rec.exc_info)
               for rec in caplog.records), "secret should reach the server log"
