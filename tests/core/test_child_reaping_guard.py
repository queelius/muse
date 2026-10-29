"""Guard: only reviewed owners may reap (or poll) a child process.

``poll()``, ``wait()``, ``communicate()`` and ``os.wait*`` reap a POSIX
child. Once an owned leader is reaped, its PID and PID==PGID group can be
reused, so a reap outside the owner's locked lifecycle breaks the
invariant every group signal relies on ("only the owner reaps, under its
owning lock"). This AST walk fails on any such call in ``src/muse`` that is
not on the reviewed allow-list below.
"""
from __future__ import annotations

import ast
from pathlib import Path

import muse

SRC = Path(muse.__file__).resolve().parent

_REAPING_METHODS = {"poll", "communicate"}
_PROCESS_WAIT_RECEIVERS = {"process", "proc", "child", "popen"}

# (module path relative to src/muse, enclosing function) -> why it may reap.
_ALLOWED: dict[tuple[str, str], str] = {
    # The shared primitives themselves.
    ("core/owned_process.py", "observe_leader"): "poll of an unpinnable "
        "child, only where WNOWAIT is missing, under the caller's owning lock",
    ("core/owned_process.py", "leader_exited_nowait"): "waitid(WNOWAIT) never reaps",
    ("core/owned_process.py", "reap_exited_leader"): "the owner's final reap",
    # The installer runner: one thread owns its child end to end.
    ("core/owned_command.py", "_target_alive"): "unsupported-platform fallback",
    ("core/owned_command.py", "_wait_owned_leader"): "unsupported-platform fallback",
    ("core/owned_command.py", "_reap_owned_leader"): "final reap after group signal",
    ("core/owned_command.py", "_terminate_owned_fallback"): "fallback reap",
}


def _receiver_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _reaping_calls(tree: ast.AST):
    """Yield (function name, lineno, description) for each reaping call."""
    stack: list[str] = []

    def visit(node: ast.AST):
        is_function = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        if is_function:
            stack.append(node.name)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            attr = node.func.attr
            receiver = _receiver_name(node.func.value)
            hit = None
            if attr in _REAPING_METHODS:
                hit = f".{attr}()"
            elif attr.startswith("wait") and receiver == "os":
                hit = f"os.{attr}()"
            elif attr == "wait" and receiver is not None and (
                receiver.lower() in _PROCESS_WAIT_RECEIVERS
                or receiver.lower().endswith(("process", "proc"))
            ):
                hit = f"{receiver}.wait()"
            if hit is not None:
                yield (stack[-1] if stack else "<module>", node.lineno, hit)
        for child in ast.iter_child_nodes(node):
            yield from visit(child)
        if is_function:
            stack.pop()

    yield from visit(tree)


def test_no_unreviewed_child_reaping_in_src():
    offenders = []
    seen_allowed = set()
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for function, lineno, hit in _reaping_calls(tree):
            key = (rel, function)
            if key in _ALLOWED:
                seen_allowed.add(key)
                continue
            offenders.append(f"src/muse/{rel}:{lineno} {function}: {hit}")
    assert not offenders, (
        "child reaping outside a reviewed owner (route it through "
        "muse.core.owned_process under the owning lock, or review and "
        "allow-list it):\n" + "\n".join(offenders)
    )
    stale = sorted(set(_ALLOWED) - seen_allowed)
    assert not stale, f"allow-list entries no longer needed: {stale}"


def test_guard_detects_the_patterns_it_forbids():
    sample = ast.parse(
        "def f(spec, proc):\n"
        "    spec.process.poll()\n"
        "    proc.wait(timeout=1)\n"
        "    os.waitpid(-1, 0)\n"
        "    p.communicate()\n"
        "    stop_event.wait(1)\n"
    )
    hits = [hit for _fn, _line, hit in _reaping_calls(sample)]
    assert hits == [".poll()", "proc.wait()", "os.waitpid()", ".communicate()"]
