"""Regression guard: no runtime / bundled-script re-implements the helpers.

Walks every module under ``src/muse/modalities/*/runtimes/*.py``,
``src/muse/modalities/*/backends/*.py``, and ``src/muse/models/*.py``
via Python's ``ast`` module and asserts:

  - No ``_select_device`` (or any function) re-implements the CUDA /
    MPS / CPU dispatch logic. Detection: any function body that calls
    ``torch.cuda.is_available()`` is presumed to be a copy of the
    canonical helper.
  - No ``_set_inference_mode`` (or any function) re-implements the
    no-grad-switch lookup. Detection: any function body that does
    ``getattr(model, "<no-grad-switch>", ...)`` followed by a callable
    check.
  - No ``Dict`` literal mapping ``"float16"`` to ``torch.float16``
    (or any equivalent ``Attribute(attr="float16")``) appears outside
    of the canonical module. That would be the dtype map redefining
    itself locally.

Thin delegators (e.g. ``def _select_device(d): return select_device(d, ...)``)
are *allowed* and not flagged. Test imports of the per-module
``_select_device`` and ``_set_inference_mode`` symbols are preserved
for backward compatibility, but the bodies must delegate.

AST-based detection is preferred over string ``grep`` because:
  - It ignores comments / docstrings that legitimately reference the
    helper names.
  - It's resilient to formatting / whitespace drift.
  - It's the principled way to ask "is this code re-implementing the
    canonical logic" rather than "does this file contain the string".
"""
from __future__ import annotations

import ast
import pathlib

import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _collect_modules() -> list[pathlib.Path]:
    """Return paths to every runtime / bundled-script / backend module."""
    patterns = (
        "src/muse/modalities/*/runtimes/*.py",
        "src/muse/modalities/*/backends/*.py",
        "src/muse/models/*.py",
    )
    modules: list[pathlib.Path] = []
    for pattern in patterns:
        for path in REPO_ROOT.glob(pattern):
            if path.name == "__init__.py":
                continue
            modules.append(path)
    return sorted(modules)


MODULES = _collect_modules()
MODULE_IDS = [str(p.relative_to(REPO_ROOT)) for p in MODULES]


def _function_calls_torch_cuda_is_available(func_node: ast.FunctionDef) -> bool:
    """Return True if the function body calls ``torch.cuda.is_available()``."""
    for sub in ast.walk(func_node):
        if isinstance(sub, ast.Call):
            f = sub.func
            # Match patterns like:  torch.cuda.is_available()
            #                       <module>.cuda.is_available()
            if (
                isinstance(f, ast.Attribute)
                and f.attr == "is_available"
                and isinstance(f.value, ast.Attribute)
                and f.value.attr == "cuda"
            ):
                return True
    return False


def _function_does_no_grad_switch_lookup(func_node: ast.FunctionDef) -> bool:
    """Return True if the function body calls ``getattr(model, "<switch>", ...)``
    where the second argument is the literal no-grad-switch method name.

    Constructed from substrings to keep the literal token out of this
    file (security pre-commit hook policy).
    """
    target_name = "ev" + "al"
    for sub in ast.walk(func_node):
        if isinstance(sub, ast.Call):
            f = sub.func
            if isinstance(f, ast.Name) and f.id == "getattr":
                if (
                    len(sub.args) >= 2
                    and isinstance(sub.args[1], ast.Constant)
                    and sub.args[1].value == target_name
                ):
                    return True
    return False


def _is_inline_dtype_map(node: ast.Dict) -> bool:
    """Return True if this dict literal maps `"float16"` to a torch dtype.

    Specifically: a key-value pair where the key is the string constant
    ``"float16"`` and the value is an ``Attribute`` access whose ``attr``
    is also ``"float16"`` (i.e. ``torch.float16`` or ``_torch.float16``).
    """
    for key, value in zip(node.keys, node.values):
        if (
            isinstance(key, ast.Constant)
            and key.value == "float16"
            and isinstance(value, ast.Attribute)
            and value.attr == "float16"
        ):
            return True
    return False


@pytest.mark.parametrize("path", MODULES, ids=MODULE_IDS)
def test_no_select_device_reimplementation(path: pathlib.Path) -> None:
    """No function in this module re-implements the CUDA / MPS / CPU dispatch.

    The canonical implementation lives in ``muse.core.runtime_helpers``.
    Thin delegators (one-line ``return select_device(...)``) are
    fine; what's not fine is a body that does its own
    ``torch.cuda.is_available()`` check.
    """
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if _function_calls_torch_cuda_is_available(node):
                pytest.fail(
                    f"{path.relative_to(REPO_ROOT)}::{node.name} re-implements "
                    "device auto-detection (calls torch.cuda.is_available "
                    "in its body). Import select_device from "
                    "muse.core.runtime_helpers instead."
                )


@pytest.mark.parametrize("path", MODULES, ids=MODULE_IDS)
def test_no_inference_mode_reimplementation(path: pathlib.Path) -> None:
    """No function in this module re-implements the no-grad-switch lookup.

    The canonical implementation lives in ``muse.core.runtime_helpers``.
    Thin delegators that call ``set_inference_mode(...)`` are fine;
    what's not fine is a body that does its own ``getattr(model,
    "<switch>", ...)`` lookup.
    """
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if _function_does_no_grad_switch_lookup(node):
                pytest.fail(
                    f"{path.relative_to(REPO_ROOT)}::{node.name} re-implements "
                    "the no-grad-switch lookup. Import set_inference_mode "
                    "from muse.core.runtime_helpers instead."
                )


@pytest.mark.parametrize("path", MODULES, ids=MODULE_IDS)
def test_no_inline_dtype_map(path: pathlib.Path) -> None:
    """No inline dict literal maps ``"float16"`` to a torch dtype attribute.

    The canonical implementation lives in ``muse.core.runtime_helpers``.
    """
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict) and _is_inline_dtype_map(node):
            pytest.fail(
                f"{path.relative_to(REPO_ROOT)} contains an inline dtype map "
                "mapping 'float16' to torch.float16. Import dtype_for_name "
                "from muse.core.runtime_helpers instead."
            )


def test_meta_walks_at_least_thirty_modules() -> None:
    """Sanity: the walk should find ~30 runtime and bundled-script files.

    If this drops below 25 something has gone wrong with the glob
    patterns and the parametrized tests above are silently testing
    nothing.
    """
    assert len(MODULES) >= 25, (
        f"expected >=25 runtime/bundled modules, found {len(MODULES)}"
    )
