"""One validator for Hugging Face artifact-receipt items.

A receipt item records one immutable snapshot a pull downloaded:
`{repo_id, revision, subdir, allow_patterns?, required_patterns?}`. The HF
resolver validates plugin receipts at resolve time, `artifacts` validates a
bundled script's `hf_artifacts`, and `catalog_store` validates receipts when
it writes catalog.json. They share these rules so a receipt that would be
rejected at catalog-write time is rejected before anything is downloaded.

`subdir` is "" (a bundled single-snapshot pull), "." (a resolver single
snapshot) or a safe directory leaf (a member of a multi-snapshot bundle).

Stdlib only.
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from muse.core.resolvers import COMMIT_RE

SAFE_LEAF_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
SHA256_HEX_RE = re.compile(r"[0-9a-f]{64}")
RECEIPT_KEYS = frozenset(
    {"repo_id", "revision", "subdir", "allow_patterns", "required_patterns"}
)
_ROOT_SUBDIRS = frozenset({"", "."})


class ReceiptError(ValueError):
    """A receipt item (or pattern list) is malformed."""


def validate_patterns(value: Any, *, label: str) -> tuple[str, ...] | None:
    """None, or a non-empty sequence of non-empty strings (as a tuple)."""
    if value is None:
        return None
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or not value
        or not all(isinstance(pattern, str) and pattern for pattern in value)
    ):
        raise ReceiptError(f"{label} must be a non-empty list of non-empty strings")
    return tuple(value)


def _check_safe_relative(patterns: tuple[str, ...], *, label: str) -> None:
    for pattern in patterns:
        parts = pattern.split("/")
        if (
            pattern.startswith(("/", "\\"))
            or "\\" in pattern
            or ":" in pattern
            or "\x00" in pattern
            or any(part in {"", ".", ".."} for part in parts)
            or parts[0] == ".cache"
        ):
            raise ReceiptError(
                f"{label} contains an unsafe relative pattern: {pattern!r}"
            )


def validate_receipt_item(
    raw: Any,
    *,
    label: str,
    allow_root: bool = True,
    default_subdir: str | None = None,
    reject_unknown: bool = False,
) -> dict[str, Any]:
    """Validate one receipt item and return its canonical copy.

    The copy carries repo_id, revision, subdir and each pattern list that is
    present and not None (as a list). `allow_root=False` requires a bundle
    member leaf (no "" or "."); `default_subdir` fills a missing subdir;
    `reject_unknown` refuses keys outside RECEIPT_KEYS.
    """
    if not isinstance(raw, Mapping):
        raise ReceiptError(f"{label} must be an object")
    if reject_unknown:
        unknown = set(raw) - RECEIPT_KEYS
        if unknown:
            raise ReceiptError(f"{label} has unknown keys: {sorted(unknown)}")
    repo_id = raw.get("repo_id")
    if (
        not isinstance(repo_id, str)
        or repo_id.count("/") != 1
        or not all(repo_id.split("/"))
        or any(char.isspace() for char in repo_id)
    ):
        raise ReceiptError(f"{label}.repo_id must be an 'org/name' repo id")
    revision = raw.get("revision")
    if not isinstance(revision, str) or COMMIT_RE.fullmatch(revision) is None:
        raise ReceiptError(f"{label}.revision must be a 40-character commit")
    subdir = raw.get("subdir", default_subdir)
    if not isinstance(subdir, str) or not (
        (allow_root and subdir in _ROOT_SUBDIRS)
        or SAFE_LEAF_RE.fullmatch(subdir) is not None
    ):
        raise ReceiptError(f"{label}.subdir must be a safe directory name")
    item: dict[str, Any] = {"repo_id": repo_id, "revision": revision, "subdir": subdir}
    allow = validate_patterns(raw.get("allow_patterns"), label=f"{label}.allow_patterns")
    if allow is not None:
        item["allow_patterns"] = list(allow)
    required = validate_patterns(
        raw.get("required_patterns"), label=f"{label}.required_patterns",
    )
    if required is not None:
        _check_safe_relative(required, label=f"{label}.required_patterns")
        item["required_patterns"] = list(required)
    return item
