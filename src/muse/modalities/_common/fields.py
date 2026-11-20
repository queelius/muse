"""Request-field checks shared by the JSON and multipart image routes.

Pydantic validates JSON bodies, but multipart Form fields arrive as plain
parameters, so routes like /v1/images/edits, /variations and /upscale
check them by hand. These helpers keep the ranges and messages of those
checks identical to their JSON twins: each raises ``FieldError``, which
``field_error_response`` turns into a 400 ``invalid_parameter``.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

_WXH_RE = re.compile(r"^(\d+)x(\d+)$")


class FieldError(ValueError):
    """A request field is out of range or malformed."""


def field_error_response(exc: FieldError):
    from muse.core.errors import error_response

    return error_response(400, "invalid_parameter", str(exc))


def check_range(name: str, value: Any, lo: float, hi: float) -> None:
    """``lo <= value <= hi`` unless ``value`` is None."""
    if value is not None and not lo <= value <= hi:
        raise FieldError(f"{name} must be in [{lo}, {hi}]")


def check_length(name: str, value: str | None, lo: int, hi: int) -> None:
    """``lo <= len(value) <= hi`` unless ``value`` is None."""
    if value is not None and not lo <= len(value) <= hi:
        raise FieldError(f"{name} must be {lo} to {hi} characters")


def check_choice(name: str, value: Any, choices: Iterable[Any]) -> None:
    options = list(choices)
    if value not in options:
        raise FieldError(f"{name} must be one of {options}")


def parse_bounded_size(
    value: str, *, max_side: int, max_pixels: int, min_side: int = 1,
) -> tuple[int, int]:
    """Parse ``"WxH"`` and bound each side and the pixel count."""
    m = _WXH_RE.match(value) if isinstance(value, str) else None
    if m is None:
        raise FieldError(f"size must be 'WIDTHxHEIGHT'; got {value!r}")
    width, height = int(m.group(1)), int(m.group(2))
    if width < min_side or height < min_side:
        raise FieldError(
            "size dimensions must be positive" if min_side <= 1
            else f"size dimensions must be at least {min_side}"
        )
    if width > max_side or height > max_side:
        raise FieldError(f"size dimensions must not exceed {max_side}")
    if width * height > max_pixels:
        raise FieldError(f"size must not exceed {max_pixels} pixels per frame")
    return width, height
