"""Pure helpers for OpenAI gpt-image style request options.

Size snapping, quality presets, output format, background and the list of
accepted-but-ignored fields. No heavy imports; routes call these.
"""
from __future__ import annotations

import math
import re
from typing import Any, Mapping

_SIZE_RE = re.compile(r"^(\d+)x(\d+)$")
_MIN_SIDE, _MAX_SIDE = 64, 2048
_QUALITIES = ("low", "medium", "high", "auto")
# OpenAI dall-e-3 quality names, accepted as aliases of the gpt-image ones.
_QUALITY_ALIASES = {"standard": "medium", "hd": "high"}
_FORMATS = ("png", "jpeg", "webp")


class OptionError(ValueError):
    """A request option is invalid or unsupported; `code` is the wire error code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def snaps(caps: Mapping) -> bool:
    """True iff the model snaps sizes (declares size_multiple or max_pixels)."""
    return "size_multiple" in caps or "max_pixels" in caps


def parse_size(size: str | None, caps: Mapping) -> tuple[int, int]:
    if size is None or size == "auto":
        if snaps(caps) and caps.get("default_size"):
            w, h = caps["default_size"]
            return (int(w), int(h))
        return (512, 512)
    m = _SIZE_RE.match(size) if isinstance(size, str) else None
    if not m:
        raise OptionError("invalid_parameter", f"size must be 'auto' or 'WxH', got {size!r}")
    w, h = int(m.group(1)), int(m.group(2))
    if not (_MIN_SIDE <= w <= _MAX_SIDE and _MIN_SIDE <= h <= _MAX_SIDE):
        raise OptionError(
            "invalid_parameter",
            f"size sides must be between {_MIN_SIDE} and {_MAX_SIDE}, got {size!r}",
        )
    return (w, h)


def resolve_quality(quality: str | None, caps: Mapping) -> tuple[str, dict]:
    quality = _QUALITY_ALIASES.get(quality, quality)
    if quality is not None and quality not in _QUALITIES:
        raise OptionError(
            "invalid_parameter",
            "quality must be one of low, medium, high, auto "
            f"(or standard, hd); got {quality!r}",
        )
    presets = caps.get("quality_presets")
    if not presets:
        return (quality or "auto", {})
    name = "medium" if quality in (None, "auto") else quality
    return (name, dict(presets.get(name, {})))


def snap_size(width: int, height: int, caps: Mapping, preset: Mapping) -> tuple[int, int]:
    if not snaps(caps):
        return (width, height)
    m = caps.get("size_multiple") or 1
    limits = [v for v in (caps.get("max_pixels"), preset.get("max_pixels")) if v]
    w, h = width, height
    if limits and w * h > min(limits):
        scale = math.sqrt(min(limits) / (w * h))
        w, h = int(w * scale), int(h * scale)
    w, h = (w // m) * m, (h // m) * m
    floor = -(-max(m, _MIN_SIDE) // m) * m
    return (max(w, floor), max(h, floor))


def check_output(output_format: str | None, output_compression: int | None) -> str:
    fmt = output_format or "png"
    if fmt not in _FORMATS:
        raise OptionError(
            "invalid_parameter", f"output_format must be png, jpeg or webp; got {fmt!r}"
        )
    if output_compression is not None and not 0 <= output_compression <= 100:
        raise OptionError("invalid_parameter", "output_compression must be between 0 and 100")
    return fmt


def check_background(background: str | None) -> str:
    if background in (None, "auto"):
        return "auto"
    if background == "opaque":
        return "opaque"
    if background == "transparent":
        raise OptionError("background_not_supported", "transparent backgrounds are not supported")
    raise OptionError(
        "invalid_parameter", f"background must be auto, opaque or transparent; got {background!r}"
    )


def ignored_fields(sent: Mapping[str, Any], caps: Mapping) -> list[str]:
    """Names of provided fields that had no effect on this model."""
    out = set()
    given = {k for k, v in sent.items() if v is not None}
    for always in ("moderation", "style"):
        if always in given:
            out.add(always)
    if "quality" in given and not caps.get("quality_presets"):
        out.add("quality")
    if "negative_prompt" in given and caps.get("supports_negative_prompt") is False:
        out.add("negative_prompt")
    if "guidance" in given and caps.get("supports_guidance") is False:
        out.add("guidance")
    if "output_compression" in given and (sent.get("output_format") or "png") == "png":
        out.add("output_compression")
    return sorted(out)
