"""Shared rich.Console + display helpers for the muse CLI.

A single Console instance is used across CLI commands so that
formatting (truecolor support, terminal width, NO_COLOR handling) is
consistent. Rich already respects the NO_COLOR env automatically; the
explicit `--no-color` flag on user-facing commands overrides via
`Console(no_color=True, highlight=False)`.

Status encoding is symbolic: each of the five model statuses gets
exactly one glyph + color pair. Glyphs are single East-Asian-Width
narrow characters so they monospace cleanly in tables. Color carries
the redundant signal for color-friendly terminals; the glyph alone
suffices in piped/no-color output.

The v0.40.0 lazy-load model splits "enabled" into a four-state lifecycle
(loaded vs unloaded) plus the original disabled / recommended / available
that catalog metadata can describe without consulting runtime state.
"""
from __future__ import annotations

import os
from functools import lru_cache

from rich.console import Console
from rich.text import Text


# Status encoding: glyph + color. The same dict is used for the table
# rendering and for the legend, so the encoding and its documentation
# can never drift out of sync.
#
# v0.40.0 split "enabled" into "enabled_loaded" (bright green filled
# circle: catalog-enabled AND currently in the director's loaded set)
# and "enabled_unloaded" (dim yellow half circle: catalog-enabled but
# not currently held by a worker, will cold-load on next request). The
# old "enabled" key is gone; callers should use "enabled_loaded" for
# the active state.
STATUS_STYLE: dict[str, tuple[str, str]] = {
    # status_name: (glyph, rich-style-name)
    "enabled_loaded": ("●", "bold green"),
    "enabled_unloaded": ("◐", "yellow"),
    "disabled": ("○", "red"),
    "recommended": ("★", "yellow"),
    "available": ("·", "dim cyan"),
}


@lru_cache(maxsize=2)
def get_console(force_no_color: bool = False) -> Console:
    """Return a singleton rich.Console.

    `force_no_color=True` is the explicit `--no-color` override on
    user-facing commands. NO_COLOR env is handled by rich automatically
    via auto-detection; passing `no_color=True` here is the harder
    override (also disables color spans in output, unlike relying on
    auto-detection alone).
    """
    if force_no_color or os.environ.get("NO_COLOR"):
        return Console(no_color=True, highlight=False)
    return Console(highlight=False)


def status_glyph(status: str) -> Text:
    """Return a colored single-char rich.Text glyph for one status.

    Falls back to a plain "?" glyph if status is unrecognized so a
    typo'd manifest doesn't crash the renderer.
    """
    glyph, style = STATUS_STYLE.get(status, ("?", "dim"))
    return Text(glyph, style=style)


def status_legend() -> Text:
    """Inline legend listing every status glyph + name.

    Used in the footer of `muse models list` so first-time users learn
    the encoding without consulting docs.
    """
    parts: list[Text] = []
    for name, (glyph, style) in STATUS_STYLE.items():
        parts.append(Text(glyph, style=style))
        parts.append(Text(f" {name}  "))
    out = Text()
    for p in parts:
        out.append(p)
    return out
