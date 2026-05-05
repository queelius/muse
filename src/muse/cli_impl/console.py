"""Shared rich.Console + display helpers for the muse CLI.

A single Console instance is used across CLI commands so that
formatting (truecolor support, terminal width, NO_COLOR handling) is
consistent. Rich already respects the NO_COLOR env automatically; the
explicit `--no-color` flag on user-facing commands overrides via
`force_terminal=False`.

Status encoding is symbolic: each of the four model statuses gets
exactly one glyph + color pair. Glyphs are single East-Asian-Width
narrow characters so they monospace cleanly in tables. Color carries
the redundant signal for color-friendly terminals; the glyph alone
suffices in piped/no-color output.
"""
from __future__ import annotations

import os
from functools import lru_cache

from rich.console import Console
from rich.text import Text


# Status encoding: glyph + color. The same dict is used for the table
# rendering and for the legend, so the encoding and its documentation
# can never drift out of sync.
STATUS_STYLE: dict[str, tuple[str, str]] = {
    # status_name: (glyph, rich-style-name)
    "enabled": ("●", "bold green"),
    "disabled": ("○", "red"),
    "recommended": ("★", "yellow"),
    "available": ("·", "dim cyan"),
}


@lru_cache(maxsize=2)
def get_console(force_no_color: bool = False) -> Console:
    """Return a singleton rich.Console.

    `force_no_color=True` is the explicit `--no-color` override on
    user-facing commands. NO_COLOR env is handled by rich automatically
    via auto-detection; passing force_terminal=False here is the harder
    override (also disables color spans in output).
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


def truncate(text: str, max_width: int, *, ellipsis: str = "…") -> str:
    """Truncate text to fit within max_width, including the ellipsis.

    A no-op when the text already fits. max_width <= len(ellipsis)
    returns just the ellipsis.
    """
    if len(text) <= max_width:
        return text
    if max_width <= len(ellipsis):
        return ellipsis
    return text[: max_width - len(ellipsis)] + ellipsis
