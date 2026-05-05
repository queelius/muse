"""`muse models list` row builder + table renderer.

The list command pulls from three sources (bundled scripts, curated
recommendations, pulled-via-resolver entries) and displays them as a
unified, status-tagged table.

The row-building logic is its own function so it can be unit-tested
without subprocess overhead. The table-rendering happens via
rich.Table for color + glyph + truncation handling; --json bypasses
rich and dumps the same row data as a JSON list.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Optional


@dataclass
class _ListRow:
    """One row for `muse models list`. Ordered so JSON output is stable."""
    id: str
    modality: str
    status: str  # one of: enabled, disabled, recommended, available
    description: str
    mem_str: str
    mem_gb: Optional[float]
    mem_device: str  # "GPU" or "CPU"

    def as_json(self) -> dict:
        return {
            "id": self.id,
            "modality": self.modality,
            "status": self.status,
            "description": self.description,
            "memory": (
                {"gb": self.mem_gb, "device": self.mem_device}
                if self.mem_gb is not None else None
            ),
        }


def build_rows() -> list[_ListRow]:
    """Build the unified row set for `muse models list`.

    Status precedence: pulled (catalog wins enabled/disabled) >
    curated-but-not-pulled (recommended) > bundled-but-not-pulled
    (available).
    """
    from muse.core.catalog import (
        _read_catalog,
        is_enabled,
        is_pulled,
        list_known,
    )
    from muse.core.curated import load_curated

    bundled_entries = {e.model_id: e for e in list_known(None)}
    curated_entries = {c.id: c for c in load_curated()}
    catalog_data = _read_catalog()

    rows: list[_ListRow] = []
    seen: set[str] = set()

    # 1. Bundled scripts and resolver-pulled entries.
    for model_id, e in bundled_entries.items():
        seen.add(model_id)
        if is_pulled(model_id):
            status = "enabled" if is_enabled(model_id) else "disabled"
        elif model_id in curated_entries:
            status = "recommended"
        else:
            status = "available"
        mem_str, mem_gb, mem_device = _model_memory_display(
            e.extra, catalog_data.get(model_id),
        )
        rows.append(_ListRow(
            id=model_id,
            modality=e.modality,
            status=status,
            description=e.description,
            mem_str=mem_str,
            mem_gb=mem_gb,
            mem_device=mem_device,
        ))

    # 2. Curated entries not already covered above (resolver-pulled
    #    curated aliases that don't have a bundled script).
    for cid, c in curated_entries.items():
        if cid in seen:
            continue
        if is_pulled(cid):
            status = "enabled" if is_enabled(cid) else "disabled"
        else:
            status = "recommended"
        mem_str, mem_gb, mem_device = _model_memory_display(
            c.capabilities or {}, catalog_data.get(cid),
        )
        rows.append(_ListRow(
            id=cid,
            modality=c.modality or "?",
            status=status,
            description=c.description or "",
            mem_str=mem_str,
            mem_gb=mem_gb,
            mem_device=mem_device,
        ))

    return rows


def filter_rows(
    rows: list[_ListRow],
    *,
    modality: str | None,
    installed: bool,
    available: bool,
) -> list[_ListRow]:
    """Apply --modality / --installed / --available filters."""
    out = list(rows)
    if modality:
        out = [r for r in out if r.modality == modality]
    if installed:
        out = [r for r in out if r.status in ("enabled", "disabled")]
    if available:
        out = [r for r in out if r.status in ("recommended", "available")]
    return out


def run_models_list(
    *,
    modality: str | None,
    installed: bool,
    available: bool,
    as_json: bool,
    no_color: bool,
) -> int:
    """Entry point invoked by the cli.py command. Returns process rc."""
    rows = build_rows()
    rows = filter_rows(
        rows, modality=modality, installed=installed, available=available,
    )

    if as_json:
        json.dump([r.as_json() for r in rows], sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    if not rows:
        suffixes = []
        if modality:
            suffixes.append(f"modality {modality!r}")
        if installed:
            suffixes.append("--installed")
        if available:
            suffixes.append("--available")
        suffix = (" matching " + ", ".join(suffixes)) if suffixes else ""
        print(f"no models{suffix}")
        return 0

    _render_table(rows, no_color=no_color)
    return 0


def _render_table(rows: list[_ListRow], *, no_color: bool) -> None:
    """Render rows. TTY -> rich.Table with color + truncation; non-TTY
    (pipe / subprocess / redirect) -> plain text with no truncation so
    scripts piping to grep get unambiguous full content.
    """
    rows = sorted(rows, key=lambda r: (r.modality, r.status, r.id))
    if sys.stdout.isatty() and not no_color:
        _render_rich_table(rows)
    else:
        _render_plain_table(rows)
    _render_footer(rows, no_color=no_color)


def _render_rich_table(rows: list[_ListRow]) -> None:
    """Pretty interactive table. Auto-fits the current terminal."""
    from rich import box
    from rich.table import Table

    from muse.cli_impl.console import get_console, status_glyph

    console = get_console()
    table = Table(
        box=box.SIMPLE,
        show_header=True,
        header_style="bold",
        pad_edge=False,
        expand=True,
    )
    # The glyph column is always 1 char; model_id and modality must
    # never truncate (they're the primary keys); description is the
    # only auto-truncated column.
    table.add_column("", width=1, no_wrap=True)
    table.add_column("model_id", no_wrap=True, style="cyan")
    table.add_column("modality", no_wrap=True)
    table.add_column("memory", justify="right", no_wrap=True)
    table.add_column("description", overflow="ellipsis", no_wrap=True, ratio=1)

    for r in rows:
        table.add_row(
            status_glyph(r.status),
            r.id,
            r.modality,
            r.mem_str,
            r.description,
        )
    console.print(table)


def _render_plain_table(rows: list[_ListRow]) -> None:
    """Plain text format for piped / redirected / non-TTY output.

    No ANSI escapes, no truncation. Each row is one line:
        <glyph> <status_padded>  <id>  <modality>  <memory>  <description>

    The status word appears verbatim per row, so `grep enabled` works
    on real rows and the legend doesn't pollute negative assertions.
    """
    from muse.cli_impl.console import STATUS_STYLE

    # Compute column widths from the longest content so columns line
    # up without truncation.
    id_w = max((len(r.id) for r in rows), default=0)
    mod_w = max((len(r.modality) for r in rows), default=0)
    mem_w = max((len(r.mem_str) for r in rows), default=0)

    for r in rows:
        glyph, _ = STATUS_STYLE.get(r.status, ("?", ""))
        line = (
            f"  {glyph} {r.status:11s}  "
            f"{r.id:<{id_w}s}  "
            f"{r.modality:<{mod_w}s}  "
            f"{r.mem_str:>{mem_w}s}  "
            f"{r.description}"
        )
        print(line)


def _render_footer(rows: list[_ListRow], *, no_color: bool) -> None:
    """Memory totals + legend. Same in both TTY and plain modes; rich
    gracefully degrades to plain text when stdout isn't a TTY."""
    from rich.text import Text

    from muse.cli_impl.console import get_console, status_legend

    console = get_console(force_no_color=no_color)
    gpu_total = sum(
        r.mem_gb for r in rows
        if r.status == "enabled" and r.mem_gb is not None
        and r.mem_device == "GPU"
    )
    cpu_total = sum(
        r.mem_gb for r in rows
        if r.status == "enabled" and r.mem_gb is not None
        and r.mem_device == "CPU"
    )
    n_enabled = sum(1 for r in rows if r.status == "enabled")

    summary = Text()
    summary.append("\nEnabled: ", style="bold")
    summary.append(f"{gpu_total:.1f} GB GPU + {cpu_total:.1f} GB CPU ")
    summary.append(f"({n_enabled} models)", style="dim")
    console.print(summary)
    console.print(status_legend())
    console.print(
        Text(
            "Measured values (from `muse models probe`) shown without prefix; "
            "annotated estimates (peak inference) shown with ~ prefix.",
            style="dim",
        )
    )


def _model_memory_display(extra: dict, catalog_entry: dict | None):
    """Return (display_str, gb_for_aggregate, device_label).

    Resolution order, in decreasing fidelity:
      1. measured peak from `muse models probe` (per-device measurement)
      2. annotated `capabilities.memory_gb`
      3. None (display "-")

    The device label ("CPU" or "GPU") is derived from
    `capabilities.device`: cpu -> CPU; everything else
    (cuda/auto/mps/unset) -> GPU.
    """
    extra = extra or {}
    cap_device = (extra.get("device") or "auto").lower()
    if cap_device == "cpu":
        device_label = "CPU"
        measurement_keys = ("cpu",)
    else:
        device_label = "GPU"
        measurement_keys = ("cuda", "auto")

    measurements = (catalog_entry or {}).get("measurements") or {}
    for key in measurement_keys:
        m = measurements.get(key)
        if not m:
            continue
        peak = m.get("peak_bytes") or 0
        if peak > 0:
            gb = peak / (1024**3)
            return f"{gb:.1f} GB {device_label}", gb, device_label

    annotation = extra.get("memory_gb")
    if annotation is not None:
        try:
            gb = float(annotation)
        except (TypeError, ValueError):
            return "-", None, device_label
        return f"~{gb:.1f} GB {device_label}", gb, device_label

    return "-", None, device_label
