"""`muse models list` row builder + table renderer.

The list command pulls from three sources (bundled scripts, curated
recommendations, pulled-via-resolver entries) and displays them as a
unified, status-tagged table.

The row-building logic is its own function so it can be unit-tested
without subprocess overhead. The table-rendering happens via
rich.Table for color + glyph + truncation handling; --json bypasses
rich and dumps the same row data as a JSON list.

v0.40.0 introduced a five-state status enum: `enabled_loaded` (running
on a worker), `enabled_unloaded` (catalog-enabled but not currently in
the director's loaded set), `disabled`, `recommended`, `available`. The
loaded-vs-unloaded split is computed by consulting the running
supervisor's LoadDirector via the admin API. When the supervisor is
unreachable from the CLI (admin token unset, or no `muse serve`
running), every catalog-enabled row falls back to `enabled_unloaded`
because the CLI cannot observe runtime state.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# Status enum values (kept here as constants so a typo in any one place
# fails type-check / lint rather than silently routing to "available").
STATUS_ENABLED_LOADED = "enabled_loaded"
STATUS_ENABLED_UNLOADED = "enabled_unloaded"
STATUS_DISABLED = "disabled"
STATUS_RECOMMENDED = "recommended"
STATUS_AVAILABLE = "available"

# Statuses that count as "in the catalog" for --installed filtering.
_INSTALLED_STATUSES = (
    STATUS_ENABLED_LOADED,
    STATUS_ENABLED_UNLOADED,
    STATUS_DISABLED,
)
# Statuses that count as "could install" for --available filtering.
_AVAILABLE_STATUSES = (STATUS_RECOMMENDED, STATUS_AVAILABLE)


@dataclass
class _ListRow:
    """One row for `muse models list`. Ordered so JSON output is stable."""
    id: str
    modality: str
    # one of: enabled_loaded, enabled_unloaded, disabled, recommended, available
    status: str
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


# Indirection seams: tests monkeypatch these symbols on the module to
# stub catalog / curated / director access without forcing each test
# to construct a full registry.

def _load_known_models():
    """Return {model_id: CatalogEntry} from bundled + resolver-pulled scripts."""
    from muse.core.catalog import list_known
    return {e.model_id: e for e in list_known(None)}


def _load_curated_entries():
    """Return {curated_id: CuratedEntry}."""
    from muse.core.curated import load_curated
    return {c.id: c for c in load_curated()}


def _read_catalog_data():
    """Return the raw catalog dict (for measurements lookup)."""
    from muse.core.catalog import _read_catalog
    return _read_catalog()


def _is_pulled_for_row(model_id: str) -> bool:
    from muse.core.catalog import is_pulled
    return is_pulled(model_id)


def _is_enabled_for_row(model_id: str) -> bool:
    from muse.core.catalog import is_enabled
    return is_enabled(model_id)


def _director_loaded_ids() -> set[str]:
    """Set of model ids the running supervisor's director reports as loaded.

    The CLI may be invoked anywhere (no running `muse serve`, no admin
    token, behind a firewall, etc.). When the director cannot be
    reached, return an empty set so callers default-classify every
    catalog-enabled row as `enabled_unloaded`. This is correct from the
    CLI's vantage point: the CLI cannot observe runtime state without
    the admin API.

    Implementation: best-effort GET /v1/admin/memory via AdminClient,
    then read the `loaded` shape if present. Falls back to inspecting
    /v1/admin/workers (for the case where the memory route doesn't
    surface director state). On any failure -> empty set.
    """
    if not os.environ.get("MUSE_ADMIN_TOKEN"):
        return set()
    try:
        from muse.admin.client import AdminClient, AdminClientError
    except Exception:  # noqa: BLE001
        return set()
    client = AdminClient(timeout=2.0)
    # The admin API does not currently expose a single endpoint that
    # lists every model the director considers loaded. We probe each
    # candidate id via /v1/admin/models/{id}/status. To keep the CLI
    # latency bounded, we only consult the catalog-enabled subset and
    # cap the per-call timeout.
    try:
        from muse.core.catalog import _read_catalog
        catalog = _read_catalog()
    except Exception:  # noqa: BLE001
        return set()

    loaded: set[str] = set()
    candidate_ids = [
        mid for mid, e in catalog.items()
        if isinstance(e, dict) and e.get("enabled", True)
    ]
    for mid in candidate_ids:
        try:
            info = client.status(mid)
        except AdminClientError:
            continue
        except Exception:  # noqa: BLE001
            return loaded
        # The status endpoint returns {"loaded": bool, "worker_port", ...}
        # under "runtime" or top-level depending on the route impl. We
        # read both for safety.
        if info.get("loaded") is True:
            loaded.add(mid)
            continue
        runtime = info.get("runtime") or {}
        if runtime.get("loaded") is True:
            loaded.add(mid)
    return loaded


def _classify_pulled(
    model_id: str,
    *,
    director_loaded_ids: set[str],
) -> str:
    """Return the status enum for a pulled catalog row.

    enabled + in director's loaded set -> enabled_loaded
    enabled + not in director's loaded set -> enabled_unloaded
    not enabled -> disabled
    """
    if not _is_enabled_for_row(model_id):
        return STATUS_DISABLED
    if model_id in director_loaded_ids:
        return STATUS_ENABLED_LOADED
    return STATUS_ENABLED_UNLOADED


def build_rows() -> list[_ListRow]:
    """Build the unified row set for `muse models list`.

    Status precedence: pulled (catalog wins enabled_loaded/enabled_unloaded
    /disabled) > curated-but-not-pulled (recommended) > bundled-but-not-
    pulled (available).
    """
    bundled_entries = _load_known_models()
    curated_entries = _load_curated_entries()
    catalog_data = _read_catalog_data()
    director_loaded = _director_loaded_ids()

    rows: list[_ListRow] = []
    seen: set[str] = set()

    # 1. Bundled scripts and resolver-pulled entries.
    for model_id, e in bundled_entries.items():
        seen.add(model_id)
        if _is_pulled_for_row(model_id):
            status = _classify_pulled(
                model_id, director_loaded_ids=director_loaded,
            )
        elif model_id in curated_entries:
            status = STATUS_RECOMMENDED
        else:
            status = STATUS_AVAILABLE
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
        if _is_pulled_for_row(cid):
            status = _classify_pulled(
                cid, director_loaded_ids=director_loaded,
            )
        else:
            status = STATUS_RECOMMENDED
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
        out = [r for r in out if r.status in _INSTALLED_STATUSES]
    if available:
        out = [r for r in out if r.status in _AVAILABLE_STATUSES]
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
    # up without truncation. The status width pads to whichever known
    # status name is longest (currently `enabled_unloaded` at 16 chars)
    # so future renamings don't silently overflow the column.
    id_w = max((len(r.id) for r in rows), default=0)
    mod_w = max((len(r.modality) for r in rows), default=0)
    mem_w = max((len(r.mem_str) for r in rows), default=0)
    status_w = max(len(s) for s in STATUS_STYLE) if STATUS_STYLE else 11

    for r in rows:
        glyph, _ = STATUS_STYLE.get(r.status, ("?", ""))
        line = (
            f"  {glyph} {r.status:<{status_w}s}  "
            f"{r.id:<{id_w}s}  "
            f"{r.modality:<{mod_w}s}  "
            f"{r.mem_str:>{mem_w}s}  "
            f"{r.description}"
        )
        print(line)


def _render_footer(rows: list[_ListRow], *, no_color: bool) -> None:
    """Memory totals + legend. Same in both TTY and plain modes; rich
    gracefully degrades to plain text when stdout isn't a TTY.

    The "Enabled" tally counts every catalog-enabled row regardless of
    whether the director currently has it loaded. Under lazy load the
    enabled set is the operator's declared intent; runtime presence
    fluctuates with traffic, so the static-state tally is the more
    actionable footer for capacity planning.
    """
    from rich.text import Text

    from muse.cli_impl.console import get_console, status_legend

    console = get_console(force_no_color=no_color)
    enabled_statuses = (STATUS_ENABLED_LOADED, STATUS_ENABLED_UNLOADED)
    gpu_total = sum(
        r.mem_gb for r in rows
        if r.status in enabled_statuses and r.mem_gb is not None
        and r.mem_device == "GPU"
    )
    cpu_total = sum(
        r.mem_gb for r in rows
        if r.status in enabled_statuses and r.mem_gb is not None
        and r.mem_device == "CPU"
    )
    n_enabled = sum(1 for r in rows if r.status in enabled_statuses)
    n_loaded = sum(1 for r in rows if r.status == STATUS_ENABLED_LOADED)

    summary = Text()
    summary.append("\nEnabled: ", style="bold")
    summary.append(f"{gpu_total:.1f} GB GPU + {cpu_total:.1f} GB CPU ")
    summary.append(f"({n_enabled} models, {n_loaded} loaded)", style="dim")
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
