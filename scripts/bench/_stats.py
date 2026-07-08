"""Pure helpers for the benchmark harness. Stdlib only; unit-tested."""
from __future__ import annotations

import json
import pathlib


def median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def tok_per_s(tokens: int, seconds: float) -> float:
    if seconds <= 0 or tokens <= 0:
        return 0.0
    return tokens / seconds


def render_md_table(headers: list, rows: list) -> str:
    out = ["| " + " | ".join(str(h) for h in headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out) + "\n"


def write_reports(results: dict, *, md_path: str | None,
                  json_path: str | None, title: str) -> None:
    """results: {scenario: {"headers": [...], "rows": [...], "raw": {...}}}"""
    if json_path:
        p = pathlib.Path(json_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(results, indent=2, default=str))
    if md_path:
        parts = [f"# {title}\n"]
        for name, sec in results.items():
            parts.append(f"## {name}\n")
            if "error" in sec:
                parts.append(f"ERROR: {sec['error']}\n")
                continue
            parts.append(render_md_table(sec["headers"], sec["rows"]))
        p = pathlib.Path(md_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(parts))
