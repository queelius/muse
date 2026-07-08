"""Unit tests for the benchmark harness pure helpers (spec 2026-07-08)."""
from __future__ import annotations

import json

from scripts.bench._stats import median, render_md_table, tok_per_s, write_reports


def test_median_odd_even_single():
    assert median([3.0, 1.0, 2.0]) == 2.0
    assert median([1.0, 2.0, 3.0, 4.0]) == 2.5
    assert median([7.0]) == 7.0


def test_tok_per_s():
    assert tok_per_s(64, 32.0) == 2.0
    assert tok_per_s(64, 0.0) == 0.0
    assert tok_per_s(0, 5.0) == 0.0


def test_render_md_table():
    md = render_md_table(["a", "b"], [["x", 1.5], ["y", "err"]])
    lines = md.strip().splitlines()
    assert lines[0] == "| a | b |"
    assert lines[1] == "|---|---|"
    assert "| x | 1.5 |" in md and "| y | err |" in md


def test_write_reports(tmp_path):
    results = {"scenario_a": {"headers": ["k", "v"], "rows": [["m", 1]],
                              "raw": {"anything": True}}}
    jp, mp = tmp_path / "r.json", tmp_path / "r.md"
    write_reports(results, md_path=str(mp), json_path=str(jp), title="T")
    assert json.loads(jp.read_text())["scenario_a"]["raw"] == {"anything": True}
    md = mp.read_text()
    assert md.startswith("# T") and "## scenario_a" in md and "| k | v |" in md
