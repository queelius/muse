#!/usr/bin/env python3
# scripts/bench/image_compare.py
"""Side-by-side image comparison: muse models vs OpenAI gpt-image-2.

Runs each case through every target and writes a contact sheet:

  python -m scripts.bench.image_compare --cases cases.yaml \
      --model flux2-klein-4b --openai gpt-image-2 --out /tmp/compare

Cases come from `--cases FILE` (YAML or JSON list of
{id, prompt, avoid?, refs?, size?, quality?}) or, best effort, from a
youtube-channel visual bible via `--from-visuals visuals.yaml` (each
character/location/prop with a description becomes a case; its reference
sheets become the case refs). Cases with refs go to /v1/images/edits as
repeated `image[]` parts; the rest go to /v1/images/generations.

OPENAI_API_KEY is read from the environment only and never written out.
Output: <out>/<target>/<case>.png, <out>/results.json, <out>/index.html.
Standalone: no muse imports.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import mimetypes
import os
import re
import struct
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import yaml

OPENAI_BASE = "https://api.openai.com/v1"
TIMEOUT = httpx.Timeout(900.0, connect=10.0)


@dataclass
class Case:
    id: str
    prompt: str
    avoid: str | None = None
    refs: list[str] = field(default_factory=list)
    size: str = "1536x1024"
    quality: str | None = None

    def full_prompt(self) -> str:
        if self.avoid:
            return f"{self.prompt} Do not include: {self.avoid}."
        return self.prompt


@dataclass
class Target:
    name: str
    model: str
    base_url: str
    openai: bool = False


def _case_from_mapping(raw: dict) -> Case:
    return Case(
        id=str(raw["id"]),
        prompt=str(raw["prompt"]),
        avoid=(str(raw["avoid"]) if raw.get("avoid") else None),
        refs=[str(r) for r in (raw.get("refs") or [])],
        size=str(raw.get("size") or "1536x1024"),
        quality=(str(raw["quality"]) if raw.get("quality") else None),
    )


def load_cases(path) -> list[Case]:
    """Load cases from a YAML or JSON file (JSON is valid YAML)."""
    data = yaml.safe_load(Path(path).read_text())
    if isinstance(data, dict):
        data = data.get("cases")
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a list of cases")
    return [_case_from_mapping(item) for item in data]


def cases_from_visuals(path) -> list[Case]:
    """Best-effort cases from a visual bible (characters/locations/props).

    Each entry with a description becomes a case: description plus the
    bible's style prompt, style.avoid as `avoid`, and its `refs` image
    paths resolved relative to the visuals file. Entries that are not
    mappings or have no description are skipped (count printed).
    """
    path = Path(path)
    data = yaml.safe_load(path.read_text()) or {}
    if isinstance(data.get("visuals"), dict):
        data = data["visuals"]
    style = data.get("style") if isinstance(data.get("style"), dict) else {}
    style_prompt = style.get("prompt")
    avoid = str(style["avoid"]) if style.get("avoid") else None
    image_model = data.get("image_model") if isinstance(data.get("image_model"), dict) else {}
    cases: list[Case] = []
    skipped = 0
    for kind in ("characters", "locations", "props"):
        group = data.get(kind) or {}
        if not isinstance(group, dict):
            continue
        for name, entry in group.items():
            if not isinstance(entry, dict) or not entry.get("description"):
                skipped += 1
                continue
            prompt = ", ".join(
                p for p in (str(entry["description"]).strip().rstrip(","),
                            str(style_prompt) if style_prompt else None) if p)
            refs = []
            for raw in entry.get("refs") or []:
                p = Path(str(raw))
                refs.append(str(p if p.is_absolute() else path.parent / p))
            case = Case(id=f"{kind[:-1]}-{name}", prompt=prompt, avoid=avoid, refs=refs)
            if image_model.get("size"):
                case.size = str(image_model["size"])
            if image_model.get("quality"):
                case.quality = str(image_model["quality"])
            cases.append(case)
    print(f"from-visuals: {len(cases)} cases, {skipped} entries skipped", file=sys.stderr)
    return cases


def _safe_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]", "_", value).lstrip(".")
    return name or "_"


class _Names:
    """Sanitised names, unique per scope: a different original that maps to
    an already-used name gets a stable sha1 suffix; the same original always
    maps to the same name."""

    def __init__(self):
        self._used: dict[str, dict[str, str]] = {}

    def get(self, scope: str, original: str) -> str:
        used = self._used.setdefault(scope, {})
        name = _safe_name(original)
        if used.get(name, original) != original:
            name = f"{name}-{hashlib.sha1(original.encode()).hexdigest()[:8]}"
        used[name] = original
        return name


def _png_size(data: bytes) -> str | None:
    if data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) >= 24:
        w, h = struct.unpack(">II", data[16:24])
        return f"{w}x{h}"
    return None


def _request(client: httpx.Client, target: Target, case: Case, api_key: str | None):
    headers = {}
    if target.openai:
        headers["Authorization"] = f"Bearer {api_key}"
    base = target.base_url.rstrip("/")
    fields = {"model": target.model, "prompt": case.full_prompt(),
              "size": case.size, "n": "1"}
    if case.quality:
        fields["quality"] = case.quality
    if not target.openai:
        fields["response_format"] = "b64_json"
    if case.refs:
        files = []
        for ref in case.refs:
            ref_path = Path(ref)
            mime = mimetypes.guess_type(ref_path.name)[0] or "image/png"
            files.append(("image[]", (ref_path.name, ref_path.read_bytes(), mime)))
        return client.post(f"{base}/images/edits", data=fields, files=files,
                           headers=headers, timeout=TIMEOUT)
    return client.post(f"{base}/images/generations", json={**fields, "n": 1},
                       headers=headers, timeout=TIMEOUT)


def run(cases: list[Case], targets: list[Target], out_dir, client: httpx.Client) -> list[dict]:
    """One cell per (case, target). Failures are recorded, never raised.

    OpenAI targets are skipped (with a notice) when OPENAI_API_KEY is unset.
    """
    out_dir = Path(out_dir)
    api_key = os.environ.get("OPENAI_API_KEY") or None
    results: list[dict] = []
    names = _Names()
    for target in targets:
        if target.openai and not api_key:
            print(f"skipping {target.name}: OPENAI_API_KEY is not set", file=sys.stderr)
            continue
        for case in cases:
            cell = {"case": case.id, "target": target.name, "path": None,
                    "seconds": None, "size": None, "revision": None,
                    "request_id": None, "error": None}
            start = time.perf_counter()
            try:
                resp = _request(client, target, case, api_key)
                cell["seconds"] = round(time.perf_counter() - start, 2)
                cell["revision"] = resp.headers.get("x-muse-model-revision")
                cell["request_id"] = resp.headers.get("x-request-id")
                if resp.status_code != 200:
                    cell["error"] = f"HTTP {resp.status_code}: {resp.text[:200]}"
                else:
                    body = resp.json()
                    png = base64.b64decode(body["data"][0]["b64_json"])
                    tdir = names.get("", target.name)
                    rel = f"{tdir}/{names.get(tdir, case.id)}.png"
                    dest = (out_dir / rel).resolve()
                    if not dest.is_relative_to(out_dir.resolve()):
                        raise ValueError(f"refusing to write outside --out: {rel}")
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(png)
                    cell["path"] = rel
                    cell["size"] = _png_size(png) or body.get("size")
            except Exception as exc:  # one bad cell must not stop the run
                if cell["seconds"] is None:
                    cell["seconds"] = round(time.perf_counter() - start, 2)
                cell["error"] = f"{type(exc).__name__}: {exc}"[:200]
            results.append(cell)
            print(f"{target.name} {case.id}: "
                  f"{cell['error'] or str(cell['seconds']) + 's'}", file=sys.stderr)
    return results


_CSS = """body{font-family:system-ui,sans-serif;margin:1rem;background:#fafafa;color:#222}
table{border-collapse:collapse}th,td{border:1px solid #ccc;padding:.4rem;vertical-align:top}
img{max-width:360px;display:block}.cap{font-size:.8rem;color:#555}.err{color:#b00020}"""


def write_index(results: list[dict], out_dir) -> Path:
    out_dir = Path(out_dir)
    targets = list(dict.fromkeys(r["target"] for r in results))
    cases = list(dict.fromkeys(r["case"] for r in results))
    by_key = {(r["case"], r["target"]): r for r in results}
    rows = ["<tr><th>case</th>" + "".join(f"<th>{html.escape(t)}</th>" for t in targets) + "</tr>"]
    for case in cases:
        cells = [f"<th>{html.escape(case)}</th>"]
        for target in targets:
            r = by_key.get((case, target))
            if r is None:
                cells.append("<td></td>")
            elif r["error"]:
                cells.append(f"<td class='err'>{html.escape(r['error'])}"
                             f"<div class='cap'>{r['seconds']}s</div></td>")
            else:
                src = html.escape(r["path"], quote=True)
                cells.append(f"<td><img src=\"{src}\" alt=\"{html.escape(case)}\">"
                             f"<div class='cap'>{r['seconds']}s, {html.escape(str(r['size']))}"
                             "</div></td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    page = ("<!doctype html><html><head><meta charset='utf-8'><title>image compare</title>"
            f"<style>{_CSS}</style></head><body><table>{''.join(rows)}</table></body></html>")
    dest = out_dir / "index.html"
    dest.write_text(page)
    return dest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--cases")
    src.add_argument("--from-visuals")
    ap.add_argument("--muse", default=os.environ.get("MUSE_SERVER", "http://localhost:8000"))
    ap.add_argument("--model", action="append", default=[])
    ap.add_argument("--openai", metavar="MODEL")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args(argv)

    cases = load_cases(args.cases) if args.cases else cases_from_visuals(args.from_visuals)
    if args.limit:
        cases = cases[: args.limit]
    muse_base = args.muse.rstrip("/") + "/v1"
    targets = [Target(m, m, muse_base) for m in args.model]
    if args.openai:
        targets.append(Target(f"openai-{args.openai}", args.openai, OPENAI_BASE, openai=True))
    if not targets:
        ap.error("give at least one --model or --openai")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with httpx.Client() as client:
        results = run(cases, targets, out, client)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(write_index(results, out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
