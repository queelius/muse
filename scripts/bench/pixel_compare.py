#!/usr/bin/env python3
# scripts/bench/pixel_compare.py
"""Contact sheet for muse pixel-art assets: generate, pixelize, autotile.

  python scripts/bench/pixel_compare.py --server http://localhost:8000 \
      --cases scripts/bench/pixel_cases.yaml --out /tmp/pixel

Case file: YAML or JSON list of cases, each with `id` and `kind`:

  sprite:   {id, kind: sprite, prompt, model, size?, seed?, steps?,
             pixelize: {form fields} | [{form fields}, ...], tool_model?}
            The prompt goes to /v1/images/generations, then each pixelize
            variant to /v1/images/pixelize (one results row per variant).
  autotile: {id, kind: autotile, fill_prompt, base_prompt?, model, size?,
             seed?, steps?, autotile: {form fields}, tool_model?}
            Generates the fill (and base) texture, posts them to
            /v1/images/autotile, and paints a random 16x12 map from the
            blob-47 tiles by neighbour mask so seams are visible.

`tool_model` is the pixel-art tool model (default `pixel-tools`); `model` is
the image generation model. Output: <out>/<case>/*.png, <out>/results.json,
<out>/index.html. Talks HTTP only; Pillow is needed for the map painting.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import io
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import httpx
import yaml

TIMEOUT = httpx.Timeout(900.0, connect=10.0)
KINDS = ("sprite", "autotile")
MAP_COLS, MAP_ROWS = 16, 12
# 8-neighbour bitmask, clockwise from north (matches the server's layout masks).
_N, _NE, _E, _SE, _S, _SW, _W, _NW = 1, 2, 4, 8, 16, 32, 64, 128


def _safe_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]", "_", str(value)).lstrip(".")
    return name or "_"


def load_cases(path) -> list[dict]:
    """Load and validate cases; a bad file raises SystemExit with a message."""
    data = yaml.safe_load(Path(path).read_text())
    if isinstance(data, dict):
        data = data.get("cases")
    if not isinstance(data, list) or not data:
        raise SystemExit(f"{path}: expected a non-empty list of cases")
    seen: set[str] = set()
    for i, case in enumerate(data):
        where = f"{path}: case #{i + 1}"
        if not isinstance(case, dict) or not case.get("id"):
            raise SystemExit(f"{where}: needs an id")
        where = f"{path}: case {case['id']!r}"
        kind = case.get("kind")
        if kind not in KINDS:
            raise SystemExit(f"{where}: unknown kind {kind!r} (use {' or '.join(KINDS)})")
        if str(case["id"]) in seen:
            raise SystemExit(f"{where}: duplicate id")
        seen.add(str(case["id"]))
        if not case.get("model"):
            raise SystemExit(f"{where}: needs a model (the image generation model)")
        need = "prompt" if kind == "sprite" else "fill_prompt"
        if not case.get(need):
            raise SystemExit(f"{where}: needs {need}")
        key = "pixelize" if kind == "sprite" else "autotile"
        spec = case.get(key)
        if kind == "sprite" and spec is None:
            case[key] = [{}]
        elif spec is None:
            case[key] = {}
        elif isinstance(spec, dict) and kind == "sprite":
            case[key] = [spec]
        if kind == "sprite":
            if not all(isinstance(v, dict) for v in case[key]) or not case[key]:
                raise SystemExit(f"{where}: pixelize must be a mapping or a list of mappings")
        elif not isinstance(case[key], dict):
            raise SystemExit(f"{where}: autotile must be a mapping of form fields")
    return data


def _form(fields: dict) -> dict[str, str]:
    return {str(k): (str(v).lower() if isinstance(v, bool) else str(v))
            for k, v in fields.items() if v is not None}


def _label(fields: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in fields.items()) or "defaults"


def _generate(client, base, case, prompt, rid_seed=None) -> tuple[bytes, float]:
    body = {"model": case["model"], "prompt": prompt, "n": 1,
            "size": str(case.get("size") or "512x512"), "response_format": "b64_json"}
    for key in ("seed", "steps"):
        if case.get(key) is not None:
            body[key] = case[key]
    if rid_seed is not None:
        body["seed"] = rid_seed
    start = time.perf_counter()
    resp = client.post(f"{base}/images/generations", json=body, timeout=TIMEOUT)
    took = round(time.perf_counter() - start, 2)
    if resp.status_code != 200:
        raise RuntimeError(f"generate HTTP {resp.status_code}: {resp.text[:200]}")
    return base64.b64decode(resp.json()["data"][0]["b64_json"]), took


class _Out:
    def __init__(self, root: Path):
        self.root = root.resolve()

    def write(self, case_id: str, name: str, data: bytes | str) -> str:
        rel = f"{_safe_name(case_id)}/{_safe_name(name)}"
        dest = (self.root / rel).resolve()
        if not dest.is_relative_to(self.root):
            raise ValueError(f"refusing to write outside --out: {rel}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data if isinstance(data, bytes) else data.encode())
        return rel


def _png_size(data: bytes):
    import struct
    if data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) >= 24:
        return list(struct.unpack(">II", data[16:24]))
    return None


def _base_tile(base_png: bytes, t: int):
    """The base texture fitted to one T x T tile the way the server fits it:
    the largest centred square that is a whole number of tiles, box-downscaled
    (a texture smaller than T is wrapped). The server's seamless pass is not
    repeated, so the last row/column may differ slightly."""
    from PIL import Image

    img = Image.open(io.BytesIO(base_png)).convert("RGBA")
    side = min(img.size)
    if side >= t:
        crop = (side // t) * t
        x0, y0 = (img.width - crop) // 2, (img.height - crop) // 2
        return img.crop((x0, y0, x0 + crop, y0 + crop)).resize((t, t), Image.Resampling.BOX)
    tile = Image.new("RGBA", (t, t))
    for dx in range(0, t, img.width):
        for dy in range(0, t, img.height):
            tile.paste(img, (dx, dy))
    return tile


def paint_map(blob_png: bytes, layout: list[dict], tile_size: int, seed: int = 0,
              cols: int = MAP_COLS, rows: int = MAP_ROWS,
              base_png: bytes | None = None) -> bytes:
    """Random blobby fill map painted from the blob-47 tiles by neighbour mask.

    Outside the map counts as filled. Unfilled cells show ``base_png`` fitted
    to one tile, or stay transparent without one. (No blob tile is pure base:
    even the isolated tile's corner quadrant holds part of the fill and its
    outline, so cropping it painted that corner into every unfilled cell.)
    """
    from PIL import Image

    t = int(tile_size)
    sheet = Image.open(io.BytesIO(blob_png)).convert("RGBA")
    pos = {e["mask"]: (e["x"] * t, e["y"] * t) for e in layout}
    rng = random.Random(seed)
    grid = [[rng.random() < 0.55 for _ in range(cols)] for _ in range(rows)]
    for _ in range(2):  # smooth into blobs
        grid = [[sum(grid[j][i] for j in range(max(0, r - 1), min(rows, r + 2))
                     for i in range(max(0, c - 1), min(cols, c + 2))) >= 4
                 for c in range(cols)] for r in range(rows)]

    def filled(c, r):
        return grid[r][c] if 0 <= c < cols and 0 <= r < rows else True

    def mask_at(c, r):
        m = 0
        for bit, dc, dr in ((_N, 0, -1), (_NE, 1, -1), (_E, 1, 0), (_SE, 1, 1),
                            (_S, 0, 1), (_SW, -1, 1), (_W, -1, 0), (_NW, -1, -1)):
            if filled(c + dc, r + dr):
                m |= bit
        for corner, a, b in ((_NE, _N, _E), (_SE, _S, _E), (_SW, _S, _W), (_NW, _N, _W)):
            if not (m & a and m & b):
                m &= ~corner
        return m

    out = Image.new("RGBA", (cols * t, rows * t))
    base = _base_tile(base_png, t) if base_png else None
    for r in range(rows):
        for c in range(cols):
            if grid[r][c]:
                x, y = pos.get(mask_at(c, r), pos.get(255, (0, 0)))
                out.paste(sheet.crop((x, y, x + t, y + t)), (c * t, r * t))
            elif base is not None:
                out.paste(base, (c * t, r * t))
    buf = io.BytesIO()
    out.save(buf, "PNG")
    return buf.getvalue()


def _run_sprite(client, base, case, out: _Out) -> list[dict]:
    rows: list[dict] = []
    tool = case.get("tool_model") or "pixel-tools"
    try:
        raw, gen_s = _generate(client, base, case, case["prompt"])
    except Exception as exc:
        return [{"case": case["id"], "kind": "sprite", "variant": None, "error":
                 f"{type(exc).__name__}: {exc}"[:200], "raw": None, "pixelized": None,
                 "seconds_generate": None, "seconds_pixelize": None}]
    raw_rel = out.write(case["id"], "raw.png", raw)
    for i, fields in enumerate(case["pixelize"]):
        row = {"case": case["id"], "kind": "sprite", "variant": _label(fields),
               "raw": raw_rel, "pixelized": None, "grid": None, "palette": None,
               "size": None, "seconds_generate": gen_s, "seconds_pixelize": None,
               "error": None}
        start = time.perf_counter()
        try:
            data = {"model": tool, "response_format": "json", **_form(fields)}
            resp = client.post(f"{base}/images/pixelize", data=data,
                               files={"image": ("raw.png", raw, "image/png")},
                               timeout=TIMEOUT)
            row["seconds_pixelize"] = round(time.perf_counter() - start, 2)
            if resp.status_code != 200:
                row["error"] = f"pixelize HTTP {resp.status_code}: {resp.text[:200]}"
            else:
                body = resp.json()
                png = base64.b64decode(body["data"][0]["b64_json"])
                row["pixelized"] = out.write(case["id"], f"v{i}.png", png)
                row["grid"] = body.get("grid")
                row["palette"] = body.get("palette")
                row["size"] = body.get("size") or _png_size(png)
        except Exception as exc:
            if row["seconds_pixelize"] is None:
                row["seconds_pixelize"] = round(time.perf_counter() - start, 2)
            row["error"] = f"{type(exc).__name__}: {exc}"[:200]
        rows.append(row)
    return rows


def _run_autotile(client, base, case, out: _Out) -> list[dict]:
    row = {"case": case["id"], "kind": "autotile", "variant": _label(case["autotile"]),
           "fill": None, "base": None, "a2": None, "blob47": None, "map": None,
           "tile_size": None, "seconds_generate": None, "seconds_autotile": None,
           "error": None}
    tool = case.get("tool_model") or "pixel-tools"
    try:
        fill, gen_s = _generate(client, base, case, case["fill_prompt"])
        row["fill"] = out.write(case["id"], "fill.png", fill)
        files = {"fill": ("fill.png", fill, "image/png")}
        base_png = None
        if case.get("base_prompt"):
            base_png, base_s = _generate(client, base, case, case["base_prompt"])
            gen_s = round(gen_s + base_s, 2)
            row["base"] = out.write(case["id"], "base.png", base_png)
            files["base"] = ("base.png", base_png, "image/png")
        row["seconds_generate"] = gen_s
        data = {"model": tool, "response_format": "json",
                "name": _safe_name(case["id"]).lower(), **_form(case["autotile"])}
        start = time.perf_counter()
        resp = client.post(f"{base}/images/autotile", data=data, files=files,
                           timeout=TIMEOUT)
        row["seconds_autotile"] = round(time.perf_counter() - start, 2)
        if resp.status_code != 200:
            row["error"] = f"autotile HTTP {resp.status_code}: {resp.text[:200]}"
            return [row]
        body = resp.json()
        a2 = base64.b64decode(body["files"]["a2"]["b64_json"])
        blob = base64.b64decode(body["files"]["blob47"]["b64_json"])
        row["a2"] = out.write(case["id"], "a2.png", a2)
        row["blob47"] = out.write(case["id"], "blob47.png", blob)
        out.write(case["id"], "tileset.tres", body["files"].get("godot_tres") or "")
        out.write(case["id"], "tileset.tsx", body["files"].get("tiled_tsx") or "")
        row["tile_size"] = body.get("tile_size")
        row["map"] = out.write(case["id"], "map.png", paint_map(
            blob, body["layout"], int(body["tile_size"]),
            seed=int(hashlib.sha1(str(case["id"]).encode()).hexdigest()[:8], 16),
            base_png=base_png))
    except Exception as exc:  # one bad case must not stop the run
        row["error"] = f"{type(exc).__name__}: {exc}"[:200]
    return [row]


def run(cases: list[dict], server: str, out_dir, client: httpx.Client) -> list[dict]:
    """Rows: one per sprite pixelize variant, one per autotile case. Failures
    are recorded in `error`, never raised."""
    base = server.rstrip("/") + "/v1"
    out = _Out(Path(out_dir))
    results: list[dict] = []
    for case in cases:
        fn = _run_sprite if case["kind"] == "sprite" else _run_autotile
        rows = fn(client, base, case, out)
        for r in rows:
            print(f"{r['case']} [{r['variant']}]: {r['error'] or 'ok'}", file=sys.stderr)
        results.extend(rows)
    return results


_CSS = """body{font-family:system-ui,sans-serif;margin:1rem;background:#fafafa;color:#222}
table{border-collapse:collapse;margin-bottom:1.5rem}th,td{border:1px solid #ccc;padding:.4rem;vertical-align:top}
img{display:block;image-rendering:pixelated}.cap{font-size:.8rem;color:#555;max-width:260px}
.err{color:#b00020}.sw{display:inline-block;width:10px;height:10px;margin:1px}"""


def _img(rel, w=None, alt=""):
    if not rel:
        return "<em>none</em>"
    style = f" style='width:{w}px'" if w else ""
    return f"<img src=\"{html.escape(rel, quote=True)}\" alt=\"{html.escape(alt)}\"{style}>"


def _swatches(palette):
    return "".join(f"<span class='sw' style='background:{html.escape(c)}'></span>"
                   for c in (palette or [])[:64] if isinstance(c, str))


def write_index(results: list[dict], out_dir) -> Path:
    out_dir = Path(out_dir)
    sprites = [r for r in results if r["kind"] == "sprite"]
    tiles = [r for r in results if r["kind"] == "autotile"]
    parts = []
    if sprites:
        rows = ["<tr><th>case</th><th>variant</th><th>raw</th><th>1x</th>"
                "<th>nearest 4x</th><th>grid / palette</th><th>timings</th></tr>"]
        for r in sprites:
            err = f"<div class='err'>{html.escape(r['error'])}</div>" if r["error"] else ""
            g = r.get("grid") or {}
            size = r.get("size")
            info = ""
            if g:
                info += (f"cell {g.get('cell')} conf {g.get('confidence', 0):.2f}"
                         f"{'' if g.get('detected') else ' (not detected)'}<br>")
            if size:
                info += f"{size[0]}x{size[1]}, {len(r.get('palette') or [])} colours<br>"
            info += _swatches(r.get("palette"))
            w = (size[0] * 4) if size else None
            rows.append(
                f"<tr><th>{html.escape(r['case'])}</th>"
                f"<td>{html.escape(str(r['variant']))}{err}</td>"
                f"<td>{_img(r['raw'], 192, r['case'])}</td>"
                f"<td>{_img(r['pixelized'], None, r['case'])}</td>"
                f"<td>{_img(r['pixelized'], w, r['case'])}</td>"
                f"<td class='cap'>{info}</td>"
                f"<td class='cap'>gen {r['seconds_generate']}s<br>"
                f"pixelize {r['seconds_pixelize']}s</td></tr>")
        parts.append("<h2>Sprites</h2><table>" + "".join(rows) + "</table>")
    for r in tiles:
        err = f"<div class='err'>{html.escape(r['error'])}</div>" if r["error"] else ""
        parts.append(
            f"<h2>{html.escape(r['case'])}</h2>{err}"
            f"<p class='cap'>{html.escape(str(r['variant']))}; gen {r['seconds_generate']}s, "
            f"autotile {r['seconds_autotile']}s, tile {r['tile_size']}px</p>"
            f"<table><tr><th>fill</th><th>base</th><th>A2</th><th>blob-47</th></tr><tr>"
            f"<td>{_img(r['fill'], 192)}</td><td>{_img(r['base'], 192)}</td>"
            f"<td>{_img(r['a2'], 192)}</td><td>{_img(r['blob47'], 384)}</td></tr></table>"
            f"<p>random {MAP_COLS}x{MAP_ROWS} map</p>{_img(r['map'], None, 'map')}")
    page = ("<!doctype html><html><head><meta charset='utf-8'><title>pixel compare</title>"
            f"<style>{_CSS}</style></head><body>{''.join(parts)}</body></html>")
    dest = out_dir / "index.html"
    dest.write_text(page)
    return dest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--server", default=os.environ.get("MUSE_SERVER", "http://localhost:8000"))
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args(argv)
    cases = load_cases(args.cases)
    if args.limit:
        cases = cases[: args.limit]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with httpx.Client() as client:
        results = run(cases, args.server, out, client)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(write_index(results, out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
