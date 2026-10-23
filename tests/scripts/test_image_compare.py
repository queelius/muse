"""Tests for the muse-vs-gpt-image-2 comparison script (mock transport)."""
from __future__ import annotations

import base64
import json
import struct
import zlib

import httpx

from scripts.bench.image_compare import (
    Case, Target, cases_from_visuals, load_cases, run, write_index,
)


def _png(w=4, h=2) -> bytes:
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))
    raw = b"".join(b"\x00" + b"\x00\x00\x00" * w for _ in range(h))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def _ok(request):
    return httpx.Response(200, json={"data": [{"b64_json": base64.b64encode(_png()).decode()}]},
                          headers={"x-muse-model-revision": "abc123", "x-request-id": "req_1"})


def _client(handler, seen=None):
    def wrapped(request):
        if seen is not None:
            seen.append(request)
        return handler(request)
    return httpx.Client(transport=httpx.MockTransport(wrapped))


MUSE = Target("flux", "flux", "http://muse.test/v1")
OAI = Target("openai-gpt-image-2", "gpt-image-2", "https://api.openai.com/v1", openai=True)


def test_generation_and_edit_requests_shaped(tmp_path):
    ref1, ref2 = tmp_path / "a.png", tmp_path / "b.png"
    ref1.write_bytes(_png()); ref2.write_bytes(_png())
    cases = [Case("plain", "a cat", avoid="text"),
             Case("edit", "a dog", refs=[str(ref1), str(ref2)])]
    seen = []
    run(cases, [MUSE], tmp_path / "out", _client(_ok, seen))
    gen, edit = seen
    assert gen.url.path == "/v1/images/generations"
    body = json.loads(gen.content)
    assert body["prompt"] == "a cat Do not include: text."
    assert body["n"] == 1
    assert body["model"] == "flux" and body["size"] == "1536x1024"
    assert edit.url.path == "/v1/images/edits"
    assert edit.headers["content-type"].startswith("multipart/form-data")
    assert edit.content.count(b'name="image[]"') == 2


def test_openai_key_only_sent_to_openai_host(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    seen = []
    results = run([Case("c", "p")], [MUSE, OAI], tmp_path, _client(_ok, seen))
    muse_req = next(r for r in seen if r.url.host == "muse.test")
    oai_req = next(r for r in seen if r.url.host == "api.openai.com")
    assert "authorization" not in muse_req.headers
    assert oai_req.headers["authorization"] == "Bearer sk-secret"
    body = json.loads(oai_req.content)
    assert "response_format" not in body and "seed" not in body
    assert "sk-secret" not in json.dumps(results)
    assert "sk-secret" not in write_index(results, tmp_path).read_text()


def test_openai_skipped_without_key(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    seen = []
    results = run([Case("c", "p")], [OAI], tmp_path, _client(_ok, seen))
    assert results == [] and seen == []


def test_cell_failure_recorded_run_continues(tmp_path):
    def handler(request):
        if b"boom" in request.content:
            return httpx.Response(500, text="server exploded")
        return _ok(request)
    results = run([Case("bad", "boom"), Case("good", "fine")], [MUSE], tmp_path, _client(handler))
    bad, good = results
    assert "500" in bad["error"] and bad["path"] is None
    assert good["error"] is None and (tmp_path / "flux" / "good.png").exists()
    assert good["size"] == "4x2"


def test_index_html_lists_every_cell(tmp_path):
    def handler(request):
        return httpx.Response(500, text="no") if b"boom" in request.content else _ok(request)
    results = run([Case("one", "x"), Case("two", "boom")], [MUSE], tmp_path, _client(handler))
    html_text = write_index(results, tmp_path).read_text()
    assert 'src="flux/one.png"' in html_text
    assert "HTTP 500" in html_text and ">two<" in html_text
    assert "http://" not in html_text.replace("<!doctype", "")


def test_revision_header_recorded(tmp_path):
    (cell,) = run([Case("c", "p")], [MUSE], tmp_path, _client(_ok))
    assert cell["revision"] == "abc123" and cell["request_id"] == "req_1"


def test_load_cases_yaml_and_json(tmp_path):
    (tmp_path / "c.json").write_text(json.dumps([{"id": "a", "prompt": "p", "quality": "low"}]))
    (case,) = load_cases(tmp_path / "c.json")
    assert case.quality == "low" and case.size == "1536x1024" and case.refs == []


def test_cases_from_visuals_maps_and_skips(tmp_path):
    d = tmp_path / "book"
    d.mkdir()
    (d / "visuals.yaml").write_text(
        "image_model: {size: 1024x1024, quality: low}\n"
        "style: {prompt: moody, avoid: text}\n"
        "characters:\n  tert: {description: tired officer, refs: [sheets/t.png]}\n"
        "  nameless: {refs: [x.png]}\n"
        "locations:\n  office: {description: dim office}\n")
    cases = cases_from_visuals(d / "visuals.yaml")
    assert [c.id for c in cases] == ["character-tert", "location-office"]
    tert = cases[0]
    assert tert.prompt == "tired officer, moody" and tert.avoid == "text"
    assert tert.refs == [str(d / "sheets/t.png")]
    assert tert.size == "1024x1024" and tert.quality == "low"


def test_unsafe_ids_and_targets_stay_inside_out(tmp_path):
    out = tmp_path / "out"
    cases = [Case("../../x", "p"), Case("a/b", "p"), Case("..", "p")]
    bad = Target("../m", "m", "http://muse.test/v1")
    results = run(cases, [bad], out, _client(_ok))
    written = [p for p in tmp_path.rglob("*.png")]
    assert len(written) == 3
    assert all(out.resolve() in p.resolve().parents for p in written)
    assert all(r["error"] is None and ".." not in r["path"].split("/")
               for r in results)
    assert results[0]["case"] == "../../x" and results[0]["target"] == "../m"


def test_colliding_sanitised_names_get_distinct_files(tmp_path):
    cases = [Case("a/b", "p"), Case("a_b", "p"), Case("a/b", "p")]
    targets = [Target("m/x", "m", "http://muse.test/v1"),
               Target("m_x", "m", "http://muse.test/v1")]
    results = run(cases[:2], targets, tmp_path, _client(_ok))
    assert len({r["path"] for r in results}) == 4
    assert len(list(tmp_path.rglob("*.png"))) == 4
    again = run(cases[:1], targets[:1], tmp_path, _client(_ok))
    assert again[0]["path"] == results[0]["path"]
