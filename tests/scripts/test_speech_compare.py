"""Speech comparison behavior against a mocked Muse server."""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import httpx
import pytest

from scripts.bench.speech_compare import (
    Case, Target, agreement, cases_from_lexicon, heard, load_cases, main,
    normalize, run, wer, write_index,
)


AUDIO = b"RIFF-test-audio"
WORDS = [{"word": "Hello", "start": 0.2, "end": 0.5, "measured": True}]


def _handler(request):
    if request.url.path == "/v1/audio/speech":
        return httpx.Response(200, json={
            "model": "kokoro-82m", "voice": json.loads(request.content)["voice"],
            "speed": 1.0, "audio": {"format": "wav",
                "b64_json": base64.b64encode(AUDIO).decode(),
                "sha256": hashlib.sha256(AUDIO).hexdigest(),
                "sample_rate": 24000, "duration_seconds": 1.0},
            "words": WORDS, "timing": {"source": "native"},
            "pronunciations": {"digest": "digest", "applied": []}},
            headers={"X-Muse-Model-Revision": "abc123"})
    if request.url.path.endswith("transcriptions"):
        assert b'name="prompt"' not in request.content
        assert b'qwen3-asr-1.7b' in request.content
        assert b'name="response_format"\r\n\r\njson' in request.content
        return httpx.Response(200, json={"text": "Hello"})
    if request.url.path.endswith("alignments"):
        assert b'qwen3-forced-aligner-0.6b' in request.content
        assert b'name="language"\r\n\r\nEnglish' in request.content
        assert b'name="text"\r\n\r\nHello' in request.content
        return httpx.Response(200, json={"words": WORDS})
    assert request.url.path == "/v1/audio/quality"
    assert b'utmos' in request.content
    return httpx.Response(200, json={"primary_score": "naturalness",
        "scores": {"naturalness": {"value": 4.2}, "other": {"value": 1.0}}})


def _client(handler=_handler):
    return httpx.Client(base_url="http://muse.test", transport=httpx.MockTransport(handler))


def _run(cases, targets, out, client):
    return run(cases, targets, out, client, listener="qwen3-asr-1.7b",
               aligner="qwen3-forced-aligner-0.6b", quality="utmos")


def test_cases_from_lexicon_ipa_and_respell(tmp_path):
    path = tmp_path / "lexicon.yaml"
    source = 'Coulomb: "[Coulomb](/kuːˈlɒm/)"\ntanh: hyperbolic tangent\n'
    path.write_text(source)
    coulomb, tanh = cases_from_lexicon(path)
    assert coulomb.pronunciations == [{"term": "Coulomb", "ipa": "kuːˈlɒm"}]
    assert tanh.pronunciations == [{"term": "tanh", "respell": "hyperbolic tangent"}]
    assert tanh.text == "Here is the word tanh in a sentence."
    assert tanh.terms == ("tanh",)
    assert path.read_text() == source


def test_heard_rules():
    assert heard("the hyperbolic tangent", "tanh", "hyperbolic tangent", ())
    assert heard("Jason files", "json", "JAY-sawn", ("jason",))
    assert not heard("jay sawn", "json", "JAY-sawn", ())
    assert heard("mah noyd", "monoid", "mah noyd", ())
    assert not heard("mah noyd", "monoid", "mah-NOYD", ())
    assert not heard("mah noyd", "monoid", "mahNoyd", ())
    assert heard("poincare map", "poincaré", None, ())
    assert heard("exclusive or", "xor", "exclusive, or", ())
    assert not heard("jsonl", "json", None, ())
    assert not heard("anything", "!!!", None, ())
    assert normalize(" POINCARÉ—map! ") == "poincare map"


def test_wer():
    assert wer("Café, here!", "cafe here") == 0
    assert wer("one two three", "one four") == pytest.approx(2 / 3)
    assert wer("one", "one two three") == 2
    assert wer("", "") == 0
    assert wer("", "extra") == 1


def test_agreement_lists_disagreements():
    native = [{"word": w, "start": t} for w, t in
              [("Café,", 0.2), ("late", 1.0), ("missing", 2.0), ("end", 3.0)]]
    aligned = [{"word": w, "start": t} for w, t in
               [("cafe", 0.3), ("late", 1.2), ("extra", 2.0), ("end", 3.0)]]
    result = agreement(native, aligned)
    assert result["within"] == 2 and result["total"] == 5
    assert result["fraction"] == 0.4
    assert len(result["disagreements"]) == 3
    assert {d["word"] for d in result["disagreements"]} == {"late", "missing", "extra"}
    assert agreement([], [])["fraction"] is None


def test_run_writes_results_and_index(tmp_path):
    seen = []
    def handler(request):
        seen.append(request)
        return _handler(request)
    case = Case("greeting", "Hello", pronunciations=[{"term": "Hello", "respell": "hello"}],
                terms=("Hello",))
    results = _run([case], [Target("kokoro-82m", "af_heart")], tmp_path, _client(handler))
    (row,) = results
    assert row["error"] is None
    assert row["transcript"] == "Hello" and row["wer"] == 0
    assert row["heard"] == {"Hello": True} and row["utmos"] == 4.2
    assert row["agreement"]["fraction"] == 1
    assert row["revision"] == "abc123" and row["voice"] == "af_heart"
    assert row["sha256"] == hashlib.sha256(AUDIO).hexdigest()
    assert row["pronunciations"]["digest"] == "digest"
    assert (tmp_path / row["path"]).read_bytes() == AUDIO
    assert json.loads((tmp_path / "results.json").read_text()) == results
    page = write_index(results, tmp_path).read_text()
    assert page.count("<audio") == 1 and row["path"] in page
    assert "Hello" in page and "abc123" in page and "disagreements" in page
    payload = json.loads(seen[0].content)
    assert payload["timestamps"] is True and payload["pronunciations"] == case.pronunciations


def test_timestamp_fallback_only_for_specific_error(tmp_path):
    speech = []
    def handler(request):
        if request.url.path.endswith("speech"):
            body = json.loads(request.content)
            speech.append(body)
            if body.get("timestamps"):
                return httpx.Response(400, json={"error": {"code": "timestamps_not_supported"}})
            return httpx.Response(200, content=AUDIO, headers={"content-type": "audio/wav",
                "X-Muse-Model-Revision": "rawrev"})
        assert not request.url.path.endswith("alignments")
        return _handler(request)
    (row,) = _run([Case("c", "Hello")], [Target("other", "v")], tmp_path, _client(handler))
    assert row["error"] is None and row["agreement"] is None
    assert row["revision"] == "rawrev" and row["sha256"] == hashlib.sha256(AUDIO).hexdigest()
    assert "timestamps" not in speech[1]
    def bad(request):
        return httpx.Response(400, json={"error": {"code": "invalid_pronunciation"}})
    (row,) = _run([Case("bad", "Hello")], [Target("m", "v")], tmp_path, _client(bad))
    assert "400" in row["error"] and row["path"] is None


def test_load_cases_and_preserve_beats_voice():
    path = Path(__file__).resolve().parents[2] / "docs/benchmarks/2026-10-24-kokoro-timing-beats.yaml"
    cases = load_cases(path)
    assert len(cases) == 50 and cases[0].voice == "bf_emma"


def test_every_take_has_distinct_safe_audio_and_voice(tmp_path):
    cases = [Case("../../same", "Hello", voice="bf_emma"),
             Case("../../same", "Hello", voice="am_puck")]
    targets = [Target("../model", "a"), Target("other", "b")]
    seen = []
    def handler(request):
        if request.url.path.endswith("speech"):
            seen.append(json.loads(request.content))
        return _handler(request)
    results = _run(cases, targets, tmp_path, _client(handler))
    assert len({r["path"] for r in results}) == 4
    assert [r["voice"] for r in results] == ["bf_emma", "am_puck"] * 2
    assert [r["voice"] for r in seen] == ["bf_emma", "am_puck"] * 2
    assert all((tmp_path / r["path"]).resolve().is_relative_to(tmp_path) for r in results)
    assert write_index(results, tmp_path).read_text().count("<audio") == 4


def test_cell_failure_continues_and_html_escapes(tmp_path):
    def handler(request):
        if request.url.path.endswith("speech") and json.loads(request.content)["input"] == "bad":
            return httpx.Response(500, text="<failed>")
        return _handler(request)
    results = _run([Case("<bad>", "bad"), Case("ok", "Hello")],
                   [Target("m", "v")], tmp_path, _client(handler))
    assert "500" in results[0]["error"] and results[1]["error"] is None
    page = write_index(results, tmp_path).read_text()
    assert "&lt;bad&gt;" in page and "&lt;failed&gt;" in page


@pytest.mark.parametrize("args, expected", [
    ([], [("kokoro-82m", "af_heart"), ("kokoro-82m", "am_puck")]),
    (["--voices", "a,b"], [("kokoro-82m", "a"), ("kokoro-82m", "b")]),
    (["--target", "hf://org/model:v"], [("hf://org/model", "v")]),
])
def test_main_targets_and_server(tmp_path, monkeypatch, args, expected):
    from scripts.bench import speech_compare
    source = tmp_path / "cases.json"
    source.write_text('[{"id":"c","text":"Hello"}]')
    observed = []
    def fake_run(cases, targets, out_dir, client, **kwargs):
        assert str(client.base_url).rstrip("/") == "http://env.test"
        observed.extend((t.model, t.voice) for t in targets)
        assert kwargs == {"listener": "qwen3-asr-1.7b", "aligner": "qwen3-forced-aligner-0.6b", "quality": "utmos"}
        return []
    monkeypatch.setenv("MUSE_SERVER", "http://env.test")
    monkeypatch.setattr(speech_compare, "run", fake_run)
    assert main(["--cases", str(source), "--out", str(tmp_path / "out"), *args]) == 0
    assert observed == expected


def test_agreement_keeps_repeated_words_and_counts_insertions():
    native = [{"word": "the", "start": 1.0}] * 210
    aligned = [{"word": "extra", "start": 0.0}, *native]
    result = agreement(native, aligned)
    assert result["within"] == 210 and result["total"] == 211
    assert result["fraction"] == pytest.approx(210 / 211)
    assert result["disagreements"] == [
        {"word": "extra", "start": 0.0, "source": "aligner", "reason": "unmatched"}]


def test_load_case_options_and_infer_terms_from_pronunciations(tmp_path):
    path = tmp_path / "cases.yaml"
    path.write_text('cases:\n- id: json\n  text: Hello\n  accept: [hello]\n'
                    '  pronunciations: [{term: json, respell: JAY-sawn}]\n')
    (case,) = load_cases(path)
    assert case.accept == ("hello",) and case.terms == ()
    (row,) = _run([case], [Target("m", "v")], tmp_path / "out", _client())
    assert row["heard"] == {"json": True}


def test_agreement_pairs_hyphenated_words_with_joined_aligner_words():
    native = [{"word": "half-seen", "start": 0.5}, {"word": "idea.", "start": 1.0}]
    aligned = [{"word": "halfseen", "start": 0.55}, {"word": "idea", "start": 1.0}]
    result = agreement(native, aligned)
    assert result["total"] == 2 and result["within"] == 2
    assert result["disagreements"] == []
