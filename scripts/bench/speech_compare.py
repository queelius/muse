#!/usr/bin/env python3
"""Compare speech takes and check a pronunciation lexicon against a Muse server.

  python scripts/bench/speech_compare.py --lexicon lexicon.yaml --out /tmp/speech
  python scripts/bench/speech_compare.py --cases beats.yaml --target kokoro-82m:af_heart --out /tmp/beats

Standalone (httpx + PyYAML); writes audio, results.json and an index.html with
players and every timing disagreement. Lexicons and case files are read-only.
"""
from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
from difflib import SequenceMatcher
import hashlib
import html
import json
import os
from pathlib import Path
import re
import sys
import unicodedata

import httpx
import yaml


TIMEOUT = httpx.Timeout(900.0, connect=10.0)
LISTENER = "qwen3-asr-1.7b"
ALIGNER = "qwen3-forced-aligner-0.6b"
QUALITY = "utmos"


@dataclass
class Case:
    id: str
    text: str
    pronunciations: list[dict] | None = None
    accept: tuple[str, ...] = ()
    terms: tuple[str, ...] = ()
    voice: str | None = None


@dataclass
class Target:
    model: str
    voice: str


def load_cases(path: str | Path) -> list[Case]:
    """Read a YAML/JSON list of cases, optionally under a `cases` key."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("cases")
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a list of cases")
    return [Case(id=str(item["id"]), text=str(item["text"]),
                 pronunciations=item.get("pronunciations"),
                 accept=tuple(item.get("accept") or ()),
                 terms=tuple(item.get("terms") or ()), voice=item.get("voice"))
            for item in data]


def cases_from_lexicon(path: str | Path) -> list[Case]:
    """Convert each term/replacement to one sentence and pronunciation entry."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or any(
        not isinstance(k, str) or not isinstance(v, str) for k, v in data.items()
    ):
        raise ValueError(f"{path}: expected a mapping of terms to strings")
    cases = []
    for term, value in data.items():
        match = re.fullmatch(r"\[(.+)\]\(/(.+)/\)", value)
        entry = {"term": term, "ipa": match[2]} if match else {"term": term, "respell": value}
        cases.append(Case(term, f"Here is the word {term} in a sentence.",
                          pronunciations=[entry], terms=(term,)))
    return cases


def normalize(text: str) -> str:
    """Casefold, strip accents, replace punctuation with spaces, collapse spaces."""
    chars = unicodedata.normalize("NFKD", text.casefold())
    return " ".join("".join(c if c.isalnum() else " " for c in chars
                            if not unicodedata.combining(c)).split())


def _ordinary_replacement(text: str) -> bool:
    """Conservative spelling heuristic, not a dictionary or phoneme recognizer.

    Reject hyphenated respellings and mid-word capitals; these require the
    written term or an explicit `accept` alias.
    """
    return bool(normalize(text)) and not any(
        unicodedata.category(c) == "Pd" for c in text
    ) and not any(a.isalpha() and b.isupper() for a, b in zip(text, text[1:]))


def heard(transcript: str, term: str, replacement: str | None,
          accept: tuple[str, ...]) -> bool:
    """Match whole normalized phrases, including explicit acceptable aliases."""
    candidates = [term, *accept]
    if replacement and _ordinary_replacement(replacement):
        candidates.append(replacement)
    transcript = f" {normalize(transcript)} "
    return any(value and f" {value} " in transcript
               for value in (normalize(c) for c in candidates))


def wer(ref: str, hyp: str) -> float:
    """Normalized word Levenshtein error rate (empty reference: insertion count)."""
    reference, hypothesis = normalize(ref).split(), normalize(hyp).split()
    previous = list(range(len(hypothesis) + 1))
    for i, word in enumerate(reference, 1):
        current = [i]
        for j, other in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (word != other)))
        previous = current
    return previous[-1] / max(1, len(reference))


def agreement(words: list[dict], aligned: list[dict], tolerance: float = 0.1) -> dict:
    """Pair normalized words in order; unmatched words on either side count."""
    native = [w for w in words if normalize(w["word"])]
    aligned = [w for w in aligned if normalize(w["word"])]
    matcher = SequenceMatcher(None, [normalize(w["word"]) for w in native],
                              [normalize(w["word"]) for w in aligned], autojunk=False)
    within = total = 0
    disagreements = []
    for op, i, j, a, b in matcher.get_opcodes():
        if op == "equal":
            for n, r in zip(native[i:j], aligned[a:b]):
                total += 1
                delta = n["start"] - r["start"]
                if abs(delta) <= tolerance + 1e-9:
                    within += 1
                else:
                    disagreements.append(dict(word=n["word"], kokoro=n["start"],
                                              aligner=r["start"], delta=delta))
        else:
            for source, group in (("kokoro", native[i:j]), ("aligner", aligned[a:b])):
                for word in group:
                    total += 1
                    disagreements.append({**word, "source": source, "reason": "unmatched"})
    return dict(within=within, total=total, fraction=within / total if total else None,
                disagreements=disagreements)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", value).lstrip(".") or "_"


def _upload(client: httpx.Client, route: str, audio: bytes, fmt: str, **data) -> dict:
    mime = {"wav": "audio/wav", "opus": "audio/ogg"}.get(fmt, "application/octet-stream")
    response = client.post(f"/v1/audio/{route}", data=data,
                           files={"file": (f"take.{fmt}", audio, mime)}, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


def run(cases: list[Case], targets: list[Target], out_dir: str | Path,
        client: httpx.Client, *, listener: str, aligner: str, quality: str) -> list[dict]:
    """Run each case on every target; record failures and keep remaining takes."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for ti, target in enumerate(targets):
        for ci, case in enumerate(cases):
            voice = case.voice or target.voice
            row = dict(case=case.id, text=case.text, target=f"{target.model}:{target.voice}",
                       model=target.model, voice=voice, path=None, revision=None,
                       sha256=None, transcript=None, wer=None, heard={}, utmos=None,
                       agreement=None, error=None)
            try:
                payload = dict(model=target.model, voice=voice, input=case.text, timestamps=True)
                if case.pronunciations is not None:
                    payload["pronunciations"] = case.pronunciations
                response = client.post("/v1/audio/speech", json=payload, timeout=TIMEOUT)
                if (response.status_code == 400 and
                    response.json().get("error", {}).get("code") == "timestamps_not_supported"):
                    del payload["timestamps"]
                    response = client.post("/v1/audio/speech", json=payload, timeout=TIMEOUT)
                response.raise_for_status()
                row["revision"] = response.headers.get("x-muse-model-revision")
                row["request_id"] = response.headers.get("x-request-id")
                is_json = response.headers.get("content-type", "").startswith("application/json")
                body = response.json() if is_json else None
                if body is not None:
                    audio = base64.b64decode(body["audio"]["b64_json"], validate=True)
                    fmt = body["audio"]["format"]
                    row.update({key: body.get(key) for key in
                                ("model", "voice", "speed", "timing", "pronunciations", "words")})
                    row["audio"] = {k: v for k, v in body["audio"].items() if k != "b64_json"}
                else:
                    audio = response.content
                    fmt = "opus" if "ogg" in response.headers.get("content-type", "") else "wav"
                    row["speed"] = response.headers.get("x-muse-speed")
                    row["pronunciations"] = {"digest": response.headers.get("x-muse-pronunciations-digest")}
                row["sha256"] = hashlib.sha256(audio).hexdigest()
                rel = (f"{ti:03d}-{_safe_name(target.model)}-{_safe_name(target.voice)}/"
                       f"{ci:04d}-{_safe_name(case.id)}-{_safe_name(voice)}.{_safe_name(fmt)}")
                dest = (out_dir / rel).resolve()
                if not dest.is_relative_to(out_dir.resolve()):
                    raise ValueError(f"refusing to write outside --out: {rel}")
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(audio)
                row["path"] = rel
                transcript = _upload(client, "transcriptions", audio, fmt,
                                     model=listener, response_format="json")["text"]
                row["transcript"] = transcript
                row["wer"] = wer(case.text, transcript)
                replacements = {p["term"]: p.get("respell") for p in case.pronunciations or []
                                if "term" in p}
                row["heard"] = {term: heard(transcript, term, replacements.get(term), case.accept)
                                for term in case.terms or tuple(replacements)}
                scores = _upload(client, "quality", audio, fmt, model=quality)
                row["utmos"] = scores["scores"][scores["primary_score"]]["value"]
                if body is not None:
                    aligned = _upload(client, "alignments", audio, fmt, model=aligner,
                                      text=case.text, language="English")
                    row["agreement"] = agreement(body["words"], aligned["words"])
            except Exception as exc:  # one bad take must not stop the run
                row["error"] = (f"HTTP {exc.response.status_code}: {exc.response.text[:200]}"
                                if isinstance(exc, httpx.HTTPStatusError)
                                else f"{type(exc).__name__}: {exc}")
            results.append(row)
            print(f"{row['target']} {case.id}: {row['error'] or 'ok'}", file=sys.stderr)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False),
                                         encoding="utf-8")
    return results


def write_index(results: list[dict], out_dir: str | Path) -> Path:
    """An audio player and all metrics/disagreements for every take."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for result in results:
        title = html.escape(f"{result['case']} — {result['target']} — {result['voice']}")
        player = (f'<audio controls preload="none" src="{html.escape(result["path"], quote=True)}"></audio>'
                  if result.get("path") else "")
        details = html.escape(json.dumps(result, indent=2, ensure_ascii=False))
        rows.append(f"<article><h2>{title}</h2>{player}<pre>{details}</pre></article>")
    dest = out_dir / "index.html"
    dest.write_text("<!doctype html><html><head><meta charset='utf-8'>"
                    "<title>speech compare</title><style>body{font-family:system-ui;margin:2rem}"
                    "pre{white-space:pre-wrap}article{border-bottom:1px solid #ccc}</style>"
                    f"</head><body>{''.join(rows)}</body></html>", encoding="utf-8")
    return dest


def _target(value: str) -> Target:
    model, sep, voice = value.rpartition(":")
    if not sep or not model or not voice:
        raise argparse.ArgumentTypeError("expected model:voice")
    return Target(model, voice)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--cases")
    source.add_argument("--lexicon")
    ap.add_argument("--target", type=_target, action="append")
    ap.add_argument("--voices", help="comma-separated voices for the default Kokoro model")
    ap.add_argument("--server", default=os.environ.get("MUSE_SERVER", "http://localhost:8000"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.target and args.voices:
        ap.error("--voices overrides the default targets; use --target model:voice for explicit targets")
    voices = args.voices.split(",") if args.voices else ["af_heart", "am_puck"]
    if any(not v.strip() for v in voices):
        ap.error("--voices requires nonempty voice names")
    targets = args.target or [Target("kokoro-82m", v.strip()) for v in voices]
    cases = load_cases(args.cases) if args.cases else cases_from_lexicon(args.lexicon)
    with httpx.Client(base_url=args.server.rstrip("/")) as client:
        results = run(cases, targets, args.out, client,
                      listener=LISTENER, aligner=ALIGNER, quality=QUALITY)
    print(write_index(results, args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
