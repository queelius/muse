# Optimization Pass Implementation Plan (v0.57.0)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Tasks 1-3 are subagent tasks; Tasks 4-5 are SESSION-DRIVER tasks (live production boxes + evidence-dependent triage) and MUST NOT be dispatched to implementer subagents.

**Goal:** Committed benchmark harness (muse-vs-ollama LLM head-to-head + per-modality latencies), llama.cpp performance kwargs exposed as capabilities, then evidence-gated tuning with a published before/after report, per `docs/superpowers/specs/2026-07-08-optimization-pass-design.md`.

**Architecture:** Two standalone scripts under `scripts/bench/` sharing a tiny `_stats.py` (pure functions, unit-tested; network paths exercised live). `LlamaCppModel` gains optional performance kwargs forwarded to `Llama(...)` only when set (default construction byte-identical). Tuning lands as catalog/curated capability values, never code defaults.

**Tech Stack:** Python 3.10+, httpx (already a dep), pytest. NO new dependencies.

## Global Constraints

- ASCII only, NO em-dash (hook rejects).
- No new pip dependencies; harness imports stdlib + httpx only.
- Harness is resilient: a failing scenario/modality records `{"error": ...}` and continues; the run never aborts.
- Triage bar (Phase 2): ship a fix only with >= 20% measured improvement on the relevant scenario AND unchanged output quality; quality-affecting knobs are documented curated-entry edits only (no-surprises rule).
- Kwargs forwarding: defaults preserve today's `Llama(...)` construction EXACTLY (new params default None; only non-None forwarded).
- Branch `feature/optimization-pass` off main. Fast lane green: `MUSE_CATALOG_DIR=$(mktemp -d) python -m pytest tests/ -q -m "not slow"` (baseline ~3892 passed; do not pin MUSE_CONFIG).
- Commit trailers:
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01X2M12d5tRULNFUHFW2Hijx`

## File Structure

| File | Responsibility |
|---|---|
| Create `scripts/bench/__init__.py` | empty package marker |
| Create `scripts/bench/_stats.py` | pure helpers: median, tok_per_s, render_md_table, write_reports |
| Create `scripts/bench/bench_llm.py` | ollama-vs-muse head-to-head, 5 scenarios |
| Create `scripts/bench/bench_modalities.py` | per-modality hot latency sweep |
| Modify `src/muse/modalities/chat_completion/runtimes/llama_cpp.py` | optional perf kwargs |
| Create `tests/scripts/test_bench_stats.py` | stats/render unit tests |
| Modify `tests/modalities/chat_completion/runtimes/test_llama_cpp.py` | kwargs forwarding tests |
| Create (at run time, Task 4) `docs/benchmarks/2026-07-08-llm.{md,json}` etc. | reports |

---

### Task 1: shared stats helpers + `bench_llm.py`

**Files:**
- Create: `scripts/bench/__init__.py` (empty)
- Create: `scripts/bench/_stats.py`
- Create: `scripts/bench/bench_llm.py`
- Create: `tests/scripts/__init__.py` (empty) and `tests/scripts/test_bench_stats.py`

**Interfaces:**
- Produces: `_stats.median(xs: list[float]) -> float`; `_stats.tok_per_s(tokens: int, seconds: float) -> float` (0.0 when seconds <= 0); `_stats.render_md_table(headers: list[str], rows: list[list]) -> str`; `_stats.write_reports(results: dict, md_path: str | None, json_path: str | None, title: str) -> None` (json dump + md with one table per scenario). Task 2 reuses all four.
- CLI of bench_llm.py: `--muse URL` (default http://localhost:8000), `--ollama URL` (default http://localhost:11434), `--muse-model ID`, `--ollama-model NAME` (default qwen2.5:3b-instruct), `--worker-port INT` (optional; enables scenario 5), `--reps N` (default 3), `--json PATH`, `--md PATH`, `--scenarios CSV` (default all).

- [ ] **Step 1: Write the failing stats tests**

```python
# tests/scripts/test_bench_stats.py
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
```

NOTE: `scripts/` has no `__init__.py` at repo root today -- check; if
`import scripts.bench._stats` fails under pytest's importlib mode, add
`scripts/__init__.py` (empty) as well. Say which you did.

- [ ] **Step 2: Run to verify RED**

Run: `python -m pytest tests/scripts/test_bench_stats.py -q`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Implement `_stats.py`**

```python
# scripts/bench/_stats.py
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
```

- [ ] **Step 4: Run stats tests GREEN**

Run: `python -m pytest tests/scripts/test_bench_stats.py -q` -> PASS

- [ ] **Step 5: Implement `bench_llm.py`**

```python
#!/usr/bin/env python3
# scripts/bench/bench_llm.py
"""Muse-vs-ollama LLM head-to-head (spec 2026-07-08).

Both servers must already be running and the target models pulled/
available. Run from the repo root:

  python -m scripts.bench.bench_llm \
      --muse http://localhost:8000 --muse-model qwen2.5-3b-instruct-gguf-q4-k-m \
      --ollama http://localhost:11434 --ollama-model qwen2.5:3b-instruct \
      --md docs/benchmarks/2026-07-08-llm.md \
      --json docs/benchmarks/2026-07-08-llm.json
"""
from __future__ import annotations

import argparse
import json
import time

import httpx

from scripts.bench._stats import median, tok_per_s, write_reports

SHORT_PROMPT = "List five uses for a brick, one line each."
LONG_PROMPT = ("Summarize the following in two sentences. " +
               "The quick brown fox jumps over the lazy dog. " * 140)
TURN_PROMPTS = [
    "Name a famous river.",
    "What country is it in?",
    "Name one city on it.",
    "What is that city known for?",
]
TIMEOUT = httpx.Timeout(600.0, connect=10.0)


class MuseClient:
    name = "muse"

    def __init__(self, base: str, model: str):
        self.base, self.model = base.rstrip("/"), model

    def chat(self, messages: list[dict], max_tokens: int) -> dict:
        t0 = time.monotonic()
        r = httpx.post(f"{self.base}/v1/chat/completions", timeout=TIMEOUT,
                       json={"model": self.model, "messages": messages,
                             "max_tokens": max_tokens})
        elapsed = time.monotonic() - t0
        r.raise_for_status()
        d = r.json()
        u = d.get("usage", {})
        return {"elapsed": elapsed,
                "completion_tokens": u.get("completion_tokens", 0),
                "prompt_tokens": u.get("prompt_tokens", 0),
                "content": d["choices"][0]["message"].get("content", "")}

    def stream_ttft(self, messages: list[dict], max_tokens: int) -> dict:
        t0 = time.monotonic()
        ttft = None
        gaps: list[float] = []
        last = None
        n = 0
        with httpx.stream(
                "POST", f"{self.base}/v1/chat/completions", timeout=TIMEOUT,
                json={"model": self.model, "messages": messages,
                      "max_tokens": max_tokens, "stream": True}) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload.strip() == "[DONE]":
                    break
                delta = (json.loads(payload)["choices"][0]
                         .get("delta", {}).get("content"))
                if not delta:
                    continue
                now = time.monotonic()
                if ttft is None:
                    ttft = now - t0
                elif last is not None:
                    gaps.append(now - last)
                last = now
                n += 1
        return {"ttft": ttft, "gap_median": median(gaps) if gaps else 0.0,
                "tokens": n, "elapsed": time.monotonic() - t0}


class OllamaClient:
    name = "ollama"

    def __init__(self, base: str, model: str):
        self.base, self.model = base.rstrip("/"), model

    def chat(self, messages: list[dict], max_tokens: int) -> dict:
        t0 = time.monotonic()
        r = httpx.post(f"{self.base}/api/chat", timeout=TIMEOUT,
                       json={"model": self.model, "messages": messages,
                             "stream": False,
                             "options": {"num_predict": max_tokens}})
        elapsed = time.monotonic() - t0
        r.raise_for_status()
        d = r.json()
        return {"elapsed": elapsed,
                "completion_tokens": d.get("eval_count", 0),
                "prompt_tokens": d.get("prompt_eval_count", 0),
                # ollama self-reported (nanoseconds); more precise than wall
                "eval_s": d.get("eval_duration", 0) / 1e9,
                "prompt_eval_s": d.get("prompt_eval_duration", 0) / 1e9,
                "content": d.get("message", {}).get("content", "")}

    def stream_ttft(self, messages: list[dict], max_tokens: int) -> dict:
        t0 = time.monotonic()
        ttft = None
        gaps: list[float] = []
        last = None
        n = 0
        with httpx.stream(
                "POST", f"{self.base}/api/chat", timeout=TIMEOUT,
                json={"model": self.model, "messages": messages,
                      "stream": True,
                      "options": {"num_predict": max_tokens}}) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line.strip():
                    continue
                d = json.loads(line)
                if d.get("done"):
                    break
                if not d.get("message", {}).get("content"):
                    continue
                now = time.monotonic()
                if ttft is None:
                    ttft = now - t0
                elif last is not None:
                    gaps.append(now - last)
                last = now
                n += 1
        return {"ttft": ttft, "gap_median": median(gaps) if gaps else 0.0,
                "tokens": n, "elapsed": time.monotonic() - t0}


def _reps(fn, reps: int):
    fn()  # discarded warmup
    return [fn() for _ in range(reps)]


def scenario_short_gen(clients, reps):
    msgs = [{"role": "user", "content": SHORT_PROMPT}]
    rows, raw = [], {}
    for c in clients:
        rs = _reps(lambda c=c: c.chat(msgs, 128), reps)
        med_el = median([r["elapsed"] for r in rs])
        med_tps = median([tok_per_s(r["completion_tokens"], r["elapsed"])
                          for r in rs])
        row = [c.name, round(med_el, 2), round(med_tps, 2)]
        if "eval_s" in rs[0]:  # ollama self-reported generation-only tok/s
            row.append(round(median([tok_per_s(r["completion_tokens"],
                                               r["eval_s"]) for r in rs]), 2))
        else:
            row.append("-")
        rows.append(row)
        raw[c.name] = rs
    return {"headers": ["server", "median elapsed s", "tok/s (wall)",
                        "tok/s (self-reported)"], "rows": rows, "raw": raw}


def scenario_stream_ttft(clients, reps):
    msgs = [{"role": "user", "content": SHORT_PROMPT}]
    rows, raw = [], {}
    for c in clients:
        rs = _reps(lambda c=c: c.stream_ttft(msgs, 128), reps)
        rows.append([c.name,
                     round(median([r["ttft"] or 0 for r in rs]), 3),
                     round(median([r["gap_median"] for r in rs]) * 1000, 1)])
        raw[c.name] = rs
    return {"headers": ["server", "median TTFT s", "median inter-token ms"],
            "rows": rows, "raw": raw}


def scenario_long_prompt(clients, reps):
    msgs = [{"role": "user", "content": LONG_PROMPT}]
    rows, raw = [], {}
    for c in clients:
        rs = _reps(lambda c=c: c.chat(msgs, 64), reps)
        rows.append([c.name, rs[0].get("prompt_tokens", "?"),
                     round(median([r["elapsed"] for r in rs]), 2),
                     round(median([r.get("prompt_eval_s", 0) for r in rs]), 2)
                     if "prompt_eval_s" in rs[0] else "-"])
        raw[c.name] = rs
    return {"headers": ["server", "prompt tokens", "median elapsed s",
                        "prompt-eval s (self-reported)"],
            "rows": rows, "raw": raw}


def scenario_multi_turn(clients, reps):
    # Per-turn elapsed with cumulative history. Flat turn-times despite a
    # growing prefix indicate KV prefix reuse; growing times indicate full
    # re-eval per turn. One pass per rep; per-turn medians reported.
    rows, raw = [], {}
    for c in clients:
        per_rep = []
        for _ in range(reps):
            history: list[dict] = []
            turns = []
            for q in TURN_PROMPTS:
                history.append({"role": "user", "content": q})
                r = c.chat(history, 48)
                history.append({"role": "assistant", "content": r["content"]})
                turns.append(r["elapsed"])
            per_rep.append(turns)
        med_turns = [round(median([rep[i] for rep in per_rep]), 2)
                     for i in range(len(TURN_PROMPTS))]
        rows.append([c.name] + med_turns)
        raw[c.name] = per_rep
    return {"headers": ["server"] + [f"turn {i+1} s"
                                     for i in range(len(TURN_PROMPTS))],
            "rows": rows, "raw": raw}


def scenario_gateway_split(muse: MuseClient, worker_port: int, reps):
    direct = MuseClient(f"http://127.0.0.1:{worker_port}", muse.model)
    return scenario_short_gen([muse, direct], reps) | {"note": (
        "second row is the same muse model via the worker port directly; "
        "delta vs row one is gateway/relay overhead")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--muse", default="http://localhost:8000")
    ap.add_argument("--muse-model", required=True)
    ap.add_argument("--ollama", default="http://localhost:11434")
    ap.add_argument("--ollama-model", default="qwen2.5:3b-instruct")
    ap.add_argument("--worker-port", type=int, default=None)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--json", dest="json_path", default=None)
    ap.add_argument("--md", dest="md_path", default=None)
    ap.add_argument("--scenarios", default="short,ttft,long,turns,split")
    a = ap.parse_args()

    muse = MuseClient(a.muse, a.muse_model)
    oll = OllamaClient(a.ollama, a.ollama_model)
    clients = [muse, oll]
    wanted = set(a.scenarios.split(","))
    results: dict = {}
    runners = {
        "short": lambda: scenario_short_gen(clients, a.reps),
        "ttft": lambda: scenario_stream_ttft(clients, a.reps),
        "long": lambda: scenario_long_prompt(clients, a.reps),
        "turns": lambda: scenario_multi_turn(clients, a.reps),
    }
    if a.worker_port:
        runners["split"] = lambda: scenario_gateway_split(
            muse, a.worker_port, a.reps)
    for name, fn in runners.items():
        if name not in wanted:
            continue
        try:
            results[name] = fn()
            print(f"[ok] {name}")
        except Exception as e:  # noqa: BLE001 -- resilient by spec
            results[name] = {"error": f"{type(e).__name__}: {e}"}
            print(f"[error] {name}: {e}")
    write_reports(results, md_path=a.md_path, json_path=a.json_path,
                  title="LLM head-to-head: muse vs ollama")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Syntax + smoke check (no server needed)**

Run: `python -m scripts.bench.bench_llm --help`
Expected: argparse help, exit 0. Also: `python -m pytest tests/scripts/ -q` -> PASS.

- [ ] **Step 7: Commit**

```bash
git add scripts/bench tests/scripts
git commit -m "feat(bench): stats helpers + muse-vs-ollama LLM harness"
```

---

### Task 2: `bench_modalities.py`

**Files:**
- Create: `scripts/bench/bench_modalities.py`

**Interfaces:**
- Consumes: `_stats.median`, `_stats.write_reports` from Task 1.
- CLI: `--server URL` (default http://192.168.0.204:8000), `--reps N` (default 3), `--json PATH`, `--md PATH`, `--only CSV` (subset of case names).

- [ ] **Step 1: Implement**

```python
#!/usr/bin/env python3
# scripts/bench/bench_modalities.py
"""Per-modality hot-latency sweep against a live muse server (spec
2026-07-08). Each case: 1 discarded warmup + N reps, median wall-clock.
Models not enabled on the target are skipped (recorded, not fatal).

  python -m scripts.bench.bench_modalities \
      --server http://192.168.0.204:8000 \
      --md docs/benchmarks/2026-07-08-modalities.md \
      --json docs/benchmarks/2026-07-08-modalities.json
"""
from __future__ import annotations

import argparse
import base64
import io
import time

import httpx

from scripts.bench._stats import median, write_reports

TIMEOUT = httpx.Timeout(600.0, connect=10.0)
SENTENCE = "The quick brown fox jumps over the lazy dog near the river bank."


def _post_json(base, path, body):
    r = httpx.post(f"{base}{path}", json=body, timeout=TIMEOUT)
    r.raise_for_status()
    return r


def _enabled_models(base) -> set[str]:
    r = httpx.get(f"{base}/v1/models", timeout=TIMEOUT)
    r.raise_for_status()
    return {m["id"] for m in r.json().get("data", [])}


def _tts_wav(base, model) -> bytes:
    r = _post_json(base, "/v1/audio/speech",
                   {"model": model, "input": SENTENCE + " " + SENTENCE})
    return r.content


CASES = [
    # (name, model_id, callable(base) -> None)  -- callables raise on failure
    ("image_gen_sd_turbo", "sd-turbo",
     lambda b: _post_json(b, "/v1/images/generations",
                          {"model": "sd-turbo", "prompt": "a red cube",
                           "size": "512x512"})),
    ("image_gen_pixel_lora", "pixelartredmond-1-5v-pixel-art-loras-for-sd-1-5",
     lambda b: _post_json(b, "/v1/images/generations",
                          {"model": "pixelartredmond-1-5v-pixel-art-loras-for-sd-1-5",
                           "prompt": "Pixel Art, PixArFK, a knight sprite",
                           "size": "512x512"})),
    ("tts_kokoro", "kokoro-82m",
     lambda b: _post_json(b, "/v1/audio/speech",
                          {"model": "kokoro-82m", "input": SENTENCE})),
    ("tts_supertonic", "supertonic-3",
     lambda b: _post_json(b, "/v1/audio/speech",
                          {"model": "supertonic-3", "input": SENTENCE})),
    ("sfx_stable_audio", "stable-audio-open-1.0",
     lambda b: _post_json(b, "/v1/audio/sfx",
                          {"model": "stable-audio-open-1.0",
                           "prompt": "a wooden door creaks", "duration": 3})),
]


def _bench_case(base, fn, reps) -> dict:
    fn(base)  # warmup (also triggers cold load; excluded from medians)
    times = []
    for _ in range(reps):
        t0 = time.monotonic()
        fn(base)
        times.append(time.monotonic() - t0)
    return {"median_s": round(median(times), 3),
            "reps_s": [round(t, 3) for t in times]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://192.168.0.204:8000")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--json", dest="json_path", default=None)
    ap.add_argument("--md", dest="md_path", default=None)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    base = a.server.rstrip("/")

    enabled = _enabled_models(base)
    only = set(a.only.split(",")) if a.only else None
    rows, raw = [], {}

    cases = list(CASES)
    # whisper: needs a wav; generate via the target's own TTS if available
    tts_model = ("kokoro-82m" if "kokoro-82m" in enabled
                 else "supertonic-3" if "supertonic-3" in enabled else None)
    whisper = next((m for m in sorted(enabled) if "whisper" in m), None)
    if tts_model and whisper:
        wav = _tts_wav(base, tts_model)

        def _transcribe(b, wav=wav, whisper=whisper):
            r = httpx.post(f"{b}/v1/audio/transcriptions", timeout=TIMEOUT,
                           files={"file": ("clip.wav", io.BytesIO(wav),
                                           "audio/wav")},
                           data={"model": whisper})
            r.raise_for_status()
        cases.append((f"transcribe_{whisper}", whisper, _transcribe))

    # embeddings: first enabled embedding-ish model by known ids
    embed = next((m for m in sorted(enabled)
                  if "minilm" in m or "embed" in m.lower()), None)
    if embed:
        def _embed(b, embed=embed):
            _post_json(b, "/v1/embeddings",
                       {"model": embed, "input": [SENTENCE] * 16})
        cases.append((f"embed_{embed}", embed, _embed))

    # smallest enabled GGUF chat model: heuristic by id
    chat = next((m for m in sorted(enabled)
                 if "gguf" in m or "qwen" in m), None)
    if chat:
        def _chat(b, chat=chat):
            _post_json(b, "/v1/chat/completions",
                       {"model": chat, "max_tokens": 64,
                        "messages": [{"role": "user",
                                      "content": "Say hello in 5 words."}]})
        cases.append((f"chat_{chat}", chat, _chat))

    for name, model, fn in cases:
        if fn is None:
            continue
        if only and name not in only:
            continue
        if model is not None and model not in enabled:
            rows.append([name, model, "SKIPPED (not enabled)"])
            continue
        try:
            r = _bench_case(base, fn, a.reps)
            rows.append([name, model, r["median_s"]])
            raw[name] = r
            print(f"[ok] {name}: {r['median_s']}s")
        except Exception as e:  # noqa: BLE001 -- resilient by spec
            rows.append([name, model, f"ERROR {type(e).__name__}"])
            raw[name] = {"error": str(e)}
            print(f"[error] {name}: {e}")

    write_reports({"modalities": {"headers": ["case", "model", "median s"],
                                  "rows": rows, "raw": raw}},
                  md_path=a.md_path, json_path=a.json_path,
                  title=f"Modality hot latencies: {base}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

IMPLEMENTER NOTES: (a) verify the TTS response is raw audio bytes
(`r.content`) by reading `src/muse/modalities/audio_speech/routes.py` --
adapt `_tts_wav` if the route returns JSON; (b) embeddings, whisper, and chat cases are appended dynamically
after model discovery, exactly as shown; (c) confirm
`/v1/audio/transcriptions` field names (file, model) against the
transcription route; (d) the base64/io imports: drop `base64` if unused
after (a).

- [ ] **Step 2: Syntax check**

Run: `python -m scripts.bench.bench_modalities --help` -> exit 0.

- [ ] **Step 3: Commit**

```bash
git add scripts/bench/bench_modalities.py
git commit -m "feat(bench): per-modality hot-latency sweep script"
```

---

### Task 3: llama.cpp performance kwargs forwarding

**Files:**
- Modify: `src/muse/modalities/chat_completion/runtimes/llama_cpp.py` (constructor, ~lines 55-100)
- Test: `tests/modalities/chat_completion/runtimes/test_llama_cpp.py` (append)

**Interfaces:**
- Produces: `LlamaCppModel.__init__` gains keyword-only params `n_threads: int | None = None`, `n_threads_batch: int | None = None`, `n_batch: int | None = None`, `flash_attn: bool | None = None`, `use_mlock: bool | None = None`, `type_k: int | None = None`, `type_v: int | None = None`. Each is forwarded to `Llama(...)` ONLY when not None. Manifest capabilities with these names flow in via the existing load_backend kwargs merge automatically (no catalog.py change needed).

- [ ] **Step 1: Write the failing tests** (append to the existing test file; follow its established fake-Llama pattern -- read the file first, reuse its fixtures/mocks):

```python
class TestPerfKwargsForwarding:
    """Spec 2026-07-08: optional llama.cpp performance kwargs are forwarded
    to Llama(...) only when set; defaults leave construction byte-identical."""

    def _construct(self, tmp_path, **extra):
        # Follow the file's existing pattern for building a LlamaCppModel
        # with a patched Llama and a dummy gguf file; return the mock's
        # call kwargs. (Reuse the existing helper/fixture if one exists.)
        ...

    def test_defaults_add_no_new_kwargs(self, tmp_path):
        kwargs = self._construct(tmp_path)
        for k in ("n_threads", "n_threads_batch", "n_batch", "flash_attn",
                  "use_mlock", "type_k", "type_v"):
            assert k not in kwargs

    def test_each_kwarg_forwarded_when_set(self, tmp_path):
        kwargs = self._construct(
            tmp_path, n_threads=12, n_threads_batch=12, n_batch=1024,
            flash_attn=True, use_mlock=True, type_k=8, type_v=8,
        )
        assert kwargs["n_threads"] == 12
        assert kwargs["n_threads_batch"] == 12
        assert kwargs["n_batch"] == 1024
        assert kwargs["flash_attn"] is True
        assert kwargs["use_mlock"] is True
        assert kwargs["type_k"] == 8 and kwargs["type_v"] == 8

    def test_false_and_zero_are_forwarded(self, tmp_path):
        """None means unset; False/0 are real values and must forward."""
        kwargs = self._construct(tmp_path, flash_attn=False, n_batch=0)
        assert kwargs["flash_attn"] is False
        assert kwargs["n_batch"] == 0
```

(Implementer: `_construct` must be real, modeled on the file's existing
constructor tests -- the `...` above is the one thing you fill from the
established pattern; assertions are the contract.)

- [ ] **Step 2: RED**

Run: `python -m pytest tests/modalities/chat_completion/runtimes/test_llama_cpp.py -q`
Expected: new tests FAIL (TypeError unexpected kwarg or KeyError)

- [ ] **Step 3: Implement**

In `LlamaCppModel.__init__`, add after `device: str = "auto",`:

```python
        # Optional llama.cpp performance kwargs (spec 2026-07-08).
        # None = unset = do not forward, so default construction stays
        # byte-identical to pre-v0.57 behavior. Set per-model via manifest
        # capabilities (they reach here through the load_backend kwargs
        # merge); tuned values live in catalog/curated entries, never as
        # code defaults.
        n_threads: int | None = None,
        n_threads_batch: int | None = None,
        n_batch: int | None = None,
        flash_attn: bool | None = None,
        use_mlock: bool | None = None,
        type_k: int | None = None,
        type_v: int | None = None,
```

and replace the `Llama(...)` call with:

```python
        perf_kwargs = {
            k: v for k, v in {
                "n_threads": n_threads,
                "n_threads_batch": n_threads_batch,
                "n_batch": n_batch,
                "flash_attn": flash_attn,
                "use_mlock": use_mlock,
                "type_k": type_k,
                "type_v": type_v,
            }.items() if v is not None
        }
        self._llama = Llama(
            model_path=str(gguf_path),
            n_ctx=context_length,
            n_gpu_layers=n_gpu_layers,
            chat_format=effective_chat_format,
            verbose=False,
            **perf_kwargs,
        )
```

Update the class docstring's kwargs list to mention the seven new keys.

- [ ] **Step 4: GREEN + no regressions**

Run: `python -m pytest tests/modalities/chat_completion/ -q` -> PASS

- [ ] **Step 5: Commit**

```bash
git add src/muse/modalities/chat_completion/runtimes/llama_cpp.py tests/modalities/chat_completion/runtimes/test_llama_cpp.py
git commit -m "feat(llama-cpp): forward optional performance kwargs (n_threads, flash_attn, ...)"
```

---

### Task 4: baseline run (SESSION DRIVER -- live boxes, do NOT dispatch)

- [ ] Pull the head-to-head model on the .102 box: `muse pull hf://bartowski/Qwen2.5-3B-Instruct-GGUF@q4_k_m && muse models enable qwen2.5-3b-instruct-gguf-q4-k-m` (local box; ollama already has qwen2.5:3b-instruct).
- [ ] Run `python -m scripts.bench.bench_llm --muse-model qwen2.5-3b-instruct-gguf-q4-k-m --md docs/benchmarks/2026-07-08-llm-baseline.md --json docs/benchmarks/2026-07-08-llm-baseline.json` on the local box (add `--worker-port <port>` from /v1/admin/workers for scenario 5).
- [ ] Run `python -m scripts.bench.bench_modalities --server http://192.168.0.204:8000 --md docs/benchmarks/2026-07-08-modalities-baseline.md --json docs/benchmarks/2026-07-08-modalities-baseline.json`.
- [ ] Commit the baseline reports; present the triage table to the user.

### Task 5: evidence-gated tuning + after-report (SESSION DRIVER)

- [ ] Apply ONLY fixes meeting the >= 20% bar (expected: capability values via catalog for the .102 GGUFs, e.g. `n_threads`; possibly curated-entry edits). Each code fix (if any beyond Task 3's mechanism) goes through TDD + review as usual.
- [ ] Re-run both harness scripts -> `docs/benchmarks/2026-07-08-*-after.md`.
- [ ] Produce the before/after table; CLAUDE.md gains a short "Benchmarks" note pointing at scripts/bench/; bump pyproject to 0.57.0; commit. NO tag/push -- release gated on user go.
```
