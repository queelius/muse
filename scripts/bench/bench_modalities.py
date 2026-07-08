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
