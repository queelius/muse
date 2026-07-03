#!/usr/bin/env python3
"""Concurrency stress test for a live muse gateway.

Fires concurrent requests across modalities to observe how the gateway's
single-event-loop director + per-model workers behave under load. Five
scenarios, each isolating one behavior:

  A. same-model concurrency   -- N identical requests to one hot model;
                                 concurrent-wall vs sequential-sum tells
                                 us whether a worker parallelizes same-model
                                 inference or serializes it.
  B. different light models    -- one request each to several co-resident
                                 light models at once; do concurrent cold
                                 loads block each other?
  C. eviction storm            -- every image model at once (way over VRAM);
                                 watch LRU eviction churn + any 503s.
  D. cold-load-blocks-loop     -- THE probe. Start one heavy cold load, then
                                 0.25s later hammer an already-hot light
                                 model. If the hot requests' latencies inflate
                                 to ~the cold-load duration, the synchronous
                                 director.acquire blocked the event loop.
  E. thundering herd           -- one request to every model simultaneously.

Outputs: results.json (all per-request records + per-scenario summaries),
loaded_snapshots.json (loaded set before/after each scenario), and binary
artifacts (PNG/WAV) under artifacts/. Analysis prose is written separately
into RESULTS.md after inspecting the data.

Run:  python examples/concurrent/stress_test.py [SERVER_URL]
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import struct
import sys
import time
from pathlib import Path

import httpx

SERVER = sys.argv[1] if len(sys.argv) > 1 else "http://192.168.0.204:8000"
HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
ART.mkdir(parents=True, exist_ok=True)

REQ_TIMEOUT = 180.0  # per request; cold image loads under contention can be slow

# ------------------------------------------------------------------ specs
# Each model maps to a request spec: (path, json_body, kind). kind drives
# response handling. Bodies are small on purpose so inference is cheap and
# the concurrency signal dominates.

def _img(model, size="512x512", extra=None):
    body = {"model": model, "prompt": "a red fox in a snowy forest",
            "size": size, "response_format": "b64_json"}
    if extra:
        body.update(extra)
    return ("/v1/images/generations", body, "image")

MODELS: dict[str, tuple] = {
    "qwen3.5-4b-q4": ("/v1/chat/completions",
        {"model": "qwen3.5-4b-q4",
         "messages": [{"role": "user", "content": "Reply with one short sentence about foxes."}],
         "max_tokens": 48, "temperature": 0.0}, "chat"),
    "qwen3-embedding-0.6b": ("/v1/embeddings",
        {"model": "qwen3-embedding-0.6b", "input": ["the quick brown fox", "a lazy dog"]}, "embed"),
    "bge-reranker-v2-m3": ("/v1/rerank",
        {"model": "bge-reranker-v2-m3", "query": "capital of France",
         "documents": ["Paris is the capital of France.", "Bananas are yellow."]}, "rerank"),
    "text-moderation": ("/v1/moderations",
        {"model": "text-moderation", "input": "I love sunny days in the park."}, "moderation"),
    "bart-large-cnn": ("/v1/summarize",
        {"model": "bart-large-cnn",
         "text": ("The quick brown fox jumps over the lazy dog. " * 12),
         "length": "short"}, "summarize"),
    "kokoro-82m": ("/v1/audio/speech",
        {"model": "kokoro-82m", "input": "Concurrency stress test in progress.",
         "voice": "af_heart"}, "tts"),
    "sd-turbo": _img("sd-turbo"),
    "sdxl-turbo": _img("sdxl-turbo"),
    "pixel-art-xl": _img("pixel-art-xl", size="768x768",
                         extra={"prompt": "pixel art, a red fox", "lora_scale": 1.0}),
    "sd_pixelart_spritesheet_generator": _img("sd_pixelart_spritesheet_generator"),
    "all-in-one-pixel-model": _img("all-in-one-pixel-model"),
}

LIGHT = ["qwen3-embedding-0.6b", "bge-reranker-v2-m3", "text-moderation",
         "bart-large-cnn", "kokoro-82m"]
IMAGE = ["sd-turbo", "sdxl-turbo", "pixel-art-xl",
         "sd_pixelart_spritesheet_generator", "all-in-one-pixel-model"]

records: list[dict] = []
snapshots: dict[str, dict] = {}


def _png_dims(raw: bytes):
    if raw[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    w, h = struct.unpack(">II", raw[16:24])
    return f"{w}x{h}"


async def snapshot_loaded(client: httpx.AsyncClient, label: str):
    try:
        r = await client.get(f"{SERVER}/v1/models", timeout=10.0)
        data = r.json()["data"]
        loaded = sorted(m["id"] for m in data if m.get("loaded"))
    except Exception as e:  # noqa: BLE001
        loaded = [f"<error: {e}>"]
    snapshots[label] = {"loaded": loaded, "t": time.time()}
    return loaded


async def fire(client, scenario, model, idx, t0):
    path, body, kind = MODELS[model]
    dispatch = time.perf_counter() - t0
    start = time.perf_counter()
    rec = {"scenario": scenario, "model": model, "idx": idx,
           "dispatch_offset_s": round(dispatch, 4)}
    try:
        r = await client.post(f"{SERVER}{path}", json=body, timeout=REQ_TIMEOUT)
        latency = time.perf_counter() - start
        rec["latency_s"] = round(latency, 3)
        rec["status"] = r.status_code
        rec["ok"] = r.is_success
        if not r.is_success:
            try:
                rec["note"] = r.json().get("error", {}).get("code", r.text[:80])
            except Exception:  # noqa: BLE001
                rec["note"] = r.text[:80]
            return rec
        if kind == "image":
            b64 = r.json()["data"][0]["b64_json"]
            raw = base64.b64decode(b64)
            fn = ART / f"{scenario}_{model}_{idx}.png"
            fn.write_bytes(raw)
            rec["note"] = f"png {_png_dims(raw)} {len(raw)}B"
            rec["out"] = fn.name
        elif kind == "tts":
            raw = r.content
            fn = ART / f"{scenario}_{model}_{idx}.wav"
            fn.write_bytes(raw)
            rec["note"] = f"wav {len(raw)}B"
            rec["out"] = fn.name
        elif kind == "chat":
            c = r.json()["choices"][0]["message"]["content"]
            rec["note"] = f"chat {len(c)}chars"
        elif kind == "embed":
            d = r.json()["data"]
            rec["note"] = f"{len(d)}x{len(d[0]['embedding'])}"
        elif kind == "rerank":
            res = r.json()["results"]
            rec["note"] = f"top={res[0]['index']} s={res[0]['relevance_score']:.3f}"
        elif kind == "moderation":
            rec["note"] = f"flagged={r.json()['results'][0]['flagged']}"
        elif kind == "summarize":
            rec["note"] = f"summary {len(r.json()['summary'])}chars"
    except Exception as e:  # noqa: BLE001
        rec["latency_s"] = round(time.perf_counter() - start, 3)
        rec["status"] = None
        rec["ok"] = False
        rec["note"] = f"EXC {type(e).__name__}: {str(e)[:80]}"
    return rec


def summarize_batch(scenario, batch, wall):
    lat = [b["latency_s"] for b in batch if b.get("latency_s") is not None]
    ok = sum(1 for b in batch if b.get("ok"))
    seq_sum = round(sum(lat), 2)
    return {
        "scenario": scenario, "n": len(batch), "ok": ok, "failed": len(batch) - ok,
        "wall_s": round(wall, 2), "sum_latency_s": seq_sum,
        "concurrency_factor": round(seq_sum / wall, 2) if wall > 0 else None,
        "min_latency_s": min(lat) if lat else None,
        "max_latency_s": max(lat) if lat else None,
    }


summaries: list[dict] = []


async def run_batch(client, scenario, calls):
    """calls: list of (model, idx). Fire all concurrently, one shared t0."""
    await snapshot_loaded(client, f"{scenario}:before")
    t0 = time.perf_counter()
    batch = await asyncio.gather(*[fire(client, scenario, m, i, t0) for (m, i) in calls])
    wall = time.perf_counter() - t0
    records.extend(batch)
    s = summarize_batch(scenario, batch, wall)
    summaries.append(s)
    await snapshot_loaded(client, f"{scenario}:after")
    print(f"[{scenario}] n={s['n']} ok={s['ok']} wall={s['wall_s']}s "
          f"sum={s['sum_latency_s']}s conc={s['concurrency_factor']}x", flush=True)
    return batch


async def main():
    print(f"stress test against {SERVER}", flush=True)
    limits = httpx.Limits(max_connections=60, max_keepalive_connections=60)
    async with httpx.AsyncClient(limits=limits) as client:
        await snapshot_loaded(client, "start")

        # --- warm chat, then A: same-model concurrency (6 concurrent) ---
        await fire(client, "warmup", "qwen3.5-4b-q4", 0, time.perf_counter())
        await run_batch(client, "A_same_model_concurrent",
                        [("qwen3.5-4b-q4", i) for i in range(6)])
        # sequential control: same 6, one at a time
        t0 = time.perf_counter()
        seq = []
        for i in range(6):
            seq.append(await fire(client, "A_same_model_sequential", "qwen3.5-4b-q4", i, t0))
        wall = time.perf_counter() - t0
        records.extend(seq)
        summaries.append(summarize_batch("A_same_model_sequential", seq, wall))
        print(f"[A_same_model_sequential] wall={round(wall,2)}s "
              f"sum={round(sum(r['latency_s'] for r in seq),2)}s", flush=True)

        # --- B: different light models, all at once ---
        await run_batch(client, "B_light_models_concurrent",
                        [(m, 0) for m in LIGHT + ["qwen3.5-4b-q4"]])

        # --- C: eviction storm, all image models at once ---
        await run_batch(client, "C_eviction_storm",
                        [(m, 0) for m in IMAGE])

        # --- D: cold-load-blocks-loop probe ---
        # warm kokoro so it is hot, pick a cold image target
        await fire(client, "warmup", "kokoro-82m", 0, time.perf_counter())
        loaded = await snapshot_loaded(client, "D_pick:loaded")
        cold_target = next((m for m in ["sdxl-turbo", "sd-turbo", "pixel-art-xl"]
                            if m not in loaded), "sdxl-turbo")
        print(f"[D] cold target = {cold_target}; kokoro hot", flush=True)
        await snapshot_loaded(client, "D_cold_load_probe:before")
        t0 = time.perf_counter()
        cold_task = asyncio.create_task(fire(client, "D_cold_load_probe", cold_target, 0, t0))
        await asyncio.sleep(0.25)  # let the cold acquire get underway
        hot = await asyncio.gather(
            *[fire(client, "D_cold_load_probe", "kokoro-82m", i, t0) for i in range(1, 7)])
        cold_rec = await cold_task
        wall = time.perf_counter() - t0
        batch = [cold_rec] + hot
        records.extend(batch)
        summaries.append(summarize_batch("D_cold_load_probe", batch, wall))
        await snapshot_loaded(client, "D_cold_load_probe:after")
        print(f"[D] cold({cold_target})={cold_rec.get('latency_s')}s "
              f"hot_kokoro=[{','.join(str(h.get('latency_s')) for h in hot)}]s", flush=True)

        # --- E: thundering herd, one request to every model ---
        await run_batch(client, "E_thundering_herd",
                        [(m, 0) for m in MODELS])

        await snapshot_loaded(client, "end")

    (HERE / "results.json").write_text(json.dumps(
        {"server": SERVER, "records": records, "summaries": summaries}, indent=2))
    (HERE / "loaded_snapshots.json").write_text(json.dumps(snapshots, indent=2))
    print(f"\nwrote {len(records)} records to results.json", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
