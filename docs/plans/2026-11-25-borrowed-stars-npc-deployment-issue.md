# Issue / Operator Handoff: Deploy Borrowed Stars v2.1 NPC SLMs & Real-Time Voice Warmup on Muse (`192.168.0.204:8000`)

**Date:** 2026-11-25  
**Source:** `borrowed-stars` game server session (`0333f6c7`) + `borrowed-stars-npc` handoff  
**Target Host:** Muse server at `http://192.168.0.204:8000`

---

## Summary

The `muse` repo (`main` @ `4fdf12b`) already includes:
- `/v1/completions`, `/completion` alias, and **GBNF pass-through** (`fa5ec7b`), and
- Three curated Borrowed Stars v2.1 NPC GGUF entries in `src/muse/curated.yaml` (`4fdf12b`) with `parallel: 4`, `cont_batching: true`, `cache_prompt: true`, and `no_jinja: true` (for Qwen3.5):
  - `borrowed-stars-npc-qwen3.5-0.8b` (`hf://queelius/borrowed-stars-npc-qwen3.5-0.8b-32k-GGUF@q4_k_m`, ~328 MB)
  - `borrowed-stars-npc-lfm2.5-350m` (`hf://queelius/borrowed-stars-npc-lfm2.5-350m-GGUF@q4_k_m`, ~219 MB)
  - `borrowed-stars-npc-gemma3-1b` (`hf://queelius/borrowed-stars-npc-gemma3-1b-32k-GGUF@q4_k_m`, ~496 MB)

However, the **running** Muse instance at `http://192.168.0.204:8000` has **not** been updated/pulled yet:
- `GET http://192.168.0.204:8000/v1/models` currently exposes only `llama-3.2-3b-q4`, `qwen3.5-4b-q4`, `qwen3-0.6b-q4`, and `smolvlm-256m-instruct` for `chat/completion`.
- Remote `POST http://192.168.0.204:8000/v1/admin/models/_/pull` returns `401 Unauthorized (missing_token: Authorization: Bearer <token> required)`, so this needs to be run directly on the Muse host (or with the Muse admin token).

---

## Action Required on the Muse Server (`192.168.0.204`)

### 1. Update Muse to `main` (`>= 4fdf12b`) and Restart
```bash
git pull origin main
# restart the muse server so curated.yaml + /completion GBNF pass-through are active
```

### 2. Pull the 3 Curated Borrowed Stars v2.1 NPC Models
Once on `>= 4fdf12b`, pull by curated alias:
```bash
muse pull borrowed-stars-npc-qwen3.5-0.8b
muse pull borrowed-stars-npc-lfm2.5-350m
muse pull borrowed-stars-npc-gemma3-1b
```

*(Or via the HTTP Admin API with `$MUSE_ADMIN_TOKEN` — note the body field is `"identifier"`):*
```bash
for id in \
  borrowed-stars-npc-qwen3.5-0.8b \
  borrowed-stars-npc-lfm2.5-350m \
  borrowed-stars-npc-gemma3-1b; do
  curl -fsS -X POST "http://192.168.0.204:8000/v1/admin/models/_/pull" \
    -H "Authorization: Bearer $MUSE_ADMIN_TOKEN" \
    -H "Content-Type: application/json" \
    -d "{\"identifier\": \"$id\"}"
  echo
done
```

### 3. Verify Models, `no_jinja` (No `<think>` Traces), and GBNF `/completion`

1. Confirm all 3 models appear in `GET http://192.168.0.204:8000/v1/models`.
2. **Verify `borrowed-stars-npc-qwen3.5-0.8b` does NOT emit `<think>` traces:**
   - During live testing against `192.168.0.204:8000`, stock `qwen3.5-4b-q4` emitted `<think>\nThinking Process:...` until hitting `max_tokens` even when `/no_think` or `chat_template_kwargs: {"enable_thinking": false}` was sent.
   - `src/muse/curated.yaml` sets `no_jinja: true` on `borrowed-stars-npc-qwen3.5-0.8b`. Please confirm a smoke call to both `/v1/completions` (or `/completion` with `grammar` + `cache_prompt: true`) and `/v1/chat/completions` returns clean JSON `{"mood": ..., "line": ..., "calls": [...]}` with zero `<think>` leakage.
3. **Optional — Warm up `borrowed-stars-npc-qwen3.5-0.8b` (and `kokoro-82m` / `qwen3-tts-0.6b-base`):**
   - `borrowed-stars` is now also ready to wire real-time NPC voices via `/v1/audio/speech`:
     - Live timing on `192.168.0.204:8000`: `kokoro-82m` (`af_heart`) returned a 160 KB WAV in **2.79 s warm** (17.07 s cold).
     - `qwen3-tts-0.6b-base` (saved character voices like `wispwood_nettle`, `vela_tide_clerk`) **timed out at 20 s** on cold load.
   - Keeping `borrowed-stars-npc-qwen3.5-0.8b` (`0.6 GB`) + `kokoro-82m` (`0.5 GB`) (and optionally `qwen3-tts-0.6b-base` if VRAM allows) warm on Muse will keep both freeform NPC turns and live voice playback snappy.

---

## Follow-Up on `borrowed-stars` Once Ready

Once `borrowed-stars-npc-qwen3.5-0.8b` (and peers) show up in `GET http://192.168.0.204:8000/v1/models`, `borrowed-stars` will:
1. Update `genai.yaml` (`text_npc` → `borrowed-stars-npc-qwen3.5-0.8b`, optional `text_npc_level_1` → `borrowed-stars-npc-lfm2.5-350m`, `text_npc_level_3` → `borrowed-stars-npc-gemma3-1b`).
2. Wire `slm_bridge.py` to support Muse's `/completion` + dynamic per-turn GBNF pass-through (with `V2_SYSTEM` prefix and `cache_prompt: true`).
