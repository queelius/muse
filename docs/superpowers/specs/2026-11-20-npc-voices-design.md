# NPC voices: designed and saved voices for audio/speech

Date: 2026-11-20. Status: approved approach, design for review.

## Intent

The owner is building a retro RPG with many NPCs, each needing a lot of spoken
lines. Each NPC should keep one recognisable voice across every line, across
sessions, and from any client. The voice should be invented from a text
description ("a gruff, elderly dwarf blacksmith, gravelly, slight Scottish
lilt"), not picked from a fixed preset list.

A throwaway spike (Qwen3-TTS on the RTX 3060, 3 NPCs x 5 lines) showed:

- VoiceDesign from a description alone already sounds good and mostly
  consistent (same-NPC x-vector cosine mean 0.972, worst 0.945).
- A fixed seed does not help (0.966 / 0.929).
- Designing once and cloning every later line from that one take with the
  0.6B Base model is the tightest (0.987 / 0.981) and cheaper (2.5 GB peak vs
  4.4 GB).
- About 10-11 s per short line with no flash-attn; fine for pre-rendering.

The owner listened and agreed the voices work well, and chose design-then-clone.

## Goals

1. Speak a line from a description: `POST /v1/audio/speech` with an
   `instructions` field (the OpenAI `gpt-4o-mini-tts` field name) on a
   VoiceDesign model.
2. Save a voice: upload a reference clip plus its transcript under a name
   (`grimbold`) to a clone-capable model. The clip is usually a take from step
   1, but any recording works.
3. Speak with a saved voice: `POST /v1/audio/speech` with `voice: "grimbold"`.
4. List saved voices with the existing `GET /v1/audio/speech/voices`; delete one.
5. Saved voices persist on the server and survive restarts and evictions.

## Non-goals

- Qwen3-TTS CustomVoice (9 preset speakers + instruct). Could be added later.
- The 1.7B Base model (clone quality step-up). Can be a later curated alias
  of the same script.
- A "give me N candidate takes" route. Auditioning is just calling speech
  again; the client keeps the take it likes.
- Per-line emotion on cloned voices. The Base model has no instruct input;
  emotional one-offs go through the VoiceDesign model with `instructions`.
- Sound effects (Stable Audio 3 small-sfx) and music: separate follow-up specs.
- Speed-ups (flash-attn, compile). Note only.

## Anti-goals

- A saved voice that silently sounds different later (e.g. the reference is
  re-encoded lossy, or the clone prompt depends on unpinned state).
- Uploads that can escape the voices directory or fill the disk.
- Breaking existing speech models: Kokoro, Soprano, Bark and Supertonic must
  behave exactly as before for requests that do not use the new fields.

## Approaches considered

1. **Description on every line** (`instructions` only). Simplest, but nothing
   guarantees the voice holds over hundreds of lines; the spike measured more
   drift. Kept as goal 1, not as the consistency mechanism.
2. **Design once, then clone (chosen).** VoiceDesign model to audition, Base
   model to clone a saved reference. Tightest in the spike, smaller model on the
   hot path, and the save route is useful for any recording.
3. **CustomVoice presets.** Most stable, but only 9 voices; a cast of NPCs
   needs more and the owner wants voices from descriptions.

## Design

### Models (two bundled scripts, one shared runtime)

- `qwen3-tts-1.7b-voicedesign`: `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`.
  Takes `instructions`; without them it uses a fixed neutral default
  description (so the probe and plain calls work). `voice` is accepted and
  ignored (OpenAI SDKs always send it); the response reports no voice.
  Calls `generate_voice_design(text, instruct, language)`.
- `qwen3-tts-0.6b-base`: `Qwen/Qwen3-TTS-12Hz-0.6B-Base`. `voice` names a
  saved voice; omitted, it uses the bundled default voice `narrator` (below).
  Refuses `instructions`. Calls
  `create_voice_clone_prompt(ref_audio, ref_text)` once per voice (cached in
  memory, keyed by name plus the stored files' identity) and
  `generate_voice_clone(text, language, voice_clone_prompt=...)`.
- Both: license Apache-2.0, pip extra `qwen-tts` (plus what it needs), revision
  pinned to a commit, `device: "auto"` (GPU strongly preferred), bf16 on cuda,
  `memory_gb` from the spike (about 5 and 3 with headroom). Loaded with
  `Qwen3TTSModel.from_pretrained(local_dir, device_map=..., dtype=...)`.
- The shared runtime goes in `audio_speech/runtimes/qwen3_tts.py`; each script
  aliases it and sets its mode in `capabilities`.
- Streaming yields the whole line as one chunk (no native streaming).
- SoX is optional (a warning only); not a system package. flash-attn is not
  installed.

### Wire changes

- `SpeechRequest.instructions: str | None` (max 4096 chars). New capability
  `supports_instructions`. Sent to a model without it: 400
  `instructions_not_supported`. Forwarded to
  `synthesize` / `synthesize_stream` as `instructions=` only when set, so old
  backends never see it.
- New typed error `UnknownVoiceError` in `audio_speech/protocol.py`; the speech
  route maps it to 400 `voice_not_found` (with up to a few known names).
  The Base model raises it for a missing or unknown voice.
- `POST /v1/audio/speech/voices` (multipart): `model`, `name`, `audio` (file),
  `transcript`, optional `overwrite` (default false). Only on models with
  capability `supports_saved_voices`; others 400 `voice_saving_not_supported`.
  Returns 200 `{"object": "audio.voice", "model", "name", "duration_seconds",
  "sha256"}` (sha256 of the stored WAV). 409 `voice_exists` without overwrite.
- `DELETE /v1/audio/speech/voices/{name}?model=...`: 200
  `{"deleted": true, ...}`, 404 `voice_not_found`.
- `GET /v1/audio/speech/voices` unchanged; for the Base model, `voices` is a
  property re-read from disk: `narrator` plus the saved names.
- `timestamps`, `align` and IPA pronunciations keep their existing capability
  gates (these models declare none of them); respellings work as for any model.
- Gateway: `extract_model_from_request` also reads the `model` query param on
  `DELETE` (today only `GET`), so the delete route reaches the worker.
- DISCOVERY gets both new routes (MCP tool `muse_save_voice`, `audio` is a
  file field; amended at planning: the delete route has no MCP tool, since the
  MCP generator has no DELETE or path-parameter support); docs updated.

### Voice storage

- Ownership: a small `VoiceStore` helper in `audio_speech/voices.py` does
  naming, layout, atomic writes, listing and deletion. The model exposes
  `save_voice(name, audio, sample_rate, transcript, overwrite)`,
  `delete_voice(name)` and the `voices` property, built on the store. The route
  does upload reading, decoding and validation, then calls the model.
- Bundled default voice `narrator`: a short (about 4 s) reference clip made once
  with the VoiceDesign model from a neutral description, committed as package
  data with its transcript. It cannot be overwritten or deleted (400
  `voice_read_only`). It also gives the probe and smoke test a voice to clone.

- `voices_dir(model_id) / <name> / {reference.wav, voice.json}`; `voice.json`
  holds transcript, duration, sha256, created time, source filename.
- Name: `^[a-z0-9][a-z0-9_-]{0,63}$`. Anything else: 400 `invalid_voice_name`.
- Upload bounded by the existing upload helpers; decoded with
  `_common/audio_decode` to mono 24 kHz (amended at planning: `decode_audio`
  needs a target rate, and 24 kHz is what the model consumes) and stored as
  PCM16 WAV. Stored once, never re-encoded. Duration 1-30 s, else 400
  `invalid_reference_audio`. Transcript 1-1000 chars.
- Writes go to a temp dir in the same parent and are renamed into place
  (overwrite replaces atomically). Use `muse.core.securefs` for the
  owned-directory and no-follow checks where they fit.
- Voices are per model id: a voice saved on `qwen3-tts-0.6b-base` is not
  visible to another model (a later 1.7B Base alias has its own set).
- Count cap: new setting `limits.max_saved_voices` (default 200 per model;
  about 1.5 MB each at most). Over it: 409 `voice_limit_reached`.
- No auth: saving and deleting are open like the other inference routes. The
  server is LAN-bound and the routes write only inside the voices directory
  under the cap. (Owner may override: gate on the admin token.)
- Load: the clone-prompt cache is keyed by name plus `voice.json` and
  `reference.wav` stat identity; files are trusted, not re-hashed. An unreadable
  or malformed voice directory is logged and returned as 500
  `voice_unreadable` without paths.

### Language

Requests carry no language today. The runtime passes the model's automatic
language mode if qwen-tts supports one, else `"English"`; a `language`
request field is out of scope.

## Constraints

- Follow CLAUDE.md conventions: deferred heavy imports, `Model` alias,
  `runtime_helpers` for device and dtype, typed OpenAI error envelopes, tests
  with fakes and mocked `qwen_tts` (no weights in CI), worktree tests with
  `PYTHONPATH=$PWD/src`.
- Do not touch the owner's running server on :8000 or `~/.muse` from tests.
- Real-model verification on the 3060 happens with an isolated
  `MUSE_CATALOG_DIR`, not the owner's catalog.

## Left to the builder

- The exact `qwen-tts` revision pins and pip pins.
- Whether qwen-tts accepts a numpy reference directly or needs a path.
- Clone-prompt cache size (a small LRU is enough).
- Where in the docs (MODALITIES.md audio/speech section, CLAUDE.md speech
  bullet) the new fields are described.
