# Narration: word timings, pronunciations, a stronger listener

Date: 2026-10-23. Status: draft for owner review.

## Why

The owner's YouTube pipeline (`~/github/youtube-channel`, read-only to muse
work) narrates reels and Manim videos with Kokoro through muse. Two problems
reach viewers:

1. **Words said wrong.** Kokoro guesses unusual words, names and initialisms.
   The pipeline patches the text before sending it (`engine/companion/lexicon.yaml`
   respellings and Kokoro `[word](/ipa/)` links, plus single-letter spelling
   in `speech_norm.py`), then listens back with `whisper-base` and keeps the
   best take. `whisper-base` is weak and tends to hear the word it expects,
   so wrong words slip through.
2. **Speech and on-screen text or animation out of step.** Recon on
   2026-10-23 found the cause is mostly in the pipeline, not muse:
   - No reel uses measured timing. 31 of 85 `reel.yaml` files set
     `align: false` and the rest inherit the default `False`
     (`prosody.Pacing.align`), so `pacing.analyze_beat` spreads each beat's
     time evenly over its words. Captions and shot cuts drift wherever the
     speaking rate changes.
   - Manim scenes (1,818 `wait_until_bookmark` calls) do align, through muse
     forced alignment in `speech.py`, but when that call fails they silently
     fall back to even spacing. `cartoon/marks.py` always uses even spacing.
     The pipeline's own 2026-10-15 process review flagged this.

Muse can remove the need for a separate alignment pass for Kokoro speech,
carry pronunciations so the text sent is the text shown, record exactly which
voice and fixes produced each take, and provide a better listener. The
pipeline changes needed to use this are listed at the end as a handoff; muse
work does not edit that repo.

## Success (the owner's bar)

The project is done when, after the pipeline adopts it:

1. On a test reel, every caption and every shot cut lands within 0.1 s of
   the word it belongs to.
2. A render never falls back to even spacing without saying so.
3. Every entry in `lexicon.yaml` is heard correctly by the stronger listener
   when spoken by the voice it was tuned for.

Muse-side acceptance, which muse work can verify on its own:

- On a set of real narration beats, Kokoro's returned word start times agree
  with the Qwen3 forced aligner's within 0.1 s for at least 95% of words, and
  every disagreement is listed with its audio so the owner can listen. (My
  call: the aligner is the independent check, but it can itself be wrong, so
  disagreements are reported rather than treated as failures.)
- A lexicon check script (below) runs every `lexicon.yaml` entry through
  Kokoro with its pronunciation and the new listener, and reports each as
  heard or not heard.

**Kokoro's times are predictions, so check them first.** Kokoro's
timestamps come from its duration predictor before the audio exists. The
audio is rendered from those same durations, so they should match closely,
but the start offset is a heuristic (upstream carries `TODO: Is -3 an
appropriate offset?`) and nobody has measured the result against real
audio. The first piece of work is therefore a throwaway agreement test on
real beats, before any API work. (My calls, from here.)

- If Kokoro meets the 95% / 0.1 s bar: build as designed.
- If it misses, a constant or per-chunk offset correction is allowed,
  recorded in the response (`timing.calibration`) and in the docs. Words
  timed this way still count as `measured: true`, because the time comes
  from the model's own durations for this take, not from spreading time
  evenly.
- If it still misses after correction: stop and bring the numbers to the
  owner. The fallback would be running muse's aligner behind the speech
  route, which this design has deliberately left out.

## Goals

1. `/v1/audio/speech` can return word timings with the audio, measured by
   the model that spoke, keyed to the words as written in `input`.
2. `/v1/audio/speech` accepts pronunciations in the request, so `input` is
   the display text and audio-only changes travel separately.
3. Every speech response records what produced it: model, weights revision,
   voice, speed, the audio's fingerprint, and the pronunciations applied.
4. Qwen3-ASR-1.7B served on `/v1/audio/transcriptions` as the listener.
5. A small speech comparison script for auditioning voices and checking the
   lexicon.
6. Soprano stops being featured; it stays pullable.

## Non-goals (now)

- **Voice cloning.** The owner wants to clone their own voice later; it gets
  its own design. Nothing here should block a later voice registry.
- **New voice models.** Kokoro remains the voice. The comparison script is
  for shopping later, not for adding models now. Candidates noted on
  2026-10-23 for that later look: Fish Audio Drama 3 (community license,
  free under $500k revenue), Ming-3.0-tts 0.5B (MIT, released 2026-10-20),
  Qwen3-TTS 0.6B/1.7B (Apache 2.0), Chatterbox (MIT), VoxCPM2 (Apache 2.0),
  Step Audio EditX (Apache 2.0, tight on 12 GB).
- **Timings for models that cannot measure their own.** Muse will not run
  its aligner behind the speech route. A model without native timings
  refuses the request (below); the client can call `/v1/audio/alignments`
  itself. (My call: a speech worker calling another model's worker is new
  architecture, and Kokoro is the only voice in use.)
- **Timings while streaming.** `stream: true` with timestamps is refused.
- Editing youtube-channel or companion; using the A100s.

## Anti-goals

- Estimated timing presented as measured. Any word whose time muse did not
  measure is marked as such; nothing is quietly spread evenly.
- Breaking existing clients. A request without the new fields gets exactly
  today's response bytes and content type (OpenAI SDK clients unchanged).
- Timings separable from their audio. Timings are only ever returned in the
  same response as the audio they describe, alongside its fingerprint.

## Constraints

- Runs on the 12 GB RTX 3060 host. Kokoro is small; Qwen3-ASR-1.7B at bf16
  is roughly 4 GB.
- Muse conventions apply (CLAUDE.md): deferred heavy imports, OpenAI error
  envelope, capability flags gate behavior, tests use isolated catalogs and
  never touch `~/.muse`, model ids stay stable for given weights.
- Kokoro in the model venv is `kokoro 0.9.4` / `misaki 0.9.4`. Its English
  pipeline already computes per-token `start_ts` / `end_ts` in seconds
  (`KPipeline.join_timestamps`, from the duration predictor), relative to
  each text chunk it splits the input into. Muse's Kokoro script currently
  discards them.

## Design

### 1. Speech with timestamps

New optional request fields on `POST /v1/audio/speech`:

- `timestamps: bool = false`
- `pronunciations: list | null` (section 2)

`timestamps: false` (default): unchanged audio response. It gains headers
(section 3) but the body and content type are the same.

`timestamps: true`: the response is JSON instead of audio:

```json
{
  "object": "audio.speech",
  "model": "kokoro-82m",
  "voice": "af_heart",
  "speed": 1.0,
  "audio": {
    "format": "wav",
    "b64_json": "<base64>",
    "sha256": "<hex of the decoded audio bytes>",
    "sample_rate": 24000,
    "duration_seconds": 12.84
  },
  "words": [
    {"word": "Coulomb's", "char_start": 0, "char_end": 9,
     "start": 0.112, "end": 0.588, "measured": true}
  ],
  "timing": {"source": "native", "measured_words": 31, "estimated_words": 0},
  "pronunciations": {"digest": "<hex>", "applied": [
    {"term": "coulomb", "kind": "ipa", "count": 1}
  ]}
}
```

Rules:

- **One entry per whitespace-separated word of `input`.** `len(words)`
  equals `len(input.split())`, in order, with `char_start`/`char_end`
  offsets into `input`. The pipeline already splits text this way
  everywhere, so it can zip timings onto its own words without matching.
  Punctuation stays attached to its word.
- **Times are seconds from the start of the returned audio**, across all of
  Kokoro's internal chunks (chunk offsets accumulated from the audio
  actually concatenated).
- **`measured: false`** marks a word Kokoro gave no time for (for example a
  token that could not be mapped back to the input). Its time is placed
  between its measured neighbours, and `timing.estimated_words` counts
  these. The caller decides whether that is acceptable.
- **Models without native timings** (`capabilities.supports_word_timestamps`
  false or absent) return 400 `timestamps_not_supported`, pointing at
  `/v1/audio/alignments`.
- `timestamps: true` with `stream: true` returns 400
  `timestamps_not_streamable`.
- `response_format` still picks the encoding inside `audio` (`wav` or
  `opus`).

The existing `align` flag and `X-Alignment` header (Soprano paragraph
timings) keep working unchanged. (My call: removing them breaks a working
feature for no gain.) The speech DISCOVERY summary, which wrongly says
`align` gives word timings, is corrected to describe `timestamps`.

MCP: the generated `muse_speak` tool must write the audio file in both cases
(raw bytes, or `audio.b64_json` in the JSON response) and return the words
and provenance. Declaring `audio.b64_json` as a `media_fields` path is the
expected route; the builder confirms the generator handles a route whose
response type depends on a request field.

### 2. Pronunciations in the request

`pronunciations` is a list of entries. Each entry has exactly one target and
exactly one replacement:

- target `term`: a word or phrase, matched whole-word and case-insensitively,
  internal whitespace flexible, longest term first, one left-to-right pass
  (the same rules as the pipeline's `apply_lexicon`, so `lexicon.yaml` maps
  over directly);
- or target `spans`: a list of `[char_start, char_end]` ranges in `input`,
  for occurrence-specific fixes (the pipeline's single-letter rule is
  context-dependent: "a" the article versus "a" the variable);
- replacement `ipa`: a phoneme string in the model's own notation, passed
  through verbatim (for Kokoro that is misaki's phoneme set, used as a
  `[original](/phonemes/)` link);
- or replacement `respell`: plain text the model reads instead.

Example:

```json
"pronunciations": [
  {"term": "coulomb", "ipa": "kuːˈlɒm"},
  {"term": "runge kutta", "respell": "Roong-uh Koo-tuh"},
  {"spans": [[14, 15]], "respell": "eigh"}
]
```

Behavior:

- Timings stay keyed to the original words. A respelling that turns one
  word into several spoken tokens ("monoid" to "mah noyd") reports one entry
  for "monoid" spanning all of them; a phrase entry spanning several written
  words reports each written word, their times divided inside the measured
  phrase span and flagged `measured: false` only if no finer measurement
  exists.
- Capability `pronunciation_kinds` lists what a model accepts (Kokoro:
  `["ipa", "respell"]`). `respell` works on any speech model, since muse
  substitutes the text before synthesis. `ipa` on a model that does not list
  it returns 400 `pronunciation_not_supported`, naming the entry.
- A `spans` entry wins over a `term` match that overlaps it. That
  occurrence of the term is skipped, and its count in `applied` excludes
  it. (My call: a span names one exact occurrence, so it is the more
  specific instruction.) Spans may not overlap each other.
- Invalid entries (both or neither of `term`/`spans`, both or neither of
  `ipa`/`respell`, overlapping or out-of-range spans) return 400
  `invalid_pronunciation` naming the entry index.
- `pronunciations.digest` is the SHA-256 of the canonical JSON of the list
  as sent (sorted keys, no whitespace), so the pipeline can pin which list
  produced a take. `applied` reports each entry that matched, with a count;
  entries that matched nothing are listed with count 0, since an unused fix
  is worth noticing.

### 3. Provenance on every speech response

Every `/v1/audio/speech` response, with or without timestamps, carries:

- existing `X-Muse-Model`, `X-Muse-Model-Revision`, `X-Muse-Model-Artifacts`;
- new `X-Muse-Voice`, `X-Muse-Speed`;
- new `X-Muse-Audio-SHA256` (of the response body when it is audio; omitted
  on streamed responses, whose bytes are not known when headers go out);
- new `X-Muse-Pronunciations-Digest` when pronunciations were sent.

The JSON body repeats these fields (section 1). The fingerprint and
`audio.format` describe the bytes actually sent: if a requested encoding
falls back (today `opus` can fall back to WAV), they say WAV.

The fingerprint identifies the take muse returned. The pipeline converts
that WAV to MP3 and joins several takes into one beat, so the file in a
render never has this hash. The pipeline records muse's hash for each take
it received and carries the timings through its own joins. Muse does not
add an MP3 output. (My call. MP3 encoders also add a short delay at the
start, about 25 ms for LAME, which the pipeline should allow for when it
offsets timings.)

Why it matters: every
pronunciation fix is tuned to one model and voice, and every timing belongs
to one exact take. With these recorded, the pipeline can tell which lexicon
entries were never checked against a new voice, and can refuse timings
whose fingerprint does not match the audio in a render.

### 4. Qwen3-ASR as the listener

- Curated `qwen3-asr-1.7b` pointing at `Qwen/Qwen3-ASR-1.7B-hf` (Apache 2.0,
  Transformers-native checkpoint; pin the revision at implementation time),
  served on the existing `/v1/audio/transcriptions` route through a new
  Transformers runtime in `audio_transcription/runtimes/`, claimed by an
  exact-repo HF plugin entry (the Qwen3 ForcedAligner plugin is the
  pattern).
- It returns transcripts only. Requests for word timestamps return 400
  `word_timestamps_not_supported` (use `/v1/audio/alignments`).
- The route's `prompt` field is passed to the model as context only when
  the client sends it. Its notes must say that a pronunciation check should
  not send one, because context biases the model toward hearing the expected
  words, which is the failure being fixed.
- Model notes (summary, strengths, limits, quality tier) in curated, as
  agent discovery requires.

### 5. Speech comparison script

`scripts/bench/speech_compare.py`, standalone like
`scripts/bench/image_compare.py` (httpx + yaml, writes `results.json` and an
`index.html` with audio players):

- input: a case file of lines (id, text, optional pronunciations), or
  `--lexicon PATH` to build one case per `lexicon.yaml` entry with the term
  in a short sentence (read-only on the pipeline's file);
- targets: one or more `model:voice` pairs on a muse server;
- per take: the audio, the listener's transcript, word error rate against
  the text, whether each lexicon term was heard, UTMOS score, and (for
  models with timestamps) the agreement report against
  `/v1/audio/alignments` used for acceptance.

**Lexicon check: what counts as heard.** After normalizing case, accents and
punctuation, the entry counts as heard if the listener's transcript contains
either:

- the written term (for example "Coulomb", "Blaise Pascal"), or
- the replacement's words, where the replacement is ordinary words (`tanh` →
  "hyperbolic tangent", "exclusive or" → "exclusive, or").

An optional case-file `accept` list adds other acceptable transcriptions
(for example "Jason" for `json`). Respellings such as "mah noyd" are not
expected to appear in the transcript, because the listener writes real
words. These checks run with voices `af_heart` and `am_puck`, the
pipeline's default voice and its most-used one, until the pipeline tags
entries with the voices they were tuned for. `--voices` overrides this.
(My calls.)

This is both the lexicon check in the success bar and the audition tool for
later voice shopping.

### 6. Soprano

- The README quickstart and examples switch from `soprano-80m` to
  `kokoro-82m`; CLAUDE.md's speech line leads with Kokoro.
- `soprano-80m` stays bundled and curated, `quality_tier: draft`. Its code
  stays, because no generic loader can run it.
- For the record: the weights are Eugene Kwek's `ekwek/Soprano-1.1-80M`, and
  muse began as a fork of his `ekwek1/soprano` repository; the Narro engine
  is that code, evolved inside muse. There is no separate repo of the
  owner's for it.

## Left to the builder

- How Kokoro tokens map back to input character offsets (misaki
  normalizes numbers and symbols; the mapping must survive that, or mark
  words `measured: false`).
- Whether Kokoro's leading and trailing silence shifts timestamps against
  the concatenated audio, and correcting for it.
- The Transformers version floor for Qwen3-ASR and its memory estimate
  (verify by probe on the 3060).
- How Qwen3-ASR answers `verbose_json`, `srt` and `vtt`, which expect
  segment times it cannot give: refuse or flatten, consistently.
- Kokoro's timestamps exist only on its English pipeline. Gate them per
  voice or language accordingly.
- Test fixtures, error message wording, and file layout within the
  conventions.

## Pipeline handoff (for the owner; not muse work)

1. Reels: request `timestamps: true` and feed the words into
   `pacing.analyze_beat` instead of estimating; or at least set
   `align: true`.
2. Manim `speech.py`: use speech timestamps for bookmarks; make the
   even-spacing fallback loud (a warning in the render log and the build
   record) instead of silent. `cartoon/marks.py` likewise.
3. Send display text as `input` and move `lexicon.yaml` respellings, IPA
   links and `speech_norm` single-letter spellings into `pronunciations`
   (occurrence-specific ones as `spans`).
4. Tag each `lexicon.yaml` entry with the model and voice it was tested
   with; all existing entries are Kokoro.
5. Store voice, speed, muse's audio SHA-256 and the pronunciation digest
   for each take in frozen speech records. Keep each take's timings with
   that take through MP3 conversion and joins, allowing for the encoder's
   start delay. Refuse a take whose timings belong to a different take.
6. Switch the check-back listener to `qwen3-asr-1.7b`, without `prompt`.
7. Earlier items still open: `.mcp.json`, `MUSE_BASE_URL` (the Whisper URL is
   a hard-coded constant in `checker.py` and `align.py`), lift the gpt-only
   reference-image gate in `image.py:84`.

## Future

- Voice cloning of the owner's voice, with a voice registry (named voices
  with reference audio and settings, stable across episodes and models).
- Auditioning and adding better voice models (candidates above).
- Timings for models without native timings, via muse's aligner.
- Timings while streaming.
