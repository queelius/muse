# Narration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `/v1/audio/speech` returns Kokoro word timings keyed to the written input, accepts pronunciations, and records provenance on every take; Qwen3-ASR-1.7B serves `/v1/audio/transcriptions` as a stronger listener; a speech comparison script checks the pipeline's lexicon.

**Architecture:** Two pure modules under `audio_speech/` do the text work: `pronunciations.py` (validate, match, render the spoken text with a spoken-to-input character map, digest) and `word_timing.py` (turn timed character spans into one entry per written word). The Kokoro script reports timed character spans in the text it was given; the route maps them back to `input` and builds the JSON response. Qwen3-ASR is a new Transformers runtime claimed by an exact-repo HF plugin, like the Qwen3 ForcedAligner.

**Tech Stack:** Python 3.10+, FastAPI/pydantic, kokoro 0.9.4 / misaki 0.9.4 (model venv only), Transformers (Qwen3-ASR venv only), httpx + PyYAML (bench script).

**Spec:** `docs/superpowers/specs/2026-10-23-narration-design.md` (approved 2026-10-24). Read it before each task; it is the authority where this plan is silent.

## Global Constraints

- Requests without `timestamps` / `pronunciations` get exactly today's response body and content type (new headers are allowed). Do not pass new kwargs to `synthesize` in that case.
- Error envelope: `muse.core.errors.error_response(status, code, message)`; codes exactly `timestamps_not_supported`, `timestamps_not_streamable`, `pronunciation_not_supported`, `invalid_pronunciation`, `word_timestamps_not_supported`, `segment_timestamps_not_supported`.
- Deferred heavy imports (torch, transformers, kokoro, misaki) per CLAUDE.md; `muse --help` must not import them.
- Tests never touch `~/.muse`; use `tmp_path` / `MUSE_CATALOG_DIR`. Model tests mock kokoro / transformers.
- Read-only: `~/github/youtube-channel`, `~/github/companion`, the Codex worktree `~/.codex/worktrees/0fa8/muse`. Never print `MUSE_ADMIN_TOKEN`. Do not signal VirtualBox processes. Use only the local RTX 3060 (12 GB).
- Do not push, tag or release. Commit on the feature branch only, in the repo's message style (`feat(speech): ...`). Do not sweep unrelated dirty files into a commit.
- Fast lane must stay green: `pytest tests/ -q -m "not slow"`.

## Review Focus

1. A term inside a longer word (`json` in `jsonl`) or a term repeated several times: whole-word only, every occurrence counted. (Task 2 test `test_term_whole_word_and_counts`.)
2. Input with leading/trailing/multiple spaces, newlines, or punctuation-only tokens (`—`, `...`): `len(words) == len(input.split())` still holds and offsets slice back to the word. (Task 3 test `test_words_match_split_with_odd_whitespace`.)
3. Kokoro splitting long input into several chunks (two paragraphs separated by a blank line): times keep increasing across the chunk boundary and stay within `duration_seconds`. (Task 4 test `test_multi_chunk_offsets_accumulate`.)
4. `response_format: "opus"` with `timestamps: true` when the opus encoder is missing: `audio.format` is `wav`, sha256 matches the decoded bytes. (Task 5 test `test_timestamps_opus_fallback_reports_wav`.)
5. Hostile `pronunciations` (not a list, 10,000 entries, a 5,000-char term, spans with floats or negative numbers): 400 `invalid_pronunciation`, never 500. (Task 2 test `test_rejects_hostile_lists`.)

---

### Task 1: Kokoro timing agreement check (throwaway, gate for everything else)

**Files:**
- Create: `scripts/bench/kokoro_timing_check.py` (committed; small, documented as a one-off)
- Create: `docs/benchmarks/2026-10-24-kokoro-timing-agreement.md`

**Interfaces:**
- Produces: a measured constant `KOKORO_TIMING_OFFSET_SECONDS` (float, may be `0.0`) and whether per-chunk correction is needed, written in the benchmark doc's first lines, for Task 4.

- [ ] **Step 1: Write the check script.** It runs under the existing Kokoro venv (`~/.muse/venvs/kokoro-82m/bin/python`, read-only use; weights from that model's `local_dir` in `~/.muse/catalog.json`, read-only). For each beat text: run `KPipeline(lang_code="a", model=KModel(config, weights))(text, voice=<voice>)`, concatenate `result.audio`, and collect each `result.tokens` token with non-None `start_ts`, offsetting by the sample length of the audio already concatenated. Save the WAV, then POST it with the text to the live server's `/v1/audio/alignments` (`MUSE_SERVER`, default `http://localhost:8000`, model `qwen3-forced-aligner-0.6b`; this is an inference call only, no admin). Pair Kokoro word starts with aligner word starts in order (match on normalized word text; unmatched pairs are reported, not counted as agreement).
- [ ] **Step 2: Pick beats.** At least 40 beats totalling at least 600 words from `~/github/youtube-channel/content/**/reel.yaml` `beats` (read-only), mixed across at least 5 reels, using each reel's own voice (`voices:` map), plus 5 beats with `af_heart` and 5 with `am_puck`. Include at least 3 beats long enough (or containing a blank line) that Kokoro splits them into more than one chunk.
- [ ] **Step 3: Run and report.** Report: fraction of words with |Δstart| ≤ 0.1 s, median and p95 Δstart, the mean signed Δ (a constant offset candidate), the same numbers after subtracting that mean, and per-chunk-boundary behaviour. List every word with |Δ| > 0.1 s (beat id, word, both times, WAV path). Keep WAVs under the scratch output dir, not in the repo.
- [ ] **Step 4: Decide (record in the doc's first lines).**
  - ≥ 95% within 0.1 s uncorrected: `KOKORO_TIMING_OFFSET_SECONDS = 0.0`.
  - ≥ 95% only after a constant offset: record that offset (sign convention: added to Kokoro's times).
  - < 95% even after correction: write `STOP` as the first line with the numbers, commit, and end the task without starting Task 2. The controller brings this to the owner.
- [ ] **Step 5: Commit** `scripts/bench/kokoro_timing_check.py` and the doc: `bench(speech): measure Kokoro timestamp agreement with the forced aligner`.

### Task 2: Pronunciations module

**Files:**
- Create: `src/muse/modalities/audio_speech/pronunciations.py`
- Test: `tests/modalities/audio_speech/test_pronunciations.py`

**Interfaces:**
- Produces:
  - `class PronunciationError(ValueError)` with attribute `index: int | None`.
  - `@dataclass(frozen=True) class Pronunciation: index: int; term: str | None; spans: tuple[tuple[int, int], ...] | None; ipa: str | None; respell: str | None` and property `kind -> Literal["ipa", "respell"]`.
  - `parse_pronunciations(raw: object, text: str) -> list[Pronunciation]` (raises `PronunciationError`).
  - `pronunciations_digest(raw: list) -> str`: `hashlib.sha256(json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()`.
  - `@dataclass(frozen=True) class Replacement: char_start: int; char_end: int; entry: Pronunciation`.
  - `match_pronunciations(text: str, entries: list[Pronunciation]) -> tuple[list[Replacement], list[dict]]`: non-overlapping replacements sorted by `char_start`, and `applied` in entry order, each `{"index", "kind", "count"}` plus `"term"` or `"spans"` (as lists).
  - `@dataclass class SpokenText: text: str; def to_input_span(self, start: int, end: int) -> tuple[int, int]`.
  - `render_spoken(text: str, replacements: list[Replacement], render_ipa: Callable[[str, str], str] | None) -> SpokenText`. Respell inserts `entry.respell`; ipa inserts `render_ipa(original_text, entry.ipa)`. Any spoken character inside an inserted replacement maps to the whole replaced input span; unchanged characters map one-to-one.
  - Constants `MAX_PRONUNCIATIONS = 2000`, `MAX_TERM_CHARS = 200`, `MAX_REPLACEMENT_CHARS = 500`.

- [ ] **Step 1: Write the failing tests**

```python
def test_term_whole_word_and_counts():
    entries = parse_pronunciations([{"term": "json", "respell": "JAY-sawn"}], "json and JSON, not jsonl")
    reps, applied = match_pronunciations("json and JSON, not jsonl", entries)
    assert [(r.char_start, r.char_end) for r in reps] == [(0, 4), (9, 13)]
    assert applied == [{"index": 0, "term": "json", "kind": "respell", "count": 2}]

def test_longest_term_first_and_flexible_whitespace():
    raw = [{"term": "kutta", "respell": "KOO"}, {"term": "runge kutta", "respell": "Roong-uh Koo-tuh"}]
    text = "the Runge  Kutta method, then kutta"
    reps, applied = match_pronunciations(text, parse_pronunciations(raw, text))
    assert [(text[r.char_start:r.char_end], r.entry.index) for r in reps] == [("Runge  Kutta", 1), ("kutta", 0)]
    assert [a["count"] for a in applied] == [1, 1]

def test_span_wins_over_overlapping_term():
    text = "a cat, not a dog"
    raw = [{"term": "a", "respell": "uh"}, {"spans": [[11, 12]], "respell": "eigh"}]
    reps, applied = match_pronunciations(text, parse_pronunciations(raw, text))
    assert [(r.char_start, r.entry.index) for r in reps] == [(0, 0), (11, 1)]
    assert applied[0]["count"] == 1 and applied[1] == {"index": 1, "spans": [[11, 12]], "kind": "respell", "count": 1}

def test_unmatched_entry_reported_with_zero():
    text = "hello"
    _, applied = match_pronunciations(text, parse_pronunciations([{"term": "tanh", "respell": "hyperbolic tangent"}], text))
    assert applied[0]["count"] == 0

@pytest.mark.parametrize("raw", [
    [{"respell": "x"}], [{"term": "a", "spans": [[0, 1]], "respell": "x"}],
    [{"term": "a"}], [{"term": "a", "ipa": "x", "respell": "y"}],
    [{"spans": [[0, 3], [2, 4]], "respell": "x"}], [{"spans": [[0, 99]], "respell": "x"}],
    [{"spans": [[0, 1]], "respell": "x"}, {"spans": [[0, 2]], "respell": "y"}],
    [{"term": "a", "respell": "x", "extra": 1}],
])
def test_invalid_entries_name_index(raw):
    with pytest.raises(PronunciationError) as e:
        parse_pronunciations(raw, "abcdef")
    assert e.value.index == len(raw) - 1

def test_rejects_hostile_lists():
    for raw in ({"term": "a"}, [{"term": "a", "respell": "b"}] * (MAX_PRONUNCIATIONS + 1),
                [{"term": "x" * (MAX_TERM_CHARS + 1), "respell": "b"}],
                [{"spans": [[0.5, 1]], "respell": "b"}], [{"spans": [[-1, 1]], "respell": "b"}],
                [{"term": "   ", "respell": "b"}], [{"spans": [[1, 1]], "respell": "b"}]):
        with pytest.raises(PronunciationError):
            parse_pronunciations(raw, "abcdef")

def test_digest_is_canonical():
    a = [{"term": "x", "respell": "y"}]
    b = [{"respell": "y", "term": "x"}]
    assert pronunciations_digest(a) == pronunciations_digest(b)
    assert len(pronunciations_digest(a)) == 64

def test_render_spoken_maps_back():
    text = "Coulomb's law and monoid"
    raw = [{"term": "monoid", "respell": "mah noyd"}, {"spans": [[0, 7]], "ipa": "kuːˈlɒm"}]
    reps, _ = match_pronunciations(text, parse_pronunciations(raw, text))
    spoken = render_spoken(text, reps, lambda orig, ipa: f"[{orig}](/{ipa}/)")
    assert spoken.text == "[Coulomb](/kuːˈlɒm/)'s law and mah noyd"
    assert spoken.to_input_span(1, 8) == (0, 7)
    i = spoken.text.index("noyd")
    assert spoken.to_input_span(i, i + 4) == (18, 24)
    j = spoken.text.index("law")
    assert spoken.to_input_span(j, j + 3) == (10, 13)

def test_render_spoken_ipa_without_renderer_raises():
    text = "tanh"
    reps, _ = match_pronunciations(text, parse_pronunciations([{"term": "tanh", "ipa": "tæn"}], text))
    with pytest.raises(PronunciationError):
        render_spoken(text, reps, None)
```

- [ ] **Step 2: Run** `pytest tests/modalities/audio_speech/test_pronunciations.py -q`; expected: import error.
- [ ] **Step 3: Implement** the interfaces above. Term matching copies the pipeline's `apply_lexicon` rule: one compiled alternation `\b(?:t1|t2|...)\b`, `re.IGNORECASE`, terms sorted by length descending, each term's words joined by `\s+`; a matched occurrence that overlaps any span is skipped (and not counted). Spans may not overlap each other (across entries too); `end > start`, both ints (reject `bool`), `0 <= start < end <= len(text)`.
- [ ] **Step 4: Run** the test file; expected: all pass.
- [ ] **Step 5: Commit** `feat(speech): pronunciation matching and spoken-text mapping`.

### Task 3: Word timing assembly

**Files:**
- Create: `src/muse/modalities/audio_speech/word_timing.py`
- Test: `tests/modalities/audio_speech/test_word_timing.py`

**Interfaces:**
- Produces: `@dataclass(frozen=True) class TimedSpan: char_start: int; char_end: int; start: float; end: float` and `assemble_words(text: str, spans: list[TimedSpan], duration: float) -> list[dict]` returning one dict per `re.finditer(r"\S+", text)` match: `{"word", "char_start", "char_end", "start", "end", "measured"}`, times rounded to 3 decimals, clamped to `[0, duration]`, non-decreasing starts.

Rules (from the spec): a span *belongs* to a word when it overlaps exactly that one word; a word with belonging spans is `measured: true`, start = min start, end = max end. A span overlapping several words is coarse: those of its words without belonging spans divide its time in proportion to character length, `measured: false`. A word with neither is placed between the previous word's end (or 0.0) and the next measured or divided word's start (or `duration`), runs of such words dividing that gap by character length, `measured: false`.

- [ ] **Step 1: Write the failing tests**

```python
def test_one_entry_per_split_word_measured():
    text = "Coulomb's law"
    words = assemble_words(text, [TimedSpan(0, 9, 0.1, 0.6), TimedSpan(10, 13, 0.62, 0.9)], 1.0)
    assert [w["word"] for w in words] == text.split()
    assert words[0] == {"word": "Coulomb's", "char_start": 0, "char_end": 9, "start": 0.1, "end": 0.6, "measured": True}

def test_respelled_word_spans_all_tokens():
    words = assemble_words("a monoid", [TimedSpan(0, 1, 0.0, 0.1), TimedSpan(2, 8, 0.2, 0.4), TimedSpan(2, 8, 0.4, 0.7)], 1.0)
    assert (words[1]["start"], words[1]["end"], words[1]["measured"]) == (0.2, 0.7, True)

def test_phrase_span_divided_unmeasured():
    words = assemble_words("Runge Kutta", [TimedSpan(0, 11, 1.0, 2.0)], 3.0)
    assert [w["measured"] for w in words] == [False, False]
    assert words[0]["start"] == 1.0 and words[1]["end"] == 2.0 and words[0]["end"] == words[1]["start"]

def test_unmeasured_word_interpolated_between_neighbours():
    words = assemble_words("one two three", [TimedSpan(0, 3, 0.0, 0.5), TimedSpan(8, 13, 1.5, 2.0)], 2.0)
    assert words[1]["measured"] is False and 0.5 <= words[1]["start"] <= words[1]["end"] <= 1.5

def test_words_match_split_with_odd_whitespace():
    text = "  Hello,\n\nworld — ok...  "
    words = assemble_words(text, [], 2.0)
    assert [w["word"] for w in words] == text.split()
    assert all(text[w["char_start"]:w["char_end"]] == w["word"] for w in words)
    assert words[-1]["end"] <= 2.0

def test_times_clamped_to_duration():
    words = assemble_words("hi", [TimedSpan(0, 2, 0.9, 1.4)], 1.0)
    assert words[0]["end"] == 1.0
```

- [ ] **Step 2: Run**; expected: import error.
- [ ] **Step 3: Implement** `assemble_words` per the rules above.
- [ ] **Step 4: Run**; expected: pass.
- [ ] **Step 5: Commit** `feat(speech): assemble per-word timings from timed spans`.

### Task 4: Kokoro native timestamps and IPA rendering

**Files:**
- Modify: `src/muse/models/kokoro_82m.py`
- Test: `tests/models/test_kokoro_82m.py`

**Interfaces:**
- Consumes: Task 1's `KOKORO_TIMING_OFFSET_SECONDS` value and chunk finding.
- Produces (backend contract the route relies on, documented in `audio_speech/protocol.py` `TTSModel` docstring as optional):
  - `synthesize(text, voice=..., speed=..., timestamps=True)` returns `AudioResult` with `metadata["token_spans"]: list[TimedSpan]` (coordinates in `text` as given) and `metadata["timing_calibration"]: dict | None` (`{"offset_seconds": x, "basis": "docs/benchmarks/2026-10-24-kokoro-timing-agreement.md"}` when `x != 0`, else `None`), plus `metadata["voice"]`.
  - `render_ipa(original: str, ipa: str) -> str` returning `f"[{original}](/{ipa}/)"`.
  - property `default_voice -> str` (`"af_heart"`).
  - MANIFEST capabilities gain `"supports_word_timestamps": True`, `"pronunciation_kinds": ["ipa", "respell"]`; `limits` drops "No word timings at synthesis..." and gains "Word timings come from the model's duration predictor (checked against the forced aligner, see docs/benchmarks/2026-10-24-kokoro-timing-agreement.md)".

- [ ] **Step 1: Write the failing tests** with a fake `KPipeline` that yields objects with `.audio` (a fake tensor with `.numpy()`) and `.tokens` (objects with `.text`, `.whitespace`, `.start_ts`, `.end_ts`):
  - `test_timestamps_off_returns_no_spans`: plain call; `"token_spans" not in result.metadata`.
  - `test_token_spans_locate_tokens_in_text`: text `"Hello, world."`, tokens `Hello`(0.1–0.4) `,`(None) `world`(0.5–0.9) `.`(0.9–0.9); spans are `[TimedSpan(0,5,0.1+off,0.4+off), TimedSpan(7,12,0.5+off,0.9+off)]` (punctuation-only tokens and tokens with `start_ts is None` are skipped; `off` = the recorded offset).
  - `test_multi_chunk_offsets_accumulate`: two yielded results, the first with 24000 samples; the second chunk's token at `start_ts=0.1` comes out at `1.1 + off`, and its char offsets point into the second paragraph of `"One.\n\nTwo."`.
  - `test_link_token_maps_inside_link`: text `"[Coulomb](/kuːˈlɒm/) law"`, token text `Coulomb`; its span is `(1, 8)`.
  - `test_unmappable_token_is_dropped`: a token whose text is not found ahead of the cursor yields no span and does not move the cursor.
  - `test_render_ipa_and_caps`: `render_ipa("Coulomb", "kuːˈlɒm") == "[Coulomb](/kuːˈlɒm/)"`; MANIFEST caps as above.
- [ ] **Step 2: Run** `pytest tests/models/test_kokoro_82m.py -q`; expected: new tests fail.
- [ ] **Step 3: Implement.** Map tokens by searching each token's text forward from a cursor in the current chunk's graphemes, and chunk graphemes forward in `text` (KPipeline splits on `\n+` and may further split long chunks; `result.graphemes` is the chunk text). Chunk time offset = samples concatenated so far / 24000. Apply Task 1's offset (and per-chunk correction if Task 1 found one). Timestamps only on the English pipelines (`lang_code in ("a", "b")`); otherwise raise `ValueError` the route turns into 400 `timestamps_not_supported`.
- [ ] **Step 4: Run**; expected: pass, and `pytest tests/models -q` green.
- [ ] **Step 5: Commit** `feat(kokoro): native word timestamps and IPA pronunciation links`.

### Task 5: Speech route: timestamps, pronunciations, provenance

**Files:**
- Modify: `src/muse/modalities/audio_speech/routes.py`, `src/muse/modalities/audio_speech/protocol.py` (docstring only)
- Test: `tests/modalities/audio_speech/test_routes_timestamps.py`

**Interfaces:**
- Consumes: Task 2 (`parse_pronunciations`, `match_pronunciations`, `render_spoken`, `pronunciations_digest`, `PronunciationError`), Task 3 (`assemble_words`, `TimedSpan`), Task 4's backend contract.
- Produces: `SpeechRequest.timestamps: bool = False`, `SpeechRequest.pronunciations: list | None = None` (raw; validated by `parse_pronunciations`, not pydantic), the JSON body from spec section 1 with `timing = {"source": "native", "measured_words", "estimated_words", "calibration"}` and `pronunciations` present only when sent, and headers `X-Muse-Voice`, `X-Muse-Speed` (`str(float(speed))`), `X-Muse-Audio-SHA256` (non-streamed), `X-Muse-Pronunciations-Digest` (when sent).

Order of checks: model lookup; `timestamps and stream` → 400 `timestamps_not_streamable`; `timestamps` and not `capabilities.supports_word_timestamps` → 400 `timestamps_not_supported` (message points at `/v1/audio/alignments`); parse pronunciations → 400 `invalid_pronunciation` naming the index; an `ipa` entry when `"ipa"` is not in `capabilities.pronunciation_kinds` or the model has no `render_ipa` → 400 `pronunciation_not_supported` naming the index. Pronunciations apply to streamed and non-streamed requests alike (synthesize the rendered spoken text). Voice header: `result.metadata.get("voice") or req.voice or getattr(model, "default_voice", None)`; omit when unknown.

- [ ] **Step 1: Write the failing tests** (FakeModel pattern, own `ModalityRegistry`, `create_app`):
  - `test_plain_request_body_unchanged`: same bytes and content type as before for `{input, model}`; `synthesize` called with exactly `(input, voice=, speed=, align=)`; headers include `X-Muse-Speed: 1.0` and `X-Muse-Audio-SHA256` equal to `sha256(body)`.
  - `test_timestamps_json_shape`: keys and types exactly as spec section 1; `len(words) == len(input.split())`; `audio.sha256 == sha256(b64decode(audio.b64_json))`; `timing.source == "native"`; header `X-Muse-Audio-SHA256 == audio.sha256`.
  - `test_timestamps_with_pronunciations_keyed_to_input`: input `"the monoid law"`, respell `monoid` → fake model receives `"the mah noyd law"` and returns spans in spoken coordinates; response words are `["the", "monoid", "law"]` with `monoid` measured; `pronunciations.digest` equals `pronunciations_digest(raw)` and header matches.
  - `test_timestamps_not_supported`, `test_timestamps_not_streamable`, `test_ipa_not_supported_names_index`, `test_invalid_pronunciation_400` (status and `error.code`).
  - `test_timestamps_opus_fallback_reports_wav`: patch `wav_bytes_to_opus` to raise `AudioFormatError`; `audio.format == "wav"`.
  - `test_respell_on_model_without_kinds`: a model with no `pronunciation_kinds` accepts `respell` and receives the substituted text.
  - `test_stream_with_pronunciations_sends_spoken_text_and_digest_header`.
  - `test_align_flag_still_sets_x_alignment` (existing behaviour, with timestamps off).
- [ ] **Step 2: Run**; expected: fail.
- [ ] **Step 3: Implement.** Keep `_non_stream`'s plain path byte-identical; add a timestamps branch that maps each `TimedSpan` through `SpokenText.to_input_span` before `assemble_words(req.input, ...)`. `measured_words` / `estimated_words` count `measured` true / false.
- [ ] **Step 4: Run** the new file and `pytest tests/modalities/audio_speech -q`; expected: pass.
- [ ] **Step 5: Commit** `feat(speech): word timestamps, pronunciations and take provenance on /v1/audio/speech`.

### Task 6: Discovery and MCP

**Files:**
- Modify: `src/muse/modalities/audio_speech/__init__.py` (DISCOVERY), `src/muse/mcp/generated.py` (`DATA_HEADERS`)
- Test: `tests/mcp/` (the existing generated-tool tests; add `test_speak_writes_audio_from_json_and_binary`), `tests/core/` capabilities test if one asserts the speech summary

**Interfaces:**
- Produces: speech DISCOVERY summary describing `timestamps: true` (JSON with base64 audio and one timing per written word, Kokoro only; models without native timings refuse) and `pronunciations`; the `align`/`X-Alignment` sentence becomes "Soprano's paragraph timings". Speech route `media_fields: ["audio.b64_json"]`. `DATA_HEADERS` adds `x-muse-voice: voice`, `x-muse-speed: speed`, `x-muse-audio-sha256: audio_sha256`, `x-muse-pronunciations-digest: pronunciations_digest`.

- [ ] **Step 1: Write the failing test**: render a fake JSON response (`audio.b64_json` = base64 WAV) and a fake `audio/wav` response through `_render` with the speech route; both write one `audio/wav` file; the JSON case's `response` keeps `words` and has `audio.b64_json` replaced by the file path; the binary case's `response` carries `voice`, `speed`, `audio_sha256`.
- [ ] **Step 2: Run**; expected: fail.
- [ ] **Step 3: Implement** the DISCOVERY and `DATA_HEADERS` changes (the existing content-type dispatch in `_render` already handles both response shapes).
- [ ] **Step 4: Run** `pytest tests/mcp tests/core -q -k "capab or generated or discovery"`; expected: pass.
- [ ] **Step 5: Commit** `feat(mcp): muse_speak returns word timings and take provenance`.

### Task 7: Qwen3-ASR-1.7B listener

**Files:**
- Create: `src/muse/modalities/audio_transcription/runtimes/qwen3_asr.py`
- Modify: `src/muse/modalities/audio_transcription/hf.py` (discovery allows one `HF_PLUGIN` per modality: `_sniff` also accepts the exact Qwen3-ASR repo, `_resolve` dispatches to a Qwen3-ASR manifest for it and keeps the faster-whisper path for everything else; `HF_PLUGIN["runtime_path"]`/`pip_extras` stay the whisper defaults), `src/muse/modalities/audio_transcription/routes.py`, `src/muse/curated.yaml`
- Test: `tests/modalities/audio_transcription/runtimes/test_qwen3_asr.py`, `tests/modalities/audio_transcription/test_hf_plugin.py`, `tests/modalities/audio_transcription/test_routes.py`

**Interfaces:**
- Produces: `Qwen3ASRRuntime` satisfying `TranscriptionModel` (`transcribe(audio_path, *, task, language, prompt, temperature, word_timestamps, vad_filter, **kwargs) -> TranscriptionResult` with one segment covering the whole file, `words=None`); curated `qwen3-asr-1.7b` → `hf://Qwen/Qwen3-ASR-1.7B-hf` with revision pinned to the current Hub commit (spec noted `bcd2b5b7f32b480ab5790554cfa8347f246a14f3`; re-check and use the commit you verify), `quality_tier: good`, capabilities `supports_word_timestamps: false`, `supports_segment_timestamps: false`, `supports_translation: false`, `device: auto`, `memory_gb: 5.0` (corrected by probe in Task 9).
- Route gates (default `True` when the capability is absent, so faster-whisper is unchanged): `"word"` in `timestamp_granularities` and not `supports_word_timestamps` → 400 `word_timestamps_not_supported` (message points at `/v1/audio/alignments`); `response_format in {"verbose_json", "srt", "vtt"}` and not `supports_segment_timestamps` → 400 `segment_timestamps_not_supported` (ruling: refuse, never fake segment times); `/v1/audio/translations` and not `supports_translation` → 400 `translation_not_supported`. Gates run after model lookup, before reading inference.

- [ ] **Step 1: Read the model card** at the pinned revision (`hf://Qwen/Qwen3-ASR-1.7B-hf/README.md`) and record in the runtime docstring: the Transformers classes and minimum version it documents, how audio and optional context are passed, and the language parameter. If it needs `trust_remote_code` or a non-Transformers package, stop and report BLOCKED with what it needs.
- [ ] **Step 2: Write the failing tests**: plugin claims exactly `Qwen/Qwen3-ASR-1.7B-hf` (any case) with the required files, refuses other Qwen repos, synthesizes the manifest above with the pinned allow-list and `pip_extras` (`torch`, `torchcodec`, `transformers>=<card floor>`); runtime (mocked transformers) passes `prompt` as context only when it is not `None`/empty (`test_prompt_only_when_sent`), returns `TranscriptionResult(text=..., segments=[Segment(0, 0.0, duration, text, None)])`; routes return the three 400s above and leave a fake whisper model's behaviour unchanged.
- [ ] **Step 3: Run**; expected: fail.
- [ ] **Step 4: Implement.** Decode with the alignment modality's `decode_audio` (16 kHz mono) or TorchCodec, bounded by the existing `limits.asr_max_mb`. Curated `summary`/`strengths`/`limits` must say: transcripts only (no word or segment times, use audio/alignment); for pronunciation checks do not send `prompt`, because context makes the model hear the expected words.
- [ ] **Step 5: Run** `pytest tests/modalities/audio_transcription tests/core -q`; expected: pass (including any curated/notes meta-tests).
- [ ] **Step 6: Commit** `feat(asr): Qwen3-ASR-1.7B transcription runtime and curated alias`.

### Task 8: Speech comparison script

**Files:**
- Create: `scripts/bench/speech_compare.py`
- Test: `tests/scripts/test_speech_compare.py`

**Interfaces:**
- Consumes: the Task 5 JSON response; `/v1/audio/transcriptions` (`qwen3-asr-1.7b`, no `prompt`), `/v1/audio/quality` (`utmos`), `/v1/audio/alignments` (`qwen3-forced-aligner-0.6b`).
- Produces: `Case(id, text, pronunciations=None, accept=(), terms=())`, `Target(model, voice)` parsed from `model:voice`, `load_cases(path)`, `cases_from_lexicon(path) -> list[Case]`, `normalize(text) -> str` (casefold, strip accents via NFKD, punctuation to spaces, collapse whitespace), `heard(transcript, term, replacement, accept) -> bool`, `wer(ref, hyp) -> float`, `agreement(words, aligned, tolerance=0.1) -> dict` (`within`, `total`, `fraction`, `disagreements: list[dict]`), `run(cases, targets, out_dir, client, *, listener, aligner, quality) -> list[dict]`, `write_index(results, out_dir) -> Path`, `main(argv=None) -> int`. CLI: `--cases PATH | --lexicon PATH`, `--target model:voice` (repeatable; default `kokoro-82m:af_heart` and `kokoro-82m:am_puck`), `--voices a,b` (overrides voices of the default model), `--server` (default `$MUSE_SERVER` or `http://localhost:8000`), `--out DIR`.

Lexicon conversion: a value matching `^\[(.+)\]\(/(.+)/\)$` becomes `{"term": key, "ipa": <group 2>}`, anything else `{"term": key, "respell": value}`; the case text is `f"Here is the word {key} in a sentence."`; the replacement counts as ordinary words (eligible for `heard`) only when it is a respell made of dictionary-looking words, i.e. no hyphen and no capital in the middle of a word (ruling: `"hyperbolic tangent"` and `"exclusive, or"` qualify, `"JAY-sawn"`, `"mah noyd"` checks fall back to the written term and `accept`). Read `lexicon.yaml` read-only.

- [ ] **Step 1: Write the failing tests** (httpx `MockTransport`, like `test_image_compare.py`): `test_cases_from_lexicon_ipa_and_respell` (Coulomb link → ipa entry; `tanh` → respell); `test_heard_rules` (`heard("the hyperbolic tangent", "tanh", "hyperbolic tangent", ())` true; `heard("Jason files", "json", "JAY-sawn", ("jason",))` true; `heard("jay sawn", "json", "JAY-sawn", ())` false; accents: `heard("poincare map", "poincaré", None, ())` true); `test_wer`; `test_agreement_lists_disagreements`; `test_run_writes_results_and_index` (one case, one target: results.json has transcript, wer, heard, utmos, agreement, audio path, sha256, model revision; index.html has an `<audio>` element per take; listener request carries no `prompt` field).
- [ ] **Step 2: Run**; expected: fail.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `pytest tests/scripts/test_speech_compare.py -q`; expected: pass.
- [ ] **Step 5: Commit** `feat(bench): speech comparison and lexicon check script`.

### Task 9: Docs, Soprano demotion, live verification

**Files:**
- Modify: `README.md` (quickstart and examples: `soprano-80m` → `kokoro-82m`), `CLAUDE.md` (audio/speech line leads with Kokoro and mentions `timestamps` / `pronunciations`; audio/transcription line adds Qwen3-ASR; a short paragraph on the speech timing contract and the Qwen3-ASR gates), `AGENTS.md` if it lists modalities, `src/muse/models/soprano_80m.py` (keep `quality_tier: draft`; add a module-docstring note on origin: weights `ekwek/Soprano-1.1-80M` by Eugene Kwek, muse began as a fork of `ekwek1/soprano`, the Narro engine is that code evolved), `docs/AGENT_DISCOVERY.md` if it describes speech `align`, `docs/benchmarks/2026-10-24-narration-acceptance.md` (new)

- [ ] **Step 1: Docs edits** above; `grep -rn "soprano-80m" README.md examples/` shows only intentional mentions.
- [ ] **Step 2: Live verification on an isolated server** (does not touch `~/.muse`): `export MUSE_CATALOG_DIR=<scratch>/muse-catalog`; `muse pull kokoro-82m`, `muse pull qwen3-asr-1.7b`, `muse pull qwen3-forced-aligner-0.6b`, `muse pull utmos` (each pull probes; record Qwen3-ASR's measured peak and set its curated `memory_gb` to that rounded up to 0.5 GB); `muse serve --port 8010` in the background. Stop it when done. If GPU memory is short because `localhost:8000` holds models, wait for its idle eviction rather than signalling it.
- [ ] **Step 3: Acceptance run**: `python scripts/bench/speech_compare.py --server http://localhost:8010 --cases <the Task 1 beats as a case file> --out <scratch>/accept` and `--lexicon ~/github/youtube-channel/engine/companion/lexicon.yaml --out <scratch>/lexicon`. Write `docs/benchmarks/2026-10-24-narration-acceptance.md`: agreement fraction (bar: ≥ 95% of words within 0.1 s), every disagreement (beat, word, both times), lexicon heard / not heard per entry and voice, and a `curl` example of `timestamps: true`. A miss is reported, not fixed by loosening the bar.
- [ ] **Step 4: Run** `pytest tests/ -q -m "not slow"`; expected: green.
- [ ] **Step 5: Commit** `docs(speech): narration timings, pronunciations, Qwen3-ASR; demote Soprano`.
