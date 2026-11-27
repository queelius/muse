# Modality notes

Per-modality design and behaviour notes, moved out of CLAUDE.md so it stays
short. CLAUDE.md keeps the one-line summary of each modality; read the matching
section here before changing a modality. Specs live under docs/superpowers/specs/.

Speech `timestamps: true` returns JSON with the audio and its SHA-256, one
entry per whitespace-separated written word, and native timing provenance.
Words without native measurements have `measured: false`; no silent timing
fallback is presented as measured. `pronunciations` keeps respellings or IPA
separate from display text. Kokoro native times come from its English pipeline
(`lang_code`, always "a" in muse today), so non-English voices get timings from
English G2P; models without native timings and streaming timestamps return 400. Legacy
`align` / `X-Alignment` remains Soprano paragraph timing. `stream: true`
returns base64 PCM16 SSE chunks, so it rejects `response_format: opus` (400
`format_not_streamable`) and `align` (400 `alignment_not_streamable`) instead
of dropping them. Speech and chat streaming share
`muse.modalities._common.streaming.threaded_sse_response`, which starts
the producer thread on the body's first iteration and, from the response's
own `__call__`, always sets the cancel flag and closes the generator when
the response ends (sse_starlette leaves a generator paused at `yield` when
the client leaves mid-send, and would otherwise rely on GC). A body that
never started still runs `on_finish`, which closes chat's decoded images. Qwen3-ASR returns
transcripts only: word timestamps and segment formats (`verbose_json`, `srt`,
`vtt`) return 400; use `/v1/audio/alignments`. Translation is unsupported.
Qwen3-ASR validates `language` before decoding and decodes at most
`limits.asr_max_duration_seconds` (`MUSE_ASR_MAX_DURATION_SECONDS`, default
600 s); longer audio returns 413 (it used to inherit alignment's 300 s
default silently).
For pronunciation checks omit `prompt`, whose context biases the listener.

Designed and saved speech voices (Qwen3-TTS; spec
`docs/superpowers/specs/2026-11-20-npc-voices-design.md`). `POST
/v1/audio/speech` takes an optional `instructions` string (the OpenAI
`gpt-4o-mini-tts` field, at most 4096 chars), accepted only by models with
capability `supports_instructions` (else 400 `instructions_not_supported`) and
forwarded to the backend only when non-blank. `qwen3-tts-1.7b-voicedesign`
speaks from that description (a neutral default when omitted) and ignores
`voice`, so the response reports no voice. `qwen3-tts-0.6b-base`
(`supports_saved_voices`) clones a saved voice named by `voice`, defaults to
the bundled read-only `narrator` (about 4 s, package data under
`audio_speech/data/`), and refuses `instructions`. Both ignore `speed` and
have no word timestamps. With `stream: true` on CUDA a line streams as it is
generated (spec `docs/plans/2026-11-26-qwen3-tts-streaming-batching.md`):
qwen-tts builds the prompt exactly as for the one-call path and is unwound at
the talker call (`FastTalker.capture`), the talker's 12 Hz frames come one at
a time from a loop the runtime owns (`FastTalker.start` / `step`), and the
speech decoder runs split into its causal front (quantizer, `pre_conv`,
transformer; over every frame so far, the clone reference's included) and
its conv stack (over the new frames plus 6 frames of context;
`audio_speech/runtimes/qwen3_tts_stream.py`), on CUDA each half as a CUDA
graph per padded shape (codes padded at the end to 64, 128, ... frames; conv
widths rounded up to 4 frames; both causal). The first chunk is
`stream_first_frames` (2, 0.16 s of audio), each later one twice the last up
to `stream_chunk_frames` (16); both are hidden capabilities. Same seed, the
streamed codes equal the one-call codes token for token and the audio sits
at the decoder's own bf16 floor against the one-pass decode (34-44 dB SNR,
median ~39). On an RTX 3060 the first audio arrives ~90 ms after the request
on both models (from ~560 ms / ~800 ms for a short line in one chunk) and
every later chunk ahead of its playback by >= 106 ms; streaming costs ~5%
more GPU per second of audio. In clone mode qwen-tts's one-call audio is one
sample longer for ~5% of line lengths (its float cut keeps one reference
sample); the stream is exactly the generated frames. A stream holds the
model until it ends, unless the hidden capability `max_batch_size` is above 1
(default 1, today's path): then one engine thread
(`audio_speech/runtimes/qwen3_tts_batch.py`) shares each 12 Hz frame among up
to that many requests (lines join at a frame boundary and leave at their own
EOS; products of more than 4 rows read float layer weights kept beside the
int8), paces streams to `stream_lead_seconds` (1.5) ahead of playback, and the
route's lock admits that many requests. There a stream's chunks are
deadline-driven instead of the doubling schedule: a line's pending frames go
out when one more frame plus a decode would no longer fit in its audio ahead
of playback (plus the hidden `stream_buffer_seconds`, the client's jitter
buffer, default 0), or at `stream_chunk_frames`; its first chunk grows with
the engine's measured frame time so the second can be as long. Whole lines
decode chunk by chunk as they generate. `warm_rows` (hidden, default 0)
allocates that many rows and captures their frame graphs and the padded
prefill lengths at load (0.6B: 2.5 -> 3.1 GiB at 8 rows, 3.6 GiB at 16);
without it the first growth to each bucket recaptures graphs mid-stream
(measured: up to 256 ms late on lines already playing). Running out of
memory while growing the state keeps the live lines (more rows: the engine
caps itself there; more KV: only the line that needed it fails); any other
engine failure makes the model serve one request at a time. Measured
capacity and costs: the streaming / batching plan. On CPU, with another
qwen-tts or after the fast talker or the split decoder has failed (a failure
before the first chunk falls back for that request) a stream is one chunk.
On CUDA both load
with SDPA attention, and qwen-tts's two nested HF `generate()` loops (the
talker's, and the 15-token code predictor it runs per 12 Hz frame) are
replaced by static-KV decoders (`audio_speech/runtimes/qwen3_tts_fast.py`).
On CUDA each frame (code predictor, codec-embedding sum, one talker token,
transformers' repetition-penalty / min-new-tokens / suppress-tokens
processors and sampling) is one CUDA graph, captured per KV capacity and
sampling parameters (about 1 s on a first request); the talker cache starts
at 512 tokens and doubles as a line needs it. Prefill runs eagerly. Without
graphs, greedy decoding reproduces upstream token for token. The graphed
frame fuses q/k/v and gate/up projections, and on CUDA with Triton
(`audio_speech/runtimes/qwen3_tts_triton.py`, lazily imported; torch ops
otherwise) runs its one- and two-row projections as Triton GEMVs and its
RMSNorms, residual adds, q/k norm + RoPE (writing K/V straight into the
cache), SiLU gating, decode attention over the live keys and
temperature/top-k/Gumbel-max sampling as fused kernels. That rounds
differently in bf16 (talker teacher-forced mean KL 5-8e-4; the code
predictor's error against an fp32 reference stays at or below upstream
bf16's own), and sampling draws from the same distributions with a
different random stream. The speech decoder's SnakeBeta activations run as
one Triton kernel each, bit-identical to upstream (`fast_decoder`). Only batch
1 is handled (larger batches use upstream). Both are installed only for
qwen-tts 0.1.1 with the expected module layout and a default talker
generation config; any other version, layout or call shape uses upstream
`generate()`, and an unexpected failure switches the failing one off for the
process. Constructor kwargs `fast_code_predictor` / `fast_talker` /
`fast_decoder` / `cuda_graphs` (default on) turn them, or just the graphs,
off (the talker needs the fast predictor). `int8_weights` (runtime default off, set by both bundled manifests; CUDA
with Triton) stores each talker
and predictor layer's projections as int8 only, one scale per output row,
quantized from the loaded checkpoint at construction; the float layer weights
are dropped. The frames' GEMVs read the int8 directly (half the bytes on a
bandwidth-bound loop: a frame takes 7.2 ms on the 0.6B and 10.2 ms on the
1.7B, RTF ~0.12 and ~0.15), and every other use of those layers (prefill,
larger batches, upstream `generate()`) computes from the same int8,
dequantized per call, which adds about 17 ms to a request's prefill on the
0.6B. Output heads stay float. Peak VRAM falls to 2.34 GiB on the 0.6B
(25 s paragraph) and 3.3 GiB on the 1.7B, whose peak is then the 3.9 GiB
load before quantization (2.5 GiB resident). It moves the token distributions
slightly (talker teacher-forced KL against a float copy 1.4-1.9e-3,
predictor KL against fp32 3-4.5e-3 where bf16's own is 0.9-1.5e-3); over 48
takes per arm and model, ASR errors were identical and UTMOS naturalness
differed by under 0.01. In float, on an RTX 3060 a short line
takes about 0.5 s on the 0.6B and 0.8 s on the 1.7B (RTF ~0.17 and ~0.24;
a frame is 11.5 ms and 17.6 ms of GPU time), from about 7 s and 11 s per
short line with upstream's loops. Every
saved voice's clone prompt stays cached once used (capacity
`limits.max_saved_voices` plus bundled). An unknown `voice` is 400
`voice_not_found` (also before a stream starts, via the backend's
`check_voice`); a stored voice that cannot be read is 500 `voice_unreadable`
with no paths. Saved voices form a catalog. Besides the clip and transcript,
each may carry metadata (all optional; old voices without it read as null):
`category` (a slash path of 1-4 segments `[a-z0-9][a-z0-9_-]{0,31}`, e.g.
`narrator/epic`, `npc/human/guard`), `tags` (at most 16 of the same shape,
deduplicated and sorted), `description` (what it sounds like, at most 500
chars), `notes` (what it is for, at most 2000 chars) and `design`
(`{model, instructions}`: how the take was designed, instructions at most
4096 chars), plus server-set `created_at` / `updated_at`. Input is
normalized (lowercased, trimmed, outer slashes stripped; a comma string is
accepted for tags; an empty string or null clears a field); a rule violation
is 400 `invalid_voice_metadata` naming the field. Voices are managed on
`/v1/audio/speech/voices`; every route but `POST` is a light route
(`muse.core.light`): the gateway answers it from disk and never loads the
model.
- `GET ?model=&category=&tag=&q=` returns `{"object": "list", model, voices:
  [names], data: [entries], categories: [{path, count}]}`. `category`
  matches the path and everything below it, `tag` repeats (all must match),
  `q` is a case-insensitive substring of name, category, description, notes
  and tags. `categories` always covers every voice. A model without saved
  voices lists its preset voices (`{object, model, name, read_only}` entries;
  the gateway forwards that case to the worker). MCP `muse_list_voices`.
- `GET /{name}?model=` returns one entry: `{"object": "audio.voice", model,
  name, category, tags, description, notes, design, transcript,
  duration_seconds, sample_rate, sha256, read_only, created_at, updated_at,
  audio_url}`. MCP `muse_get_voice`.
- `GET /{name}/audio?model=` returns the reference clip as `audio/wav`, to
  audition a voice without synthesizing. MCP `muse_get_voice_audio`.
- `PATCH /{name}` (JSON `{model, category?, tags?, description?, notes?,
  design?}`) changes only the fields sent (null clears) and returns the
  entry; audio and transcript are not editable. 422 `invalid_request` for an
  unknown field or wrong type. MCP `muse_update_voice`.
- `POST` (multipart `model`, `name`, `transcript`, optional `overwrite`
  default false, file `audio`, and the optional metadata form fields
  `category`, `tags` (comma-separated), `description`, `notes`,
  `design_model`, `design_instructions`) returns 200 `{"object":
  "audio.voice", model, name, duration_seconds, sha256, category, tags,
  description, notes, design}` (sha256 of the stored WAV). Errors: 400
  `invalid_voice_name` (name must match `^[a-z0-9][a-z0-9_-]{0,63}$`),
  `invalid_transcript` (1-1000 chars), `invalid_reference_audio` (1-30 s),
  `invalid_voice_metadata`, `voice_read_only` (`narrator`),
  `voice_saving_not_supported`; 500 `voice_storage_unavailable` (fixed
  message, no paths; a read failure on GET is `voice_unreadable`); 409
  `voice_exists` (without `overwrite`) and `voice_limit_reached`; 413
  `payload_too_large` (`limits.voice_reference_max_bytes`, default 20 MB);
  415 `unsupported_media_type` (undecodable audio). MCP tool `muse_save_voice`.
- `DELETE /{name}?model=` returns 200 `{"object": "audio.voice.deleted",
  model, name, "deleted": true}`; 404 `voice_not_found`, 409
  `voice_read_only`. MCP `muse_delete_voice`.
- Every route but `POST` answers 400 `voice_saving_not_supported` for a model
  without saved voices (except the preset listing above), 404
  `voice_not_found` for an unknown or malformed name, and 409
  `voice_read_only` for PATCH or DELETE of a bundled voice.

Storage: `<paths.home>/voices/<model_id>/<name>/{reference.wav, voice.json}`
(voices are per model id; the store root is created 0700). The reference is
decoded to mono 24 kHz and stored once as PCM16 WAV, never re-encoded;
`voice.json` holds transcript, sample rate, duration, sha256, created and
updated times, source filename and the metadata above. Saves write a temp
dir and rename it into place; metadata edits replace `voice.json` atomically.
Every process (gateway and worker) takes the voice root's advisory lock
`<root>/.lock` (shared for reads, exclusive for writes; `VoiceCatalog.lock`,
stdlib-only `voice_catalog.py`), so a reader in one process never sees
another's half-finished swap. The count cap `limits.max_saved_voices`
(default 200 per model) is read when the model loads, so a change needs a
reload. The clone prompt is cached per voice (LRU of 32) keyed by the files'
stat identity, so an edit or delete from the gateway is picked up on the next
line (a metadata edit costs one re-encode). Design-then-clone, the intended
workflow for a recurring character:
1. Audition on the VoiceDesign model with `instructions` until a take fits.
2. Save that take with its exact text as the transcript: `POST /v1/audio/speech/voices` on `qwen3-tts-0.6b-base`, `name: grimbold`, plus `category`, `description` and the `design_instructions` you used.
3. Speak every later line on the Base model with `voice: "grimbold"`; it holds tighter than re-describing each line.
4. For a one-off emotional line, go back to the VoiceDesign model with `instructions`.

`audio/quality` is muse's twenty-first modality. `POST
/v1/audio/quality` accepts multipart `file` plus optional `model` and
returns `{id, object, model, primary_score, scores, metadata}`. Each
named score carries `value`, nominal `minimum` / `maximum`, and a
direction (`higher_is_better`, `lower_is_better`, or `descriptive`),
because a UTMOS 1-5 naturalness MOS is not numerically interchangeable
with Audiobox's 1-10 axes. Both runtimes use TorchCodec to decode 16kHz
mono ranges sequentially (10-second windows), return a duration-weighted
aggregate plus `metadata.segments`, the literal `metadata.worst_segment`,
and `metadata.worst_review_segment` (which ignores subsecond tail windows
when a longer review candidate exists), and reject audio over
`MUSE_AUDIO_QUALITY_MAX_DURATION_SECONDS` (600s default) with 413. The
priority-100 HF plugin claims only two
verified custom checkpoint shapes: `Blinorot/UTMOS-PyTorch` routes to
the local TorchScript runtime and `facebook/audiobox-aesthetics`
loads the package's official model class from Hub safetensors directly
onto Muse's selected device. Audiobox must
win before the broader priority-110 `audio/classification` plugin,
since its Hub card carries that generic tag. Curated aliases are
`utmos` (MIT, primary `naturalness`) and `audiobox-aesthetics`
(CC-BY-4.0, primary `production_quality`). Pairwise/text-conditioned
judges such as SpeechJudge are deferred to a separate comparison
route/capability rather than overloading this single-ended contract.

`audio/alignment` is muse's twenty-second modality. `POST
/v1/audio/alignments` accepts multipart `file` + trusted reference `text`,
plus optional `model` and `language`, and returns `{id, object, model, text,
language, duration_seconds, words, metadata}`. Each word has `word`, `start`,
`end`, and an uncalibrated confidence derived from the mean probability of its
start/end timestamp tokens. This is forced alignment, not ASR: the model
locates supplied words and does not decide what was spoken. The priority-90 HF
plugin claims only `Qwen/Qwen3-ForcedAligner-0.6B-hf`, downloads its safe
safetensors checkpoint, and installs Transformers 5.13+, TorchCodec, Nagisa,
and SoyNLP. Audio is bounded before inference, resampled to 16kHz mono, and
limited by `MUSE_AUDIO_ALIGNMENT_MAX_BYTES` (50 MB),
`MUSE_AUDIO_ALIGNMENT_MAX_DURATION_SECONDS` (300s), and
`MUSE_AUDIO_ALIGNMENT_MAX_TEXT_CHARS` (50,000). The curated alias is
`qwen3-forced-aligner-0.6b` (Apache 2.0; 11 languages). Before audio decode,
the runtime tokenizes the reference and rejects more than 2,048 alignable
words; after processor preparation it enforces the checkpoint's 8,192-token
context before moving inputs to the device. Confidence calculation selects
timestamp-marker logits before its float32 softmax, avoiding a full-sequence
probability tensor. Only the typed `AudioAlignmentDecodeError` emitted at the
TorchCodec boundary maps to HTTP 415; model alignment failures remain 500s.

TorchCodec decoding for alignment, audio/quality and Qwen3-ASR lives in one
module, `muse.modalities._common.audio_decode` (`decode_audio`,
`WindowedAudio`). It raises typed errors: `AudioDecodeError` (undecodable
upload, 415; `AudioAlignmentDecodeError` is an alias),
`AudioDurationExceededError` (413) and `AudioDecoderUnavailableError` (a
missing TorchCodec/FFmpeg, 500). Routes catch only those types. The
message heuristic `_common.uploads.looks_like_audio_decode_error` remains
only for libraries that raise untyped errors (faster-whisper's PyAV,
librosa, soundfile): it matches decoder exception class names and
decoder-specific phrases, never bare "decode"/"ffmpeg"/"torchcodec", and
never an ImportError.

`decision/evaluation` is muse's twenty-fourth modality (docs/DECISIONS.md).
`POST /v1/systemone` takes `{model, state, questions}` (a text-only subset of
TypeSafe's System One: Choice, Score, Noul; `model` is always required, no
default) and returns `{model, answers, usage}` built by the shared codec from
complete per-question distributions; confidence is concentration (Choice
`(max p - 1/K)/(1 - 1/K)`, Score an ordered-spread statistic), never
correctness, and `output_tokens` is 0. `validation.parse_request` is the one
strict parser (raw bytes, duplicate keys at any depth, non-finite numbers,
unknown/media fields, UTF-8 byte limits); the package exports
`STRICT_REQUEST_PARSERS = {"/v1/systemone": "...:parse_request"}`, a generic
AST-read seam the gateway runs on the raw body BEFORE model selection (its
lenient `json.loads` would keep the last of two `model` keys), routing by the
parsed `.model`. Two runtimes: `DecisionLlamaCppModel` renders
evalengine/decision-0.8b-gguf's published no-thinking prompt (framing parsed
for special tokens, user JSON as literal text), rejects any question over
2,048 tokens before evaluating any, and softmaxes `get_logits_ith(-1)` over the
option letters A-X (never `Llama.scores`, empty with `logits_all=False`);
`DecimaOnnxModel` uses decima-ai 2.2.0's sessions/tokenizer/ordinal head but
not its `decide*`/System One paths (they truncate, shorten options and may
download), and rejects state+question over 512 tokens or an option over 64.
Both reset/isolate per question and mark themselves `usable = False` after a
native failure; `muse.core.server`'s `/health` returns 503 when any registered
backend reports `usable is False`, so the supervisor's monitor replaces the
worker (a generic hook). The decision GGUF runtime is in
`muse.core.placement.NATIVE_GGUF_BACKENDS`, so admission resolves it on the
native-GGUF pool. The priority-80 HF plugin claims exactly the two repos in
its `SOURCES` table at their pinned commits (a moved Hub head is refused; the
curated ids carry the pin), downloads exact allow-lists and checks each pinned
file's size and the large files' SHA-256; this fixed the old misdispatch of
Decima (card tag zero-shot-classification) to the NLI classifier and of the
Decision GGUF to the chat runtime. In the worker, one `DecisionExecutor` per
model runs batches on its own thread behind a cap-1/32-deep `ConcurrencyGate`
and releases the slot from the native call's completion callback, so a
cancelled request cannot free the backend while native work continues.
Cancellation does stop the batch at the next question boundary: `evaluate`
takes `should_stop` and raises `DecisionCancelled` (native state stays
usable). The worker route wraps execution in
`muse.core.disconnect.unless_disconnected` (Starlette/uvicorn never cancel a
non-streaming handler on client disconnect, so it races the work against
the `http.disconnect` message), and the package exports
`CANCEL_ON_DISCONNECT_PATHS = ("/v1/systemone",)`, a generic AST-read seam
(`discovery.disconnect_cancellable_paths`) that makes the gateway close its
worker connection, release its slot/refcount and record 499 when the
client leaves while the worker is still producing headers. Only declare it
where the worker owns admission, or a freed gateway slot could put two
runs on one backend.
PROBE_DEFAULTS runs one Choice, Score and Noul question. No per-model
inference authorization exists in muse, so there is no 403 path.

Text routes (`/v1/moderations`, `/v1/text/classifications`, `/v1/embeddings`,
`/v1/rerank`, `/v1/summarize`, `/v1/translate`) share
`muse.modalities._common.inference`: `run_locked` runs the backend call
off-loop under its `_inference_lock` and turns a failure into an opaque 500,
and `check_text_batch` applies the batch and per-item caps. Every cap is read
per request with `config.get_positive`, so 0 or a bad value falls back to
the setting's default rather than rejecting every request (the same holds
for `MUSE_ASR_MAX_MB`). Zero-shot classification also bounds inputs x
`candidate_labels` at 16,384 NLI pairs per request (400 `invalid_parameter`).

`text/rerank` is muse's first Cohere-compat modality (rather than
OpenAI-compat): OpenAI has no rerank API, and Cohere's `/v1/rerank` is
the de-facto standard that downstream tooling (LangChain, LlamaIndex,
Haystack) expects. Response envelope mirrors Cohere's: `results[]`
with `index` + `relevance_score`, optional `document.text`, plus
`meta.billed_units.search_units` for SDK compatibility.

`text/summarization` is muse's second Cohere-compat modality (after
text/rerank). OpenAI has no summarization API; Cohere's `/v1/summarize`
was the de-facto reference until its 2024 deprecation, and the wire
shape is what summarization tooling expects. Request: `{text, length,
format, model}`. Response: `{id, model, summary, usage, meta}`.
`length` ("short"|"medium"|"long") deterministically maps to
`max_new_tokens` in the runtime: short=80, medium=180, long=400.
`format` ("paragraph"|"bullets") is recorded in `meta.format` and is
metadata-only for non-instruction summarizers like BART-CNN; future
instruction-tuned summarizers can consult it. The bundled
`bart-large-cnn` (Apache 2.0, ~400MB, CPU-friendly) is the default;
the curated `bart-cnn-samsum` (`supports_dialog_summarization=true`)
is dialog-tuned. The HF resolver sniffs any `summarization`-tagged
repo at priority 110 and serves it via `BartSeq2SeqRuntime` over
`transformers.AutoModelForSeq2SeqLM` (BART, PEGASUS, T5).

`text/translation` (v0.58.0) is muse's twentieth modality and its
first LibreTranslate-compat one -- self-hosted translation clients
(LibreTranslate SDKs, the `libretranslate` CLI) work against muse
unmodified. Wire shape: `POST /v1/translate` and the bare `POST
/translate` alias (LT clients hardcode that path) take `{q, source,
target, format?, model?}` (`q` is a string or list[str]; scalar in ->
scalar out) and return `{translatedText}`; `GET /languages` returns
`[{code, name, targets}, ...]` derived live from the resolved model's
tokenizer, not a hand-maintained table. `model` is a muse extension
LT clients never send -- see the default-model seam below. Single
runtime `TranslationRuntime` over `transformers.AutoModelForSeq2SeqLM`
+ `AutoTokenizer`, dispatching per family (`_family_for(hf_repo)`,
mirroring the 3D modality's precedent): m2m100 and NLLB use
`forced_bos_token_id` language-token injection (NLLB via a structured
ISO-639-1 -> FLORES-200 table); Opus-MT checkpoints are fixed-pair
(`capabilities.source_language`/`target_language`, any other pair
400s `invalid_language`); MADLAD-400 (curated-only, GPU-oriented)
prepends a T5-style `<2xx>` target prefix. The bundled `m2m100-418m`
(facebook/m2m100_418M, MIT, ~2GB, `device: auto`, covers all pairs
across 100 languages) is the default; curated additions are
`nllb-200-distilled-600m` (**CC-BY-NC-4.0**, 200 languages, non-
commercial license called out in its description) and the permissive
per-pair `opus-mt-en-es` / `opus-mt-en-de` (Helsinki-NLP, ~300MB
each, best per-pair quality). The HF resolver plugin sniffs
`translation`-tagged repos (plus `text2text-generation` + name-
pattern fallback) at **priority 109** -- one tier below the 110 group
that includes text/summarization's plugin -- so translation's both-
tagged disambiguation rule (a repo tagged both `summarization` and
`translation` goes to translation only on a translation-family name
match) is actually exercised by real `HFResolver` dispatch instead of
being shadowed by summarization's unconditional `"summarization" in
tags` sniff (an as-built fix; the design originally shipped this
plugin at the same 110 tier, which made the rule correct in isolated
unit tests but unreachable in practice since `"text/summarization" <
"text/translation"` alphabetically). Because LibreTranslate clients
never send `model`, the modality exports `MODEL_OPTIONAL_PATHS =
("/v1/translate", "/translate", "/languages")`; `discover_modalities`
aggregates every modality's declared paths into a `{path: modality}`
map, and the gateway's catch-all proxy resolves `model` from that map
(first ENABLED catalog model of the matching modality, by discovery
order -- bundled m2m100 wins on a fresh install) instead of 400ing
`model_required` when a declared path's request omits `model`; no
enabled model of that modality -> 503 `no_default_model`. Any other
path with a missing `model` keeps today's 400 exactly. This seam is
generic: a future modality can opt in by exporting its own
`MODEL_OPTIONAL_PATHS`, no gateway changes needed. Input size is
capped by `limits.translate_max_chars` (env `MUSE_TRANSLATE_MAX_CHARS`,
default 20000, read per-request) -> 400 `input_too_long`.
`source: "auto"` (language auto-detection) and `POST /detect` are
explicitly deferred; v1 returns a structured 400
`source_detection_not_supported`.

`image/embedding` is muse's first image-to-vector modality. The wire
envelope at `POST /v1/images/embeddings` mirrors `/v1/embeddings`
exactly (`{object: "list", data, model, usage}`) so OpenAI SDK clients
that already consume embeddings can reuse helper code. Each `input`
entry is a `data:image/...;base64,...` URL or `http(s)://...` URL
pointing at PNG/JPEG/WEBP; image decoding goes through the shared
`muse.modalities._common.images.decode_image_input` helper, up to
`limits.image_embeddings_decode_concurrency` (default 8) inputs at a time
within one `limits.image_embeddings_decode_deadline_seconds` (default 60 s)
deadline (a missed deadline is 400 `invalid_parameter`). The response `model` is the registry id.
The bundled `dinov2-small` (Apache 2.0, 88MB, 384-dim, CPU-friendly)
is the default; curated additions cover SigLIP2 and CLIP. The HF
resolver sniffs any repo with an image-feature-extraction-class tag
plus a `preprocessor_config.json` sibling at priority 105 (between
embedding/text and image-generation file-pattern) and serves it via
`ImageEmbeddingRuntime` over `transformers.AutoModel` +
`AutoProcessor`. The runtime's `_extract_embeddings` dispatch picks
the right pooling per architecture: CLIP `image_embeds` >
SigLIP/DINOv2 `pooler_output` > DINOv2 base `last_hidden_state[:, 0]`
(CLS token).

`audio/embedding` is muse's first audio-to-vector modality. Wire shape
is multipart-in (one or more `file` parts, mirroring
`/v1/audio/transcriptions`) and `/v1/embeddings`-shaped JSON out
(`{object: "list", data, model, usage}`). Audio decoding goes through
`librosa` (already installed for Whisper) inside the runtime/script,
which resamples on the way in to each model's preferred rate (CLAP
48kHz, MERT 24kHz). The bundled `mert-v1-95m` (CC-BY-NC-4.0
non-commercial, 95M parameters, 768-dim music understanding via
mean-pool over time, `trust_remote_code=True` for its custom model
architecture/config) is the default; the curated
`clap-htsat-fused` adds 512-dim audio + text-aligned embeddings
(BSD-3, supports_text_embeddings_too=True). The HF resolver sniffs
any repo with `feature-extraction` tag plus a name pattern matching
`clap`, `mert`, `audio-encoder`, `wav2vec`, or `audio-embedding` at
priority 105, and serves it via `AudioEmbeddingRuntime` over
`transformers.AutoModel` + `AutoFeatureExtractor` (with
`AutoProcessor` preferred for newer repos). The runtime's
`_extract_embeddings` dispatch picks the right pooling per
architecture: CLAP `audio_embeds` > pooler_output > MERT/wav2vec
`last_hidden_state.mean(dim=1)` (mean-pool over time). Per-file size
cap via `MUSE_AUDIO_EMBEDDINGS_MAX_BYTES` (default 50MB), and an aggregate
cap of max(64 MiB, the per-file cap) across up to 32 files. Duration is
bounded per model by `capabilities.max_duration_seconds` (default 60s):
librosa stops decoding there (`duration=`), so a long file is never fully
materialized. audio/classification takes the same capability (default
60s). The four upload routes (alignment, classification, quality,
embedding) resolve the model before reading the upload, so an unknown
model 404s without reading up to 50 MB, and `run_on_spooled_upload` writes
the temp file on the worker thread.

`audio/generation` is muse's first modality with TWO URL routes mounted
on ONE MIME tag. `/v1/audio/music` and `/v1/audio/sfx` share the same
request body, codec, registry surface, and runtime. The only per-route
difference is the manifest capability key consulted: `supports_music`
or `supports_sfx`. When a flag is False (or a future MusicGen-only
model lacks `supports_sfx`), the unsupported route returns 400. The
two-URL split is for legibility: a "footsteps on gravel" prompt sent
to `/v1/audio/music` would silently produce a 30-second loop of
footsteps treated as music; routing the same prompt to `/v1/audio/sfx`
makes the user's intent explicit to operators reading logs and to the
model itself.

ACE-Step v1 3.5B (v0.49.0) is the second audio/generation backend and
muse's first *song* model: a genre/style `prompt` plus optional
structured `lyrics` (`[verse]`/`[chorus]` tags) produce a sung song;
empty/None lyrics produce an instrumental (the runtime substitutes
ACE-Step's literal `[instrumental]` tag). It is a bundled script
(`ace-step-v1-3.5b`) that aliases the shared `AceStepRuntime` as `Model`
(the VLM-bundled pattern), declares `supports_music: true` /
`supports_sfx: false` (so `/v1/audio/sfx` returns 400 for it), and pins
`device: cuda` (3.5B is impractical on CPU). The optional `lyrics` field
(max 8000 chars) was added to `AudioGenerationRequest` and the
`generate` protocol; models that ignore lyrics (Stable Audio) simply
drop it. The route `duration` ceiling was raised 120s -> 240s for
ACE-Step's long clips; the per-model runtime still clamps to its own
`max_duration`. Unlike StableAudio (which returns in-memory numpy
arrays), ACE-Step *writes WAV file(s) to disk and returns the path(s)*:
`AceStepRuntime` hands the pipeline a temp `save_path`, resolves the
on-disk output defensively (returned path if present, else the
`save_path`), reads it back via `soundfile`, and deletes the temp dir in
a `finally` (leak-safe, per the #200 temp-WAV lesson). `trust_remote_code`
is NOT needed (`from acestep.pipeline_ace_step import ACEStepPipeline` is
a direct package import). Install pins the git source (`ace-step @
git+https://github.com/ace-step/ACE-Step.git` -- the distribution name
is `ace-step` with a hyphen even though the import is `acestep`; the
PyPI `ace-step` is a stale v0.1.0). `pip_extras` also pins `torchcodec`:
ACE-Step's `save_wav_file` calls `torchaudio.save`, which on modern
torchaudio (>=2.8) delegates encoding to torchcodec (backed by ffmpeg's
libav, already in `system_packages`); without it generation fails at
save time. Both the dist-name and torchcodec gaps were caught by
real-API verification on the GPU box (Step B1), not by the fully-mocked
unit tests. The `cpu_offload` / `overlapped_decode` / `torch_compile`
low-VRAM knobs are exposed as optional manifest capabilities (default
off) and forwarded to the pipeline constructor.

Image inputs and the error table (all image, video and 3D routes, and
chat's VLM path). Ingress lives in `muse.modalities._common.images`: the
bounded read or URL fetch is async I/O, and the pixel decode (uploads and
data URLs alike) runs in a worker thread. Its typed errors map to one
table, adopted from image/pixelart in this release (a behaviour change:
these were 400 `invalid_parameter` on most routes): byte cap -> 413
`payload_too_large`; side or pixel cap (per image or the batch total) ->
413 `input_too_large`; empty, malformed or undecodable -> 400
`invalid_image`; a refused or failed URL fetch and other bad fields -> 400
`invalid_parameter`. An http(s) image URL whose body exceeds the byte cap
is 413 `payload_too_large` too (`muse.core.net_fetch` raises the typed
`ResponseTooLargeError`). PIL's
decompression-bomb refusal counts as the pixel cap (413). Size messages
name the setting that raises the cap.
Out-of-range JSON fields are still 422 `invalid_request` (Pydantic); their
multipart twins (`/edits`, `/variations`, `/upscale`) are 400
`invalid_parameter` with the same ranges, via `_common/fields.py`. Routes
run the backend call AND the output encoding (PNG/WebP/GIF/MP4/GLB/depth/
masks) off the event loop through `_common/inference.offload_call`
(n-result routes use `produce_each`: lock per result, each result closed
once encoded, remaining results skipped after a cancel), so a long encode
can no longer stall `/health` into a monitor restart. Backend failures
reach the app's generic 500 handler.

The `/v1/images/generations` route also accepts optional `image` (data URL or http(s):// URL) + `strength` (0.0 to 1.0, default 0.5) fields for img2img since v0.17.0. OpenAI SDK clients pass them via `extra_body`:

    client.images.generate(prompt="oil painting", model="sdxl-turbo",
                           extra_body={"image": "data:image/png;base64,...", "strength": 0.6})

Models advertise support via `capabilities.supports_img2img`. Requests for non-supporting models return 400.

The `image/generation` modality also exposes `/v1/images/edits` (inpainting) and `/v1/images/variations` (alternates of one image, no prompt) since v0.21.0. Both are multipart/form-data routes that mount on the same modality. Inpainting takes `image` + `mask` + `prompt` and routes to `backend.inpaint(...)`, which lazy-loads `AutoPipelineForInpainting.from_pipe(self._pipe)` to share VRAM with the loaded t2i pipeline. Variations takes `image` only and routes to `backend.vary(...)`, which delegates to the existing img2img path with empty prompt and high strength (default 0.85). Capability flags `supports_inpainting` and `supports_variations` gate the routes; OpenAI SDK clients use `client.images.edit(image=..., mask=..., prompt=..., model=...)` and `client.images.create_variation(image=..., model=...)` natively.

Multi-reference edits and gpt-image request fields (v0.68.0, spec docs/superpowers/specs/2026-10-22-local-image-models-design.md). `/v1/images/edits` merges `image` parts then `image[]` parts (the OpenAI SDK list shape) and `mask` is optional. With a mask: exactly one image, inpaint as before (unused `steps`/`seed`/`guidance`/`negative_prompt` are reported as ignored). Without a mask: the model must declare `supports_reference_images` (else 400 `reference_images_not_supported`, pointing at `/v1/capabilities?modality=image/generation`), at most `max_reference_images` (400 `too_many_images`; `MAX_EDIT_IMAGES = 16` is checked before any decode), each reference at least 64 px with aspect ratio at most 8:1; the route calls the optional `backend.edit(prompt, images=[...], ...)` (`ReferenceImageModel` protocol). Both routes go through `image_generation/options.py`: `size` omitted or `auto` means the model's `default_size` only for models that declare `size_multiple`/`max_pixels` (others keep 512x512); those models get sizes scaled down (never up) to `max_pixels` and the quality preset's cap, then rounded down to the multiple; `quality` low/medium/high/auto (dall-e `standard`/`hd` alias medium/high) maps to `capabilities.quality_presets`; `output_format` png/jpeg/webp with `output_compression`; `background` auto/opaque (`transparent` is 400 `background_not_supported`); `moderation` and `style` are accepted and ignored. Responses add `size` (as produced), `quality`, `output_format`, `background` and `metadata.ignored` (accepted fields that had no effect); models without quality presets, and variations, echo `quality: "auto"`/`background: "auto"`, values OpenAI does not document for responses (its Python SDK does not validate them). `runtimes/reference.py:DiffusersReferenceModel` loads any diffusers pipeline class named by `capabilities.pipeline_class` that takes `image=[...]`, placed by `muse.modalities._common.offload.place_pipeline(..., override_key=None)` (the video-only `server.video_cpu_offload` override never applies to image models). The HF plugin routes `flux.2-klein`/`flux2-klein` repos (not `base` ones, unless a curated entry for that repo sets `reference_runtime: true`, which also gives it and LoRAs on it undistilled presets: 28 steps, real guidance 4.0, low/medium/high 20/28/50) to it with `cpu_offload: model`, 4-step presets, `max_reference_images: 4`, `size_multiple: 16`, `max_pixels: 4194304`; its download allow-list includes `*/*.jinja` (klein's Qwen3 chat template lives only in `tokenizer/chat_template.jinja`). Klein's offload parks ~16 GB in host RAM that the LoadDirector does not count. The image probe also runs one `edit` for models with `supports_reference_images`, and fails (so a pull disables the model) when the generate or edit output looks degenerate (`image_generation/probe_check.py`, numpy+PIL imported lazily in the model venv): on luminance box-downsampled to <=256 px, std < 2.0 (flat) or lag-1 spatial autocorrelation (mean of horizontal and vertical) < 0.5 (noise-like); the autocorrelation test is skipped when the source image's short side (before downsampling) is under 128 px (1x pixel art reads ~0.45 at 16 px, and nearest-upsampling would hide noise). Calibrated: noise reads ~0, 57 local real images >= 0.62 (raw generations >= 0.84). LoRA on klein: the reference runtime accepts `lora_adapter` / `base_model` and a per-request `lora_scale` (`set_adapters` with `lora_scale * capabilities.lora_scale_factor`; shared base resolution in `runtimes/_lora.py:resolve_lora_base`). When `lora_scale_factor` is not set, both LoRA runtimes (this one and the SDXL diffusers one, whose `cross_attention_kwargs` scale it multiplies) derive it at load from the adapter file's safetensors header (stdlib only, no tensor bytes; `_lora.resolve_lora_scale_factor`): sqrt(r)/lora_alpha when diffusers' `__metadata__["lora_adapter_metadata"]` declares `<component>.use_rslora` with one r/alpha across components, no rank/alpha patterns, and the file has no `.alpha` tensors (peft would scale by alpha/sqrt(r), ComfyUI by 1); else 1.0, with a warning when the header is corrupt or ambiguous. An explicit capability always wins. The HF plugin routes klein-based LoRAs to the reference runtime, preferring `pytorch_lora_weights.safetensors`. The `prompt_prefix` capability is prepended by the route unless the prompt already contains its phrase (the prefix without its trailing separator), case-insensitively (`metadata.prompt_prefix_applied`). Curated `klein-pixel-art` (hf://Limbicnation/pixel-art-lora, base `flux2-klein-4b`, 512x512, 4 steps, prefix "pixel art sprite, ", `lora_scale_factor: 0.0625` because its rsLoRA metadata makes diffusers scale it 16x and an adapter weight of 0.25+ destroys the image, so `lora_scale` 1.0 equals the card's ComfyUI strength 1.0) feeds `/v1/images/pixelize`. Curated `flux2-klein-base-4b` (opted in) carries `klein-pixel-walk` (hf://svntax-dev/pixel_spritesheet_4walk_small_lora_v1, a plain rank-32 ai-toolkit LoRA so no `lora_scale_factor`; `prompt_prefix` is the card's 4x4 template sentence; 512x512): one image is a 4x4 walk sheet (down/left/right/up rows), pixelized with `sheet=4x4 grid=4 palette=auto:24` to 32x32 frames (the go/no-go spike passed 4/4; docs/benchmarks/2026-10-28-pixel-assets-acceptance.md). When a LoRA repo ships its final weight next to numbered intermediate checkpoints (`<stem>_<digits>.safetensors`), the plugin downloads only the final one. An unprobed LoRA (any manifest with `capabilities.base_model`, no `memory_gb` and no measurement of its own) is sized by the supervisor's ladder from its base's catalog entry, found by muse id or, for a raw `hf://` pull whose Hub tag names an HF repo, by that entry's `hf_repo`: the base's probe measurement for the LoRA's pool, else its `memory_gb`, never the base's weights on disk (see "Lazy load + LRU eviction"). When the base has neither (or is not in the catalog), the LoRA is sized from its own adapter weights, so a non-curated `hf://` klein LoRA pulled with `--no-probe` starts under-sized until probed; the pull probe measures it and later loads self-heal the estimate upward. `scripts/bench/image_compare.py` runs a case file (or a youtube-channel `visuals.yaml`, read-only) through muse models and, with `OPENAI_API_KEY`, gpt-image-2, writing `results.json` and an `index.html` contact sheet.

`image/upscale` (v0.25.0) is muse's super-resolution modality: a separate MIME tag from `image/generation` because the runtime backbone is different (`StableDiffusionUpscalePipeline`, not `AutoPipelineForText2Image`). Wire shape at `POST /v1/images/upscale` is multipart/form-data (mirroring `/v1/images/edits`), with `image` as the source file plus `model`, `scale`, optional `prompt`, `negative_prompt`, `steps`, `guidance`, `seed`, `n`, and `response_format` as Form fields. Output envelope mirrors `/v1/images/generations`: `{created, data: [{b64_json|url, revised_prompt}]}`. The bundled `stable-diffusion-x4-upscaler` (CreativeML OpenRAIL++-M, ~3GB, fixed 4x) is the default; the HF resolver plugin (priority 105) sniffs other diffusers-shape upscalers (`model_index.json` + `image-to-image` tag + upscaler-name allowlist). The `supported_scales` capability gates the request `scale` parameter (returns 400 for unsupported values; SD x4 supports `[4]` only); an omitted `scale` uses `capabilities.default_scale`. An env-tunable input-side cap (`MUSE_UPSCALE_MAX_INPUT_SIDE`, default 1024; over it is 413 `input_too_large`) prevents runaway VRAM use on oversized inputs; the cap is read per-request, so changes take effect on the next request, not at supervisor restart. GAN-based upscalers (AuraSR, Real-ESRGAN) need separate non-diffusers runtimes and are deferred to v1.next.

`image/vectorization` (v0.59.0) is muse's raster-to-SVG modality. `POST /v1/images/vectorize` accepts multipart `image` plus optional `model`, `max_new_tokens` (1-7680), `temperature` (0-2), `top_p`, `num_beams` (1-8), `seed`, and `response_format` (`json` or `svg`). JSON returns `{id, object, model, mime_type, svg, source_size, width, height, view_box, seed, usage, metadata}`; raw mode returns `image/svg+xml`, ready to save or pass to an SVG consumer such as Manim. The bundled and curated `starvector-1b-im2svg` model uses the Apache-2.0 official mixed-precision checkpoint (~5.14GB on disk, loaded fp16 with ~7GB working memory, suitable for a 12GB GPU). Supply-chain policy is exact-only: `muse pull` pins HF revision `380ab95d25a8e9ab1dc825debe238b4953ae13b9`, downloads only config/tokenizer/safetensors data, and instantiates Muse's audited inference-only architecture; there is intentionally no broad HF resolver and no `trust_remote_code`. The output boundary accepts a static SVG subset and rejects scripts, event handlers, foreign objects, embedded rasters, remote/data URLs (protocol-relative `//host` included), backslashes in attribute or style values (a CSS escape such as `\75rl(` spells `url(`), DTDs/entities, oversized documents, or excessive element nesting. Input long side defaults to 2048 pixels and is configurable with `MUSE_VECTORIZATION_MAX_INPUT_SIDE`. This model is aimed at icons, logos, flat illustrations, and technical diagrams rather than photographs.

`image/segmentation` (v0.26.0) is muse's promptable-segmentation modality. Wire shape at `POST /v1/images/segment` is multipart/form-data: `image` as the source file plus `model`, `mode` (auto/points/boxes/text), `prompt` (text mode), `points` (JSON-encoded `[[x, y], ...]`), `boxes` (JSON-encoded `[[x1, y1, x2, y2], ...]`), `mask_format` (`png_b64` or `rle`), and `max_masks` as Form fields. Output: `{id, model, mode, image_size, masks: [{index, score, mask, bbox, area}]}`. Mode dispatch is capability-gated end-to-end: `supports_automatic`, `supports_point_prompts`, `supports_box_prompts`, `supports_text_prompts`. The gate defaults to deny: a mode whose flag the manifest does not declare as true returns 400 before the runtime is invoked. The bundled `sam2-hiera-tiny` (Apache 2.0, ~40MB, point/box/auto, no text) is the default; curated `sam2-hiera-base-plus` and `sam2-hiera-large` extend the family. The HF resolver plugin (priority 110) claims only what `SAM2Runtime` (`AutoModelForMaskGeneration`) can load: `mask-generation`-tagged repos, plus `image-segmentation`-tagged repos with a SAM-family name; SegFormer, Mask2Former, OneFormer, BiRefNet, DETR-panoptic and CLIPSeg are not claimed (text prompts need a CLIPSeg runtime, deferred). Mask formats: `png_b64` (portable, viewable) or `rle`, muse's own compact run-length format `{"size": [H, W], "counts": "p:<r0>,<r1>,..."}` (column-major runs alternating 0/1, starting with the leading 0-run). It follows COCO's size and order conventions but is NOT COCO's compressed counts, so pycocotools cannot decode it; use `image_segmentation.codec.decode_mask_rle` or split the counts on commas. Encoding is vectorised with numpy and runs off the event loop. Axis-order discipline at the wire layer: `image_size` is `[W, H]` (PIL convention); RLE `size` is `[H, W]` (COCO convention); `bbox` is `[x, y, w, h]` (COCO bbox convention).

`image/cv` (v0.37.0) is muse's image-CV-primitives umbrella, hosting three routes on one MIME tag: `POST /v1/images/depth` (depth estimation), `POST /v1/images/keypoints` (pose / keypoint detection), `POST /v1/images/detect` (object detection). Three sibling runtimes: `HFDepthRuntime` over `AutoModelForDepthEstimation`, config-dispatched `HFKeypointRuntime` (explicit `VitPoseForPoseEstimation` for ViTPose; the generic auto class for other supported keypoint families), and `HFObjectDetectionRuntime` over `AutoModelForObjectDetection`. Capability flags `supports_depth` / `supports_keypoints` / `supports_detection` on each model's manifest gate which route accepts it; mismatch returns 400 `wrong_primitive`. The depth route's `response_format` field selects between `png16` (16-bit grayscale PNG, default; quantized into the value range so the full 16-bit precision tracks the actual depth bounds) and `float32` (raw little-endian bytes, fully precise but ~4x payload). The keypoint runtime's v1 ViTPose flow takes a single full-image bbox per call; multi-person pose extraction needs an upstream person detector and is deferred. Bundled defaults: `depth-anything-v2-small` (relative depth, ~25M, Apache 2.0), `vitpose-base-simple` (COCO 17-keypoint pose, ~85M, Apache 2.0; Transformers >=4.48 plus SciPy), `detr-resnet-50` (COCO 80-class detection, ~41M, Apache 2.0; `pip_extras` includes `timm` for the ResNet backbone). Curated additions: `depth-anything-v2-base` (larger), `zoedepth-nyu-kitti` (metric meters), `dpt-large` (classic), `rtdetr-r50vd` (faster detection). The single HF plugin (priority 110) dispatches per primitive via tag (`depth-estimation`, `keypoint-detection`, `object-detection`) with repo-name fallbacks matched on the repo name only (not the org) and naming loadable families (`depth-anything`, `dpt`, `zoedepth`, `midas`; `detr`, `yolos`, `rt-detr`; no bare `depth`, which matched ControlNet-depth repos, and no OWL-ViT, a zero-shot detector outside `AutoModelForObjectDetection`); `metric_depth: True` auto-derives from the repo name (`zoedepth`, `metric`).

`image/ocr` (v0.36.0) is muse's image-to-text modality. Wire shape at `POST /v1/images/ocr` is multipart/form-data: `image` as the source file plus optional `model`, `prompt` (Nougat-style task hint; TrOCR ignores), and `max_new_tokens` (validated to [1, 4096]). Output envelope mirrors `/v1/audio/transcriptions`: `{id, model, text, usage: {completion_tokens}}`. Single runtime `HFVision2SeqRuntime` over `transformers.AutoModelForVision2Seq` + `AutoProcessor`; the processor handles per-family preprocessing (TrOCRProcessor, NougatProcessor) automatically. The bundled `trocr-base-printed` (MIT, 334M, English printed text) is the default; curated additions cover handwriting (`trocr-large-handwritten`, MIT), academic papers with math/LaTeX (`nougat-base`, **CC-BY-NC-4.0**, non-commercial), and math-formula-only (`texteller`, MIT). The HF plugin (priority 110) sniffs `image-to-text`-tagged repos but explicitly excludes `image-text-to-text` (those are VLMs, reserved for the future #97 `image/description` modality). Capability flags `supports_handwritten` and `supports_math` are advisory only (surfaced in `/v1/models` for client-side filtering, not server-enforced). Donut/DocVQA structured-document variants are out of scope for v1; they may land later as a separate `image/document` modality.

`image/pixelart` (spec docs/superpowers/specs/2026-10-28-retro-pixel-assets-design.md, recipe docs/PIXELART.md) is muse's pixel-art toolbox: CPU-only, deterministic, no weights. `POST /v1/images/pixelize` takes multipart `image` plus `grid` (`auto`/N) or `target_size` (mutually exclusive), `palette` (`pico-8`, `db16`, `db32`, `endesga-32`, `nes`, `gameboy`, `auto:N`, `none`, or `#rrggbb,...`; default `auto:32`), `background`, `remove_enclosed` (default false; with `background=auto` also keys enclosed background-coloured regions of at least max(4 cells, 0.05% of the grid), measured after the grid is known and before the cell vote), `alpha_threshold`, `cleanup` (default: clean only after a grid snap), `outline`, `seamless`, `sheet`, `preview_scale` and `response_format` (`json` -> `{id, object: "image.pixelize", model, data, grid, size, palette, background_removed, metadata}`, or raw `png`). `POST /v1/images/autotile` takes `a2` (an existing RPG Maker block, alpha-hardened at 128) OR `fill` (+ optional `base`) plus edge style fields, `tile_size`, `palette`, `name` and `response_format` (`json` with `files.{a2,blob47}.b64_json`, `godot_tres`, `tiled_tsx` and `layout`, or a `zip` of `<name>_a2.png`, `<name>_blob47.png`, `<name>.tres`, `<name>.tsx`, `<name>_layout.json`). The blob sheet follows RPG Maker MV `FLOOR_AUTOTILE_TABLE` (rpgtkoolmv/corescript@182e3144, MIT; attribution in the source), keeps shape 46 and drops 47; `BlobTile` x/y are in tile units and `mask` is the reduced 8-neighbour bitmask (N=1, NE=2, E=4, SE=8, S=16, SW=32, W=64, NW=128). Caps: `limits.pixelart_max_input_side` (`MUSE_PIXELART_MAX_INPUT_SIDE`, default 4096, read per request) bounds each input side and the pixelize preview's long side (art size x `preview_scale`, rejected before the preview is allocated) -> 413 `input_too_large`; byte-cap overflow is 413 `payload_too_large`. Both paths are in the package's `MODEL_OPTIONAL_PATHS`, so a request without `model` (JSON or multipart) gets the first enabled image/pixelart model through the gateway's default-model seam. The runtime `PixelTools` (Pillow + numpy, imported lazily) is the bundled `pixel-tools` (CPU, `memory_gb` 1.5) and is a weightless model (below). `scripts/bench/pixel_compare.py` runs `pixel_cases.yaml` (8 sprite prompts x `db32` / `auto:16` / `db32` at `target_size` 64x64, plus a grass-over-dirt autotile; generated sprites have no consistent grid, so `grid=auto` usually reports none and `target_size` is the recipe's default for them) against a live server and writes `results.json` plus an `index.html` sheet whose autotile section paints a random 16x12 map from the blob tiles by mask so seams show.

`video/generation` (v0.27.0) is muse's narrative-clip modality, the heaviest yet. Wire shape at `POST /v1/video/generations` is JSON-only (no multipart) with `prompt` plus optional `model`, `duration_seconds` (0.5 to 30), `fps` (1 to 60), `size` (WxH string), `seed`, `negative_prompt`, `steps`, `guidance`, `response_format` (`mp4` default, `webm`, or `frames_b64`), and `n` (capped at 2 because each video is heavy). Output envelope mirrors `/v1/images/animations`: `{data: [{b64_json}], model, metadata: {frames, fps, duration_seconds, format, size}}`. Two distinct runtimes ship under the same MIME tag: `WanRuntime` (`diffusers.WanPipeline` or `DiffusionPipeline` fallback) and `CogVideoXRuntime` (`diffusers.CogVideoXPipeline`); the HF resolver dispatches per architecture. The bundled `wan2-1-t2v-1-3b` (Apache 2.0, 5s clips at 832x480) bundles a UMT5-XXL text encoder (~11GB at fp16) alongside its 1.3B transformer, so a full-resident load is ~11-12GB and OOMs a 12GB card; the earlier "~3GB / fits 8GB" claim counted only the transformer and was wrong. Curated additions: `cogvideox-2b` (~6GB at fp16, 6s at 720x480, fits 12GB) and `ltx-video` (~13GB, 30fps at 1216x704, requires 16GB+). The HF plugin (priority 105) sniffs `text-to-video`-tagged repos whose name (not org) matches one of `wan` (as a name token, so `wangfuyun/AnimateLCM` stays with image/animation), `cogvideox`, `ltx-video`, `mochi`, or `hunyuan`. LTX/Mochi/Hunyuan currently fall back to `WanRuntime`; their dedicated runtimes ship in v1.next. Distinction from `image/animation`: animation is short looping clips (16 frames @ 8fps = 2s, default `loop=true`, animated WebP), video is narrative clips (5s+, single play, no loop field, mp4). The mp4 / webm / frames_b64 encoders are shared with image/animation (`_common/video_codec.py`, one `to_pil` convention: float frames are [0, 1]); webm is vp9 with vp8 fallback (when ffmpeg lacks vp9). A missing imageio for mp4/webm is 400 before generation, and so is a `frames_b64` clip whose frame count (`duration_seconds` x `fps` x `n`) is known to exceed `MUSE_VIDEO_MAX_FRAMES_B64`; otherwise the cap is checked as results arrive. All bundled video models declare `device: "cuda"`; CPU inference would take 10 to 30 minutes per clip and isn't a useful default.

CPU offload (v0.52.1) is a per-model capability, not a bool: `capabilities.cpu_offload` is `"model"` (`pipe.enable_model_cpu_offload(device=...)`, whole-component granularity, peak VRAM ~ largest single component), `"sequential"` (`pipe.enable_sequential_cpu_offload(device=...)`, sub-module granularity, fits <=12GB but slower), or absent/`false` (today's plain `.to(device)` move, unchanged default for models that do not set it). Offload and `.to(device)` are mutually exclusive; offload applies on any non-cpu device (cuda, and mps too), the `device != "cpu"` guard. `capabilities.vae_tiling: true` calls `enable_vae_tiling()` plus `enable_vae_slicing()` (best-effort, guarded by `hasattr`) after placement, to cap the VAE-decode spike at higher resolutions. `WanRuntime` and `CogVideoXRuntime` (which the bundled `wan2-1-t2v-1-3b` script aliases as its `Model`) dispatch through one shared helper, `muse.modalities._common.offload.place_pipeline`. The registry setting `server.video_cpu_offload` (env `MUSE_VIDEO_CPU_OFFLOAD`, opt_str, default unset) globally overrides the per-model capability across every video model: `model`, `sequential`, or `off`/`false`/`none` to force plain placement even on a model that declares offload; unset falls through to the capability. `wan2-1-t2v-1-3b` sets `cpu_offload: "sequential"` and `vae_tiling: true` by default (guarantees load+generate on 8-12GB out of the box) and a corrected `memory_gb: 5.0` (an honest sequential-peak estimate; `muse models probe` self-heals it to the measured value). A bigger-card operator who wants speed over headroom sets `server.video_cpu_offload model` (or `off`).

`3d/generation` (v0.41.0 through v0.45.0) hosts four concrete runtimes under one MIME tag, dispatched per-family via `_family_for(repo_id)` in `model_3d_generation/hf.py`. `TripoSRRuntime` (v0.41.0+) is the bundled default for image-to-3D. `ShapERuntime` (v0.43.0+) wraps `diffusers.ShapEPipeline` for text-to-3D; it follows the deferred-imports pattern (module-level sentinels for `torch`, `ShapEPipeline`, `trimesh`; `_ensure_deps()` populates them lazily). `TRELLISRuntime` (v0.44.0+) wraps Microsoft's TRELLIS SDK via `transformers` AutoPipeline with `trust_remote_code=True` for image-to-3D. `Hunyuan3DRuntime` (v0.45.0+) is the fourth runtime and the first to support BOTH image-to-3D and text-to-3D from one model; it wraps Tencent's `hy3dgen` SDK with `trust_remote_code=True`. Internally, text-to-3D is a two-stage pipeline: text generates an image via `HunyuanDiTPipeline`, then the image is lifted to a mesh via `Hunyuan3DDiTFlowMatchingPipeline`. The `_Family.trust_remote_code` flag flows into the synthesized manifest's `capabilities.trust_remote_code`, which the runtime constructor reads via the kwargs splat. The per-family dispatch adds one `_Family` constant plus one dispatch branch in `_family_for` plus one runtime file under `runtimes/` per new runtime. Shap-E is text-only: the curated `shap-e` entry declares `supports_image_to_3d: False` and `supports_text_to_3d: True`; a request to `/v1/3d/from-image` with `model=shap-e` returns 400 before the runtime is invoked. TRELLIS is image-only: the curated entry (renamed from `trellis-text-image` to `trellis-image` in v0.44.0) declares `supports_text_to_3d: False`. Hunyuan3D-2 declares both flags True. GLB output flows through the existing codec: `trimesh.Trimesh` assembles vertices and faces from the pipeline output, then `mesh.export(file_type="glb")` yields the bytes. A companion `_pip_extras_for(runtime_path)` helper mirrors the dispatch shape so each runtime's venv installs only the deps it needs. After v0.45.0, task #108 is closed. Wonder3D is deferred indefinitely (multi-view plus NeuS reconstruction does not fit muse's one-pipeline-call runtime model). The HF plugin sniffs only family names with a runtime (TripoSR, Shap-E, TRELLIS, Hunyuan3D), matched on the repo name; an `image-to-3d` / `text-to-3d` tag alone, Wonder3D or InstantMesh no longer resolve (they fell back to TripoSR and failed at load). Curated pulls name the modality and skip the sniff, so `_resolve` applies the same family check and raises `ResolverError` for anything else; the old `wonder3d-base` curated entry is removed. `/v1/3d/from-image` normalizes the decoded image off the event loop: RGBA when it has transparency (so TRELLIS and Hunyuan3D see the alpha cutout), RGB otherwise (P, L, CMYK, I;16, ...); TripoSR converts to RGB itself. The byte cap (`MUSE_3D_INPUT_MAX_BYTES`) is 413 `payload_too_large`.

`chat/completion` (v0.42.0+) accepts vision-language models via the OpenAI multimodal `messages` shape: `content: [{type: text}, {type: image_url, image_url: {url: ...}}]`. The route's pre-dispatch step (`_decode_image_parts` in `routes.py`) walks `messages`, validates capability flags (`supports_vision`, `supports_multi_image`) on the loaded model, decodes each `image_url` via the existing `decode_image_input` helper (data URLs + `http(s)://` with SSRF guard), and rewrites the part to a muse-internal `{type: image, image: <PIL.Image>}` shape before calling the backend's `chat()` / `chat_stream()`. Capability mismatches return 400 `vision_not_supported` or `too_many_images`; bad images return 400 `invalid_image`. The runtime is `HFVisionLanguageModel` (`muse.modalities.chat_completion.runtimes.transformers_vlm`); bundled default is `smolvlm-256m-instruct` (~500MB, CPU-runnable). Curated extends with SmolVLM-Instruct, Qwen2-VL-2B/7B, and LLaVA-1.5-7B. Its stream opens with an `{"role": "assistant"}` delta; closing the stream (client disconnect) sets a stopping criterion and joins the `generate()` thread before the route releases the model lock, and a request `seed` runs under `torch.random.fork_rng` so it never reseeds the process RNG.
