# audio/transcription modality (Whisper family): design

**Date:** 2026-04-24
**Status:** approved
**Target release:** v0.13.0

## Goal

Add muse's 5th modality: `audio/transcription` served by a generic
`FasterWhisperModel` runtime over any Systran faster-whisper
(CT2-format) HuggingFace repo. OpenAI wire-compat for
`/v1/audio/transcriptions` and `/v1/audio/translations`. Three
newbie-friendly curated aliases: `whisper-tiny`, `whisper-base`,
`whisper-large-v3`.

## Scope

**In v1:**
- `POST /v1/audio/transcriptions` and `/v1/audio/translations`
- Multipart/form-data input (first muse modality with file uploads)
- 5 response formats: `json`, `text`, `srt`, `vtt`, `verbose_json`
- Word-level timestamps gated by `timestamp_granularities[]=word`
- Muse extension: `vad_filter: bool` form field (default false)
- Generic `FasterWhisperModel` runtime via faster-whisper (CT2 backend)
- HF resolver sniff for CT2 faster-whisper repos
- 3 curated entries
- `ffmpeg` in `system_packages` for audio decoding

**Not in v1 (explicitly deferred):**
- Streaming segments via SSE (OpenAI's Whisper endpoint doesn't stream either)
- Distil-Whisper or other Whisper-family variants in curated (resolver URI still works)
- Diarization
- Real-time microphone capture
- GGML/whisper.cpp path via pywhispercpp

## Backend choice

**faster-whisper** over pywhispercpp (whisper.cpp binding) or
openai-whisper. Rationale:
- Mature, well-maintained Python API.
- CT2 (CTranslate2) engine: int8 quantization on CPU, float16 on GPU, significantly faster than openai-whisper at equivalent quality.
- Mirrors the muse pattern: one generic runtime class (`FasterWhisperModel`) serving many Systran repos, analogous to `SentenceTransformerModel` and `LlamaCppModel`.
- pywhispercpp has less API stability; we lose the GGUF-style symmetry with chat/completion but gain it on robustness.

## Package layout

```
src/muse/modalities/audio_transcription/
├── __init__.py          # exports MODALITY + build_router
├── protocol.py          # TranscriptionModel Protocol + dataclasses
├── routes.py            # FastAPI multipart routes
├── codec.py             # TranscriptionResult to {json,text,srt,vtt,verbose_json}
├── client.py            # TranscriptionClient
└── runtimes/
    └── faster_whisper.py  # FasterWhisperModel generic runtime
```

No bundled model scripts under `src/muse/models/`. Every Whisper
variant routes through the resolver + runtime, the same pattern v0.12
established for ST embeddings.

## Protocol

```python
@dataclass
class Word:
    word: str
    start: float
    end: float

@dataclass
class Segment:
    id: int
    start: float
    end: float
    text: str
    words: list[Word] | None  # populated iff word_timestamps

@dataclass
class TranscriptionResult:
    text: str                   # full concatenated transcript
    language: str               # detected or specified
    duration: float             # seconds
    segments: list[Segment]
    task: Literal["transcribe", "translate"]

class TranscriptionModel(Protocol):
    def transcribe(
        self,
        audio_path: str,
        *,
        task: Literal["transcribe", "translate"] = "transcribe",
        language: str | None = None,
        prompt: str | None = None,
        temperature: float = 0.0,
        word_timestamps: bool = False,
        vad_filter: bool = False,
        **kwargs: Any,
    ) -> TranscriptionResult: ...
```

## Wire contract

Both endpoints accept `multipart/form-data`.
`/v1/audio/transcriptions` sets `task="transcribe"`;
`/v1/audio/translations` sets `task="translate"` and ignores the
`language` form field.

**Request form fields:**

| Field | Type | Required | Notes |
|---|---|---|---|
| `file` | file | yes | Audio in any PyAV/ffmpeg-decodable format |
| `model` | string | yes | Muse catalog id |
| `language` | string | no | ISO-639-1; auto-detect if unset; ignored on translations |
| `prompt` | string | no | Forwarded to faster-whisper as `initial_prompt` |
| `response_format` | string | no | `json` (default), `text`, `srt`, `vtt`, `verbose_json` |
| `temperature` | float | no | Default 0.0 |
| `timestamp_granularities[]` | string, repeatable | no | `segment` (default) and/or `word` |
| `vad_filter` | bool | no | Muse extension, default false |

**Response bodies by format:**

- `json`: `application/json`, `{"text": "..."}`
- `text`: `text/plain`, raw transcript
- `srt`: `application/x-subrip`, standard SubRip
- `vtt`: `text/vtt`, `WEBVTT\n\n` header + `HH:MM:SS.mmm` timestamps
- `verbose_json`: `application/json`, `{task, language, duration, text, segments[], words[]?}`; `words` at top level (OpenAI shape), flattened across segments, populated only when `timestamp_granularities[]` includes `word`

**Error envelopes** (muse-standard OpenAI-shape):
- 404 `model_not_found`: `model` field unknown
- 400 `invalid_parameter`: bad `response_format` or missing `file`
- 413 `payload_too_large`: file > size cap (env `MUSE_ASR_MAX_MB`, default 100)
- 415 `unsupported_media_type`: PyAV/ffmpeg decode failure

## Runtime

`FasterWhisperModel` follows the existing generic-runtime pattern:

- Deferred imports (module-level `torch`, `WhisperModel` sentinels; `_ensure_deps` lazy-imports on first instantiation)
- Constructor receives `model_id`, `hf_repo`, `local_dir`, `device`, `compute_type`, `beam_size`, `**_`
- `device="auto"` resolves via the existing `_select_device` helper (cuda > mps > cpu)
- `compute_type` defaults: `float16` on cuda, `int8` on cpu
- `transcribe` method drives `WhisperModel(src).transcribe(...)` with kwargs threaded through
- Segments iterator materialized into `list[Segment]`; per-segment `words` collected when `word_timestamps=True`
- Single result object constructed and returned

## HF resolver extension

`_sniff_repo_shape` gains a third branch (after GGUF and ST):

```python
def _looks_like_faster_whisper(siblings: list[str], tags: list[str]) -> bool:
    names = {Path(f).name for f in siblings}
    has_ct2_shape = (
        "model.bin" in names
        and "config.json" in names
        and ("vocabulary.txt" in names or "tokenizer.json" in names)
    )
    has_asr_tag = "automatic-speech-recognition" in tags
    return has_ct2_shape and has_asr_tag
```

Resolve branch returns a `ResolvedModel` with:
- `manifest.model_id`: repo-name slug (e.g. `faster-whisper-large-v3`)
- `manifest.modality`: `"audio/transcription"`
- `manifest.pip_extras`: `("faster-whisper>=1.0.0",)`
- `manifest.system_packages`: `("ffmpeg",)`
- `manifest.capabilities`: `{}` (no resolver-sniffed extras; curated overlay fills in if needed)
- `backend_path`: `"muse.modalities.audio_transcription.runtimes.faster_whisper:FasterWhisperModel"`
- `download`: `snapshot_download(repo_id, ...)` with no `allow_patterns` restriction (CT2 Whispers are small)

Search: `muse search <q> --modality audio/transcription` delegates to
`HfApi.list_models(filter="automatic-speech-recognition", search=q)`.
Per-repo CT2 filtering is NOT done at search time (one extra repo_info
call per result would be expensive); we let `resolve` reject non-CT2
repos with a clear error.

## Curated entries

```yaml
- id: whisper-tiny
  uri: hf://Systran/faster-whisper-tiny
  modality: audio/transcription
  size_gb: 0.08
  description: "Whisper tiny (39M, ~75MB): CPU-friendly smoke test, multilingual"

- id: whisper-base
  uri: hf://Systran/faster-whisper-base
  modality: audio/transcription
  size_gb: 0.15
  description: "Whisper base (74M, ~142MB): balanced CPU default"

- id: whisper-large-v3
  uri: hf://Systran/faster-whisper-large-v3
  modality: audio/transcription
  size_gb: 3.0
  description: "Whisper large-v3 (1550M, ~2.9GB): SotA quality, 8GB+ GPU"
```

No curated `medium`/`small`; users who want them pull via URI or
`muse search whisper`.

## Multipart handling

First muse modality with file uploads. Pattern:

```python
@router.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form(...),
    language: str | None = Form(None),
    prompt: str | None = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
    timestamp_granularities: list[str] = Form(default_factory=list, alias="timestamp_granularities[]"),
    vad_filter: bool = Form(False),
):
    # save to tempfile, call backend, encode response
```

Keep inline for v1. If a second multipart modality lands (images/edits,
audio/generation with audio input), factor out into
`src/muse/modalities/_common/uploads.py`. Noting the TODO in
`CLAUDE.md` so the refactor trigger is visible.

## Test strategy

Unit coverage is the bulk: codec formatters are pure functions
(testable without FastAPI), routes use FastAPI's `TestClient` with a
mocked `TranscriptionModel`, runtime mocks `WhisperModel` at the
import boundary.

One slow e2e test (`@pytest.mark.slow`): builds a real muse app with
a mocked runtime, uploads a synthesized 1s sine-wave WAV through the
FastAPI multipart path, asserts the full codec chain returns correct
SRT bytes. Skipped unless `ffmpeg` is on PATH. This is the only place
we exercise `UploadFile` + tempfile + codec end-to-end.

One opt-in integration test (`tests/integration/test_remote_asr.py`):
requires `MUSE_REMOTE_SERVER` AND a pulled Whisper model; uploads a
bundled short audio fixture; asserts non-empty transcript.

## Documentation

- `CLAUDE.md`: add `audio/transcription` to modality list; one-line
  note that routes.py holds the first multipart pattern and a TODO
  flag for refactor trigger.
- `README.md`: update modality list and endpoints block.
- No new long-form doc; the spec + plan serve as reference.

## File inventory

See the plan for exhaustive line-level file list.

## Release

**v0.13.0.** Minor bump: new modality, no breaking changes. Tag
message calls out new endpoints, new curated entries, and the
multipart-modality milestone.

## Out of scope

- Streaming
- Diarization
- Live mic capture
- Auto-sniff of Whisper variants other than Systran (Distil-Whisper has different repo layout)
- pywhispercpp / GGUF Whisper path
