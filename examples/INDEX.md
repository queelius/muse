# muse exercise - run summary

A running `muse serve` (gateway on :8000, lazy per-model workers, `--device auto`)
exercised across 9 modalities. Each modality directory holds the generated
artifact(s) plus `metadata.json` with full provenance (model, endpoint, request
params, timing, muse version, server commit, timestamp, derivation chain).

- muse version: 0.47.0 (server commit recorded per item in each metadata.json)
- host: 4-core CPU, 31 GB RAM, NVIDIA RTX 3060 12 GB
- `latency_seconds` INCLUDES one-time model cold-load on the first request to
  each worker (lazy loading). Warm calls are much faster.

## Modalities exercised (9)

| Modality | Model | Endpoint | Artifact(s) | Notes / provenance |
|---|---|---|---|---|
| audio/speech | kokoro-82m | POST /v1/audio/speech | kokoro-82m_hello.wav | text -> 6.7s of speech (CPU) |
| audio/transcription | whisper-base | POST /v1/audio/transcriptions | transcript.{json,txt} | transcribes the speech wav back to ~original text (chain) |
| image/generation | sd-turbo | POST /v1/images/generations | sd-turbo_mountain_lake.png (512x512) | text -> image (GPU, 1-step) |
| image/cv | depth-anything-v2-small, detr-resnet-50 | POST /v1/images/{depth,detect} | depth_map.png, detect_input.png, detections.json | depth of the landscape; detection on a generated street scene -> 48 objects (cars/people/bicycles/traffic lights) (chain) |
| embedding/text | qwen3-embedding-0.6b | POST /v1/embeddings | embeddings.json | 1024-dim; paraphrase cos=0.72 > unrelated |
| text/classification | text-moderation | POST /v1/moderations | moderations.json | benign vs threatening text |
| text/rerank | bge-reranker-v2-m3 | POST /v1/rerank | rerank.json | correctly ranks the password-reset doc #1 |
| text/summarization | bart-large-cnn | POST /v1/summarize | summary.{txt,json} | JWST paragraph -> short summary |
| chat/completion | smolvlm-256m-instruct | POST /v1/chat/completions | response.{txt,json} | text generation (see P4 + perf note) |

## Provenance chains demonstrated

- text -> kokoro speech -> whisper transcription (round-trips to ~original text)
- text -> sd-turbo image -> depth map + (separate generated street scene) object detection
- the catalog is self-describing: every `metadata.json` records `derived_from`
  for chained inputs.

## Problems encountered and fixes

Four real muse bugs were found and FIXED (with regression tests); one was a
bug in the exercise driver itself.

**P1 - kokoro /v1/audio/speech returned HTTP 500 (muse, FIXED).** The route
declares `voice: str | None = None` and always forwards `voice=req.voice`, so an
omitted voice reaches the model as explicit `None`. KokoroModel used
`kwargs.get("voice", "af_heart")`, whose default only fires when the key is
ABSENT, so `None` passed through and the kokoro pipeline raised
`ValueError: Specify a voice`. Every client that omits a voice hit this.
Fix: `kwargs.get("voice") or "af_heart"` in `synthesize` + `synthesize_stream`
(src/muse/models/kokoro_82m.py) + 2 regression tests. Verified end-to-end.

**P2 - probed bundled GPU model stuck "no memory estimate" -> 503 (muse, FIXED).**
Boot validation `_has_memory_data` (src/muse/cli_impl/supervisor.py) reads the
device from the persisted `manifest.capabilities.device`, defaulting to "cpu".
Bundled models have NO persisted manifest in catalog.json, so device resolved to
"cpu" and the lookup read `measurements.cpu` while `muse models probe` had
written `measurements.cuda` - so probing could never clear the flag. Fix: when
the manifest-derived device has no measurement, fall back to any present
measurement and adopt its recorded device. Regression test added. Verified:
serve boots "0 unservable".

**P3 - depth-anything + smolvlm failed to load: missing torchvision (muse, FIXED).**
Their `pip_extras` listed torch + transformers but not torchvision; transformers
5.x builds their image processors via torchvision-backed ops, so AutoImageProcessor
raised "requires the Torchvision library" (depth) / "Unrecognized image processor"
(smolvlm). detr-resnet-50 worked only because `timm` pulls torchvision in
transitively. Fix: add `torchvision` to both scripts' `pip_extras`
(src/muse/models/{depth_anything_v2_small,smolvlm_256m_instruct}.py). Verified:
both probe + serve cleanly.

**P4 - VLM chat 400 'vision_not_supported' despite supports_vision=True (muse, FIXED).**
The smolvlm script aliases `Model = HFVisionLanguageModel` (the shared VLM
runtime), so its CatalogEntry.backend_path points at the runtime module, which
has no MANIFEST. `get_manifest` re-imported that module and returned `{}` - so
the worker registered an empty manifest and the chat route read
supports_vision=False, 400ing every image request. Fix: in
`get_manifest` (src/muse/core/catalog.py), when backend_path's module lacks a
matching MANIFEST, recover the real one from discovery (non-lossy - keeps
license + all capabilities). Regression test added. Verified: get_manifest now
returns supports_vision=True and the route accepts image input.

**D1 - driver used wrong client class name (driver bug, not muse).**
`TranscriptionsClient` -> `TranscriptionClient` (singular). Fixed in the driver.

## Notes

- **smolvlm VLM image inference is impractically slow on this CPU.** After P4,
  the chat route correctly ACCEPTS image input, but a 256M VLM tiling/encoding
  an image on a 4-core CPU takes minutes (it exceeded a 300s client timeout).
  The demonstrated chat call is therefore text-only; the vision *correctness*
  bug is fixed, the remaining limitation is pure CPU performance.
- **Models not exercised here:** image/embedding (dinov2-small),
  image/segmentation (sam2-hiera-tiny), image/ocr (trocr-base-printed),
  audio/embedding (mert-v1-95m), audio/classification (ast-audioset),
  video/generation, 3d/generation. The driver (`_driver.py`) has functions for
  the first five; they were not run because each `muse pull` builds a per-model
  venv with its own ~3-4 GB CUDA-torch install, which is slow on this 4-core
  box. They can be added with `muse pull <id> --no-probe && muse models probe
  <id> --device cpu && muse models enable <id>`, then
  `python examples/_driver.py <name>`.
- **Pre-existing test failures (NOT caused by these fixes):** the full fast lane
  shows ~12 failures in sd-turbo / wan / chat-streaming / mcp / resolver tests.
  Verified by stashing all changes and re-running: identical failures with and
  without the fixes. They stem from the dev env's very new transformers 5.12.1 /
  torch 2.12 breaking older test mocks, and are independent of this work.

## Reproduce

    muse serve --device auto                       # gateway on :8000 (lazy workers)
    python examples/_driver.py speech transcription image_gen embedding \
        classification rerank summarization image_cv chat
    # each function writes its artifact + metadata.json; independently re-runnable
