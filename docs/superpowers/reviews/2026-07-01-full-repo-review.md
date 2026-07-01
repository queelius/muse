# Full-repo code review: muse (museq) @ v0.49.0

- **Date:** 2026-07-01
- **Scope:** entire repository (`src/muse/`), 257 Python files / ~38.5k LOC
- **Commit reviewed:** `26f480a` (chore(release): v0.49.0)
- **Method:** multi-agent adversarial Workflow. 23 review units fanned out in
  parallel (one reviewer per high-risk control-plane file, modalities clustered
  by pattern). **Every** finding was re-verified by an independent
  refute-by-default skeptic that had to construct a concrete failing input, or
  the finding was dropped.
- **Outcome:** 35 raw findings, **33 confirmed** after verification (2 refuted).
  - **HIGH: 7** &nbsp; **MEDIUM: 10** &nbsp; **LOW: 16**
- **Agents:** 58 &nbsp; **Tool calls:** 681 &nbsp; **Wall clock:** ~16 min

No fixes have been applied. This report is the diagnosis; it is written before
any code change so the campaign survives a context compaction (per the release
workflow memory).

---

## Cross-cutting themes

Several findings share a root cause. Fixing the theme fixes multiple findings:

1. **Missing `capabilities.device` on 3 bundled scripts** (H3, H4, H5). The
   v0.48.0 invariant is "every CUDA-safe bundled model declares `device: auto`."
   `nv_embed_v2`, `sd_turbo`, `bark_small` omit it. The constructor default loads
   them on CUDA, but every control-plane accounting site defaults the *missing*
   key to `"cpu"` (`.get("device", "cpu")`), so their VRAM is sized, admitted, and
   evicted against host RAM, giving GPU OOM under pressure. **One-line fix each.**
   A regression guard (a test asserting every bundled script whose runtime can hit
   CUDA declares `device`) would prevent recurrence.

2. **`device_override` invisible to the control plane** (M6 `load_director`,
   M9 `probe_worker`). Only `load_backend` reads the catalog `device_override`;
   the director's sizing/eviction and the probe's device resolution read
   `manifest.capabilities.device` instead. An operator `set-device` pin makes
   accounting diverge from where the worker actually loads.

3. **Dead worker specs are never removed from `state.workers`** (H1 enable, L3
   remove-model 409, and it feeds M10 supervisor-thread-death). The monitor sets
   `status="dead"` after 10 restart attempts and then `continue`s, leaving a spec
   with `job_id=None` that downstream code misreads as "already running / loaded."

4. **Backend inference not wrapped, so 500 instead of the OpenAI/Cohere error
   envelope** (L14 rerank/summarize/embedding/video, L11 segmentation, L13
   DecompressionBomb, L5 VLM `part["text"]`). Task #201 fixed this only for
   `/v1/moderations`; the same one-line `try/except` to `error_response(500, ...)`
   wrap is missing on ~6 other routes.

5. **StableAudio bare-tensor bug is duplicated** (H2). The identical batch-dim
   defect lives in both the resolver runtime (`stable_audio.py`) and the bundled
   default (`stable_audio_open_1_0.py`); the mock-shaped-as-`list` in both test
   files is exactly why it was never caught.

---

## HIGH (7)

### H1: `enable` / lazy-load treats a *dead* worker as "already running"
- **File:** `src/muse/admin/operations.py:137` (and identical at `:336`)
- **Unit:** admin-ops
- The guard `existing.status == "running" or existing.job_id is None` classifies a
  dead/unhealthy spec (`status="dead"`, `job_id=None`) as `already_running`, so
  `enable_model` returns `{loaded: True, worker_port: <dead port>}` and never
  respawns. The gateway only routes `status=="running"` specs, so requests to the
  model get no route while the admin API reports success. The identical condition
  in `load_model_into_worker` makes the director commit a hot `LoadEntry` pointing
  at the dead port, so it never retries either: permanent silent breakage of the
  affected model, in exactly the crash/failed-load case the auto-restart + lazy
  machinery exists to handle.
- **Fix:** exclude non-`running` specs from the "already running" branch; fall
  through to respawn.

### H2: StableAudio 500s on every real generation (bare-tensor batch dim)
- **File:** `src/muse/modalities/audio_generation/runtimes/stable_audio.py:200`
  (dup: `src/muse/models/stable_audio_open_1_0.py:227`)
- **Unit:** audio-gen-emb-cls
- `generate` strips the batch dim only when the pipeline output is a `list/tuple`.
  Real `diffusers.StableAudioPipeline` (default `output_type="pt"`) returns a bare
  `torch.Tensor` of shape `(1, 2, N)`, which is not a list/tuple, so the 3-D array
  reaches `_normalize_pipeline_output` (handles only ndim 1/2) and raises
  `ValueError`, i.e. HTTP 500. Every generation of the bundled *default* audio
  model fails. Unit tests pass only because they mock `.audios` as a Python list.
- **Fix:** index `[0]` (strip leading batch dim) for ndarray/tensor outputs too,
  or handle a size-1 leading batch dim in `_normalize_pipeline_output`. Fix both
  copies; update both tests to use a bare tensor.

### H3: `nv-embed-v2` omits `capabilities.device`, 16GB accounted vs host RAM
- **File:** `src/muse/models/nv_embed_v2.py:85` (Unit: bundled-models)
- On a GPU host the 16GB Mistral-7B backbone loads on CUDA (ctor default `auto`),
  but `load_director`/`supervisor`/`idle_sweeper` default the missing key to
  `"cpu"` and size it against host RAM. The director believes 16GB always fits,
  never evicts it under VRAM pressure, giving GPU OOM. Largest bundled model,
  worst mis-accounting. **Fix:** add `"device": "auto"`.

### H4: `sd-turbo` (default image model) omits `capabilities.device`
- **File:** `src/muse/models/sd_turbo.py:97` (Unit: bundled-models)
- Same root cause as H3; ~4GB VRAM sized against host RAM. As the *default*
  image/generation model this is the most-hit path: a co-resident 6GB model plus a
  first sd-turbo request admits with no eviction, giving CUDA OOM at load.
  **Fix:** add `"device": "auto"`.

### H5: `bark-small` omits `capabilities.device`
- **File:** `src/muse/models/bark_small.py:61` (Unit: bundled-models)
- Same root cause; ~3GB VRAM sized against host RAM. **Fix:** add
  `"device": "auto"`.

### H6: Diffusers t2i unconditionally requests `variant="fp16"`, crashing repos without fp16 weights
- **File:** `src/muse/modalities/image_generation/runtimes/diffusers.py:124`
- **Unit:** image-gen
- The runtime passes `variant="fp16"` whenever `dtype=="float16"` (the default),
  but the HF downloader only fetches `.fp16.` weights when `has_fp16_variants` is
  True. For a repo like **flux-schnell** (curated) that ships no fp16-named files,
  the local snapshot holds only bare weights, and
  `from_pretrained(local_dir, variant="fp16")` raises
  `ValueError: ... no such modeling files are available`, so the worker crashes on
  cold load. `sd-turbo`/`sdxl-turbo` mask the bug because they ship fp16 variants.
  Affects flux-schnell and any resolver-pulled bf16-only diffusers t2i repo.
- **Fix:** request `variant="fp16"` only when the snapshot actually has fp16
  variants (thread `has_fp16_variants` through, or try/fallback to no-variant).

### H7: Image-embedding runtime crashes for CLIP/SigLIP (2 of 3 advertised families)
- **File:** `src/muse/modalities/image_embedding/runtimes/transformers_image.py:245`
- **Unit:** image-ocr-emb-3d
- `embed()` calls `self._model(**inputs)` with only `pixel_values`. `AutoModel`
  returns full `CLIPModel`/`SiglipModel` whose `forward()` runs the text tower and
  requires `input_ids`, raising `ValueError: You have to specify input_ids`
  (reproduced on transformers 5.12.1). Every `/v1/images/embeddings` request to
  curated `clip-vit-base-patch32` or `siglip2-base` 500s; the CLIP `image_embeds`
  branch in `_extract_embeddings` is unreachable dead code. Only DINOv2
  (vision-only) works. **Fix:** call `get_image_features()` for CLIP/SigLIP
  families (dispatch on model type / presence of the method).

---

## MEDIUM (10)

### M1: `find_free_port` TOCTOU, concurrent cold loads can pick the same port
- **File:** `src/muse/admin/operations.py:172` (and `:356`) (Unit: admin-ops)
- The lazy load paths call `find_free_port()` without a `used_ports` set (unlike
  boot's `plan_workers`) and with no retry. Two concurrent cold loads of different
  models can both get 9001 (neither child has bound yet); the loser fails to bind,
  `wait_for_ready` times out (~120s), and the load 503s despite ~999 free ports.
- **Fix:** exclude ports already reserved by pending specs in `state.workers`;
  retry `find_free_port` on bind failure per its docstring contract.

### M2: `muse models refresh` broken for PyPI-installed museq
- **File:** `src/muse/cli_impl/refresh.py:144` (Unit: cli-display)
- `_muse_repo_root()` returns `Path.cwd()` when no ancestor `pyproject.toml`
  exists (the wheel case), and `refresh_one` unconditionally runs
  `pip install -e <path>[server,...]`. For a `pip install museq` user running
  refresh from home, this errors ("not a Python project") or editable-installs an
  unrelated cwd project, contradicting the docstring's promise to pull museq from
  PyPI. **Fix:** emit a bare `museq[server,...]` spec when not in a source tree.

### M3: Re-pulling a resolver model by its bare id corrupts the catalog
- **File:** `src/muse/core/catalog.py:523` (Unit: core-catalog)
- A raw-URI pull persists under a synthesized bare id with `manifest`+`source`.
  Re-pulling by that displayed id hits the bare-id branch, so `_pull_bundled`
  overwrites the entry with a dict lacking `manifest`/`source`. The next
  `known_models()` rebuild skips it (`if not manifest: continue`) and the model
  vanishes, giving `unknown model` 404/500 until re-pulled via full URI.
- **Fix:** in the bare-id branch, if the existing entry carries `source`/`manifest`
  (resolver-pulled), route back through `_pull_via_resolver` (or short-circuit
  already-pulled).

### M4: SSRF guard is validate-then-connect (DNS-rebinding TOCTOU)
- **File:** `src/muse/core/net_fetch.py:65` (Unit: core-security)
- `validate_public_host` resolves via `socket.gethostbyname` once and discards the
  IP; httpx re-resolves the hostname independently at connect time. A low-TTL DNS
  flip (or a multi-A-record host) lets the connect reach `127.0.0.1` /
  `169.254.169.254` after the check passed. The docstring's "closes the DNS-rebind
  gap" claim is overstated. Reachable from user image `url` fields and the MCP
  `fetch_url_bytes` path. Static-private-IP / redirect / IPv6 paths are correctly
  blocked; only an active DNS-controlling attacker bypasses. **Fix:** resolve
  once, verify the resolved IP is public, connect to that pinned IP with the
  original `Host` header.

### M5: Concurrent evict-needing acquires 503 spuriously
- **File:** `src/muse/cli_impl/load_director.py:734` (Unit: load-director)
- The `evict_and_retry` path does not claim an `in_flight_loads` slot (unlike the
  `load` path), so two concurrent cold acquires for the same model both enter
  `_evict_lru_until_fits`. The loser sees the single LRU victim already popped,
  hits `if not candidates:` and raises 503 `model_too_large_for_device`, without
  re-checking whether the model now fits against current free memory. The gateway
  surfaces it verbatim with no retry, even though the model becomes serviceable
  the instant the winner commits. **Fix:** collapse same-model evict-path acquires
  via `in_flight_loads`, and/or re-check fit against live memory before raising.

### M6: Director sizes/evicts against manifest device, ignoring `device_override`
- **File:** `src/muse/cli_impl/load_director.py:562` (and `:610`) (Unit: load-director)
- `device_override` (operator `set-device`) determines where the worker loads but
  is never folded into the manifest the director reads. An `auto`/`cuda` model
  pinned to `cpu` makes the director needlessly evict GPU models to "make room"
  for a load that actually goes to host RAM; the inverse pin over-commits VRAM.
  **Fix:** fold `device_override` into the device the control plane sizes against
  (e.g. `backfill_manifest_memory` also carries device, or `_decide` consults the
  override).

### M7: MCP `resolve_binary_input` empty-string slot silently wins
- **File:** `src/muse/mcp/binary_io.py:68` (Unit: mcp-core)
- The "exactly one provided" guard uses truthiness (`if v`) but dispatch uses
  `is not None`. `image_b64=""` + a real `image_url` passes the guard (only "url"
  counted) yet enters the b64 branch, so `b64decode("")==b""` ignores the URL.
  LLM clients routinely emit `""` for unused slots. **Fix:** make guard and
  dispatch agree (treat `""` as absent, i.e. use truthiness in both).

### M8: `muse_list_models` `filter_status` always returns zero rows
- **File:** `src/muse/mcp/tools/admin.py:61` (Unit: mcp-tools)
- `/v1/models` entries carry `loaded` (bool), never a `status` key. The filter
  `[r for r in rows if r.get("status") == s]` matches `None == "enabled"` for
  every row, i.e. a confidently-wrong empty result. Test passes only because it
  mocks a fabricated `status` key. **Fix:** filter on `loaded` (and note disabled
  models are omitted from the gateway list entirely).

### M9: Probe device resolution ignores `device_override`
- **File:** `src/muse/cli_impl/probe_worker.py:139` (Unit: probe)
- `_resolve_device` reads only manifest capabilities; `load_backend` honors
  `device_override`. With a `cpu` pin on an `auto` model on a GPU host, the probe
  measures VRAM (`~0`) while the model actually loads on CPU, then persists a bogus
  `~0` record under `measurements["cuda"]` while never writing the `cpu` bucket the
  sizing ladder reads. **Fix:** make `_resolve_device` consult `device_override`
  (mirror `load_backend`).

### M10: `OSError` from `Popen` kills the auto-restart monitor thread
- **File:** `src/muse/cli_impl/supervisor.py:395` (Unit: supervisor)
- `_attempt_restart` catches only `subprocess.SubprocessError`/`TimeoutError`.
  A missing/broken venv python makes `Popen` raise `FileNotFoundError`
  (`OSError`), which escapes the loop and terminates the `muse-monitor` daemon
  thread, disabling health-monitoring and auto-restart for **all** workers, with
  only a bare stderr traceback. Realistic trigger: venv deleted, or python symlink
  broken by a system minor-version upgrade. **Fix:** broaden the except to include
  `OSError` (and/or wrap the monitor loop body).

---

## LOW (16)

- **L1: `AdminClient` error parsing raises `AttributeError` on string/list `detail`.**
  `src/muse/admin/client.py:85`. A `{"detail":"Not Found"}` 404 (wrong base_url /
  unmounted router) or a 422 with a list `detail` makes `.get("error")` on a str
  throw, masking the real status. **Fix:** `isinstance` guard before `.get`.

- **L2: `verify_admin_token` 500s on a non-ASCII bearer token.**
  `src/muse/admin/auth.py:69`. `secrets.compare_digest` on a token that contains
  any non-ASCII character raises `TypeError` (not an HTTPException), giving 500
  instead of the intended 403. No bypass. **Fix:** catch/encode non-ASCII, treat
  as invalid (403).

- **L3: `remove_model` returns 409 for a *dead* worker spec.**
  `src/muse/admin/operations.py:676`. `find_worker_for_model` returns dead specs
  (no status filter); the 409's "holding open FDs" rationale doesn't apply to an
  exited process. Operator must `disable` first to clear the stale spec.
  **Fix:** ignore non-running specs in the loaded-check (theme #3).

- **L4: Time normalizer emits `{'45'}` set-repr in H:MM:SS.**
  `src/muse/modalities/audio_speech/utils/text_normalizer.py:155`. Fallback is the
  set literal `{minutes}` instead of the string `minutes`; `clean_text("at 9:45:30.")`
  gives `"at nine, 'forty-five', thirty."`, degrading TTS prosody for times whose
  minutes don't start with `0`. **Fix:** `{minutes}` to `minutes`.

- **L5: VLM `part["text"]` hard subscript, 500 on a text part missing `"text"`.**
  `src/muse/modalities/chat_completion/runtimes/transformers_vlm.py:285`.
  Asymmetric with the image-part path (graceful 400). Same class at `msg["role"]`.
  **Fix:** `.get("text", "")` / validate part shape to 400 (theme #4).

- **L6: `ChatClient.chat_stream` ignores the SSE `event: error` frame.**
  `src/muse/modalities/chat_completion/client.py:61`. The client filters on
  `data: ` lines only, so a mid-stream backend error is yielded as a normal chunk;
  callers iterating `chunk["choices"]` `KeyError` or silently truncate. **Fix:**
  recognize the `event: error` frame and raise.

- **L7: `muse models list` never shows measured memory on Apple Silicon.**
  `src/muse/cli_impl/models_list.py:406`. Non-CPU lookup uses keys
  `("cuda","auto")` but the probe writes the `"mps"` bucket; `"auto"` is a dead key
  never written. `muse models info` shows it correctly (list/info inconsistency).
  **Fix:** include `"mps"`, drop dead `"auto"`.

- **L8: Idle sweeper re-checks refcount but not `last_touched_at` (TOCTOU).**
  `src/muse/cli_impl/idle_sweeper.py:203`. A model touched during the
  snapshot-to-lock window (which includes a catalog read) is idle-evicted anyway,
  giving one spurious cold reload. **Fix:** re-validate `last_touched_at` under the
  lock.

- **L9: `_reset_known_models_cache()` writes the shared global lock-free.**
  `src/muse/core/catalog.py:321`. Races the lock-guarded rebuild; a slow-path
  rebuild reading a pre-mutation catalog can resurrect a stale cache (no mtime
  self-invalidation), briefly hiding a just-pulled model. **Fix:** take
  `_KNOWN_MODELS_LOCK` in the invalidator.

- **L10: `error_response()` hardcodes `type="invalid_request_error"` for 5xx.**
  `src/muse/core/errors.py:15`. 500s carry a client-error type string; the HTTP
  status is still 500 (what real SDKs use), so impact is limited to bespoke
  `error.type` branching. **Fix:** derive type from status (server_error for 5xx).

- **L11: Segmentation route maps `RuntimeError` to 400, no generic 500 catch.**
  `src/muse/modalities/image_segmentation/routes.py:230`. CUDA OOM (a
  `RuntimeError`) returns 400 `invalid_parameter` (a "don't retry" signal) for a
  valid request; other exceptions escape the OpenAI envelope. **Fix:** narrow the
  RuntimeError handling to capability mismatches; add `except Exception` to 500
  (theme #4).

- **L12: img2img `strength` in `[0, 0.01)` crashes (floor applied to steps, not strength).**
  `src/muse/modalities/image_generation/runtimes/diffusers.py:231`. `strength=0.0`
  (schema-valid, `ge=0.0`) yields `int(steps*0.0)=0` effective denoise steps, so
  empty timesteps give the VAE crash the comment claims to prevent. **Fix:** floor
  the strength value actually passed to the pipeline, not just the step
  computation.

- **L13: `DecompressionBombError` escapes `_bytes_to_pil`, 500 not 400.**
  `src/muse/modalities/image_generation/image_input.py:174`. Catches only
  `(UnidentifiedImageError, OSError)`; the bomb error is a plain `Exception`. A
  tiny PNG declaring huge dimensions 500s instead of returning a clean 400
  (PIL still prevents the memory DoS). Affects edits/variations/upscale/img2img.
  **Fix:** also catch `Image.DecompressionBombError`, convert to `ValueError`.

- **L14: rerank/summarize/embedding/video routes don't wrap inference, giving bare 500.**
  `src/muse/modalities/text_rerank/routes.py:114` (+ `text_summarization:97`,
  `embedding_text:109`, `video_generation:95`). A backend exception yields
  Starlette's default 500, not the documented `{"error":{...}}` envelope: the gap
  #201 closed only for `/v1/moderations`. **Fix:** the same `try/except` to
  `error_response(500, ...)` wrap (theme #4).

- **L15: MCP audio output hardcodes `mimeType="audio/wav"`.**
  `src/muse/mcp/tools/inference_audio.py:52` (+ `:80`, `:93`). `response_format=opus`
  (music/sfx/speech) returns ogg/mpeg bytes labeled `audio/wav`; block metadata
  disagrees with the JSON summary. **Fix:** derive the mime from `response_format`.

- **L16: Gateway body read after `acquire()` outside try/finally, stranded refcount.**
  `src/muse/cli_impl/gateway.py:581`. For a body-bearing GET routed via the
  director, a `ClientDisconnect` during `await request.body()` skips
  `director.release()`, wedging the refcount so the model can never be evicted.
  Narrow today (no modality routes a body GET through the director) but violates
  the documented "refcount is never stranded" invariant. **Fix:** read the body
  before `acquire`, or wrap it in try/except that releases.

---

## Refuted at verification (2)

Two raw findings did not survive the refute-by-default skeptic (no reproducible
failing input) and are intentionally excluded. The verify stage's job is exactly
this: kill plausible-but-wrong findings before they reach a fix campaign.

---

## Suggested fix ordering

1. **Batch 1, HIGH, mostly one-liners (highest value/effort):** H3/H4/H5
   (`device: auto` x3 + regression guard), H1 (dead-spec guard), H2 (StableAudio
   x2 + tests), H6 (fp16 variant guard), H7 (`get_image_features`).
2. **Batch 2, MEDIUM control-plane correctness:** M10, M1, M5, M6+M9
   (`device_override` theme), M3, M8, M2, M7, M4 (SSRF pinning).
3. **Batch 3, LOW error-envelope + robustness sweep:** theme #4 group
   (L14, L5, L11, L13) + L1, L2, L3, L4, L6, L7, L8, L9, L10, L12, L15, L16.

Each batch is a candidate named cleanup release (v0.49.1 / v0.49.2 / v0.49.3) in
the usual ship-then-review-then-fix rhythm.
