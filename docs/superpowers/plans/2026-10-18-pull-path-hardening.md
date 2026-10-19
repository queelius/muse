# Pull Path Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the five defects found by the 2026-10-18 install audit directly in the existing pull path, so the Decision API can ship on it.

**Architecture:** No new installation subsystem. Each defect gets a targeted fix in the resolver plugin or CLI that owns it, plus a regression test that fails before the fix. Native GGUF placement (Task 1 of the deferred safe-install plan) is already finished on Codex's branch and is cherry-picked as-is.

**Tech Stack:** Python 3.10+, Hugging Face Hub, existing catalog/venv/probe helpers, pytest.

**Replaces:** Tasks 1–10 of the [safe installation plan](2026-10-18-safe-model-installation.md), which is deferred. Follow with the [decision API plan](2026-10-18-decision-api.md).

## Why this instead of the safe-install plan

The audit's five findings are each local bugs. The existing pull path already has revision-pinned curated entries, artifact receipts, transactional venv updates with backups, atomic catalog writes and the shared readiness gate. Rebuilding installation (generations, full transitive locks, a frozen Muse wheel, verification leases, a systemd/cgroup build runner) would put ten tasks in front of the Decision API and make strict installs Linux/systemd-only.

What this plan deliberately does not provide, and the deferred plan would:

- automatic rollback to a previous generation when an update fails (venv transactions cover only part of this);
- pinning of every transitive package, rather than top-level requirements;
- workers running a frozen Muse build instead of the editable checkout;
- cross-process leases preventing two verifications competing for one device;
- `--dry-run` cost/plan output.

The work for those lives, unmerged, on `codex/safe-install-and-decisions` (Tasks 1–4 and half of 5). Revisit only if real failures justify it.

## Global Constraints

- Keep `hf://` and every existing catalog id. A fix may refuse a new pull; it must not rename or rewrite an existing entry.
- Each fix is preceded by a regression test that fails on current `main` for the audited reason, not an import or fixture error.
- Resolver plugins stay single-file imports with no ML imports (see `docs/HF_PLUGINS.md`).
- Tests use isolated catalogs/caches and mocked Hub calls. Do not restart the user's service, signal unrelated processes or touch VMs.
- Do not push, tag or release without a separate request.

---

### Task 1: Cherry-pick Native GGUF Placement

**Done 2026-10-18** as `13aa01a`, `267edc7` (fast lane 5228 passed). **Files:** From `codex/safe-install-and-decisions`: `d29dc69` (unify native GGUF placement and accounting) and `8c10c12` (prepare native placement outside admission lock). They add `src/muse/core/placement.py` and touch `cli_impl/{gateway,load_director,probe_worker,supervisor}.py`, `admin/routes/memory.py` and the llama.cpp runtime.

- [x] `git cherry-pick d29dc69 8c10c12` onto `main`; resolve any conflict against the readiness-gate commit (`6c36f56`) in `gateway.py`.
- [x] Run `pytest tests/cli_impl/test_gguf_placement.py tests/modalities/chat_completion/runtimes/test_llama_cpp.py tests/admin/routes/test_memory_routes.py tests/cli_impl/test_load_director.py tests/cli_impl/test_gateway_lazy.py -q`, then the fast lane; require GREEN.

### Task 2: Refuse Cross-Publisher Catalog Id Collisions

**Done 2026-10-18.** As built: tests in `tests/core/test_catalog_source_conflict.py`; the comparison is (scheme, repo) case-insensitively, ignoring variant (a variant switch on one repo is an update); a resolver pull onto a model-script id is now refused too, because the script would otherwise load the new weights (the plan originally kept that shadowing).

**Files:** Modify `src/muse/core/catalog.py` (`_pull_via_resolver`), `src/muse/cli.py` (`pull`); test `tests/core/test_catalog.py`, `tests/test_cli.py`.

The GGUF, VLM and other resolver plugins derive the catalog id from the repo basename (`chat_completion/hf.py:65`, `:150`), so `hf://alice/Foo-GGUF@q4_k_m` and `hf://bob/Foo-GGUF@q4_k_m` both become `foo-gguf-q4-k-m`, and the second pull overwrites the first.

- [x] Write tests: pulling a resolver URI whose synthesized id already exists for a *different* source (compare `hf_repo` case-insensitively, plus variant) fails before any download or venv work, leaves the existing entry byte-identical, and the error names both sources. Re-pulling the *same* source keeps today's behavior. A bundled-id collision keeps today's shadowing behavior.
- [x] Add `muse pull --id <model_id>` so the user can install the second source under an explicit id; the error message suggests it. Validate the id with the existing model-id rules and apply the same collision check to it.
- [x] Implement the check once in the shared resolver-pull path, not per plugin.
- [x] Run the new tests plus `pytest tests/core/ -q -k "catalog or resolver or pull"`; require GREEN.

### Task 3: Download Every Shard of a Split GGUF

**Done 2026-10-18.** As built: shard naming lives in a new stdlib-only `src/muse/core/gguf.py` shared by the plugin and the sizer; the runtime needed no change; tests in `tests/modalities/chat_completion/test_hf_plugin_split_gguf.py`, `tests/core/test_gguf_shards.py` and `tests/cli_impl/test_supervisor_lazy.py`.

**Files:** Modify `src/muse/modalities/chat_completion/hf.py` (`_match_gguf_variant`, `_resolve`'s `_download`), the llama.cpp runtime if it needs the first-shard path; test `tests/modalities/chat_completion/test_hf_plugin.py`.

`_match_gguf_variant` returns the first file whose variant matches, and `_download` allow-lists only that file, so a `*-00001-of-00003.gguf` variant pulls one third of the weights.

- [x] Write tests against sibling lists modeled on Qwen3-32B-GGUF's split layout: a sharded variant allow-lists all `-0000N-of-0000M` files; `capabilities.gguf_file` is the first shard (what llama.cpp loads); a missing or extra shard count is a resolve-time error; a single-file variant is unchanged; search still emits one row per variant.
- [x] Make the weights-on-disk sizer (`supervisor.py`, GGUF exception) count every shard of the declared variant, not only `gguf_file`; add a test.
- [x] Implement, then run `pytest tests/modalities/chat_completion/ tests/cli_impl/test_supervisor*.py -q`; require GREEN.

### Task 4: Include SentencePiece Tokenizer Files

**Done 2026-10-18.** As built: `*.model` added to `embedding_text`, `text_classification` (DeBERTa-v3 `spm.model`) and `text_rerank` (previously only `spiece.model`, missing XLM-R's `sentencepiece.bpe.model`), and `*/*.model` to the `image_generation` and `image_upscale` diffusers filters (T5 `tokenizer_2/spiece.model`). Summarization, translation, OCR and the embedding modalities already listed their SentencePiece files; video and 3D download whole repos. One test, `tests/modalities/test_sentencepiece_allow_patterns.py`, checks the patterns with huggingface_hub's own filter.

**Files:** Modify `src/muse/modalities/embedding_text/hf.py` (allow-patterns around lines 67–73); test `tests/modalities/embedding_text/test_hf_plugin.py`.

- [x] Write a test that a repo shipping `sentencepiece.bpe.model` / `tokenizer.model` gets it in the allow-list, in both fp16 and non-fp16 branches.
- [x] Add `*.model` to the allow-list. Grep the other plugins' allow-patterns (`grep -n allow_patterns src/muse/modalities/*/hf.py`) for the same omission on text-tokenizing modalities (summarization, translation, classification, rerank); fix and test each one that filters with explicit globs and omits it.
- [x] Run `pytest tests/modalities/ -q -k hf`; require GREEN.

### Task 5: A Failed Post-Pull Probe Fails the Pull

**Done 2026-10-18.** As built: `catalog.pull()` now returns the installed id so the probe targets the right entry (a URI re-pull previously could not be identified, which would have become a false failure); tests in `tests/cli_impl/test_pull_probe_failure.py`; the admin job needed no change. Follow-up after the 0.61.0 push: the probe worker exits 0 when the model loads but its inference raises (the record carries `inference_error`), so `run_probe` now returns 6 for that record; before the fix a model that loaded but could not infer still pulled enabled.

**Files:** Modify `src/muse/cli.py` (`pull`, ~lines 385–397), `src/muse/cli_impl/probe.py` (`run_probe_for_pull`); test `tests/cli_impl/test_probe.py`, `tests/test_cli.py`.

Today the pull exits 0 and leaves the model `enabled: true` even when the probe could not load it or run inference.

- [x] Write tests: when the probe fails, `muse pull` exits nonzero, the catalog entry and weights remain (so a retry needs no re-download), the entry is set `enabled: false`, and stderr says to run `muse models probe <id>` then `muse models enable <id>`. A probe that could not identify the new entry is reported as a failure, not silently skipped. `--no-probe` keeps exit 0 and leaves the entry enabled, as today. An internal error in the probe helper also exits nonzero.
- [x] Add a test that the admin pull job (`POST /v1/admin/models/_/pull`, which runs `muse pull` as a subprocess in `admin/operations.py`) now ends `failed` with the probe reason; no admin code change should be needed.
- [x] Implement; run `pytest tests/cli_impl/ tests/admin/ tests/test_cli.py -q`; require GREEN.
- [x] Update the "`muse pull` runs probe at the end" paragraph in `CLAUDE.md` and the README pull section to describe the new exit behavior.

### Task 6: Release Gate

**Done 2026-10-18.** Preflight: all 31 fast-lane deps present. Full lane: 5346 passed, 116 skipped, 1 failed (the known `test_e2e_supervisor_vlm`). There is no changelog file; the bump commit message carries the release summary.

- [x] Run `python scripts/preflight.py` and `pytest tests/ -q`; require zero unexpected failures (known flakes: `test_e2e_supervisor_vlm`, order-dependent `test_memory_probe`).
- [x] Commit the pending `pyproject.toml` bump to 0.61.0 with a changelog/README note covering the hardening commits and Tasks 1–5.
- [x] Ask before pushing or tagging. (Pushed and tagged `v0.61.0` 2026-10-18.)

## Out of scope here

Generic-tag misdispatch (Decima claimed by the NLI classifier) is fixed by the decision plugin in the decision API plan's Task 4: it claims the exact repos at a priority ahead of `text/classification`, with a regression test that `hf://amyrmahdy/decima-base` resolves to `decision/evaluation`.
