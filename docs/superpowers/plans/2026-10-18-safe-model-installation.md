# Safe Model Installation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make supported model pulls complete, verified, atomic and recoverable without breaking existing installations.

**Architecture:** Extend existing resolvers, artifact receipts, locks and owned subprocess helpers with a small installation package. Build immutable generations privately, verify through serving-compatible admission, and publish one catalog pointer only after success. Keep legacy loading intact and stage the rollout behind an internal adapter until the public command contract is ready.

**Tech Stack:** Python 3.10+, dataclasses, Hugging Face Hub, existing venv/subprocess helpers, FastAPI admin jobs, pytest.

**Spec:** [Safe model installation design](../specs/2026-10-18-safe-model-installation-design.md). Follow with the [decision API plan](2026-10-18-decision-api.md).

## Global Constraints

- A normal successful pull includes complete artifacts, environment preparation, load, representative inference, resource release and atomic publication.
- Keep `hf://`; support validated runtime recipes, not arbitrary Hub repositories. Curated aliases and direct references share the installer.
- Resolve all model, base and executable-code repositories to immutable commits. Pins establish identity, not trust.
- Preserve existing working records, canonical venvs and aliases. Do not claim new verification for legacy installations.
- Create generation venvs at their final absolute paths. Stable generations must not use a moving editable checkout.
- Keep candidate state separate from active readiness. Unverified candidates never route; `/health` remains liveness.
- Reject updates to loaded/leased models before mutation. No automatic hot replacement, serving-worker eviction, drivers or system-package changes.
- Use existing catalog/resource/cache locks; never hold the global catalog lock across network, build or inference work.
- Retain one previous verified generation per model by default; reuse identical artifact content and existing cleanup grace/lease rules.
- First strict slices: single/split GGUF, then both decision recipes. Advertised entries without recipes must be explicitly experimental before strict pull becomes the default.
- Tests use isolated catalogs/caches and owned child processes. Do not restart the user's service, signal unrelated processes or touch VMs during implementation.

## Review Focus

- Escaping, publisher case and bundled aliases must not let a different model replace an existing slot — Tasks 2–3.
- An installed package version can stay constant while editable code or a native library changes; cached verification must become stale — Tasks 1 and 5.
- Cancellation or ENOSPC between receipt durability and pointer replacement must leave either the old or complete new generation, never a broken active record — Tasks 4 and 7.
- A gateway starting during standalone verification, or a second gateway, must not oversubscribe a device reservation — Task 6.
- Cleanup concurrent with publication or shared-artifact reuse must retain every still-referenced resource — Task 9.

---

## File Responsibilities and Shared Records

New production modules live under `src/muse/core/installation/`:

- `types.py`: versioned, JSON-serializable plan, generation and result contracts; no ML imports.
- `identity.py`: canonical slot keys and collision-resistant display aliases.
- `planning.py`: immutable resolution, recipe selection, inventory and cost/build planning.
- `generations.py`: private directories, validated receipts and publication transaction.
- `environment.py`: fixed-prefix venvs, package locks, code fingerprints and environment receipts.
- `loading.py`: shared backend construction from a frozen generation; used inside its interpreter.
- `verification.py`: gateway delegation or standalone coordinated private probes.
- `coordinator.py`: lifecycle, idempotence, resumability and stable errors.
- `__init__.py`: small public exports, without side effects.

Other focused additions: `core/placement.py` for shared actual placement; `core/verification_leases.py` for cross-process device ownership; `cli_impl/pull.py` for CLI presentation; `cli_impl/private_probe.py` for generation loading; `admin/routes/installations.py` for validated verification jobs. Keep scheduler logic in `load_director.py`, filesystem checks in existing helpers, and CLI option wiring in `cli.py`.

Define these frozen records in Task 2; `JsonValue` is a recursive alias for finite JSON primitives, lists and string-keyed objects:

| Record | Fields |
| --- | --- |
| `SourceRef` | `provider: str`, `repository: str`, `revision: str`, `variant: str \| None` |
| `ArtifactFile` | `source: SourceRef`, `path: str`, `destination: str`, `size_bytes: int \| None`, `sha256: str \| None` |
| `EnvironmentSpec` | `requirements: tuple[str, ...]`, `locked_packages: tuple[dict[str, JsonValue], ...]`, `code_identity: str`, `python_requires: str`, `platform_tags: tuple[str, ...]`, `build: dict[str, JsonValue]`, `trusted_code: bool` |
| `ProbeSpec` | `call_path: str`, `timeout_seconds: int`, `threads: int`, `processes: int`, `memory_bytes: int` |
| `RuntimeRecipe` | `profile_id: str`, `version: int`, `modality: str`, `backend_path: str`, `environment: EnvironmentSpec`, `device_policy: str`, `limits: dict[str, JsonValue]`, `probe: ProbeSpec` |
| `InstallOptions` | `update: bool = False`, `revision/task/runtime/name: str \| None = None`, `dry_run/download_only/allow_build/allow_remote_code/development/experimental: bool = False`, `base_override: str \| None = None` |
| `InstallPlan` | `fingerprint: str`, `slot_key: str`, `model_id: str`, `sources: tuple[SourceRef, ...]`, `recipe: RuntimeRecipe`, `files: tuple[ArtifactFile, ...]`, `manifest: dict[str, JsonValue]`, `device: str`, `estimates: dict[str, int \| None]`, `expected_active: str \| None` |
| `GenerationRef` | `model_id: str`, `generation_id: str` — identifiers only, never caller-supplied paths |
| `EnvironmentReceipt` | `identity: str`, `python_path: str`, `packages: tuple[dict[str, JsonValue], ...]`, `code_identity: str`, `platform: str`, `build: dict[str, JsonValue]`, `development: bool` |
| `VerificationReceipt` | `plan_fingerprint: str`, `environment_identity: str`, `code_identity: str`, `device: str`, `pool: str`, `measurements: dict[str, JsonValue]`, `probe_shape: str`, `completed_at: str` |
| `InstalledGeneration` | `ref: GenerationRef`, `plan: InstallPlan`, `environment: EnvironmentReceipt`, `local_dir: str`, `verification: VerificationReceipt \| None` |
| `InstallError` | Exception with `code/stage/message/remedy: str`, `retryable: bool`; safe strings, no raw exception payload |
| `InstallResult` | `model_id: str`, `state: Literal["dry_run", "staged", "published", "unchanged", "failed"]`, `active_generation/candidate_generation: str \| None`, `previous_preserved: bool`, `error: InstallError \| None`, `warnings: tuple[str, ...]` |

Persist records with `schema_version: 1`; reject unknown future schema versions. Fingerprints include resolved runtime-affecting inputs, code identity and platform lock, not mutable display metadata, estimates or `expected_active` (which is only a publication precondition). Verification additionally binds the actual environment receipt identity. Unknown cost estimates remain `None`, not zero. Test fixtures in `tests/core/installation/conftest.py` supply a fake Hub, declarative GGUF recipe, private generation and an old active generation, all under `tmp_path`.

### Task 1: Finish Actual Native Placement

**Files:** Create `src/muse/core/placement.py`; modify `src/muse/modalities/chat_completion/runtimes/llama_cpp.py`, `src/muse/cli_impl/{supervisor,probe_worker,load_director}.py`, `src/muse/admin/routes/memory.py`; test `tests/cli_impl/test_gguf_placement.py`, `tests/modalities/chat_completion/runtimes/test_llama_cpp.py`, `tests/admin/routes/test_memory_routes.py`.

**Interfaces:** Produce `Placement(device: str, pool: str, n_gpu_layers: int)`; `native_placement(device: str, n_gpu_layers: int, *, gpu_offload: bool) -> Placement`; `resolve_model_placement(model_id: str, manifest: dict, *, requested: str = "auto", entry: dict | None = None) -> Placement`. Query native capability inside the recorded model interpreter; cache by interpreter/package/native-library identity, not host CUDA visibility.

- [ ] Write/retain regression tests asserting CPU-only `auto` gives `device == pool == "cpu"` and `n_gpu_layers == 0`; explicit unsupported CUDA raises; explicit CPU or zero layers never queries GPU support. Test operator `auto` precedence, native-library replacement invalidation, probes with misleading Torch CUDA, resident-entry pool accounting and dead-worker exclusion. Preserve the existing failing assertions; adapt mocks only if the subprocess ownership boundary changes.
- [ ] Run `pytest tests/cli_impl/test_gguf_placement.py tests/modalities/chat_completion/runtimes/test_llama_cpp.py -q`; confirm RED for the known placement defects, not fixture/import errors.
- [ ] Implement the placement interfaces and use them in native construction, admission and probes. Reuse `run_owned_command` for bounded interpreter queries; preserve already-resident pools until unload rather than recomputing placement from edited preferences.
- [ ] Run `pytest tests/cli_impl/test_gguf_placement.py tests/modalities/chat_completion/runtimes/test_llama_cpp.py tests/admin/routes/test_memory_routes.py tests/cli_impl/test_load_director.py -q`; require GREEN.
- [ ] Commit only this task's reviewed hunks as `fix(runtime): unify native GGUF placement and accounting`; do not sweep other dirty supervisor/admin changes into it.

### Task 2: Define Installation Records and Slot Identity

**Files:** Create `src/muse/core/installation/{__init__,types,identity}.py`, `tests/core/installation/{conftest,test_types,test_identity}.py`; modify `src/muse/core/resolvers.py`.

**Interfaces:** Produce all records above, `slot_key(source: SourceRef, runtime_profile: str) -> str`, `generated_alias(source: SourceRef, runtime_profile: str) -> str`. Extend `ResolvedModel` with optional `recipe: RuntimeRecipe | None = None` and `artifact_files: tuple[ArtifactFile, ...] = ()`; make `download: Callable[[Path], Path] | None = None` so strict results need no callback. Existing legacy callbacks remain supported only by the legacy adapter.

- [ ] Add schema round-trip and identity tests with these assertions:
  ```python
  assert slot_key(source_at_commit_a, "chat-gguf") == slot_key(source_at_commit_b, "chat-gguf")
  assert slot_key(source_at_commit_a, "chat-gguf") != slot_key(source_at_commit_a, "decision-gguf")
  assert generated_alias(publisher_one, "chat-gguf") != generated_alias(publisher_two, "chat-gguf")
  ```
  Fixtures use the same repository basename but different publishers; add case/escaping pairs that sanitize identically, invalid relative paths, non-finite values and unknown schema versions.
- [ ] Run `pytest tests/core/installation/test_types.py tests/core/installation/test_identity.py -q`; expect RED for missing contracts.
- [ ] Implement schema validation and canonical JSON hashing. Preserve provider-canonical repository case; use a readable provider/publisher prefix plus stable identity hash, within existing filesystem-ID limits. Reject remaining alias collisions when binding, rather than truncating identity or overwriting.
- [ ] Run the Task 2 command plus `pytest tests/core/test_resolvers.py -q`; require GREEN and backward-compatible legacy construction.
- [ ] Commit `feat(install): define plans generations and canonical model identity` with only Task 2 files.

### Task 3: Plan Complete Supported Recipes Before Download

**Files:** Create `src/muse/core/installation/planning.py`, `src/muse/modalities/chat_completion/recipes.json`, `tests/core/installation/test_planning.py`; modify `src/muse/core/{resolvers,resolvers_hf,curated}.py`, `src/muse/modalities/chat_completion/hf.py`, `src/muse/modalities/embedding_text/hf.py`, `docs/HF_PLUGINS.md`; test `tests/core/test_hf_resolver_dispatch.py`.

**Interfaces:** Produce `build_install_plan(identifier: str, options: InstallOptions) -> InstallPlan`, `load_recipe_definitions(path: Path) -> tuple[RuntimeRecipe, ...]`; add `CuratedEntry.runtime_profile: str | None = None` and `experimental: bool = False`. Extend `HFResolver.resolve` and central `resolve` with `task: str | None = None`, `runtime: str | None = None`, `strict: bool = False`. An optional plugin `plan(repo_id: str, variant: str | None, info: Any) -> ResolvedModel` returns a declarative recipe and exact inventory; it cannot download weights or execute source. Strict dispatch considers all compatible candidates and rejects ambiguity; legacy `resolve` remains an explicitly separate adapter. Plugins load sibling `recipes.json` as data through the lightweight core loader, never by importing modality packages.

- [ ] Write parameterized planner tests for single GGUF, all `00001-of-00002` shards, absent shard, indexed tensor dependencies, SentencePiece `.model`, adapters/base/code commits, invalid metadata, traversal/duplicate destinations, incompatible architecture and two valid recipes. Assert `--dry-run` causes zero weight/package-payload downloads, venv and catalog-write calls; only metadata may be cached. Assert canonical identity follows the provider response rather than lowercasing URI spelling. Assert an explicit alias colliding with a bundled ID fails without mutation. Assert plain re-pull keeps the recorded commit, `update` advances it, and `update + revision` fails. Assert private/gated failures are not retried; transient metadata failure makes at most three attempts.
- [ ] Run `pytest tests/core/installation/test_planning.py tests/core/test_hf_resolver_dispatch.py -q`; expect RED for strict planning/inventory/identity checks.
- [ ] Implement metadata-only planning, curated recipe selection and frozen runtime overlays. Select the platform dependency lock from recipe data/provider package metadata without installing packages or fetching payloads in dry-run; unsupported lock/platform/build requirements fail clearly. Validate architecture/configuration, shard/index/tokenizer/base/code closure and platform/build policy; tags discover candidates only. Display candidate plus rollback/temp/environment disk costs and unknown estimates; reject insufficient headroom before preparation. Keep `allow_remote_code`/`--trust-remote-code` consent distinct from `allow_build`; reviewed recipes may explicitly trust their pinned code, arbitrary Hub metadata may not. Improve the existing embedding artifact filter without advertising a strict embedding recipe prematurely.
- [ ] Run the Task 3 command plus `pytest tests/core/test_resolvers_hf.py tests/core/test_curated.py tests/core/test_catalog_security.py -q`; require GREEN.
- [ ] Commit `feat(install): validate runtime recipes and complete artifact plans` with Task 3 paths only.

### Task 4: Fetch and Validate Private Artifact Generations

**Files:** Create `src/muse/core/installation/generations.py`, `tests/core/installation/test_generations.py`; modify `src/muse/core/artifacts.py`; test `tests/core/test_artifacts.py`.

**Interfaces:** Produce `create_generation(plan: InstallPlan) -> GenerationRef`, `generation_path(ref: GenerationRef) -> Path`, `fetch_artifacts(plan: InstallPlan, ref: GenerationRef) -> Path`, `read_generation(ref: GenerationRef) -> InstalledGeneration`, `write_generation(generation: InstalledGeneration) -> None`. Layout: `<catalog>/generations/<model_id>/<generation_id>/` with `plan.json`, `artifacts/`, `venv/`, `receipt.json`; incomplete markers are not complete receipts.

- [ ] Write isolated tests for exact immutable downloads, missing/truncated/hash-mismatched files, required licenses, duplicate destinations, ENOSPC and cancellation. Assert approved cache-root symlinks work, links outside approved roots fail, and materialized bundles still reject symlinks. Assert a dependency-only update shares artifact content and does not recopy weights. Assert no private generation enters `known_models()` or active catalog state.
- [ ] Run `pytest tests/core/installation/test_generations.py tests/core/test_artifacts.py -q`; expect RED for generation operations and trusted-cache policy.
- [ ] Implement core-owned fetching and fsync/atomic receipt writes using existing path/ownership/cache checks. Expand the existing artifact machinery to single snapshots without weakening bundle validation. Keep immutable content references and bounded retries; never follow model-supplied download callbacks on the strict path. Resolve generation references only inside the owned generation root and reject symlinked generation paths.
- [ ] Run the Task 4 command; require GREEN with all failure cases retaining the old active record.
- [ ] Commit `feat(install): prepare validated private artifact generations` with Task 4 paths only.

### Task 5: Build Reproducible Environments at Final Paths

**Files:** Create `src/muse/core/installation/environment.py`, `tests/core/installation/test_environment.py`; modify `src/muse/core/venv.py`; test `tests/core/test_venv.py`.

**Interfaces:** Produce `prepare_environment(plan: InstallPlan, ref: GenerationRef, *, allow_build: bool, development: bool = False) -> EnvironmentReceipt`, `environment_is_current(receipt: EnvironmentReceipt) -> bool`, `muse_code_identity(*, development: bool) -> str`. Consume the generation's final `venv/` path; reuse existing owned process execution, capture and cleanup.

- [ ] Write tests asserting console-script shebangs reference the final path; a stable install loads a hashed Muse wheel rather than `-e`; full dependency/source hashes, Python, platform and build options are recorded. Assert unchanged package version plus edited runtime code invalidates developer evidence. Assert missing wheel without `allow_build` launches no compiler, incompatible Python fails early, remote code without trust is rejected, and cancelled/incomplete environments resume only by validated reuse or safe rebuild.
- [ ] Run `pytest tests/core/installation/test_environment.py tests/core/test_venv.py -q`; expect RED for fixed-prefix/code-identity behavior.
- [ ] Implement final-path venv creation without changing legacy canonical-venv semantics. Resolve and freeze the complete platform dependency graph before installation, preserve wheel/source hashes, and install the exact recorded artifacts. Build or obtain a versioned Muse wheel whose content identity includes packaged runtime code/assets. Bound recipe-approved source builds with explicit job/thread/time budgets; never run host package managers. Distinguish complete receipts from partial environments.
- [ ] Run the Task 5 command; require GREEN, including executable-script and edited-code regressions.
- [ ] Commit `feat(install): freeze generation environments and runtime code` with Task 5 paths only.

### Task 6: Verify Privately Under Coordinated Admission

**Files:** Create `src/muse/core/verification_leases.py`, `src/muse/core/installation/{verification,loading}.py`, `src/muse/cli_impl/private_probe.py`, `src/muse/admin/routes/installations.py`, `tests/core/installation/test_verification.py`, `tests/cli_impl/test_private_probe.py`, `tests/admin/routes/test_installations.py`; modify `src/muse/core/config.py`, `src/muse/cli.py`, `src/muse/cli_impl/{load_director,supervisor,probe_worker}.py`, `src/muse/admin/routes/__init__.py`.

**Interfaces:** Produce `verify_generation(ref: GenerationRef, *, gateway_url: str | None = None) -> VerificationReceipt`, parent-side `run_private_probe(ref: GenerationRef, *, reservation: dict[str, JsonValue]) -> VerificationReceipt`, child-side `run_private_probe_worker(ref: GenerationRef) -> int`, `load_generation_backend(generation: InstalledGeneration, *, device: str) -> Any`, and `device_lease(pool: str, *, owner: Literal["gateway", "probe"], catalog_identity: str) -> ContextManager[None]`. Add `LoadDirector.reserve_probe(*, pool: str, memory_gb: float) -> str` and `release_probe(token: str) -> None`; reservations participate in existing capacity accounting but never evict. Add authenticated `POST /v1/admin/installations/verify` with exactly `model_id` and `generation_id`, returning the existing job envelope. Hidden `_generation_probe` CLI takes identifiers, not command/source paths, and runs inside the candidate interpreter.

- [ ] Add tests for load/inference/output-validation/cleanup failure, offline-only execution, explicit unsupported placement, insufficient CPU/GPU headroom, bounded owned descendants and cancellation. Assert probe cleanup must complete before success. Assert admin rejects arbitrary paths/commands, wrong catalog, nonlocal callers and missing auth. Add a slow race test where standalone verification owns a device lease as gateway startup attempts ownership; assert only one proceeds. Two uncoordinated gateways must yield a staged capacity error, not concurrent probes.
- [ ] Run `pytest tests/core/installation/test_verification.py tests/cli_impl/test_private_probe.py tests/admin/routes/test_installations.py -q`; expect RED for private verification/admission.
- [ ] Implement private backend construction from frozen receipts using the worker loader's effective settings and recipe probe. Delegate to a matching local authenticated gateway when present; absent admin credentials/support leaves the model staged, never an independent competing probe. Do not recursively acquire the CLI-held model lock inside that job. Otherwise take an exclusive device verification lease; participating gateways hold device ownership for their lifetime and verify the same catalog identity before accepting delegation. Use a shared per-user coordination root, configurable for isolated tests; reject unverifiable coordination rather than falling back to an independent probe. Check memory headroom and enforce recipe thread/process/time/RSS budgets with the existing owned process runner plus a budget watchdog. Unsupported enforcement leaves a candidate staged. Report actual native placement and validated finite measurements; release reservations in every completion path.
- [ ] Run the Task 6 command plus `pytest tests/cli_impl/test_load_director.py tests/cli_impl/test_probe_worker.py tests/admin/test_e2e_admin_router.py -q`; require GREEN, including the marked slow lease race separately on isolated processes.
- [ ] Commit `feat(install): verify staged generations through safe admission` with Task 6 paths only.

### Task 7: Publish Atomically and Preserve Serving Semantics

**Files:** Modify `src/muse/core/installation/generations.py`, `src/muse/core/catalog.py`, `src/muse/cli_impl/{gateway,supervisor,worker}.py`, `src/muse/admin/routes/models.py`; create `tests/core/installation/test_publication.py`; test `tests/cli_impl/{test_gateway_default_model,test_model_readiness,test_gateway_lazy}.py`, `tests/core/test_catalog_security.py`.

**Interfaces:** Produce `publish_generation(generation: InstalledGeneration, *, expected_active: str | None) -> None` and `verification_is_current(generation: InstalledGeneration, *, manifest: dict) -> bool`. Consume Task 6's `load_generation_backend(generation: InstalledGeneration, *, device: str) -> Any` for active worker loading. Catalog entries gain `active_generation`, separate `candidate_generation`, and `default_for: list[str]`; `get_manifest`, `load_backend` and interpreter lookup keep their caller contracts. Private probing and worker loading share backend construction rather than creating a second runtime.

- [ ] Add fault-injection tests before/after receipt fsync, catalog replacement and directory fsync. Assert old-or-complete-new visibility; stale `expected_active` rejects; post-commit cleanup failure reports committed success plus warning. Race admin enable/device edits with publication and assert controls survive, with changed verification assumptions stale. Assert staged-only explicit requests return 503, candidates do not poison old readiness, runtime overlays remain frozen, legacy paths load unchanged, a running gateway sees a new publication, and adding a model preserves existing defaults across restart.
- [ ] Run `pytest tests/core/installation/test_publication.py tests/cli_impl/test_gateway_default_model.py tests/cli_impl/test_model_readiness.py -q`; expect RED for pointer/evidence/default semantics.
- [ ] Implement publication under the existing short catalog write lock, checking model lease ownership and expected prior identity. Fsync complete receipts before the single active-pointer commit. Preserve operator state and snapshot established implicit defaults into `default_for` in that same catalog transaction; an unset modality chooses its first eligible publication. Read generation runtime data without reapplying mutable curated runtime settings. Refresh by publication identity; distinguish verification, eligibility, residency and capacity in management output. Retain former active resources and invalidate relevant caches, not the running gateway process.
- [ ] Run the Task 7 command plus `pytest tests/core/test_catalog.py tests/core/test_catalog_security.py tests/cli_impl/test_gateway_lazy.py tests/admin/routes/test_models_routes.py -q`; require GREEN. Run owned crash-boundary subprocess tests with `-m slow` separately.
- [ ] Commit `feat(install): publish verified generations without disrupting serving` with Task 7 paths only.

### Task 8: Route Pull and Probe Commands Through the Coordinator

**Files:** Create `src/muse/core/installation/coordinator.py`, `src/muse/cli_impl/pull.py`, `tests/core/installation/test_coordinator.py`, `tests/cli_impl/test_pull.py`; modify `src/muse/cli.py`, `src/muse/cli_impl/{probe,pull_errors}.py`, `src/muse/core/catalog.py`, `src/muse/admin/{operations,client}.py`, `src/muse/admin/routes/models.py`; test `tests/admin/test_operations.py`, `tests/cli_impl/test_probe.py`.

**Interfaces:** Produce `install(identifier: str, options: InstallOptions) -> InstallResult`, `verify_staged(model_id: str) -> InstallResult`, and CLI adapter `run_pull(identifier: str, options: InstallOptions, *, as_json: bool = False, verbose: bool = False) -> int`. Keep `catalog.pull(identifier, *, base_override=None)` as a compatibility wrapper with proper failure propagation. CLI/admin use the same result/error schema, not parsed log text.

- [ ] Add tests asserting plain verified idempotent pull creates no new environment; matching staged work resumes; changed code/device/recipe re-verifies. Assert normal probe failure gives nonzero and preserves the old active model. Assert `--download-only` and `--no-probe` produce `state == "staged"`, exit zero and no route; `models probe --staged` publishes, while ordinary probe targets active. Test `--update`, `--revision`, `--task`, `--runtime`, `--name`, `--allow-build`, `--trust-remote-code`, `--dry-run`, `--json`, unknown experimental profiles, concurrent pulls, cancellation and redacted credential-bearing exceptions. Assert loaded/leased updates fail before download/build.
- [ ] Run `pytest tests/core/installation/test_coordinator.py tests/cli_impl/test_pull.py tests/cli_impl/test_probe.py tests/admin/test_operations.py -q`; expect RED for the new command contract.
- [ ] Implement the lifecycle with existing per-model/resource/cache locks and bounded same-input retries. Before strict public rollout, keep the coordinator behind an internal adapter. Wire admin result transport directly to structured outcomes, including failures after publication. Preserve previous commits on plain re-pull and keep failed candidate metadata separate. Make expected errors friendly and stage-specific; verbose tails remain bounded and redacted. Do not offer opaque legacy callbacks as strict recipes: unmigrated preparation requires explicit experimental/download-only use until a verified recipe exists.
- [ ] Run the Task 8 command plus `pytest tests/core/test_catalog_pull_lock.py tests/cli_impl/test_pull_errors.py tests/admin/routes/test_models_routes.py -q`; require GREEN.
- [ ] Commit `feat(cli): expose verified pull staging and resumable updates` with Task 8 paths only.

### Task 9: Integrate Generation Retention and Recovery With Storage

**Files:** Modify `src/muse/core/storage.py`, `src/muse/core/config.py`, `src/muse/core/installation/generations.py`, `docs/STORAGE.md`; create `tests/core/installation/test_recovery.py`; test `tests/core/test_storage.py`.

**Interfaces:** Extend existing `inspect_storage`, `plan_prune`, `execute_prune` contracts without a new cleanup group. Produce `recover_generations() -> tuple[dict[str, JsonValue], ...]` for reports only, never heuristic promotion. Add config `storage.retained_generations` with default `1` for previous verified generations; active is additional.

- [ ] Add tests asserting active + one prior verified generation + leased/in-progress work survive cleanup; older unreferenced generations honor the current grace period. Assert a shared artifact survives until its last reference disappears, a newly published pointer defeats a stale prune plan, and identity-swapped directories are retained. Assert interrupted ownership-ambiguous environments are reported, not deleted/promoted, and failed-attempt bytes appear in inventory. Assert dependency-only generations count shared weights once.
- [ ] Run `pytest tests/core/installation/test_recovery.py tests/core/test_storage.py -q`; expect RED for generation references and retention.
- [ ] Implement receipt-based ownership/reachability in existing storage inventory and deletion-time revalidation. Use the established lock order for cleanup versus publication; hold necessary resource/cache protection until deletion completes. Recover from durable active pointers and complete receipts; explicit compatible resume may reuse work, but missing/incomplete environment evidence requires rebuild. Keep legacy rollback backups protected until ownership is established.
- [ ] Run the Task 9 command; require GREEN, including controlled publication/cleanup race tests.
- [ ] Commit `feat(storage): account for generation retention and interrupted installs` with Task 9 paths only.

### Task 10: Gate Strict Rollout With Catalog and Live Conformance

**Files:** Modify `src/muse/curated.yaml`, `src/muse/core/curated.py`, `src/muse/cli_impl/pull.py`, `README.md`, `docs/HF_PLUGINS.md`, `docs/MODEL_SCRIPTS.md`, `scripts/preflight.py`; create `tests/core/installation/test_recipe_conformance.py`, `tests/integration/test_installation_lifecycle.py`, `docs/INSTALLATION.md`.

**Interfaces:** Consume Task 3's curated `runtime_profile` and `experimental` fields; ordinary strict resolution fails early for unvalidated entries. Shared conformance tests accept each registered `RuntimeRecipe` and use its own representative probe. The companion decision plan registers its recipes through these same interfaces, not another installer.

- [ ] Write a curated coverage test asserting every advertised entry has a validated profile or an explicit experimental designation. Add opt-in pinned live cases for single/split GGUF: fresh pull, offline load/inference, unchanged re-pull, download-only then staged verification, failed update preserving old inference, unload and safe cleanup. Assert no manual catalog registration or undeclared HTTP downloads. Unsupported platform/build combinations must report staged/actionable outcomes, not a readiness claim.
- [ ] Run `pytest tests/core/installation/test_recipe_conformance.py -q`; expect RED until coverage/rollout gates exist.
- [ ] Enable the coordinator as the normal public path only after Tasks 1–9 pass; designate unmigrated profiles explicitly experimental in catalog display/help. Document supported families/platforms, alias identity, trust/build consent, plan estimates, offline guarantees, retention costs and changed pull/update/no-probe success semantics. Register the conformance gate with preflight. Document integration environment variables in the live test module; use `MUSE_REMOTE_SERVER` for remote gateway cases and an explicit opt-in for downloads/builds.
- [ ] Run `pytest -m "not slow" -q`, `pytest tests/ -q`, then `python scripts/preflight.py`; require zero unexpected failures. Separately run opt-in live tests for each claimed platform/profile using isolated catalog/cache roots and a pinned gateway running the implementation. Report skipped platforms and experimental families explicitly. Never repurpose the user's active service for these tests.
- [ ] Commit `feat(install): enable strict supported recipes with lifecycle conformance` with Task 10 paths only, after fresh final review.

## Execution Boundaries

Use an isolated worktree at execution time. The starting workspace contains earlier uncommitted runtime/security fixes and intentionally failing native-placement regressions; explicitly carry the required reviewed changes into the execution baseline without committing unrelated edits. Tasks 1–10 are sequential integration gates, though isolated tests may run concurrently. Finish this foundation before shipping the companion modality; its recipes can then be added without redesigning installation.

Do not upgrade packages in existing model environments, remove user artifacts, restart services, commit unrelated files or push/tag as part of this plan without a separate scoped request. A green unit lane does not replace pinned live conformance or establish model quality.
