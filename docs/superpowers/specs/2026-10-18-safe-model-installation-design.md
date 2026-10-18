# Safe model installation design

**Date:** 2026-10-18
**Status:** Draft for user review
**Companion:** [Decision API and curated models](2026-10-18-decision-api-design.md)

## Purpose and contract

Keep `hf://` as a model source reference, without promising that Muse can run
every Hugging Face repository. Accept repositories that satisfy a supported
runtime recipe. Curated aliases and direct references use the same installer.

A normal successful pull means that Muse has downloaded the complete artifact
set, prepared an environment, loaded the model, completed a bounded representative
inference, and published the verified installation. A failure explains the stage,
cause and next action without replacing a working installation.

Verification is evidence for a particular generation, environment, device and
probe shape. It does not guarantee future free memory, arbitrary request sizes,
model accuracy, or immunity to upstream and hardware failures.

## Scope and alternatives

Use validated runtime families, rather than a curated-only allowlist or a
best-effort universal loader. A repository need not be individually curated when
its format and configuration satisfy a supported recipe. Custom architectures
remain an explicit extension path through model scripts and modality plugins.

This work covers resolution, installation, verification, catalog publication,
storage ownership and ordinary serving integration. It does not replace Muse's
scheduler, introduce a package sandbox, train models, change host drivers, or
manage unrelated processes or virtual machines.

Preserve existing working installations. Finish the outstanding native GGUF
device and memory-pool consistency work before depending on its placement report
for verification or admission.

## Evidence motivating the change

The 2026-10-18 audit completed a fresh non-curated single-file GGUF pull and
offline CPU inference. The existing targeted catalog, resolver, venv and plugin
lane passed 348 tests. Separate diagnostic cases exposed missing protections:

- Repository basenames become catalog IDs, allowing two publishers' same-named
  repositories to replace the same record.
- GGUF selection downloads only the first matching shard. The omission also
  occurs against the published split-file metadata of Qwen3-32B-GGUF.
- Embedding filters omit SentencePiece `.model` files. The filter omission was
  reproduced; a real model-load failure from it was not established in the audit.
- The pull CLI intentionally ignores a failed post-pull probe's return code.
- Generic task tags can select an incompatible loader, as with Decima being
  treated as an ordinary NLI classifier.

These require regression tests and a stronger publication boundary, not removal
of the source URI syntax alone.

## Model identity and runtime recipes

Evolve `ResolvedModel` and the existing artifact receipts into explicit,
serializable installation inputs. Avoid a second independent installation
framework. The conceptual records are:

| Record | Responsibility |
| --- | --- |
| Source reference | Provider, canonical publisher/repository, immutable revision and selected variant |
| Runtime recipe | Versioned profile, modality, compatible architecture/layout, required artifacts, environment requirements, device support, request limits and probe |
| Install plan | Validated source and recipe, exact file set, dependency/build policy, intended device, resource estimates and proposed catalog identity |
| Installed generation | Plan fingerprint, actual environment/artifact receipts, verification result and immutable runtime locations |

Runtime-affecting curated overlays are resolved into the plan and frozen with
the installed generation. A later YAML edit must not silently substitute new
runtime settings or artifact pins on reads of that generation. Display-only
metadata may refresh independently. Operator overrides remain separate controls;
changes that alter verification assumptions mark that evidence stale and require
re-verification before eligibility on the new configuration.

Separate identity from presentation. A model slot includes provider, full
repository identity, variant and runtime profile. The revision identifies a
generation of that slot, not a new user-visible model on every update. The
runtime profile distinguishes, for example, a decision scorer from a chat loader
using the same source weights.

Use the provider's canonical repository identity, not an unconditional lowercase
transformation. Generated aliases include publisher information and a stable
identity suffix so filesystem escaping cannot create a collision. Preserve
existing curated aliases. Reject an explicit alias already bound to a different
slot; never silently overwrite or auto-rename it. Existing aliases are not
renamed during migration.

Recipes own compatibility checks. Hub tags and filenames discover candidates;
they do not establish compatibility. Validate model configuration, architecture,
artifact layout and required backend capabilities. When several recipes remain
valid, require an explicit task/runtime selection rather than first-match wins.
A curated entry makes that selection itself.

## Installation lifecycle

The lifecycle is:

```text
resolve -> validate plan -> prepare private generation -> verify -> publish
                                      |
                                      +-> staged or failed, never routable
```

### Resolve and validate

Resolve metadata to immutable commits for every repository involved, including
adapter bases and separately hosted executable code. Expand file patterns into
an exact required inventory before fetching weights. Validate paths, duplicate
destinations and shard completeness. Select one compatible precision/layout;
do not download every alternate framework or precision by default.

Show the chosen task, runtime, device policy, artifact size, environment/build
requirements and expected disk use. Distinguish known sizes from estimates.
Check disk headroom for the candidate plus retained prior generation. Resource
estimates do not substitute for a bounded real probe.

`muse pull <ref> --dry-run` resolves and displays this plan without creating a
model environment, downloading weights, or changing catalog records. Small
provider metadata caching is permitted. Unsupported/ambiguous inputs fail here
where possible, with a supported profile or custom-script hint.

### Prepare privately

Create a generation under a Muse-owned generation directory. Build its venv at
its final absolute path: moving an already-built venv would invalidate absolute
paths in scripts. Privacy here means absent from active catalog publication,
not protection from other programs running as the same user.

Core installation code owns artifact fetching and validation. Recipes supply
declarative requirements rather than unconstrained download callbacks. Pin and
record dependencies, wheel/source hashes, Python/platform information and build
options. A source build must be announced, resource-bounded and allowed by the
recipe/operator policy; never change drivers or system packages automatically.
Repeatability of a source recipe is not a claim of bit-identical native builds.
Record Muse and runtime code identity as well as dependency versions. Stable
worker generations must not import code from an untracked moving editable
checkout. Explicit development-mode editable installs become stale when their
relevant code fingerprint changes; a package version alone cannot validate them.

Remote-code execution and custom Python source hooks require an explicitly
trusted recipe or operator opt-in. A pinned commit establishes identity, not
trust; neither a Hub tag nor an arbitrary model-supplied install instruction
may enable executable code. A venv does not protect host files or credentials.

Validate all required files, available content hashes and expected sizes before
loading. Missing shards, truncated transfers and unresolved tokenizer/base
dependencies are errors. Shared Hub cache links may resolve only into approved
cache roots; materialized bundles retain their stricter regular-file policy.
Do not bypass existing filesystem and provenance checks.

Use the existing pull/resource/cache locking responsibilities. Reject an update
to a loaded or leased model with an unload/retry instruction before mutation;
automatic live replacement is not part of this release. Never hold the global
catalog lock through network calls, installation or inference.

### Verify with the serving runtime

Probe the private generation through the same backend construction and request
contract used by workers, without adding it to global active discovery. Run
load plus a small modality-specific inference, check the result's structure and
finite values, and release all owned resources before publication.

Verification uses local artifacts only. A runtime that needs additional weights,
code or configuration must declare them in the plan; it must not silently fetch
them during the probe or first HTTP request. Artifact completeness is checked
structurally as well as through this small probe.

Use the same effective device resolution and memory-pool policy as serving.
Native GPU capability must come from the installed native backend, not merely
host GPU visibility. Display `auto` placement explicitly; an explicit unsupported
GPU request fails rather than silently falling back.

When a gateway is running for the target catalog, the CLI delegates verification
to a local authenticated admin job. That job accepts only a validated private
generation reference and reserves probe capacity through the gateway's existing
admission machinery; it does not accept arbitrary commands. Without a gateway,
standalone verification must hold an exclusive verification lease for the
selected device, also honored by participating gateway startup and other probes.
If multiple Muse instances cannot coordinate a safe reservation, leave the model
staged with an actionable capacity error rather than guessing.

Refuse or defer verification when a safe reservation cannot be obtained; do not
evict serving workers as a pull side effect. CPU memory/process/thread limits
and bounded execution are required. Reserve headroom for unrelated host work.
GPU admission is a cooperative budget, not a hard security boundary for arbitrary
native code or a guarantee against unrelated applications allocating memory.

### Publish atomically

Write and fsync the complete generation receipt before atomically changing the
catalog's active-generation reference. Verify that publication still owns the
model lease and that its expected prior generation has not changed. Re-read and
preserve operator enable/device settings under the short catalog write lock.

Only publication makes the new generation routable. A failed candidate never
replaces the active reference. A crash before the pointer commit leaves an
inactive candidate; a crash after it leaves a complete referenced generation.
Cleanup failures after publication are reported separately and must not undo or
misreport a committed installation. Retain the preceding generation under the
existing storage retention policy until safe cleanup; do not delete it during
publication.

## User commands and failure behavior

- `muse pull <ref>` verifies and publishes, or resumes a matching pending plan.
  An unchanged installed plan with valid receipts and current verification is
  idempotent. A changed environment/device/recipe requires re-verification.
- Existing installations keep their recorded commits on a plain re-pull.
  `--update` requests a new upstream resolution; `--revision <commit>` explicitly
  selects one. These options are mutually exclusive. Show old and new pins.
- `--task` and, when needed, `--runtime` resolve ambiguity without guessing.
  Curated aliases already specify these choices.
- `--name <alias>` assigns an unused friendly alias to a new slot; it cannot
  replace another slot or rename an existing installation implicitly.
- `--allow-build` permits the recipe's declared bounded native source build.
  Without a compatible wheel or this explicit permission, fail with the build
  requirement before launching a compiler. It never authorizes driver changes.
- `--download-only` prepares but does not publish an unverified generation.
  Retain `--no-probe` as a documented compatibility alias for this behavior.
  Exit zero means the explicitly requested preparation completed, not readiness.
- `muse models probe <id> --staged` verifies and publishes a staged candidate.
  Without `--staged`, probing continues to target the installed active generation.

Normal pull returns nonzero on unsupported input, dependency/download failure,
unsafe or incomplete artifacts, unavailable verification capacity, load failure,
inference failure, or cancellation. Output names the failed stage and whether
the previous generation is unchanged. No default traceback for expected failures;
verbose diagnostics retain bounded details and never credentials.

Use stable machine-readable error codes and a JSON result mode, shared with
admin pull jobs. Distinguish permanent incompatibility from retryable network or
capacity failures. Retries are bounded and reuse only matching immutable inputs;
they must not select a different model, precision or device behind the user.

## Serving and observability

Keep preparation state separate from active installation state. A slot may have
a working active generation and a failed candidate at the same time. CLI/admin
output shows both without calling the working model failed. Candidate failures
are not copied into the active generation's readiness field.

Active model reporting distinguishes verification evidence, present eligibility,
loaded state and current capacity. A staged-only model is visible to management
commands, but is absent from normal inference selection and returns a clear
unverified-model error if explicitly requested. `/health` remains liveness.

Published models use the ordinary modality registry, gateway admission/queues,
worker supervisor, telemetry, idle unload and storage ownership rules. No special
HF worker lifecycle. Refresh catalog views from publication identity so ordinary
pulls require no gateway restart. Do not change an existing implicit or explicit
default as a side effect of installation; preserve an established default before
adding an eligible model. The first eligible model may establish an unset
default under a documented deterministic rule.

## Storage and interrupted work

Generation receipts record all artifact/environment ownership and references.
Storage cleanup protects active generations, live leases, in-progress attempts,
retained rollback generations and shared artifacts. Extend the existing storage
inventory rather than adding another cleanup command group.

Reuse identical artifact content across generations rather than copying model
weights for a dependency-only update. Retain one previous verified generation
per model by default. Older unreferenced generations become eligible for the
existing cleanup grace period, subject to leases and shared references. Show
retention and temporary disk costs in the install plan; failed attempts must not
accumulate as an unreported unlimited cache.

After interruption, recovery trusts the active catalog pointer and complete
receipts. It never promotes a staging directory heuristically. Resume compatible
work explicitly; mark incomplete environments for safe rebuild. Interrupted
attempts with ambiguous ownership are reported and retained. Preserve current
rollback backups until their publication status is established.

## Compatibility and rollout

Existing working catalog records and canonical venv paths remain loadable and
are not silently migrated or relabeled as newly verified. A successful explicit
re-pull creates the first new-format generation. Report old verification as
legacy evidence, not as a newly completed probe.

Introduce the plan/receipt boundary alongside existing helpers, then migrate
recipe families. Before strict mode becomes the normal public path, every
advertised curated entry must have a validated recipe or an explicit experimental
designation. Legacy opaque callbacks are not silently accepted as strict recipes;
unmigrated pulls require an explicit experimental path. Existing active installs
continue working during this transition. Document the change in pull success,
update and `--no-probe` semantics as a compatibility change.

Implementation boundaries are `core/resolvers*`, `core/artifacts`, catalog and
venv helpers, a focused installation coordinator, CLI/admin job adapters, private
probe support, storage inventory and readiness consumers. Do not grow catalog.py
into a second scheduler. The companion decision modality must exercise this
same path rather than introducing a special installer.

## Acceptance and release gates

Required automated cases include namespaced/case/escaping collisions; aliases
against bundled models; complete GGUF shards; required tokenizer/base/code
artifacts; ambiguous loaders; malformed metadata; private/gated access; bounded
network retries; disk exhaustion; install/load/inference/probe-cleanup failure;
interruptions around each receipt/pointer boundary; stale plans; concurrent pulls;
admin setting races; active worker leases; cleanup races; and offline reload.

Assert that failed updates preserve the prior working model, unverified models
never route, publication refreshes a running gateway, defaults remain stable,
and repeated pulls do not duplicate active environments or artifacts. Use owned
subprocess fault injection for real cancellation/crash behavior; mark these tests
slow and isolate all host configuration and caches.

Run the full suite and preflight before release. Maintain an opt-in pinned live
smoke matrix per supported recipe/platform, including fresh pull, offline
inference, re-pull, staged verification, failed-update preservation, unload and
safe cleanup. Unit-test counts alone are not evidence that every Hub model works.

The first strict vertical slices are single/split GGUF installation and both
decision recipes. Broader recipe migration follows the same conformance gate;
the release must describe the supported set precisely.
