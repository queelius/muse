# Storage lifecycle

Muse keeps model environments and newly pulled weights below the configured
catalog directory (`~/.muse` by default):

```text
~/.muse/
  catalog.json
  venvs/<model-id>/
  weights/<artifact-or-Hugging-Face-cache>/
```

Older Muse releases put bundled-model weights in the user-wide Hugging Face
cache. Those legacy paths remain usable and are reported, but Muse never
deletes them because other applications may share the same blobs. New pulls
use the Muse-owned weights root so their lifecycle can be managed safely.

## Inspecting storage

```bash
muse doctor storage
muse doctor storage --json
```

The report distinguishes:

- catalog-referenced venvs and weights;
- old incomplete downloads and abandoned staging workspaces;
- unreferenced resources, which may have been intentionally retained by
  `muse models remove`;
- missing or unsafe catalog paths;
- user-wide Hugging Face and pip caches, reported separately;
- filesystem free space.

The doctor is read-only. It exits `1` when it finds actionable storage health
conditions and `2` when state cannot be inspected safely.

## Reclaiming space

```bash
# Delete old incomplete downloads and definitively abandoned staging data.
muse storage prune

# Preview the identical selection without changing files.
muse storage prune --dry-run

# Also consider unregistered Muse-owned venvs and weights.
muse storage prune --include-unreferenced --dry-run
muse storage prune --include-unreferenced

# Change the default 24-hour grace period.
muse storage prune --older-than-hours 168
```

Default pruning is deliberately narrow. It never evicts a cataloged model,
even when disabled, and it never touches shared Hugging Face or pip caches.
`--include-unreferenced` is explicit because current catalogs cannot
distinguish an accidental orphan from files deliberately retained for a fast
future re-pull.

Cleanup and pulls coordinate through process and filesystem locks. Before a
planned path is deleted, Muse checks its filesystem identity, newest descendant
timestamp, and current catalog references again. Catalog references take priority
over staging-like names, including custom model IDs, and references through
symlinked parent directories identify the same owned resource. Venv cleanup
also respects the candidate's actual model lifetime lease rather than trusting
the owner recorded in an earlier cleanup plan.

Venv transaction workspaces are always preserved for manual recovery. There is
no durable commit receipt that lets a later cleanup distinguish an interrupted
update from a successfully committed update whose backup cleanup failed. Even
an executable Python in the canonical venv does not prove its packages are
complete. Normal successful transactions remove their own backups; leftover
`.MODEL.transaction-*` directories are never bulk-pruned, even with
`--include-unreferenced`. Inspect them and verify the canonical environment
before manually recovering or removing a backup.

## Deduplicating venvs

Every model venv carries its own copy of the same large payloads: Torch,
Triton, the CUDA runtime libraries, llama.cpp builds. `muse storage dedupe`
replaces content-identical copies with hardlinks to one file:

```bash
muse storage dedupe --dry-run   # report only (it still reads and hashes)
muse storage dedupe
```

- **What it links.** Regular files of at least 1 MiB under each venv's
  `lib/python*/site-packages`, when SHA-256, size, permissions and owner all
  match. `bin/` scripts, `__pycache__`, transaction workspaces
  (`.MODEL.transaction-*`) and files changed in the last minute are skipped.
- **Why it is safe.** pip never edits an installed file in place: its wheel
  installer unlinks a path before writing new contents, and uninstall moves
  files aside. An upgrade in one venv therefore breaks that venv's link and
  leaves every other copy alone.
- **How it swaps.** Each replacement re-checks both files, links to a
  temporary name and renames it over the target. A running worker keeps
  the file it already has open.
- **Memory too.** Workers that load the same shared library from one inode
  share its page-cache pages, so host RAM use drops when several venvs load
  the same Torch or CUDA libraries. VRAM is unaffected: CUDA loads kernels
  into each process.
- **Automatic after installs.** After every successful pull or refresh,
  Muse links that venv's files to identical copies in the others
  (`storage.dedupe_venvs`, default `true`). A refresh first clones the venv,
  so the clone is full size until this pass relinks it. Nothing else ever
  needs a manual pass, except a venv built before this feature.

On the 2026-10-19 development host, the first pass linked 2,341 of 2,762
large files across 33 venvs and reclaimed 118.5 GiB (the venvs went from
154 GiB to 36 GiB on disk).

## Automatic maintenance

Before a model pull, Muse checks filesystem headroom. By default, if free space
falls below either 50 GiB or 5%, Muse automatically runs the same narrow safe
cleanup for transient data untouched for at least 24 hours. It never includes
unreferenced model environments or weights, and cleanup problems are reported
without hiding the pull's own result.

The policy is configurable through `config.yaml`, `muse config set`, or the
corresponding environment variables:

```yaml
storage:
  auto_prune_before_pull: true
  auto_prune_grace_hours: 24
  auto_prune_min_free_gb: 50
  auto_prune_min_free_percent: 5
```

Set `storage.auto_prune_before_pull` to `false` to disable it. Explicit
`muse storage prune --include-unreferenced` remains the only bulk cleanup that
can remove retained/unregistered model resources.

## Shared caches

The reported Hugging Face and pip caches are user-wide and may be shared with
other applications, so Muse never deletes from them. After reviewing their
contents, manage them with their owning tools:

```bash
hf cache ls
hf cache prune
python -m pip cache info
python -m pip cache purge
```

`hf cache prune` targets detached revisions and incomplete downloads; use
`hf cache rm` for explicitly selected repositories or revisions. Pip's purge
removes the whole pip download/wheel cache, not installed packages.

Hugging Face's cache tools do not consult Muse's catalog. A detached snapshot
can still be pinned by a cataloged Muse model, especially a legacy model using
the shared cache. Compare proposed removals with `muse doctor storage` and the
catalog's `local_dir` paths before pruning; do not assume "detached" means
unused by Muse. This also applies when pointing HF tools at Muse's owned cache.

**Legacy models in the shared cache.** Models pulled before 0.61.0 loaded
their weights straight from the shared Hugging Face cache and carry no pinned
revision or download receipt, so current Muse refuses to serve them
(`model_unservable`, "does not match the current immutable artifact
provenance"). `muse doctor storage` lists them under the Hugging Face cache.
`muse pull <id>` fixes one, moving its weights into Muse's own store.

**Pulls reuse the shared cache.** Before a cache-mode download (bundled
scripts and resolver pulls), Muse hardlinks the shared cache's blobs for that
repo into `<catalog_dir>/weights` (`muse.core.hf_seed`). Each blob is verified
first: an LFS blob against the SHA-256 in its name, a regular blob against its
git blob SHA-1. huggingface_hub then finds what it needs under the same names,
so nothing already on disk is downloaded again and the two caches share the
same disk blocks. Seeded blobs the download did not use are removed afterwards.
Once the shared copy is deleted, Muse's link keeps the data. (Artifact-bundle
downloads already copy files from the shared cache when they match.)

## Removing one model

```bash
muse models remove MODEL_ID
muse models remove MODEL_ID --purge
```

Without `--purge`, Muse unregisters the model and retains its files. With
`--purge`, Muse deletes its per-model venv and unshared Muse-owned weights.
Hugging Face snapshots are cleaned at repository scope so the sibling `blobs/`
directory is reclaimed; a repository is preserved while any other catalog
entry references one of its revisions. Legacy shared-cache weights are always
preserved.

## Why venvs are still separate

Each model keeps a dedicated environment, so one model's upgrade can never
break another model's runtime. A shared mutable base environment is
intentionally avoided. Hardlink deduplication (above) recovers the disk cost
of identical payloads without giving up that isolation: a venv that installs
a different version simply gets its own file.
