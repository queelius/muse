# Handoff: GPU runtime-upgrade builds (ACE-Step + Wan2.2)

**For:** a fresh Claude Code session running on a remote GPU host.
**Written:** 2026-06-18, from a CPU host, after shipping v0.47.0.
**Repo:** muse (PyPI distribution name `museq`; import path `muse`; CLI `muse`).

## TL;DR - start here

1. `git pull origin main` (you want at least the commit that added this file).
2. Read `CLAUDE.md` (architecture) and the auto-memory at
   `~/.claude/projects/-home-spinoza-github-repos-muse/memory/` (release workflow
   + hard-won lessons). The single most load-bearing memory:
   `feedback_release_workflow.md`.
3. Confirm the GPU: `nvidia-smi` (need a real CUDA GPU; ACE-Step wants ~12GB+,
   Wan2.2 wants 16GB+).
4. Pick the next sub-project (recommended order below) and run the SAME pipeline
   every prior runtime used: **brainstorming -> spec -> writing-plans ->
   subagent-driven-development**, with a **B1 real-API verification step FIRST**.

## Where things stand

muse is a self-hostable OpenAI-compatible multimodal server, ~20 modalities,
published on PyPI as `museq`. Latest release: **v0.47.0** (Supertonic-3 CPU TTS).

This GPU session is the back half of a 4-item "runtime upgrade" roadmap that
came out of a HuggingFace model survey. Status:

- DONE (CPU, v0.47.0): **Supertonic-3** on-device TTS. Shipped.
- REMOVED: **TRELLIS.2** (3D). Cut by the user (its `trellis2`+`o_voxel`
  standalone SDK with CUDA build deps was the gnarliest install of the four).
  Its spec/plan were deleted; recoverable from git history if ever wanted.
- TODO (this session): **ACE-Step** (music) and **Wan2.2** (image-to-video).

Neither remaining item has a spec or plan yet - they were GPU-gated, so they
were left at the "scoped, not designed" stage. You start each with a fresh
brainstorm.

## The two remaining sub-projects

### 1. ACE-Step (music generation) - recommended first

- Model: `ACE-Step/Ace-Step1.5` (HF). `transformers` + `diffusers`,
  `custom_code` (trust_remote_code), multi-component (Qwen3-Embedding-0.6B +
  acestep LM 1.7B + turbo + VAE). Full song generation (vocals + lyrics +
  style). GPU. License: MIT.
- Fits the EXISTING `audio/generation` modality (`/v1/audio/music`,
  `/v1/audio/sfx`). New runtime in
  `src/muse/modalities/audio_generation/runtimes/` (mirror the existing
  `stable_audio.py`). The wire mostly exists; you will likely add a `lyrics`
  request field (ACE-Step is lyrics-conditioned).
- Effort: medium. Risk: trust_remote_code SDK + multi-model load; verify the
  real pipeline API at B1 before mocking.
- Suggested release: v0.48.0.

### 2. Wan2.2 image-to-video - heavier, do second

- Model: `Wan2.2-TI2V-5B` (TI2V = text+image to video). GPU 16GB+.
- This needs a WIRE CHANGE, not just a model swap: the `video/generation`
  request shape is **text-only today** (`VideoGenerationRequest` has `prompt` +
  `negative_prompt`, NO image input). TI2V adds image conditioning, so you add
  an `image` input (data-URL / http URL / multipart) to the video modality plus
  the Wan2.2 runtime. Closer to a new capability than a model add.
- Mirror `src/muse/modalities/video_generation/runtimes/wan_runtime.py`
  (the existing Wan 2.1 runtime). Decode the input image via the shared
  `muse.modalities.image_generation.image_input.decode_image_input` helper
  (SSRF-guarded; see how chat/completion VLM does it).
- Effort: high (wire + runtime). Suggested release: v0.49.0.

## The workflow to follow (non-negotiable for this project)

Every runtime in this codebase (ShapE, TRELLIS, Hunyuan3D, VLM, Supertonic)
went through this. Do not freelance.

1. **superpowers:brainstorming** - explore the modality's existing code, ask the
   user 1 question at a time, propose approaches, present a design, get approval,
   write a spec to `docs/superpowers/specs/YYYY-MM-DD-<topic>-design.md`, commit,
   user reviews.
2. **superpowers:writing-plans** - bite-sized TDD plan to
   `docs/superpowers/plans/YYYY-MM-DD-<topic>.md`. Front-load a **B1 step**.
3. **superpowers:subagent-driven-development** - fresh implementer subagent per
   task + spec/quality review between tasks. Direct-to-main (no feature branch -
   the user's established workflow). Controller (you) runs the release task
   directly (twine upload / push / gh release are irreversible).

### B1 real-API verification (the most important lesson)

Before writing any runtime code or mock, DOWNLOAD AND RUN the real SDK on the
GPU and record the actual API: pipeline class + import path, load signature,
the run/generate signature, device placement, output extraction, and the git
install URL if not on PyPI. v0.43 Shap-E shipped BROKEN because the mock matched
the implementer's mental model, not reality. The Supertonic B1 (this same
session) caught a `(1,N)`-vs-1-D shape bug and a non-HF cache dir that the README
did not mention. On a GPU box you can run B1 for real - do it.

## Hard constraints / gotchas (these will bite you)

- **ASCII only in committed files.** A pre-commit hook (soul voice check)
  REJECTS em-dashes, en-dashes, smart quotes, arrows, ellipsis. Write plain
  ASCII: `-`, `:`, `,`, `()`, `->`, `...`. If a commit is blocked, strip the
  non-ASCII (e.g. `perl -CSD -i -pe 's/\x{2014}/ - /g; s/\x{2013}/-/g; ...'`)
  and retry. NEVER use `--no-verify`.
- **Release ritual** (the v0.46.0 packaging lesson): bump `pyproject.toml`
  version, run the full fast lane (`pytest tests/ -m "not slow" -q`), then
  `python -m build`, `twine check dist/*`, and a **fresh-venv wheel
  smoke-install** that imports the built wheel and asserts the new model is
  discoverable + curated loads (editable installs hide packaging bugs - data
  files like curated.yaml were silently absent from every wheel until v0.46.0).
  Then `twine upload dist/*`, tag, `git push origin main && git push origin
  <tag>`, `gh release create`. PyPI token is in `~/.pypirc`.
- **Bundled-model data files** ship via `[tool.setuptools.package-data] "*" =
  ["*.yaml","*.json"]` in pyproject - already configured; just do not remove it.
- **Deferred imports**: heavy/SDK imports go inside the Model/`__init__` or an
  `_ensure_deps()` with module-level sentinels; tests mock them. `muse --help`
  and discovery must work without the SDK installed.
- **runtime_helpers**: use `muse.core.runtime_helpers` (select_device,
  dtype_for_name, set_inference_mode, LoadTimer); a meta-test
  (`tests/core/test_runtime_helpers_meta.py`) fails on re-implementations.
- **pip_extras audit**: `tests/models/test_pip_extras_audit.py` may require a
  `MODULE_TO_PYPI` entry for a new top-level import.
- **Curated entries**: heavy GPU models are curated `device: cuda` entries (not
  bundled defaults) and are NOT added to the free-tier CI smoke matrix
  (`.github/workflows/fresh-venv-smoke.yml`).

## Patterns to mirror (read these first)

- Music runtime: `src/muse/modalities/audio_generation/runtimes/stable_audio.py`
  + `src/muse/modalities/audio_generation/{routes,protocol,codec,hf}.py`.
- Video runtime + wire: `src/muse/modalities/video_generation/runtimes/wan_runtime.py`
  + `routes.py` (the text-only `VideoGenerationRequest` you will extend).
- A trust_remote_code SDK runtime done right (B1-documented docstring):
  `src/muse/modalities/model_3d_generation/runtimes/hunyuan3d.py`.
- Image-input decode (for Wan2.2 TI2V): `muse.modalities.image_generation.image_input`
  and how chat/completion's `routes.py` pre-decodes `image_url` parts.
- The just-shipped Supertonic build (clean recent example of the full pipeline):
  spec `docs/superpowers/specs/2026-06-18-supertonic-tts-design.md`, plan
  `docs/superpowers/plans/2026-06-18-supertonic-tts.md`, code
  `src/muse/models/supertonic_3.py`.

## First concrete steps on the GPU box

```bash
cd <repo>/muse
git pull origin main
nvidia-smi                      # confirm GPU + VRAM
pip install -e ".[dev,server,audio,images,embeddings]"   # dev env
pytest tests/ -m "not slow" -q  # confirm green baseline (expect ~3160 passed)
```

Then tell Claude: "Build ACE-Step (music) next per the handoff doc" and it should
invoke superpowers:brainstorming for the ACE-Step sub-project. Ask the user the
real open questions (e.g. lyrics field shape, which ACE-Step variant/size,
turbo vs base, bundled-vs-curated) before designing.

## Roadmap recap

- v0.47.0 Supertonic-3 TTS (CPU) - DONE
- v0.48.0 ACE-Step music (GPU) - next
- v0.49.0 Wan2.2 image-to-video (GPU, wire change) - after
- TRELLIS.2 3D - removed (recover from git if revived)
