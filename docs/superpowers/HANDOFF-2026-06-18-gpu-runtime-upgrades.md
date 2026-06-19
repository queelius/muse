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
4. **ACE-Step is already designed** - start here: execute
   `docs/superpowers/plans/2026-06-18-acestep-music.md` via
   **superpowers:subagent-driven-development**. Its **Task 1 is the B1 step**
   (install the real SDK on the GPU and confirm the API) - do it first and
   correct the marked spots in the later tasks from its findings. Wan2.2 still
   needs a fresh brainstorm (no spec/plan yet).

## Where things stand

muse is a self-hostable OpenAI-compatible multimodal server, ~20 modalities,
published on PyPI as `museq`. Latest release: **v0.47.0** (Supertonic-3 CPU TTS).

This GPU session is the back half of a 4-item "runtime upgrade" roadmap that
came out of a HuggingFace model survey. Status:

- DONE (CPU, v0.47.0): **Supertonic-3** on-device TTS. Shipped.
- REMOVED: **TRELLIS.2** (3D). Cut by the user (its `trellis2`+`o_voxel`
  standalone SDK with CUDA build deps was the gnarliest install of the four).
  Its spec/plan were deleted; recoverable from git history if ever wanted.
- DESIGNED, ready to build (this session): **ACE-Step** (music). Spec +
  implementation plan committed on the CPU host:
  `docs/superpowers/specs/2026-06-18-acestep-music-design.md` and
  `docs/superpowers/plans/2026-06-18-acestep-music.md`. Execute the plan; Task 1
  is B1 on the GPU.
- TODO, still needs a fresh brainstorm: **Wan2.2** (image-to-video). Scoped only;
  no spec/plan yet (it also needs a video-modality wire change).

## The two remaining sub-projects

### 1. ACE-Step (music generation) - DESIGNED, build first

- **Spec:** `docs/superpowers/specs/2026-06-18-acestep-music-design.md`
- **Plan (5 tasks, TDD):** `docs/superpowers/plans/2026-06-18-acestep-music.md`
- Model: `ACE-Step/Ace-Step1.5`. Full song generation (vocals + lyrics + style).
  GPU, MIT.
- **Scope correction discovered during design** (do not trust the older
  "mirror stable_audio / medium effort" framing): ACE-Step 1.5 is a TWO-handler
  system (`AceStepHandler` DIT + `LLMHandler` planner, default `backend="vllm"`)
  driven by a functional `generate_music(...)`, installed from
  github.com/ace-step/ACE-Step-1.5 (the `ace-step` PyPI v0.1.0 is the OLD v1).
  Closer to the standalone-SDK 3D runtimes than to Stable Audio. The API was
  researched from the repo's `docs/en/INFERENCE.md` and is captured in the
  plan's code (best-evidence; Task 1 B1 confirms init args, vllm necessity,
  tensor orientation, sample rate, VRAM).
- Fits the EXISTING `audio/generation` modality; new `ACEStepRuntime`; the wire
  adds one optional `lyrics` field (lyrics-only per the design); curated
  GPU-only `ace-step-1.5` entry. Ships v0.48.0.
- **Execute:** superpowers:subagent-driven-development on the plan; Task 1 (B1)
  runs the real install + generate on the GPU before the runtime body is fixed.

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

Then tell Claude: "Execute the ACE-Step plan
(docs/superpowers/plans/2026-06-18-acestep-music.md) via
subagent-driven-development" - it is already brainstormed + spec'd + planned
(design approved on the CPU host). Task 1 of that plan is the GPU B1 step; run it
first and correct the B1-marked spots in the runtime/curated tasks from its
findings. After ACE-Step ships, brainstorm Wan2.2 fresh.

## Roadmap recap

- v0.47.0 Supertonic-3 TTS (CPU) - DONE
- v0.48.0 ACE-Step music (GPU) - spec + plan READY; execute on GPU
- v0.49.0 Wan2.2 image-to-video (GPU, wire change) - needs brainstorm
- TRELLIS.2 3D - removed (recover from git if revived)
