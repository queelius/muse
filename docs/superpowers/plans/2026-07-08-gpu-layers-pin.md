# GPU-Layers Operator Pin Implementation Plan (v0.56.0)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `muse models set-gpu-layers <id> <N|--clear>` pins a GGUF model's llama.cpp GPU/CPU layer split via a top-level catalog field, honored at load time and by the probe, per `docs/superpowers/specs/2026-07-08-gpu-layers-pin-design.md`.

**Architecture:** Clone the `set-device` pattern end to end: catalog setter (`set_gpu_layers_override`), `load_backend` precedence injection (pin > `capabilities.n_gpu_layers` > runtime default -1), inline typer verb in `cli.py`, render in `models_info_display`. The probe needs NO new code: `probe_worker` constructs via `load_backend`, which reads the pin live.

**Tech Stack:** Python 3.10+, typer, pytest. No new deps.

## Global Constraints

- ASCII only in code/comments/commits; NO em-dash (hook rejects).
- `N >= -1` only: `-1` = all layers on GPU, `0` = pure CPU, `N > 0` = first N layers on GPU.
- The verb REFUSES models whose capabilities lack `gguf_file` (exit 2, message names the constraint).
- Field name is exactly `gpu_layers_override` (top-level catalog field, NOT in the manifest).
- Precedence at load: catalog `gpu_layers_override` > manifest `capabilities.n_gpu_layers` > runtime default. The pin is applied AFTER the kwargs merge (same as `device_override`) so it also beats caller kwargs.
- Takes effect on next cold load; all user-facing messages say so (mirror set-device wording).
- Backward compatible: no field set = behavior unchanged.
- Branch: `feature/gpu-layers-pin` off main. Fast lane must stay green: `MUSE_CATALOG_DIR=$(mktemp -d) python -m pytest tests/ -q -m "not slow"` (baseline ~3862 passed; do NOT pin MUSE_CONFIG -- it spuriously breaks 4 config-cli tests).
- Commit trailers on every commit:
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01X2M12d5tRULNFUHFW2Hijx`

## File Structure

| File | Responsibility |
|---|---|
| Modify `src/muse/core/catalog.py` | `set_gpu_layers_override()` next to `set_device_override` (~line 1056); pin injection in `load_backend`'s precedence block (~line 1150) |
| Modify `src/muse/cli.py` | `@models_app.command("set-gpu-layers")` next to `set-device` (~line 601) |
| Modify `src/muse/cli_impl/models_info_display.py` | render the pin beside the device override (~line 243) |
| Modify `CLAUDE.md` + `pyproject.toml` | device-placement docs note + version 0.56.0 |
| Test `tests/core/test_catalog.py` (append) | setter round-trip + load precedence |
| Create `tests/cli_impl/test_set_gpu_layers_cli.py` | CLI verb (mirrors `test_set_device_cli.py`) |

---

### Task 1: catalog setter + load_backend precedence

**Files:**
- Modify: `src/muse/core/catalog.py` (after `set_device_override`, ~line 1092; and inside `load_backend`'s precedence block, ~line 1150)
- Test: `tests/core/test_catalog.py` (append)

**Interfaces:**
- Produces: `muse.core.catalog.set_gpu_layers_override(model_id: str, n: int | None) -> None` (None pops the field; raises KeyError for unknown model, ValueError for n < -1 or non-int); `load_backend` passes `n_gpu_layers=<pin>` into the backend constructor when the catalog field is present.

- [ ] **Step 1: Write the failing tests** (append to `tests/core/test_catalog.py`; reuse that file's existing `tmp_catalog`-style fixture -- grep the file for how existing set_device tests seed a pulled model, and follow the same shape):

```python
class TestGpuLayersOverride:
    """Spec 2026-07-08: operator pin for llama.cpp n_gpu_layers."""

    def _seed(self, tmp_path, monkeypatch, capabilities=None):
        """Seed a resolver-pulled-style catalog entry with a persisted
        manifest so known_models() picks up capabilities."""
        import json
        monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
        from muse.core.catalog import _reset_known_models_cache
        entry = {
            "pulled_at": "...", "hf_repo": "org/repo", "local_dir": "/w",
            "venv_path": "/v", "python_path": "/v/bin/python",
            "enabled": True, "source": "hf://org/repo",
            "manifest": {
                "model_id": "test-gguf", "modality": "chat/completion",
                "hf_repo": "org/repo",
                "backend_path": "muse.modalities.chat_completion.runtimes.llama_cpp:LlamaCppModel",
                "capabilities": capabilities or {"gguf_file": "m.gguf"},
            },
        }
        (tmp_path / "catalog.json").write_text(json.dumps({"test-gguf": entry}))
        _reset_known_models_cache()

    def test_set_and_clear_round_trip(self, tmp_path, monkeypatch):
        from muse.core.catalog import _read_catalog, set_gpu_layers_override
        self._seed(tmp_path, monkeypatch)
        set_gpu_layers_override("test-gguf", 30)
        assert _read_catalog()["test-gguf"]["gpu_layers_override"] == 30
        set_gpu_layers_override("test-gguf", None)
        assert "gpu_layers_override" not in _read_catalog()["test-gguf"]

    def test_minus_one_and_zero_are_valid(self, tmp_path, monkeypatch):
        from muse.core.catalog import _read_catalog, set_gpu_layers_override
        self._seed(tmp_path, monkeypatch)
        set_gpu_layers_override("test-gguf", -1)
        assert _read_catalog()["test-gguf"]["gpu_layers_override"] == -1
        set_gpu_layers_override("test-gguf", 0)
        assert _read_catalog()["test-gguf"]["gpu_layers_override"] == 0

    def test_invalid_values_raise(self, tmp_path, monkeypatch):
        from muse.core.catalog import set_gpu_layers_override
        self._seed(tmp_path, monkeypatch)
        with pytest.raises(ValueError):
            set_gpu_layers_override("test-gguf", -2)
        with pytest.raises(ValueError):
            set_gpu_layers_override("test-gguf", "thirty")

    def test_unknown_model_raises_keyerror(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
        from muse.core.catalog import set_gpu_layers_override
        with pytest.raises(KeyError):
            set_gpu_layers_override("never-pulled", 10)

    def test_load_backend_pin_beats_capability(self, tmp_path, monkeypatch):
        """Precedence: catalog pin > capabilities.n_gpu_layers > default."""
        from unittest.mock import MagicMock, patch as mpatch
        from muse.core.catalog import load_backend, set_gpu_layers_override
        self._seed(tmp_path, monkeypatch,
                   capabilities={"gguf_file": "m.gguf", "n_gpu_layers": 10})
        set_gpu_layers_override("test-gguf", 30)
        fake_cls = MagicMock()
        fake_module = MagicMock()
        fake_module.LlamaCppModel = fake_cls
        with mpatch("muse.core.catalog._import_backend_module",
                    return_value=fake_module), \
             mpatch("muse.core.catalog.is_pulled", return_value=True):
            load_backend("test-gguf")
        assert fake_cls.call_args.kwargs["n_gpu_layers"] == 30

    def test_load_backend_capability_used_without_pin(self, tmp_path, monkeypatch):
        from unittest.mock import MagicMock, patch as mpatch
        from muse.core.catalog import load_backend
        self._seed(tmp_path, monkeypatch,
                   capabilities={"gguf_file": "m.gguf", "n_gpu_layers": 10})
        fake_cls = MagicMock()
        fake_module = MagicMock()
        fake_module.LlamaCppModel = fake_cls
        with mpatch("muse.core.catalog._import_backend_module",
                    return_value=fake_module), \
             mpatch("muse.core.catalog.is_pulled", return_value=True):
            load_backend("test-gguf")
        assert fake_cls.call_args.kwargs["n_gpu_layers"] == 10

    def test_load_backend_absent_everywhere_passes_nothing(self, tmp_path, monkeypatch):
        """No pin + no capability: n_gpu_layers not in kwargs; the runtime
        default (-1) governs."""
        from unittest.mock import MagicMock, patch as mpatch
        from muse.core.catalog import load_backend
        self._seed(tmp_path, monkeypatch)  # gguf_file only
        fake_cls = MagicMock()
        fake_module = MagicMock()
        fake_module.LlamaCppModel = fake_cls
        with mpatch("muse.core.catalog._import_backend_module",
                    return_value=fake_module), \
             mpatch("muse.core.catalog.is_pulled", return_value=True):
            load_backend("test-gguf")
        assert "n_gpu_layers" not in fake_cls.call_args.kwargs
```

IMPLEMENTER NOTE: verify the seeded-entry shape against how `known_models()`
merges persisted manifests (grep `test_supervisor_lazy.py` for `"manifest":`
seeds); if `backend_path` inside the manifest is not the key `known_models()`
reads, adapt the seed (NOT the product code) until `load_backend` resolves --
the assertions above are the contract.

- [ ] **Step 2: Run to verify RED**

Run: `python -m pytest tests/core/test_catalog.py::TestGpuLayersOverride -q`
Expected: FAIL with `ImportError: cannot import name 'set_gpu_layers_override'`

- [ ] **Step 3: Implement**

In `src/muse/core/catalog.py`, directly after `set_device_override`:

```python
def set_gpu_layers_override(model_id: str, n: int | None) -> None:
    """Set or clear the per-model llama.cpp GPU-layer pin for a pulled model.

    `n` is the llama.cpp `n_gpu_layers` value: -1 = offload every layer the
    GPU fits, 0 = pure CPU, N > 0 = first N layers on GPU (rest on CPU).
    Stored as the TOP-LEVEL catalog field `gpu_layers_override` (operator
    state, mirroring `device_override` -- NOT part of the manifest). Passing
    ``None`` removes the pin (revert to capabilities.n_gpu_layers / the
    runtime default).

    Catalog state only: takes effect on the model's next cold load. To
    apply it to an already-resident worker, evict or restart that worker.

    Raises ValueError for a non-int or n < -1 and KeyError when the model
    is not pulled. Holds _CATALOG_WRITE_LOCK for the full
    read->mutate->write (mirrors `set_device_override`).
    """
    if n is not None:
        if isinstance(n, bool) or not isinstance(n, int) or n < -1:
            raise ValueError(
                f"invalid gpu layers {n!r}; expected an int >= -1 "
                "(-1 = all layers on GPU, 0 = pure CPU) or None to clear"
            )
    with _CATALOG_WRITE_LOCK:
        catalog = _read_catalog()
        if model_id not in catalog:
            raise KeyError(f"model {model_id!r} is not pulled")
        if n is None:
            catalog[model_id].pop("gpu_layers_override", None)
        else:
            catalog[model_id]["gpu_layers_override"] = n
        _write_catalog(catalog)
    _reset_known_models_cache()
```

In `load_backend`, extend the existing precedence block (right after the
`device_override` lines):

```python
    # GPU-layers precedence (spec 2026-07-08), most authoritative first:
    #   1. catalog `gpu_layers_override` (operator, via
    #      `muse models set-gpu-layers`)
    #   2. manifest `capabilities.n_gpu_layers` (already in `merged` via the
    #      capabilities splat above)
    #   3. runtime default (-1 in LlamaCppModel: everything the GPU fits)
    # Applied AFTER the kwargs merge, like device_override, so the operator
    # pin also beats caller kwargs: it is a placement preference.
    gpu_layers = entry_data.get("gpu_layers_override")
    if gpu_layers is not None:
        merged["n_gpu_layers"] = gpu_layers
```

- [ ] **Step 4: Run to verify GREEN**

Run: `python -m pytest tests/core/test_catalog.py -q`
Expected: PASS (new class green, existing catalog tests untouched)

- [ ] **Step 5: Commit**

```bash
git add src/muse/core/catalog.py tests/core/test_catalog.py
git commit -m "feat(catalog): gpu_layers_override setter + load_backend precedence"
```

---

### Task 2: CLI verb + models info rendering

**Files:**
- Modify: `src/muse/cli.py` (new command directly after `models_set_device`, ~line 660)
- Modify: `src/muse/cli_impl/models_info_display.py:243-249` (render beside the device override)
- Create: `tests/cli_impl/test_set_gpu_layers_cli.py`

**Interfaces:**
- Consumes: `set_gpu_layers_override` from Task 1.
- Produces: `muse models set-gpu-layers <id> <N>` / `--clear`; `models info` line `gpu layers pin:  <N> (operator pin via \`muse models set-gpu-layers\`)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/cli_impl/test_set_gpu_layers_cli.py
"""Tests for the `muse models set-gpu-layers` CLI verb (spec 2026-07-08).

Mirrors test_set_device_cli.py: the verb writes a per-model
`gpu_layers_override` into the catalog; load_backend honors it on the
next cold load. GGUF-only: models without capabilities.gguf_file are
refused (honest error beats a silently ignored pin).
"""
from __future__ import annotations

import json

import pytest
import typer


@pytest.fixture
def tmp_catalog(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
    from muse.core.catalog import _reset_known_models_cache
    _reset_known_models_cache()
    yield tmp_path
    _reset_known_models_cache()


def _seed_gguf(tmp_path, model_id="test-gguf", capabilities=None):
    from muse.core.catalog import _reset_known_models_cache
    entry = {
        "pulled_at": "...", "hf_repo": "org/repo", "local_dir": "/w",
        "venv_path": "/v", "python_path": "/v/bin/python",
        "enabled": True, "source": "hf://org/repo",
        "manifest": {
            "model_id": model_id, "modality": "chat/completion",
            "hf_repo": "org/repo",
            "backend_path": "muse.modalities.chat_completion.runtimes.llama_cpp:LlamaCppModel",
            "capabilities": capabilities if capabilities is not None
                            else {"gguf_file": "m.gguf"},
        },
    }
    (tmp_path / "catalog.json").write_text(json.dumps({model_id: entry}))
    _reset_known_models_cache()


def test_set_writes_override(tmp_catalog):
    from muse.cli import models_set_gpu_layers
    from muse.core.catalog import _read_catalog
    _seed_gguf(tmp_catalog)
    models_set_gpu_layers("test-gguf", 30, clear=False)
    assert _read_catalog()["test-gguf"]["gpu_layers_override"] == 30


def test_clear_removes_override(tmp_catalog):
    from muse.cli import models_set_gpu_layers
    from muse.core.catalog import _read_catalog
    _seed_gguf(tmp_catalog)
    models_set_gpu_layers("test-gguf", 30, clear=False)
    models_set_gpu_layers("test-gguf", None, clear=True)
    assert "gpu_layers_override" not in _read_catalog()["test-gguf"]


def test_minus_one_and_zero_accepted(tmp_catalog):
    from muse.cli import models_set_gpu_layers
    from muse.core.catalog import _read_catalog
    _seed_gguf(tmp_catalog)
    models_set_gpu_layers("test-gguf", -1, clear=False)
    assert _read_catalog()["test-gguf"]["gpu_layers_override"] == -1
    models_set_gpu_layers("test-gguf", 0, clear=False)
    assert _read_catalog()["test-gguf"]["gpu_layers_override"] == 0


def test_below_minus_one_exits_2(tmp_catalog):
    from muse.cli import models_set_gpu_layers
    _seed_gguf(tmp_catalog)
    with pytest.raises(typer.Exit) as exc:
        models_set_gpu_layers("test-gguf", -2, clear=False)
    assert exc.value.exit_code == 2


def test_no_n_and_no_clear_exits_2(tmp_catalog):
    from muse.cli import models_set_gpu_layers
    _seed_gguf(tmp_catalog)
    with pytest.raises(typer.Exit) as exc:
        models_set_gpu_layers("test-gguf", None, clear=False)
    assert exc.value.exit_code == 2


def test_unknown_model_exits_2(tmp_catalog):
    from muse.cli import models_set_gpu_layers
    with pytest.raises(typer.Exit) as exc:
        models_set_gpu_layers("never-pulled", 10, clear=False)
    assert exc.value.exit_code == 2


def test_non_gguf_model_refused(tmp_catalog):
    """A model without capabilities.gguf_file is refused: the pin would be
    silently ignored by non-llama.cpp runtimes."""
    from muse.cli import models_set_gpu_layers
    from muse.core.catalog import _read_catalog
    _seed_gguf(tmp_catalog, capabilities={})  # no gguf_file
    with pytest.raises(typer.Exit) as exc:
        models_set_gpu_layers("test-gguf", 30, clear=False)
    assert exc.value.exit_code == 2
    assert "gpu_layers_override" not in _read_catalog()["test-gguf"]


def test_clear_on_non_gguf_still_allowed(tmp_catalog):
    """--clear must work even on a non-GGUF entry (e.g. an operator
    removing a stale pin after a manifest change)."""
    from muse.cli import models_set_gpu_layers
    _seed_gguf(tmp_catalog, capabilities={})
    models_set_gpu_layers("test-gguf", None, clear=True)  # must not raise


def test_info_renders_pin(tmp_catalog):
    from muse.cli import models_set_gpu_layers
    _seed_gguf(tmp_catalog)
    models_set_gpu_layers("test-gguf", 30, clear=False)
    from muse.cli_impl.models_info_display import build_info_lines
    from muse.core.catalog import _read_catalog, known_models
    entry = known_models()["test-gguf"]
    text = "\n".join(build_info_lines(
        "test-gguf", entry, _read_catalog().get("test-gguf", {}),
    ))
    assert "gpu layers" in text and "30" in text
```

IMPLEMENTER NOTE on the last test: `models_info_display`'s public builder
may not be named `build_info_lines` or take those exact args -- read the
file and its existing tests (grep tests/ for models_info_display) and adapt
the TEST to the real entry point; the contract is "the rendered info text
contains the pin".

- [ ] **Step 2: Run to verify RED**

Run: `python -m pytest tests/cli_impl/test_set_gpu_layers_cli.py -q`
Expected: FAIL with `ImportError: cannot import name 'models_set_gpu_layers'`

- [ ] **Step 3: Implement**

In `src/muse/cli.py`, directly after the `models_set_device` command:

```python
@models_app.command("set-gpu-layers")
def models_set_gpu_layers(
    model_id: Annotated[str, typer.Argument()],
    n: Annotated[
        Optional[int],
        typer.Argument(
            help="llama.cpp n_gpu_layers: -1 = all layers on GPU, 0 = pure "
                 "CPU, N > 0 = first N layers on GPU; omit with --clear",
        ),
    ] = None,
    clear: Annotated[
        bool,
        typer.Option("--clear", help="remove the pin (revert to manifest / runtime default)"),
    ] = False,
) -> None:
    """Pin a GGUF model's llama.cpp GPU/CPU layer split (operator override).

    Writes a per-model `gpu_layers_override` to the catalog. Precedence at
    load time: pin > manifest capabilities.n_gpu_layers > runtime default
    (-1, everything the GPU fits). GGUF-only: refuses models without
    capabilities.gguf_file, since other runtimes silently ignore the kwarg.

    Catalog-only state: takes effect on the model's NEXT cold load. To
    apply it to an already-resident worker, evict it or restart the
    supervisor. Run `muse models probe <id>` after pinning so admission
    sizing measures the split's real (smaller) VRAM peak.
    """
    from muse.core.catalog import known_models, set_gpu_layers_override

    if clear:
        target = None
    elif n is None:
        typer.echo(
            "error: provide a layer count (int >= -1) or pass --clear",
            err=True,
        )
        raise typer.Exit(2)
    else:
        target = n

    if target is not None:
        entry = known_models().get(model_id)
        capabilities = (entry.extra or {}) if entry is not None else {}
        if not capabilities.get("gguf_file"):
            typer.echo(
                f"error: {model_id!r} is not a GGUF model (no "
                "capabilities.gguf_file); n_gpu_layers only applies to "
                "llama.cpp runtimes and would be silently ignored",
                err=True,
            )
            raise typer.Exit(2)

    try:
        set_gpu_layers_override(model_id, target)
    except KeyError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)
    except ValueError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)

    if target is None:
        typer.echo(
            f"cleared gpu-layers pin for {model_id} "
            "(takes effect on next cold load)"
        )
    else:
        typer.echo(
            f"set {model_id} gpu layers -> {target} "
            "(takes effect on next cold load; run `muse models probe "
            f"{model_id}` to re-measure VRAM)"
        )
```

In `src/muse/cli_impl/models_info_display.py`, directly after the
`device override` render block (~line 249):

```python
    layers_pin = catalog_data.get("gpu_layers_override") if is_pulled else None
    if layers_pin is not None:
        lines.append(
            f"  gpu layers pin:  {layers_pin} "
            "(operator pin via `muse models set-gpu-layers`)"
        )
```

(NOTE: `if layers_pin is not None`, not truthiness -- 0 is a valid pin.)

- [ ] **Step 4: Run to verify GREEN**

Run: `python -m pytest tests/cli_impl/test_set_gpu_layers_cli.py tests/cli_impl/test_set_device_cli.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/muse/cli.py src/muse/cli_impl/models_info_display.py tests/cli_impl/test_set_gpu_layers_cli.py
git commit -m "feat(cli): muse models set-gpu-layers verb + info rendering"
```

---

### Task 3: probe flow verification + docs + version bump

**Files:**
- Test: `tests/core/test_catalog.py` (append one test to `TestGpuLayersOverride`)
- Modify: `CLAUDE.md` (device-placement section) + `pyproject.toml` (0.55.0 -> 0.56.0)

**Interfaces:** consumes Tasks 1-2; produces the release candidate.

- [ ] **Step 1: Write the probe-flow verification test** (append to the Task 1 class):

```python
    def test_probe_flow_gets_pin_via_load_backend(self, tmp_path, monkeypatch):
        """probe_worker constructs via load_backend(model_id, device=...),
        so the pin flows into the probed construction with zero probe code.
        This test binds that seam: a caller-passed device kwarg must NOT
        displace the injected n_gpu_layers."""
        from unittest.mock import MagicMock, patch as mpatch
        from muse.core.catalog import load_backend, set_gpu_layers_override
        self._seed(tmp_path, monkeypatch)
        set_gpu_layers_override("test-gguf", 25)
        fake_cls = MagicMock()
        fake_module = MagicMock()
        fake_module.LlamaCppModel = fake_cls
        with mpatch("muse.core.catalog._import_backend_module",
                    return_value=fake_module), \
             mpatch("muse.core.catalog.is_pulled", return_value=True):
            load_backend("test-gguf", device="cuda")  # probe-style call
        assert fake_cls.call_args.kwargs["n_gpu_layers"] == 25
        assert fake_cls.call_args.kwargs["device"] == "cuda"
```

- [ ] **Step 2: Run RED-or-GREEN honestly**

Run: `python -m pytest tests/core/test_catalog.py::TestGpuLayersOverride -q`
Expected: PASS already (the seam exists by construction after Task 1). This
test is a REGRESSION GUARD, not TDD RED -- state that in the commit body.

- [ ] **Step 3: Docs**

CLAUDE.md, in the "### Device placement precedence (v0.48.0+)" section, append
after the numbered ladder:

```markdown
**GPU-layers pin (v0.56.0, GGUF only).** `muse models set-gpu-layers <id>
<N|--clear>` writes a top-level catalog `gpu_layers_override` (the
`device_override` pattern): llama.cpp `n_gpu_layers`, -1 = all layers on
GPU, 0 = pure CPU, N > 0 = a static GPU/CPU layer split for GGUFs bigger
than VRAM. Precedence: pin > `capabilities.n_gpu_layers` > runtime default
(-1). Refuses non-GGUF models (other runtimes silently ignore the kwarg).
Takes effect on next cold load; run `muse models probe <id>` after pinning
so admission sizing measures the split's real (smaller) VRAM peak. The
probe honors the pin automatically (it constructs via `load_backend`).
Known limitation: a split model occupies both VRAM and host RAM, but the
director accounts it against its resolved device's pool only -- the
Tier-1 static-offload simplification (automatic offload was evaluated and
rejected; see docs/superpowers/specs/2026-07-08-gpu-layers-pin-design.md).
```

`pyproject.toml`: `sed -i 's/^version = "0.55.0"/version = "0.56.0"/' pyproject.toml`

- [ ] **Step 4: Full verification**

Run: `MUSE_CATALOG_DIR=$(mktemp -d) python -m pytest tests/ -q -m "not slow"`
Expected: >= 3862 passed + the new tests, zero new failures.

- [ ] **Step 5: Commit (NO tag/push -- release is user-gated)**

```bash
git add -A
git commit -m "feat(models): gpu-layers pin docs + probe seam guard + v0.56.0 bump"
```

Post-merge deploy notes (session driver, not implementer): release ritual on
user go (FF main, tag v0.56.0, gh release, build+twine, deploy frodo with
supervisor restart). Live validation on frodo: `muse pull` the 32B GGUF
there (weights ~20 GB; check disk first), `muse models set-gpu-layers <id>
30`, `muse models probe <id>`, one chat request; compare tok/s vs the CPU
box's ~1.3. Tune N so probed VRAM peak sits ~1-2 GB under the card.
```
