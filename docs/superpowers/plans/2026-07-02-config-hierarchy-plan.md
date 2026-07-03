# Config Hierarchy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace ~35 scattered `os.environ.get("MUSE_*", default)` reads with one declarative settings registry (`muse.core.config`), a persistent `~/.muse/config.yaml`, a standard precedence chain (override > env > file > default), and a `muse config` CLI group.

**Architecture:** A single `SETTINGS` list of `Setting` rows is the sole source of truth. A `Config` object resolves each key by precedence, reading env LIVE per call and the yaml file once (cached). Every former env-read site calls `config.get("group.key")`. A meta-test forbids stray `os.environ.get("MUSE_` outside `config.py`.

**Tech Stack:** Python 3.10+, pyyaml (already a core dep), typer + rich (CLI), pytest.

**Reference:** `docs/superpowers/specs/2026-07-02-config-hierarchy-design.md`. Its **Authoritative settings inventory** table is the migration work-list; the plan references it rather than re-listing all 35 rows (DRY).

## Global Constraints

- **ASCII only** in all file content and commit messages. NO em-dash characters anywhere (a hook rejects them). Use `--`, colon, comma, or parentheses.
- **TDD (red-green)** on every task: write the failing test, watch it fail, minimal code to pass, watch it pass, commit.
- **Commit locally after each task.** Do NOT push, tag, or bump the package version -- release is gated on an explicit later "go".
- **Commit trailers** (both, exactly):
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_01J2SDRmdTMP3sVoBpAqZ3VV`
- **Never echo the admin token** (`MUSE_ADMIN_TOKEN` / `admin.token`) in logs, errors, or `config show` output -- redact to a fixed placeholder.
- **Fast lane must stay green:** `pytest tests/ -q -m "not slow"` after every task.
- **Deferred imports discipline:** `muse.core.config` imports only stdlib + `yaml`. No torch/transformers/fastapi. `muse --help` and `muse pull` must work with zero ML deps.
- **`muse` is the import name / CLI / identity; `museq` is only the PyPI label.** Do not rename anything to `museq` in code.

---

### Task 1: Registry core -- `Setting`, `SETTINGS`, coercion, path bootstrap

**Files:**
- Create: `src/muse/core/config.py`
- Test: `tests/core/test_config_registry.py`

**Interfaces:**
- Produces:
  - `class Setting` (frozen dataclass): `key: str`, `env: str`, `type: str`, `default`, `group: str`, `help: str`.
  - `SETTINGS: list[Setting]` -- every row from the spec's inventory table.
  - `SETTINGS_BY_KEY: dict[str, Setting]` and `SETTINGS_BY_ENV: dict[str, Setting]`.
  - `coerce(setting: Setting, raw: str) -> Any` -- parse a string per `setting.type`; raises `ConfigError` on failure (callers decide lenient vs strict).
  - `class ConfigError(ValueError)`.
  - `_catalog_dir() -> pathlib.Path` -- `MUSE_CATALOG_DIR` env expanded, else `~/.muse`. Standalone; does NOT import `catalog.py`.
  - `config_path() -> pathlib.Path` -- `MUSE_CONFIG` env (expanded full path) else `_catalog_dir() / "config.yaml"`.

- [ ] **Step 1: Write failing tests**

```python
# tests/core/test_config_registry.py
import os
import pytest
from muse.core import config as cfg


def test_registry_keys_unique_and_dotted():
    keys = [s.key for s in cfg.SETTINGS]
    assert len(keys) == len(set(keys)), "duplicate keys"
    assert all("." in k for k in keys), "keys must be group.leaf"


def test_registry_envs_unique_and_prefixed():
    envs = [s.env for s in cfg.SETTINGS]
    assert len(envs) == len(set(envs)), "duplicate env vars"
    assert all(e.startswith("MUSE_") for e in envs)


def test_key_group_matches_prefix():
    for s in cfg.SETTINGS:
        assert s.key.split(".")[0] == s.group


def test_lookup_maps_cover_all():
    assert set(cfg.SETTINGS_BY_KEY) == {s.key for s in cfg.SETTINGS}
    assert set(cfg.SETTINGS_BY_ENV) == {s.env for s in cfg.SETTINGS}


def test_expected_settings_present():
    # spot-check representative rows across groups
    for key in (
        "server.idle_timeout_seconds",
        "server.gpu_headroom_gb",
        "admin.token",
        "client.server_url",
        "paths.catalog_dir",
        "fetch.allow_private",
        "limits.upscale_max_input_side",
        "limits.rerank_max_documents",
    ):
        assert key in cfg.SETTINGS_BY_KEY, key


def test_idle_timeout_default_is_600():
    assert cfg.SETTINGS_BY_KEY["server.idle_timeout_seconds"].default == 600.0


@pytest.mark.parametrize("t,raw,expected", [
    ("int", "42", 42),
    ("float", "1.5", 1.5),
    ("str", "hi", "hi"),
    ("bool", "1", True),
    ("bool", "true", True),
    ("bool", "0", False),
    ("bool", "false", False),
    ("opt_int", "", None),
    ("opt_int", "7", 7),
    ("opt_float", "", None),
    ("opt_str", "", None),
    ("opt_str", "x", "x"),
])
def test_coerce_types(t, raw, expected):
    s = cfg.Setting(key="g.k", env="MUSE_K", type=t, default=None, group="g", help="h")
    assert cfg.coerce(s, raw) == expected


def test_coerce_bad_int_raises():
    s = cfg.Setting(key="g.k", env="MUSE_K", type="int", default=0, group="g", help="h")
    with pytest.raises(cfg.ConfigError):
        cfg.coerce(s, "abc")


def test_catalog_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
    assert cfg._catalog_dir() == tmp_path


def test_config_path_defaults_under_catalog_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
    monkeypatch.delenv("MUSE_CONFIG", raising=False)
    assert cfg.config_path() == tmp_path / "config.yaml"


def test_config_path_explicit_override(monkeypatch, tmp_path):
    p = tmp_path / "custom.yaml"
    monkeypatch.setenv("MUSE_CONFIG", str(p))
    assert cfg.config_path() == p
```

- [ ] **Step 2: Run tests, verify they fail** (`ModuleNotFoundError` / attribute errors).
  Run: `pytest tests/core/test_config_registry.py -q`

- [ ] **Step 3: Implement `src/muse/core/config.py` (registry portion)**

Header docstring, then:

```python
from __future__ import annotations

import logging
import os
import pathlib
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("muse.config")

_MB = 1024 * 1024


class ConfigError(ValueError):
    """A config value could not be coerced to its declared type."""


@dataclass(frozen=True)
class Setting:
    key: str          # dotted "group.leaf"
    env: str          # "MUSE_*"
    type: str         # int|float|str|bool|opt_int|opt_float|opt_str
    default: Any
    group: str
    help: str


_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off", ""}


def coerce(setting: Setting, raw: str) -> Any:
    """Parse a raw string per setting.type. Raises ConfigError on failure.
    Callers choose lenient (Config.get) vs strict (config set)."""
    t = setting.type
    if t.startswith("opt_"):
        if raw is None or raw.strip() == "":
            return None
        t = t[len("opt_"):]
    try:
        if t == "int":
            return int(raw)
        if t == "float":
            return float(raw)
        if t == "bool":
            low = raw.strip().lower()
            if low in _TRUE:
                return True
            if low in _FALSE:
                return False
            raise ValueError(f"not a boolean: {raw!r}")
        if t == "str":
            return raw
    except (TypeError, ValueError) as e:
        raise ConfigError(
            f"{setting.env} / {setting.key} must be {setting.type}, got {raw!r}: {e}"
        ) from e
    raise ConfigError(f"unknown type {setting.type!r} for {setting.key}")


SETTINGS: list[Setting] = [
    # --- server ---
    Setting("server.idle_sweep_interval_seconds", "MUSE_IDLE_SWEEP_INTERVAL_SECONDS",
            "float", 30.0, "server", "Seconds between idle-eviction sweeps."),
    Setting("server.idle_timeout_seconds", "MUSE_DEFAULT_IDLE_TIMEOUT_SECONDS",
            "opt_float", 600.0, "server",
            "Global default idle timeout (s) before an untouched model is evicted; 0/negative disables."),
    Setting("server.shutdown_grace_seconds", "MUSE_SHUTDOWN_GRACE_SECONDS",
            "opt_float", None, "server",
            "Grace period (s) for workers to exit on shutdown; None uses the built-in default."),
    Setting("server.gpu_budget_gb", "MUSE_GPU_BUDGET_GB",
            "opt_float", None, "server",
            "Declared GPU memory cap (GB); muse uses min(declared, live)."),
    Setting("server.cpu_budget_gb", "MUSE_CPU_BUDGET_GB",
            "opt_float", None, "server", "Declared host-RAM cap (GB)."),
    Setting("server.gpu_headroom_gb", "MUSE_GPU_HEADROOM_GB",
            "float", 1.0, "server",
            "GB subtracted from live free VRAM before deciding fit."),
    Setting("server.cpu_headroom_gb", "MUSE_CPU_HEADROOM_GB",
            "float", 2.0, "server",
            "GB subtracted from live free RAM before deciding fit."),
    Setting("server.device", "MUSE_DEVICE",
            "str", "auto", "server",
            "Default device for models (auto|cpu|cuda|mps); `muse serve --device` overrides."),
    # --- admin ---
    Setting("admin.token", "MUSE_ADMIN_TOKEN",
            "opt_str", None, "admin",
            "Bearer token that unlocks /v1/admin/*; unset keeps admin closed."),
    # --- client ---
    Setting("client.server_url", "MUSE_SERVER",
            "str", "http://localhost:8000", "client",
            "Base URL muse clients + CLI target."),
    # --- paths (bootstrap: catalog_dir/config_file resolve env+default only) ---
    Setting("paths.catalog_dir", "MUSE_CATALOG_DIR",
            "str", "~/.muse", "paths", "Directory for catalog.json, venvs, config.yaml."),
    Setting("paths.home", "MUSE_HOME",
            "str", "~/.muse", "paths", "Base dir for bundled voices/assets."),
    Setting("paths.models_dir", "MUSE_MODELS_DIR",
            "opt_str", None, "paths", "Extra directory scanned for model scripts."),
    Setting("paths.modalities_dir", "MUSE_MODALITIES_DIR",
            "opt_str", None, "paths", "Extra directory scanned for modality packages."),
    Setting("paths.config_file", "MUSE_CONFIG",
            "opt_str", None, "paths",
            "Explicit config.yaml path; overrides <catalog_dir>/config.yaml."),
    # --- fetch ---
    Setting("fetch.allow_private", "MUSE_ALLOW_PRIVATE_FETCH",
            "bool", False, "fetch",
            "Allow image/URL fetches to non-public IPs (SSRF guard off)."),
    Setting("fetch.mcp_allowed_path_prefixes", "MUSE_MCP_ALLOWED_PATH_PREFIXES",
            "str", "", "fetch",
            "Colon-separated dir prefixes MCP *_path inputs may read from."),
    # --- limits (per-modality request caps) ---
    Setting("limits.image_input_max_bytes", "MUSE_IMAGE_INPUT_MAX_BYTES",
            "opt_int", 10 * _MB, "limits", "Max bytes per image upload / data URL."),
    Setting("limits.audio_cls_max_bytes", "MUSE_AUDIO_CLS_MAX_BYTES",
            "opt_int", 50 * _MB, "limits", "Max bytes per audio-classification upload."),
    Setting("limits.audio_embeddings_max_bytes", "MUSE_AUDIO_EMBEDDINGS_MAX_BYTES",
            "opt_int", 50 * _MB, "limits", "Max bytes per audio-embedding upload."),
    Setting("limits.asr_max_mb", "MUSE_ASR_MAX_MB",
            "int", 100, "limits", "Max MB per transcription/translation upload."),
    Setting("limits.embeddings_max_batch", "MUSE_EMBEDDINGS_MAX_BATCH",
            "int", 2048, "limits", "Max inputs per /v1/embeddings request."),
    Setting("limits.embeddings_max_chars_per_item", "MUSE_EMBEDDINGS_MAX_CHARS_PER_ITEM",
            "int", 100000, "limits", "Max chars per embedding input."),
    Setting("limits.image_embeddings_max_batch", "MUSE_IMAGE_EMBEDDINGS_MAX_BATCH",
            "int", 64, "limits", "Max inputs per /v1/images/embeddings request."),
    Setting("limits.segmentation_max_input_side", "MUSE_SEGMENTATION_MAX_INPUT_SIDE",
            "int", 2048, "limits", "Max px on the long side of a segmentation input."),
    Setting("limits.upscale_max_input_side", "MUSE_UPSCALE_MAX_INPUT_SIDE",
            "int", 1024, "limits", "Max px on the long side of an upscale input."),
    Setting("limits.model_3d_input_max_bytes", "MUSE_3D_INPUT_MAX_BYTES",
            "opt_int", 20 * _MB, "limits", "Max bytes per image-to-3D upload."),
    Setting("limits.moderations_max_batch", "MUSE_MODERATIONS_MAX_BATCH",
            "int", 1024, "limits", "Max inputs per /v1/moderations request."),
    Setting("limits.moderations_max_chars_per_item", "MUSE_MODERATIONS_MAX_CHARS_PER_ITEM",
            "int", 100000, "limits", "Max chars per moderation input."),
    Setting("limits.classifications_max_labels", "MUSE_CLASSIFICATIONS_MAX_LABELS",
            "int", 200, "limits", "Max candidate labels per zero-shot classification."),
    Setting("limits.rerank_max_documents", "MUSE_RERANK_MAX_DOCUMENTS",
            "int", 1000, "limits", "Max documents per /v1/rerank request."),
    Setting("limits.rerank_max_query_chars", "MUSE_RERANK_MAX_QUERY_CHARS",
            "int", 4000, "limits", "Max chars in a rerank query."),
    Setting("limits.rerank_max_doc_chars", "MUSE_RERANK_MAX_DOC_CHARS",
            "int", 100000, "limits", "Max chars per rerank document."),
    Setting("limits.summarize_max_text_chars", "MUSE_SUMMARIZE_MAX_TEXT_CHARS",
            "int", 100000, "limits", "Max chars per /v1/summarize request."),
    Setting("limits.video_max_frames_b64", "MUSE_VIDEO_MAX_FRAMES_B64",
            "int", 240, "limits", "Max frames returned as base64 from /v1/video/generations."),
]

SETTINGS_BY_KEY: dict[str, Setting] = {s.key: s for s in SETTINGS}
SETTINGS_BY_ENV: dict[str, Setting] = {s.env: s for s in SETTINGS}


def _catalog_dir() -> pathlib.Path:
    """Resolve the catalog/config directory from env+default only.
    Standalone by design: must NOT import catalog.py (import cycle)."""
    raw = os.environ.get("MUSE_CATALOG_DIR")
    base = raw if raw else "~/.muse"
    return pathlib.Path(base).expanduser()


def config_path() -> pathlib.Path:
    """Resolve the config.yaml path from env+default only (bootstrap)."""
    raw = os.environ.get("MUSE_CONFIG")
    if raw:
        return pathlib.Path(raw).expanduser()
    return _catalog_dir() / "config.yaml"
```

Note: `_catalog_dir` / `config_path` read `os.environ` directly on purpose (bootstrap; they resolve the file's own location). The meta-test in Task 8 exempts `config.py`.

- [ ] **Step 4: Run tests, verify pass.** `pytest tests/core/test_config_registry.py -q`
- [ ] **Step 5: Commit.** `feat(config): settings registry + coercion + path bootstrap`

---

### Task 2: `Config` resolver -- precedence, file cache, lenient get, singleton

**Files:**
- Modify: `src/muse/core/config.py`
- Test: `tests/core/test_config_resolve.py`

**Interfaces:**
- Consumes: `SETTINGS_BY_KEY`, `coerce`, `ConfigError`, `config_path` (Task 1).
- Produces:
  - `class Config`: `__init__(self, *, path: pathlib.Path | None = None, overrides: dict[str, Any] | None = None)`.
    - `get(self, key: str, override: Any | None = None) -> Any`
    - `source(self, key: str) -> str` returns `"override"|"env"|"file"|"default"`.
    - `file_values(self) -> dict` (parsed nested file, `{}` if absent).
  - `get_config() -> Config` -- process singleton (lazy).
  - `reset_config() -> None` -- test hook to clear the singleton.
  - Module helper `get(key, override=None)` and `source(key)` delegating to `get_config()`.

**Resolution (get):** override arg (if not None) > `os.environ[setting.env]` (live) > file value (cached) > `setting.default`. Any raw string from env/file is `coerce`d; on `ConfigError` log ONE warning (`logger.warning`) and fall back to `setting.default`. `default` is returned as-is (already typed). Unknown `key` -> `KeyError` (programmer error, not user input).

**File cache:** parse `path` once in `__init__` (or first `file_values()`), store nested dict. Missing file -> `{}`. Top-level keys not matching a group, or leaves not matching a setting, -> `logger.warning("unknown config key ...")` once each and ignore. File lookup for `group.leaf` is `data.get(group, {}).get(leaf)`.

- [ ] **Step 1: Write failing tests**

```python
# tests/core/test_config_resolve.py
import pytest
from muse.core import config as cfg


@pytest.fixture(autouse=True)
def _reset():
    cfg.reset_config()
    yield
    cfg.reset_config()


def _cfg(tmp_path, text=None):
    p = tmp_path / "config.yaml"
    if text is not None:
        p.write_text(text)
    return cfg.Config(path=p)


def test_default_when_nothing_set(tmp_path):
    c = _cfg(tmp_path)
    assert c.get("limits.rerank_max_documents") == 1000
    assert c.source("limits.rerank_max_documents") == "default"


def test_env_overrides_default(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSE_RERANK_MAX_DOCUMENTS", "5")
    c = _cfg(tmp_path)
    assert c.get("limits.rerank_max_documents") == 5
    assert c.source("limits.rerank_max_documents") == "env"


def test_file_overrides_default(tmp_path):
    c = _cfg(tmp_path, "limits:\n  rerank_max_documents: 7\n")
    assert c.get("limits.rerank_max_documents") == 7
    assert c.source("limits.rerank_max_documents") == "file"


def test_env_beats_file(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSE_RERANK_MAX_DOCUMENTS", "9")
    c = _cfg(tmp_path, "limits:\n  rerank_max_documents: 7\n")
    assert c.get("limits.rerank_max_documents") == 9


def test_override_beats_env(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSE_RERANK_MAX_DOCUMENTS", "9")
    c = _cfg(tmp_path)
    assert c.get("limits.rerank_max_documents", override=3) == 3
    assert c.source("limits.rerank_max_documents") == "env"  # source ignores per-call override


def test_env_live_reread(tmp_path, monkeypatch):
    c = _cfg(tmp_path)
    assert c.get("limits.rerank_max_documents") == 1000
    monkeypatch.setenv("MUSE_RERANK_MAX_DOCUMENTS", "11")
    assert c.get("limits.rerank_max_documents") == 11  # not cached


def test_bad_env_warns_and_defaults(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("MUSE_RERANK_MAX_DOCUMENTS", "abc")
    c = _cfg(tmp_path)
    with caplog.at_level("WARNING"):
        assert c.get("limits.rerank_max_documents") == 1000  # lenient
    assert any("MUSE_RERANK_MAX_DOCUMENTS" in r.message for r in caplog.records)


def test_opt_float_empty_is_none(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSE_SHUTDOWN_GRACE_SECONDS", "")
    c = _cfg(tmp_path)
    assert c.get("server.shutdown_grace_seconds") is None


def test_idle_timeout_default_600(tmp_path, monkeypatch):
    monkeypatch.delenv("MUSE_DEFAULT_IDLE_TIMEOUT_SECONDS", raising=False)
    c = _cfg(tmp_path)
    assert c.get("server.idle_timeout_seconds") == 600.0


def test_unknown_file_key_ignored_with_warning(tmp_path, caplog):
    with caplog.at_level("WARNING"):
        c = _cfg(tmp_path, "limits:\n  bogus_key: 1\nnope:\n  x: 2\n")
        c.file_values()
    msgs = " ".join(r.message for r in caplog.records)
    assert "bogus_key" in msgs and "nope" in msgs


def test_unknown_key_raises_keyerror(tmp_path):
    c = _cfg(tmp_path)
    with pytest.raises(KeyError):
        c.get("no.such.key")


def test_singleton_and_reset(monkeypatch):
    a = cfg.get_config()
    b = cfg.get_config()
    assert a is b
    cfg.reset_config()
    assert cfg.get_config() is not a
```

- [ ] **Step 2: Run tests, verify they fail.**
- [ ] **Step 3: Implement the `Config` class + singleton in `config.py`.**

```python
import yaml  # add to imports

_GROUPS = {s.group for s in SETTINGS}


class Config:
    def __init__(self, *, path: pathlib.Path | None = None,
                 overrides: dict[str, Any] | None = None):
        self._path = path if path is not None else config_path()
        self._overrides = dict(overrides or {})
        self._file: dict | None = None  # lazy

    def file_values(self) -> dict:
        if self._file is None:
            self._file = self._load_file()
        return self._file

    def _load_file(self) -> dict:
        try:
            text = self._path.read_text()
        except (FileNotFoundError, NotADirectoryError):
            return {}
        try:
            data = yaml.safe_load(text) or {}
        except yaml.YAMLError as e:
            logger.warning("could not parse config file %s: %s", self._path, e)
            return {}
        if not isinstance(data, dict):
            logger.warning("config file %s is not a mapping; ignoring", self._path)
            return {}
        cleaned: dict = {}
        for group, leaves in data.items():
            if group not in _GROUPS or not isinstance(leaves, dict):
                logger.warning("unknown config section %r in %s; ignoring", group, self._path)
                continue
            for leaf, val in leaves.items():
                if f"{group}.{leaf}" not in SETTINGS_BY_KEY:
                    logger.warning("unknown config key %r in %s; ignoring",
                                   f"{group}.{leaf}", self._path)
                    continue
                cleaned.setdefault(group, {})[leaf] = val
        return cleaned

    def _file_raw(self, setting: Setting):
        group, leaf = setting.key.split(".", 1)
        return self.file_values().get(group, {}).get(leaf, _MISSING)

    def get(self, key: str, override: Any | None = None) -> Any:
        setting = SETTINGS_BY_KEY[key]  # KeyError on unknown key (programmer error)
        if override is not None:
            return override
        if key in self._overrides:
            return self._overrides[key]
        env_raw = os.environ.get(setting.env)
        if env_raw is not None:
            return self._coerce_lenient(setting, env_raw, "env")
        file_raw = self._file_raw(setting)
        if file_raw is not _MISSING:
            # file values come from yaml already typed; still route through
            # coerce via str() so a yaml "true"/"5" and a python bool/int both work
            return self._coerce_lenient(setting, str(file_raw), "file")
        return setting.default

    def _coerce_lenient(self, setting: Setting, raw: str, origin: str) -> Any:
        try:
            return coerce(setting, raw)
        except ConfigError as e:
            logger.warning("%s; using default %r", e, setting.default)
            return setting.default

    def source(self, key: str) -> str:
        setting = SETTINGS_BY_KEY[key]
        if key in self._overrides:
            return "override"
        if os.environ.get(setting.env) is not None:
            return "env"
        if self._file_raw(setting) is not _MISSING:
            return "file"
        return "default"


_MISSING = object()
_CONFIG: Config | None = None


def get_config() -> Config:
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = Config()
    return _CONFIG


def reset_config() -> None:
    global _CONFIG
    _CONFIG = None


def get(key: str, override: Any | None = None) -> Any:
    return get_config().get(key, override=override)


def source(key: str) -> str:
    return get_config().source(key)
```

Place `_MISSING = object()` ABOVE the `Config` class (it is referenced in methods). Adjust ordering so module imports cleanly.

- [ ] **Step 4: Run tests, verify pass.**
- [ ] **Step 5: Commit.** `feat(config): Config resolver with override>env>file>default precedence`

---

### Task 3: File writers -- `render_template()` and `set_value()`

**Files:**
- Modify: `src/muse/core/config.py`
- Test: `tests/core/test_config_writers.py`

**Interfaces:**
- Produces:
  - `render_template() -> str` -- a commented `config.yaml` body: grouped sections, each setting as `  leaf: <default-as-yaml>` preceded by a `# <help> (env: MUSE_X)` comment. `None` defaults render as `null`. Bootstrap paths (catalog_dir, config_file) included but commented-out with a note that env resolves them.
  - `set_value(key: str, raw: str, *, path: pathlib.Path | None = None) -> Any` -- validate `raw` against the registry (strict: raises `ConfigError`), then read-modify-write the yaml file at `path` (or `config_path()`), creating parent dir + file if absent, preserving other keys. Returns the coerced value written.

**Strictness contract:** `set_value` is the strict path -- an un-coercible `raw` raises `ConfigError` and writes nothing.

- [ ] **Step 1: Write failing tests**

```python
# tests/core/test_config_writers.py
import pytest
import yaml
from muse.core import config as cfg


def test_template_has_every_setting():
    body = cfg.render_template()
    for s in cfg.SETTINGS:
        assert s.env in body           # env name in a comment
    # parseable once comment-only lines / commented bootstrap are stripped by yaml
    assert "server:" in body and "limits:" in body


def test_set_value_creates_and_coerces(tmp_path):
    p = tmp_path / "config.yaml"
    out = cfg.set_value("limits.rerank_max_documents", "42", path=p)
    assert out == 42
    data = yaml.safe_load(p.read_text())
    assert data["limits"]["rerank_max_documents"] == 42


def test_set_value_preserves_other_keys(tmp_path):
    p = tmp_path / "config.yaml"
    cfg.set_value("limits.rerank_max_documents", "42", path=p)
    cfg.set_value("server.gpu_headroom_gb", "2.5", path=p)
    data = yaml.safe_load(p.read_text())
    assert data["limits"]["rerank_max_documents"] == 42
    assert data["server"]["gpu_headroom_gb"] == 2.5


def test_set_value_bad_value_raises_and_no_write(tmp_path):
    p = tmp_path / "config.yaml"
    with pytest.raises(cfg.ConfigError):
        cfg.set_value("limits.rerank_max_documents", "abc", path=p)
    assert not p.exists()


def test_set_value_unknown_key_raises(tmp_path):
    with pytest.raises(KeyError):
        cfg.set_value("no.such.key", "1", path=tmp_path / "config.yaml")
```

- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement `render_template` + `set_value`.**

```python
def render_template() -> str:
    lines = ["# muse configuration (~/.muse/config.yaml)",
             "# Precedence: MUSE_* env var > this file > built-in default.",
             "# Generated by `muse config generate`; edit freely.", ""]
    bootstrap = {"paths.catalog_dir", "paths.config_file"}
    for group in sorted({s.group for s in SETTINGS}):
        lines.append(f"{group}:")
        for s in [x for x in SETTINGS if x.group == group]:
            leaf = s.key.split(".", 1)[1]
            default_yaml = yaml.safe_dump(s.default, default_flow_style=True).strip()
            lines.append(f"  # {s.help} (env: {s.env})")
            if s.key in bootstrap:
                lines.append(f"  # {leaf}: {default_yaml}   # resolved from env/default; file cannot set its own path")
            else:
                lines.append(f"  {leaf}: {default_yaml}")
        lines.append("")
    return "\n".join(lines)


def set_value(key: str, raw: str, *, path: pathlib.Path | None = None) -> Any:
    setting = SETTINGS_BY_KEY[key]           # KeyError on unknown
    value = coerce(setting, raw)             # strict: raises ConfigError
    target = path if path is not None else config_path()
    data = {}
    if target.exists():
        data = yaml.safe_load(target.read_text()) or {}
    group, leaf = key.split(".", 1)
    data.setdefault(group, {})[leaf] = value
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(yaml.safe_dump(data, default_flow_style=False, sort_keys=True))
    tmp.replace(target)                      # atomic write-then-rename
    return value
```

- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit.** `feat(config): render_template + atomic set_value writers`

---

### Task 4: `muse config` CLI group

**Files:**
- Create: `src/muse/cli_impl/config_cmd.py`
- Modify: `src/muse/cli.py` (register a `config` typer sub-app)
- Test: `tests/cli_impl/test_config_cli.py`

**Interfaces:**
- Consumes: `muse.core.config` (`SETTINGS`, `get`, `source`, `render_template`, `set_value`, `config_path`, `get_config`, `reset_config`, `ConfigError`).
- Produces CLI verbs (mirror `muse models`): `generate [--force]`, `show [--json]`, `path`, `get <key>`, `set <key> <value>`.

**Follow the CLI conventions:** thin typer command functions in `cli.py`; logic in `cli_impl/config_cmd.py`. TTY -> `rich.Table` via `cli_impl/console.py`; non-TTY / `--json` -> plain / JSON. Errors -> `raise typer.Exit(<nonzero>)` (propagates via `main`, per v0.48.0). **Redact `admin.token` in `show`** (display `set`/`unset`, never the value).

- [ ] **Step 1: Write failing tests** (in-process via typer's `CliRunner` on the sub-app, mirroring existing CLI tests):

```python
# tests/cli_impl/test_config_cli.py
import json
import yaml
from typer.testing import CliRunner
from muse.core import config as cfg
from muse.cli import app

runner = CliRunner()


def test_config_path(monkeypatch, tmp_path):
    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
    cfg.reset_config()
    r = runner.invoke(app, ["config", "path"])
    assert r.exit_code == 0
    assert str(tmp_path / "config.yaml") in r.stdout


def test_config_generate_writes(monkeypatch, tmp_path):
    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
    cfg.reset_config()
    r = runner.invoke(app, ["config", "generate"])
    assert r.exit_code == 0
    body = (tmp_path / "config.yaml").read_text()
    assert "server:" in body
    # refuses overwrite without --force
    r2 = runner.invoke(app, ["config", "generate"])
    assert r2.exit_code != 0
    r3 = runner.invoke(app, ["config", "generate", "--force"])
    assert r3.exit_code == 0


def test_config_get(monkeypatch, tmp_path):
    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
    monkeypatch.setenv("MUSE_RERANK_MAX_DOCUMENTS", "5")
    cfg.reset_config()
    r = runner.invoke(app, ["config", "get", "limits.rerank_max_documents"])
    assert r.exit_code == 0 and "5" in r.stdout


def test_config_set_then_show_json(monkeypatch, tmp_path):
    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
    cfg.reset_config()
    r = runner.invoke(app, ["config", "set", "server.gpu_headroom_gb", "2.5"])
    assert r.exit_code == 0
    data = yaml.safe_load((tmp_path / "config.yaml").read_text())
    assert data["server"]["gpu_headroom_gb"] == 2.5
    cfg.reset_config()
    r2 = runner.invoke(app, ["config", "show", "--json"])
    assert r2.exit_code == 0
    rows = json.loads(r2.stdout)
    row = next(x for x in rows if x["key"] == "server.gpu_headroom_gb")
    assert row["value"] == 2.5 and row["source"] == "file"


def test_config_set_bad_value_nonzero(monkeypatch, tmp_path):
    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
    cfg.reset_config()
    r = runner.invoke(app, ["config", "set", "limits.rerank_max_documents", "abc"])
    assert r.exit_code != 0


def test_config_show_redacts_admin_token(monkeypatch, tmp_path):
    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
    monkeypatch.setenv("MUSE_ADMIN_TOKEN", "super-secret-value")
    cfg.reset_config()
    r = runner.invoke(app, ["config", "show", "--json"])
    assert "super-secret-value" not in r.stdout
```

- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement `config_cmd.py`** (functions: `run_path()`, `run_generate(force)`, `run_get(key)`, `run_set(key, value)`, `run_show(as_json)`), and wire a typer sub-app in `cli.py`. `show` builds rows `{key, value, source, env}`; for `admin.token` replace `value` with `"set"`/`"unset"` (never the token). `reset_config()` at the start of each command so a just-written file / env is seen. Unknown key -> message + `typer.Exit(2)`; `ConfigError` from `set` -> message + `typer.Exit(2)`; generate-over-existing without `--force` -> message + `typer.Exit(1)`.
- [ ] **Step 4: Run, verify pass;** also run the whole CLI suite (`pytest tests/cli_impl/ tests/test_cli.py -q -m "not slow"`).
- [ ] **Step 5: Commit.** `feat(cli): muse config generate/show/path/get/set`

---

### Task 5: Migrate server/supervisor/director/paths/fetch/admin; wire budgets+headroom

**Files (modify + their tests):**
- `src/muse/cli_impl/supervisor.py` (idle sweep/timeout reads; construct `LoadDirector` with config-derived budgets/headroom)
- `src/muse/cli_impl/serve_util.py:43` (shutdown grace)
- `src/muse/core/catalog.py:129,398` (models_dir, catalog_dir)
- `src/muse/core/discovery.py:129,391` and `src/muse/cli_impl/worker.py:37` (modalities_dir)
- `src/muse/modalities/audio_speech/backends/base.py:14` (home)
- `src/muse/core/net_fetch.py:73` (allow_private)
- `src/muse/mcp/binary_io.py:117` (mcp_allowed_path_prefixes)
- `src/muse/admin/client.py`, `src/muse/cli.py`, `src/muse/cli_impl/mcp_server.py`, `src/muse/mcp/client.py`, `src/muse/mcp/server.py` (admin.token)

**Transform (per site):** replace `os.environ.get("MUSE_X", "D")` with `from muse.core import config` + `config.get("group.key")`. Keep each site's surrounding logic (the `<=0` guards, the None checks) unchanged -- `config.get` returns the SAME coerced value the old inline parse produced for valid input. Delete now-dead local parse/try-except that only guarded a bad env value (the registry handles it), BUT keep any semantic guard (e.g. supervisor's "idle_timeout <= 0 disables").

**Budget/headroom wiring (the doc-drift fix):** at `supervisor.py:942` `LoadDirector(...)` construction, pass:
```python
gpu_budget_gb=config.get("server.gpu_budget_gb"),
cpu_budget_gb=config.get("server.cpu_budget_gb"),
gpu_headroom_gb=config.get("server.gpu_headroom_gb"),
cpu_headroom_gb=config.get("server.cpu_headroom_gb"),
```
Defaults (None/None/1.0/2.0) equal the current `LoadDirector.__init__` defaults, so nothing changes unless an operator sets the env/file.

**Idle-timeout default flip:** the supervisor's current `_raw_default_idle` block (supervisor.py:1023-1030) that parses `MUSE_DEFAULT_IDLE_TIMEOUT_SECONDS` and defaults to None becomes `config.get("server.idle_timeout_seconds")` -- which now defaults to 600.0. Keep the "invalid -> off" log intent via the registry's lenient warn (which returns the 600 default; if an operator explicitly wants OFF they set 0, and the existing `<=0 disables` guard in `IdleSweeper` handles 0). Update the supervisor log line to report the resolved value.

**`server.device` wiring (new):** find where `muse serve --device` is defined (typer option in `cli.py`) and where its value flows into the supervisor / worker kwargs / `LoadDirector`. Flip the typer option default from `"auto"` to `None`, and at the point the effective device is resolved, compute `device = config.get("server.device", override=<cli --device value or None>)`. The registry default `"auto"` preserves current behavior when neither the flag nor config is set; an explicit `--device cuda` still wins (it is the override arg). Add a test that (a) with no flag and `MUSE_DEVICE=cuda` set, the resolved device is `cuda`, and (b) an explicit `--device cpu` beats `MUSE_DEVICE=cuda`.

- [ ] **Step 1: Write failing tests** in `tests/cli_impl/test_config_integration_server.py`:

```python
import pytest
from muse.core import config as cfg


@pytest.fixture(autouse=True)
def _reset():
    cfg.reset_config()
    yield
    cfg.reset_config()


def test_director_gets_headroom_from_config(monkeypatch):
    monkeypatch.setenv("MUSE_GPU_HEADROOM_GB", "3.0")
    cfg.reset_config()
    from muse.cli_impl import supervisor
    director = supervisor.build_load_director(...)   # use the real constructor entry
    assert director.gpu_headroom_gb == 3.0


def test_default_idle_timeout_is_600(monkeypatch):
    monkeypatch.delenv("MUSE_DEFAULT_IDLE_TIMEOUT_SECONDS", raising=False)
    cfg.reset_config()
    assert cfg.get("server.idle_timeout_seconds") == 600.0
```

(Adapt `build_load_director(...)` to the actual factory at supervisor.py:942 -- name the function the implementer finds; if the director is built inline in `run_supervisor`, extract a tiny testable factory as part of this task and call it from both places.)

- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Apply the migrations + budget/headroom wiring + idle default.** Confirm the existing idle-sweeper / director tests still pass unchanged.
- [ ] **Step 4: Run** `pytest tests/cli_impl/ tests/core/test_catalog*.py tests/core/test_discovery*.py -q -m "not slow"`.
- [ ] **Step 5: Commit.** `refactor(config): route server/paths/fetch/admin through the registry; wire budgets+headroom`

---

### Task 6: Migrate per-modality request-cap sites (limits.*)

**Files (modify + touch each modality's route test):** every `limits.*` site in the spec inventory table:
`image_generation/image_input.py`, `audio_classification/routes.py`, `audio_embedding/routes.py`, `audio_transcription/routes.py`, `embedding_text/routes.py`, `image_embedding/routes.py`, `image_segmentation/routes.py`, `image_upscale/routes.py`, `model_3d_generation/routes.py`, `text_classification/routes.py` (3 caps), `text_rerank/routes.py` (3 caps), `text_summarization/routes.py`, `video_generation/routes.py`.

**Transform, preserving read timing:**
- Import-time constant sites (`_MAX = int(os.environ.get("MUSE_X", "D"))`): become `_MAX = config.get("limits.x")` at module top. (Behavior on a bad env value improves from import-crash to warn+default -- acceptable per spec.)
- Per-request function sites (`def _cap(): ... os.environ.get(...) ... return`): the whole function body collapses to `return config.get("limits.x")`. Delete the now-redundant local parse + warn + fallback (the registry does exactly that). Keep any downstream error message that references the env name (update the message to name the env var via `config.SETTINGS_BY_KEY["limits.x"].env` OR keep the literal string -- either is fine; literal is simplest).

- [ ] **Step 1: Add/adjust a focused test per representative modality** proving the cap reads from config. Pattern:

```python
# in the relevant modality's route test
def test_upscale_cap_reads_config(monkeypatch):
    from muse.core import config as cfg
    monkeypatch.setenv("MUSE_UPSCALE_MAX_INPUT_SIDE", "77")
    cfg.reset_config()
    from muse.modalities.image_upscale import routes
    assert routes._max_input_side() == 77   # name matches the actual accessor
```

Cover at least: one opt_int byte-cap (image_input), one per-request int (upscale), one import-time const (embeddings_max_batch), one multi-cap file (rerank). Existing route tests that monkeypatch these env vars MUST still pass (they exercise the live-env path); add `cfg.reset_config()` where a test sets the env after import if needed.

- [ ] **Step 2: Run, verify new tests fail** (accessors still read os.environ).
- [ ] **Step 3: Apply the transform at every `limits.*` site.**
- [ ] **Step 4: Run** `pytest tests/modalities/ -q -m "not slow"`.
- [ ] **Step 5: Commit.** `refactor(config): route per-modality request caps through the registry`

---

### Task 7: Migrate client + CLI + MCP `MUSE_SERVER` / `MUSE_ADMIN_TOKEN` reads

**Files (modify):** every `client.py` reading `MUSE_SERVER` (audio_speech, audio_transcription, audio_embedding, audio_classification, audio_generation, chat_completion, embedding_text, image_generation, image_animation, image_cv, image_embedding, image_ocr, image_segmentation, image_upscale, model_3d_generation, text_classification (2), text_rerank, text_summarization, video_generation), plus `cli.py:304`, `cli_impl/runtime_state.py:34`, `cli_impl/mcp_server.py:55`, `mcp/client.py`, `mcp/server.py`, `admin/client.py` (server + token).

**Transform:** `arg or os.environ.get("MUSE_SERVER", "http://localhost:8000")` becomes `arg or config.get("client.server_url")`. `token or os.environ.get("MUSE_ADMIN_TOKEN")` becomes `token or config.get("admin.token")`. Preserve the explicit-arg-wins precedence (arg is truthy-checked first, exactly as today).

- [ ] **Step 1: Add a focused test** (one client + admin token):

```python
def test_client_server_url_from_config(monkeypatch):
    from muse.core import config as cfg
    monkeypatch.setenv("MUSE_SERVER", "http://box:9000")
    cfg.reset_config()
    from muse.modalities.embedding_text.client import EmbeddingsClient
    assert EmbeddingsClient().server_url.rstrip("/") == "http://box:9000"
```

Existing client tests that set `MUSE_SERVER` must still pass (add `cfg.reset_config()` if the test sets the env inside the test body after import).

- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Apply the transform at every client/CLI/MCP site.**
- [ ] **Step 4: Run** `pytest tests/ -q -m "not slow"` (whole fast lane -- this touches many modules).
- [ ] **Step 5: Commit.** `refactor(config): route client/CLI/MCP server-url + admin-token through the registry`

---

### Task 8: Meta-test guard -- no stray `MUSE_*` env reads

**Files:**
- Create: `tests/core/test_no_stray_env_reads.py`

**Interface:** AST-walk (or line-scan) every `.py` under `src/muse/`, flag any `os.environ.get("MUSE_...`, `os.environ["MUSE_...`, or `os.getenv("MUSE_...`. Allow ONLY `src/muse/core/config.py` (the two bootstrap reads: `MUSE_CATALOG_DIR`, `MUSE_CONFIG`). Mirrors `tests/core/test_runtime_helpers_meta.py`.

- [ ] **Step 1: Write the test** (it should PASS if Tasks 5-7 are complete):

```python
# tests/core/test_no_stray_env_reads.py
import ast
import pathlib

SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "muse"
ALLOW = {"core/config.py"}  # only the bootstrap reads live here


def _muse_env_reads(tree) -> list[str]:
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # os.environ.get("MUSE_...") / os.getenv("MUSE_...")
        args = node.args
        if not args or not isinstance(args[0], ast.Constant):
            continue
        if not (isinstance(args[0].value, str) and args[0].value.startswith("MUSE_")):
            continue
        fn = node.func
        if isinstance(fn, ast.Attribute) and fn.attr in {"get", "getenv"}:
            hits.append(args[0].value)
    # os.environ["MUSE_..."] subscripts
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            v = node.slice.value
            if isinstance(v, str) and v.startswith("MUSE_"):
                hits.append(v)
    return hits


def test_no_stray_muse_env_reads_outside_config():
    offenders = {}
    for path in SRC.rglob("*.py"):
        rel = path.relative_to(SRC).as_posix()
        if rel in ALLOW:
            continue
        hits = _muse_env_reads(ast.parse(path.read_text()))
        if hits:
            offenders[rel] = hits
    assert not offenders, f"stray MUSE_* env reads (route via muse.core.config): {offenders}"
```

- [ ] **Step 2: Run it.** If it FAILS, the failure names the remaining sites -- migrate them (they belong to Tasks 5-7; fix in place) until green. Do not weaken the test or widen `ALLOW` beyond `core/config.py`.
- [ ] **Step 3: Run** the full fast lane `pytest tests/ -q -m "not slow"`.
- [ ] **Step 4: Commit.** `test(config): guard against stray MUSE_* env reads`

---

### Task 9: Documentation

**Files:**
- Modify: `CLAUDE.md` (rewrite the scattered env-var references into a "Configuration" section: the registry, precedence, config.yaml, the `muse config` verbs; fix the budget/headroom section to say they are now live; note the idle-timeout default is 600s). Keep existing per-modality env-var mentions but point them at `muse config`.
- Modify: `README.md` (short "Configuration" subsection: `muse config generate`, precedence one-liner).
- Create: `docs/CONFIG.md` (the full settings table -- can be generated from `render_template()` output; include the precedence rules and the lenient/strict policy).

**No version bump, no tag, no publish.** Those are gated on an explicit "go".

- [ ] **Step 1: Update the docs.** Verify every setting in `docs/CONFIG.md` matches `SETTINGS` (copy from `render_template()`).
- [ ] **Step 2: Run** `pytest tests/ -q -m "not slow"` one final time.
- [ ] **Step 3: Commit.** `docs(config): document the settings registry, config.yaml, and muse config`

---

## Self-Review

- **Spec coverage:** registry (T1), precedence/lenient/singleton (T2), writers (T3), CLI (T4), server+budgets+idle-default (T5), limits (T6), client/admin (T7), meta-guard (T8), docs (T9). Every inventory row is migrated across T5-T7; the meta-test (T8) proves completeness.
- **Read-timing preserved:** T6 explicitly splits import-time constants vs per-request functions.
- **Bootstrap cycle avoided:** `_catalog_dir`/`config_path` in T1 read env directly and never import `catalog.py`; T8 `ALLOW`s exactly those two reads.
- **Type consistency:** `config.get(key, override=None)`, `config.source(key)`, `set_value(key, raw, *, path=None)`, `render_template()` names are identical across T2-T4 tests and impl.
- **No placeholder steps:** new-code tasks carry full code; migration tasks carry the exact transform + the spec's enumerated site table.
