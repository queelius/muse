"""Tests for discover_hf_plugins: per-modality HF plugin loader."""
from pathlib import Path

import pytest

from muse.core.discovery import discover_hf_plugins, REQUIRED_HF_PLUGIN_KEYS


def _write_plugin(modality_dir: Path, plugin_dict_literal: str) -> None:
    """Helper: write a hf.py with a given HF_PLUGIN literal into a temp dir."""
    modality_dir.mkdir(parents=True, exist_ok=True)
    (modality_dir / "__init__.py").write_text("")
    (modality_dir / "hf.py").write_text(plugin_dict_literal)


def test_required_keys_constant_complete():
    assert set(REQUIRED_HF_PLUGIN_KEYS) == {
        "modality", "runtime_path", "pip_extras", "system_packages",
        "priority", "sniff", "resolve", "search",
    }


def test_discovers_valid_plugin(tmp_path):
    _write_plugin(tmp_path / "audio_transcription", '''
HF_PLUGIN = {
    "modality": "audio/transcription",
    "runtime_path": "muse.modalities.audio_transcription.runtimes.faster_whisper:FasterWhisperModel",
    "pip_extras": ("faster-whisper>=1.0.0",),
    "system_packages": ("ffmpeg",),
    "priority": 100,
    "sniff": lambda info: True,
    "resolve": lambda repo_id, variant, info: None,
    "search": lambda api, query, **kw: iter(()),
}
''')
    plugins = discover_hf_plugins([tmp_path])
    assert len(plugins) == 1
    p = plugins[0]
    assert p["modality"] == "audio/transcription"
    assert p["priority"] == 100
    assert callable(p["sniff"])


def test_skips_plugin_with_missing_required_key(tmp_path, caplog):
    _write_plugin(tmp_path / "broken", '''
HF_PLUGIN = {
    "modality": "x/y",
    "runtime_path": "a:B",
    "pip_extras": (),
    "system_packages": (),
    # priority intentionally missing
    "sniff": lambda info: True,
    "resolve": lambda *a, **kw: None,
    "search": lambda *a, **kw: iter(()),
}
''')
    import logging
    with caplog.at_level(logging.WARNING):
        plugins = discover_hf_plugins([tmp_path])
    assert plugins == []
    assert any("missing required keys" in r.message for r in caplog.records)


def test_skips_plugin_with_no_hf_plugin_attr(tmp_path, caplog):
    _write_plugin(tmp_path / "noattr", "x = 1\n")
    import logging
    with caplog.at_level(logging.WARNING):
        plugins = discover_hf_plugins([tmp_path])
    assert plugins == []
    assert any("no top-level HF_PLUGIN" in r.message for r in caplog.records)


def test_skips_plugin_with_syntax_error(tmp_path, caplog):
    (tmp_path / "syntax").mkdir()
    (tmp_path / "syntax" / "__init__.py").write_text("")
    (tmp_path / "syntax" / "hf.py").write_text("def def def\n")
    import logging
    with caplog.at_level(logging.WARNING):
        plugins = discover_hf_plugins([tmp_path])
    assert plugins == []
    assert any("import failed" in r.message for r in caplog.records)


def test_orders_by_priority_then_modality(tmp_path):
    _write_plugin(tmp_path / "alpha", '''
HF_PLUGIN = {
    "modality": "z/last", "runtime_path": "a:B",
    "pip_extras": (), "system_packages": (),
    "priority": 200,
    "sniff": lambda info: False,
    "resolve": lambda *a, **kw: None,
    "search": lambda *a, **kw: iter(()),
}
''')
    _write_plugin(tmp_path / "beta", '''
HF_PLUGIN = {
    "modality": "a/first", "runtime_path": "a:B",
    "pip_extras": (), "system_packages": (),
    "priority": 100,
    "sniff": lambda info: False,
    "resolve": lambda *a, **kw: None,
    "search": lambda *a, **kw: iter(()),
}
''')
    _write_plugin(tmp_path / "gamma", '''
HF_PLUGIN = {
    "modality": "b/second", "runtime_path": "a:B",
    "pip_extras": (), "system_packages": (),
    "priority": 100,
    "sniff": lambda info: False,
    "resolve": lambda *a, **kw: None,
    "search": lambda *a, **kw: iter(()),
}
''')
    plugins = discover_hf_plugins([tmp_path])
    assert [p["modality"] for p in plugins] == ["a/first", "b/second", "z/last"]


def test_returns_empty_when_dirs_empty():
    assert discover_hf_plugins([]) == []


def test_skips_nonexistent_dirs(tmp_path):
    bogus = tmp_path / "does-not-exist"
    assert discover_hf_plugins([bogus]) == []


def test_default_dirs_includes_bundled():
    from muse.core.discovery import _default_hf_plugin_dirs
    dirs = _default_hf_plugin_dirs()
    assert dirs[0].name == "modalities"
    assert dirs[0].is_dir()
