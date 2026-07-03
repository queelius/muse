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
