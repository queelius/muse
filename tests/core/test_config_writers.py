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


def test_template_is_valid_yaml_and_roundtrips():
    import yaml
    from muse.core import config as cfg
    body = cfg.render_template()
    # no bare document-end markers
    assert not any(line.strip() == "..." for line in body.splitlines())
    data = yaml.safe_load(body)          # must parse without raising
    assert isinstance(data, dict)
    # active (non-bootstrap) settings round-trip to their declared default
    for key in ("server.idle_timeout_seconds", "limits.rerank_max_documents",
                "client.server_url", "fetch.allow_private"):
        group, leaf = key.split(".", 1)
        assert data[group][leaf] == cfg.SETTINGS_BY_KEY[key].default
    # bootstrap keys are commented out -> NOT present as active keys
    assert "catalog_dir" not in data.get("paths", {})
    assert "config_file" not in data.get("paths", {})


# --- unset_value: remove a key so it falls back to env/default ---

def test_unset_value_removes_key_preserves_others(tmp_path):
    import yaml
    from muse.core import config as cfg
    p = tmp_path / "config.yaml"
    cfg.set_value("limits.rerank_max_documents", "42", path=p)
    cfg.set_value("server.gpu_headroom_gb", "2.5", path=p)
    assert cfg.unset_value("limits.rerank_max_documents", path=p) is True
    data = yaml.safe_load(p.read_text())
    assert "rerank_max_documents" not in data.get("limits", {})
    assert data["server"]["gpu_headroom_gb"] == 2.5


def test_unset_value_prunes_empty_group(tmp_path):
    import yaml
    from muse.core import config as cfg
    p = tmp_path / "config.yaml"
    cfg.set_value("limits.rerank_max_documents", "42", path=p)
    cfg.unset_value("limits.rerank_max_documents", path=p)
    data = yaml.safe_load(p.read_text()) or {}
    assert "limits" not in data


def test_unset_value_absent_key_is_noop(tmp_path):
    import yaml
    from muse.core import config as cfg
    p = tmp_path / "config.yaml"
    cfg.set_value("server.gpu_headroom_gb", "2.5", path=p)
    assert cfg.unset_value("limits.rerank_max_documents", path=p) is False
    assert yaml.safe_load(p.read_text())["server"]["gpu_headroom_gb"] == 2.5


def test_unset_value_no_file_is_noop(tmp_path):
    from muse.core import config as cfg
    p = tmp_path / "config.yaml"
    assert cfg.unset_value("server.gpu_headroom_gb", path=p) is False
    assert not p.exists()


def test_unset_value_unknown_key_raises(tmp_path):
    import pytest
    from muse.core import config as cfg
    with pytest.raises(KeyError):
        cfg.unset_value("no.such.key", path=tmp_path / "config.yaml")


# --- singleton reset: writes to the ACTIVE config path must invalidate
# the process-wide Config singleton so a later config.get() in the SAME
# process sees the new value instead of a stale cached parse. Writes to
# an explicit non-active test path (all the tests above) must NOT reset
# the singleton.


@pytest.fixture
def _reset_singleton():
    from muse.core import config as cfg
    cfg.reset_config()
    yield
    cfg.reset_config()


def test_set_value_default_path_resets_singleton_for_get(
    tmp_path, monkeypatch, _reset_singleton,
):
    from muse.core import config as cfg
    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
    monkeypatch.delenv("MUSE_CONFIG", raising=False)
    monkeypatch.delenv("MUSE_GPU_HEADROOM_GB", raising=False)

    # Prime the singleton with the default value before any write.
    assert cfg.get("server.gpu_headroom_gb") == 1.0

    cfg.set_value("server.gpu_headroom_gb", "3.5")  # default (active) path

    assert cfg.get("server.gpu_headroom_gb") == 3.5, (
        "set_value on the active config path must reset the module "
        "singleton so a same-process get() sees the new value"
    )


def test_unset_value_default_path_resets_singleton_for_get(
    tmp_path, monkeypatch, _reset_singleton,
):
    from muse.core import config as cfg
    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
    monkeypatch.delenv("MUSE_CONFIG", raising=False)
    monkeypatch.delenv("MUSE_GPU_HEADROOM_GB", raising=False)

    cfg.set_value("server.gpu_headroom_gb", "3.5")  # active path, resets singleton
    assert cfg.get("server.gpu_headroom_gb") == 3.5

    cfg.unset_value("server.gpu_headroom_gb")  # active path

    assert cfg.get("server.gpu_headroom_gb") == 1.0, (
        "unset_value on the active config path must reset the module "
        "singleton so a same-process get() reverts to the default"
    )


def test_set_value_explicit_test_path_does_not_reset_singleton(
    tmp_path, monkeypatch, _reset_singleton,
):
    """Writing to an explicit, non-active path (the common test pattern
    used throughout this file) must NOT clobber the process singleton --
    only writes to the resolved active config_path() do."""
    from muse.core import config as cfg
    monkeypatch.delenv("MUSE_CATALOG_DIR", raising=False)
    monkeypatch.delenv("MUSE_CONFIG", raising=False)
    monkeypatch.delenv("MUSE_GPU_HEADROOM_GB", raising=False)

    # Prime the singleton before touching an unrelated explicit path.
    assert cfg.get("server.gpu_headroom_gb") == 1.0

    other_path = tmp_path / "unrelated.yaml"
    cfg.set_value("server.gpu_headroom_gb", "9.0", path=other_path)

    assert cfg.get("server.gpu_headroom_gb") == 1.0, (
        "a write to an explicit non-active path must not reset the "
        "process singleton"
    )
