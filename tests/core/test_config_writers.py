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
