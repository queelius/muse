from pathlib import Path
from muse.federation.nodes import NodeSpec, load_nodes


def test_cli_nodes_plain_and_named():
    nodes = load_nodes(cli_nodes=["http://a:8000/", "b=http://b:8000"])
    assert nodes[0] == NodeSpec(url="http://a:8000", name="a", token=None)  # trailing slash stripped, name from host
    assert nodes[1] == NodeSpec(url="http://b:8000", name="b", token=None)


def test_yaml_nodes_and_dedup(tmp_path):
    p = tmp_path / "federation.yaml"
    p.write_text("nodes:\n  - {url: http://a:8000, name: alpha, token: t1}\n  - url: http://a:8000\n")
    nodes = load_nodes(config_path=p)
    assert len(nodes) == 1 and nodes[0].name == "alpha" and nodes[0].token == "t1"  # dedup by url, first wins


def test_config_setting_defaults():
    from muse.core import config
    assert config.get("federation.refresh_interval_seconds") == 3.0
